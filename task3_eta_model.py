import argparse
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless-safe: write PNGs without a display / blocking show()
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import pickle
import networkx as nx
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH = "delivery_data.csv"
NODE_METRICS_PATH = "outputs/node_metrics.csv"
OUTPUT_DIR = "outputs"
MODEL_DIR = "models"
GRAPH_PICKLE = os.path.join(OUTPUT_DIR, "logistics_graph.pkl")
EMBEDDINGS_PATH = os.path.join(OUTPUT_DIR, "node_emb_graphsage.csv")

TARGET = "actual_time"
WITHIN_PCT = 0.15         # business accuracy threshold: within 15% of actual
RANDOM_STATE = 42

BASELINE_MODEL_PATH = os.path.join(MODEL_DIR, "baseline_eta_xgb.joblib")
GRAPH_MODEL_PATH = os.path.join(MODEL_DIR, "graph_eta_xgb.joblib")
COMPARISON_PATH = os.path.join(OUTPUT_DIR, "model_comparison.csv")
IMPORTANCE_PLOT_PATH = "feature_importance.png"

XGB_PARAMS = dict(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=RANDOM_STATE,
    verbosity=0,
)

BASELINE_FEATURES = [
    "osrm_time",
    "osrm_distance",
    "is_ftl",
    "hour_of_day",
    "day_of_week",
]

# Source/destination hub metrics pulled from the logistics graph.
GRAPH_METRIC_COLS = [
    "src_betweenness", "src_in_degree", "src_out_degree",
    "src_avg_delay", "src_bottleneck_score",
    "dst_betweenness", "dst_in_degree", "dst_out_degree",
    "dst_avg_delay", "dst_bottleneck_score",
]
EMB_DIM = 8
SRC_EMB_COLS = [f"src_emb_{i}" for i in range(EMB_DIM)]
DST_EMB_COLS = [f"dst_emb_{i}" for i in range(EMB_DIM)]
GRAPH_FEATURES = BASELINE_FEATURES + GRAPH_METRIC_COLS + SRC_EMB_COLS + DST_EMB_COLS


# ---------------------------------------------------------------------------
# 1. Load & feature engineering
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(NODE_METRICS_PATH):
        raise FileNotFoundError(
            f"{NODE_METRICS_PATH} not found. Run build_graph.py first - it "
            "produces the per-hub graph metrics this model depends on."
        )
    df = pd.read_csv(DATA_PATH)
    node_metrics = pd.read_csv(NODE_METRICS_PATH)
    return df, node_metrics


def merge_graph_features(df: pd.DataFrame, node_metrics: pd.DataFrame) -> pd.DataFrame:
    """Attach source- and destination-hub graph metrics to every trip row."""
    keep = ["center", "betweenness", "in_degree", "out_degree",
            "avg_incoming_delay_factor", "bottleneck_score"]
    # Shorten avg_incoming_delay_factor -> avg_delay for compact feature names.
    nm = node_metrics[keep].rename(columns={"avg_incoming_delay_factor": "avg_delay"})

    for side, join_key in [("src", "source_center"), ("dst", "destination_center")]:
        prefixed = {c: f"{side}_{c}" for c in nm.columns if c != "center"}
        df = df.merge(
            nm.rename(columns=prefixed),
            left_on=join_key, right_on="center", how="left",
        ).drop(columns="center")

    missing = df[GRAPH_METRIC_COLS].isnull().any(axis=1).sum()
    print(f"  graph metrics merged | rows missing any hub metric: {missing:,} / {len(df):,}")
    return df


def load_graph(path: str = GRAPH_PICKLE):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run build_graph.py first to produce the graph pickle.")
    with open(path, "rb") as f:
        G = pickle.load(f)
    return G


def compute_graphsage_embeddings(G: nx.DiGraph, node_metrics: pd.DataFrame, emb_dim: int = EMB_DIM) -> pd.DataFrame:
    """Train a small GraphSAGE (PyTorch) model to produce node embeddings.

    This implementation trains a 1-layer GraphSAGE that aggregates mean of
    1-hop neighbors and learns embeddings by predicting the node's
    `avg_incoming_delay_factor` (regression). The learned embeddings are the
    penultimate-layer outputs and are saved to `EMBEDDINGS_PATH`.
    """
    feat_cols = ["betweenness", "in_degree", "out_degree", "avg_incoming_delay_factor", "bottleneck_score"]
    nm = node_metrics.set_index("center")[feat_cols]

    # Node ordering
    nodes = list(G.nodes())
    idx_map = {n: i for i, n in enumerate(nodes)}

    # Feature matrix and target vector
    X = np.vstack([nm.loc[n].values if n in nm.index else np.zeros(len(feat_cols), dtype=float) for n in nodes])
    y = np.array([nm.loc[n]["avg_incoming_delay_factor"] if n in nm.index else 0.0 for n in nodes], dtype=float)

    # Neighbor lists
    nbrs: List[List[int]] = []
    for n in nodes:
        neigh = set(G.predecessors(n)) | set(G.successors(n))
        nbrs.append([idx_map[nb] for nb in neigh if nb in idx_map])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(1)

    class GraphSAGE(nn.Module):
        def __init__(self, in_dim: int, emb_dim: int):
            super().__init__()
            self.fc1 = nn.Linear(in_dim * 2, 128)
            self.fc2 = nn.Linear(128, emb_dim)
            self.reg = nn.Linear(emb_dim, 1)

        def forward(self, x, neigh_mean):
            h = torch.cat([x, neigh_mean], dim=1)
            h = torch.relu(self.fc1(h))
            emb = torch.relu(self.fc2(h))
            out = self.reg(emb)
            return emb, out

    model = GraphSAGE(X.shape[1], emb_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # Training loop
    epochs = 100
    for ep in range(epochs):
        model.train()
        # build neighbor means each epoch (simple python loops)
        neigh_means = []
        for nbr_idx in nbrs:
            if nbr_idx:
                neigh_means.append(X_t[nbr_idx].mean(dim=0, keepdim=True))
            else:
                neigh_means.append(torch.zeros(1, X_t.size(1), device=device))
        neigh_means_t = torch.cat(neigh_means, dim=0)

        emb, preds = model(X_t, neigh_means_t)
        loss = loss_fn(preds, y_t)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (ep + 1) % 25 == 0:
            print(f"    [GraphSAGE] epoch {ep+1}/{epochs} loss={loss.item():.6f}")

    # Compute final embeddings and save
    model.eval()
    with torch.no_grad():
        neigh_means = []
        for nbr_idx in nbrs:
            if nbr_idx:
                neigh_means.append(X_t[nbr_idx].mean(dim=0, keepdim=True))
            else:
                neigh_means.append(torch.zeros(1, X_t.size(1), device=device))
        neigh_means_t = torch.cat(neigh_means, dim=0)
        embeddings, _ = model(X_t, neigh_means_t)

    emb_np = embeddings.cpu().numpy()
    emb_df = pd.DataFrame(emb_np, columns=[f"emb_{i}" for i in range(emb_np.shape[1])])
    emb_df["center"] = nodes
    emb_df = emb_df[["center"] + [c for c in emb_df.columns if c != "center"]]
    emb_df.to_csv(EMBEDDINGS_PATH, index=False)
    print(f"  Trained GraphSAGE embeddings saved -> {EMBEDDINGS_PATH} (dim={emb_np.shape[1]})")
    return emb_df


def merge_embeddings(df: pd.DataFrame, emb_df: pd.DataFrame) -> pd.DataFrame:
    """Attach src_ and dst_ embeddings (prefix emb columns) to trip rows."""
    if emb_df is None or emb_df.empty:
        return df

    emb_cols = [c for c in emb_df.columns if c != "center"]
    # merge source
    df = df.merge(
        emb_df.rename(columns={c: f"src_{c}" for c in emb_cols}),
        left_on="source_center", right_on="center", how="left",
    ).drop(columns="center")
    # merge destination
    df = df.merge(
        emb_df.rename(columns={c: f"dst_{c}" for c in emb_cols}),
        left_on="destination_center", right_on="center", how="left",
    ).drop(columns="center")

    # Report any missing embeddings
    src_missing = df[[f"src_{c}" for c in emb_cols]].isnull().any(axis=1).sum()
    dst_missing = df[[f"dst_{c}" for c in emb_cols]].isnull().any(axis=1).sum()
    print(f"  src embeddings missing: {src_missing:,} | dst embeddings missing: {dst_missing:,}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df["trip_creation_time"] = pd.to_datetime(df["trip_creation_time"], errors="coerce")
    df["hour_of_day"] = df["trip_creation_time"].dt.hour
    df["day_of_week"] = df["trip_creation_time"].dt.dayofweek
    df["is_ftl"] = (df["route_type"] == "FTL").astype(int)  # FTL=1, Carting=0
    return df


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["data"] == "training"].copy()
    test = df[df["data"] == "test"].copy()
    print(f"  train: {len(train):,} rows | test: {len(test):,} rows")
    return train, test


# ---------------------------------------------------------------------------
# 2. Train / evaluate / persist
# ---------------------------------------------------------------------------

def build_design_matrix(frame: pd.DataFrame, features: list, fill_values: dict) -> pd.DataFrame:
    """Select feature columns and fill NaNs per the model's fill strategy."""
    return frame[features].copy().fillna(fill_values)


def evaluate(y_true, y_pred) -> dict:
    """MAE plus % of predictions within WITHIN_PCT of actual."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true > 0
    within = np.mean(np.abs(y_pred[mask] - y_true[mask]) / y_true[mask] <= WITHIN_PCT) * 100
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "within_15_pct": float(within),
    }


def train_model(name, features, fill_values, train, test) -> dict:
    """Fit one XGBoost regressor and bundle it with everything needed to reuse it."""
    X_train = build_design_matrix(train, features, fill_values)
    X_test = build_design_matrix(test, features, fill_values)
    y_train = train[TARGET]
    y_test = test[TARGET]

    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train)
    metrics = evaluate(y_test, model.predict(X_test))

    return {
        "name": name,
        "model": model,
        "features": features,
        "fill_values": fill_values,
        "target": TARGET,
        "within_pct": WITHIN_PCT,
        "xgb_params": XGB_PARAMS,
        "metrics": metrics,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }


def get_model(name, path, features, fill_values, train, test, retrain) -> dict:
    """Load a cached model artifact if present, otherwise train and save it."""
    if os.path.exists(path) and not retrain:
        print(f"  [{name}] loading cached model -> {path}")
        return joblib.load(path)

    print(f"  [{name}] training...")
    artifact = train_model(name, features, fill_values, train, test)
    joblib.dump(artifact, path)
    print(f"  [{name}] saved -> {path}  "
          f"(MAE {artifact['metrics']['mae']:.2f}, "
          f"within15 {artifact['metrics']['within_15_pct']:.1f}%)")
    return artifact


# ---------------------------------------------------------------------------
# 3. Reporting
# ---------------------------------------------------------------------------

def report_comparison(baseline_art: dict, graph_art: dict) -> pd.DataFrame:
    b, g = baseline_art["metrics"], graph_art["metrics"]
    comparison = pd.DataFrame(
        [
            {"model": "baseline", "n_features": len(baseline_art["features"]), **b},
            {"model": "graph_enhanced", "n_features": len(graph_art["features"]), **g},
        ]
    )

    mae_impr = (b["mae"] - g["mae"]) / b["mae"] * 100
    within_impr = g["within_15_pct"] - b["within_15_pct"]
    wins_both = (g["mae"] < b["mae"]) and (g["within_15_pct"] > b["within_15_pct"])

    print("\n" + "=" * 60)
    print("MODEL COMPARISON  (evaluated on held-out test set)")
    print("=" * 60)
    print(comparison.to_string(index=False))
    print("-" * 60)
    print(f"MAE improvement        : {mae_impr:+.1f}%  "
          f"({b['mae']:.2f} -> {g['mae']:.2f} min)")
    print(f"Within-15% improvement : {within_impr:+.1f} pts  "
          f"({b['within_15_pct']:.1f}% -> {g['within_15_pct']:.1f}%)")
    print(f"Graph advantage on BOTH metrics: "
          f"{'YES - confirmed' if wins_both else 'NO'}")
    print("=" * 60)

    comparison.to_csv(COMPARISON_PATH, index=False)
    print(f"Comparison saved -> {COMPARISON_PATH}")
    return comparison


def plot_feature_importance(graph_art: dict, path: str) -> None:
    model = graph_art["model"]
    features = graph_art["features"]
    importances = model.feature_importances_
    order = np.argsort(importances)  # ascending -> barh plots biggest on top

    plt.figure(figsize=(10, 6))
    plt.barh([features[i] for i in order], importances[order], color="steelblue")
    plt.xlabel("Importance Score")
    plt.title("Feature Importance - Graph-Enhanced XGBoost Model")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Feature-importance plot saved -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/benchmark graph-enhanced ETA models.")
    p.add_argument(
        "--retrain", action="store_true",
        help="Force retraining even if cached models exist in ./models/.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    both_cached = os.path.exists(BASELINE_MODEL_PATH) and os.path.exists(GRAPH_MODEL_PATH)

    if both_cached and not args.retrain:
        # Fast path: skip the 55 MB CSV load entirely and reuse saved models.
        print("Cached models found - loading (pass --retrain to rebuild).")
        baseline_art = joblib.load(BASELINE_MODEL_PATH)
        graph_art = joblib.load(GRAPH_MODEL_PATH)
    else:
        print("Loading data...")
        df, node_metrics = load_data()
        # Merge per-node graph metrics
        df = merge_graph_features(df, node_metrics)

        # Ensure GraphSAGE-like embeddings exist (recompute when retraining or missing)
        emb_df = None
        if args.retrain or not os.path.exists(EMBEDDINGS_PATH):
            print("  computing GraphSAGE-like embeddings...")
            G = load_graph()
            emb_df = compute_graphsage_embeddings(G, node_metrics, emb_dim=EMB_DIM)
        else:
            try:
                emb_df = pd.read_csv(EMBEDDINGS_PATH)
                print(f"  loaded existing embeddings -> {EMBEDDINGS_PATH}")
            except Exception:
                print("  failed to load embeddings file; recomputing...")
                G = load_graph()
                emb_df = compute_graphsage_embeddings(G, node_metrics, emb_dim=EMB_DIM)

        # Merge embeddings into trip rows (src_/dst_ prefixed columns)
        df = merge_embeddings(df, emb_df)
        df = engineer_features(df)

        print("\nSplitting train/test...")
        train, test = split_train_test(df)

        # Fill strategy: baseline cols -> 0; graph metrics -> training median so
        # unmatched hubs don't get distorting zeros. Stored in the artifact so
        # prediction-time filling stays identical.
        graph_medians = train[GRAPH_METRIC_COLS].median().to_dict()
        baseline_fill = {f: 0 for f in BASELINE_FEATURES}
        # Embedding fill: zeros for missing embeddings
        emb_fill = {c: 0.0 for c in (SRC_EMB_COLS + DST_EMB_COLS)}
        graph_fill = {**baseline_fill, **graph_medians, **emb_fill}

        print("\nTraining / loading models...")
        baseline_art = get_model(
            "baseline", BASELINE_MODEL_PATH,
            BASELINE_FEATURES, baseline_fill, train, test, args.retrain,
        )
        graph_art = get_model(
            "graph_enhanced", GRAPH_MODEL_PATH,
            GRAPH_FEATURES, graph_fill, train, test, args.retrain,
        )

    report_comparison(baseline_art, graph_art)
    plot_feature_importance(graph_art, IMPORTANCE_PLOT_PATH)

    print("\nDone!\n")
    return baseline_art, graph_art


if __name__ == "__main__":
    baseline_art, graph_art = main()
