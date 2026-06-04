import argparse
import os
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless-safe: write PNGs without a display / blocking show()
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from sklearn.decomposition import PCA

from graphsage_eta_model import compute_graphsage_embeddings


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
GRAPHSAGE_MODEL_PATH = os.path.join(MODEL_DIR, "graphsage_eta_xgb.joblib")
GRAPHSAGE_LGBM_MODEL_PATH = os.path.join(MODEL_DIR, "graphsage_eta_lgbm.joblib")
GRAPHSAGE_MLP_MODEL_PATH = os.path.join(MODEL_DIR, "graphsage_eta_mlp.joblib")
GRAPHSAGE_RESIDUAL_MODEL_PATH = os.path.join(MODEL_DIR, "graphsage_residual_xgb.joblib")
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
GRAPHSAGE_FEATURES = BASELINE_FEATURES + SRC_EMB_COLS + DST_EMB_COLS
RESIDUAL_FEATURES = BASELINE_FEATURES + GRAPH_METRIC_COLS + SRC_EMB_COLS + DST_EMB_COLS


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


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_model(name, features, fill_values, train, test, model_kind: str = "xgb") -> dict:
    """Fit one ETA regressor and bundle timing/quality metrics for comparison."""
    X_train = build_design_matrix(train, features, fill_values)
    X_test = build_design_matrix(test, features, fill_values)
    use_residual = model_kind == "residual"

    if use_residual:
        y_train = train[TARGET] - train["osrm_time"]
        y_test = test[TARGET] - test["osrm_time"]
    else:
        y_train = train[TARGET]
        y_test = test[TARGET]

    if model_kind == "lgbm":
        model = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=-1,
            random_state=RANDOM_STATE,
            verbosity=-1,
            objective="regression_l1",
        )
        train_start = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - train_start

        infer_start = time.perf_counter()
        y_pred = model.predict(X_test)
        infer_time = time.perf_counter() - infer_start
    elif model_kind == "mlp":
        X_train_np = X_train.to_numpy(dtype=np.float32)
        X_test_np = X_test.to_numpy(dtype=np.float32)
        y_train_np = y_train.to_numpy(dtype=np.float32)
        y_test_np = y_test.to_numpy(dtype=np.float32)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MLPRegressor(X_train_np.shape[1]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        train_start = time.perf_counter()
        model.train()
        Xtr = torch.tensor(X_train_np, dtype=torch.float32, device=device)
        ytr = torch.tensor(y_train_np, dtype=torch.float32, device=device)
        for _ in range(40):
            optimizer.zero_grad()
            pred = model(Xtr)
            loss = loss_fn(pred, ytr)
            loss.backward()
            optimizer.step()
        train_time = time.perf_counter() - train_start

        infer_start = time.perf_counter()
        model.eval()
        with torch.no_grad():
            y_pred = model(torch.tensor(X_test_np, dtype=torch.float32, device=device)).cpu().numpy()
        infer_time = time.perf_counter() - infer_start
    else:
        model = XGBRegressor(**XGB_PARAMS)
        train_start = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - train_start

        infer_start = time.perf_counter()
        y_pred = model.predict(X_test)
        infer_time = time.perf_counter() - infer_start

    if use_residual:
        y_pred = test["osrm_time"].to_numpy(dtype=float) + y_pred
        y_true_for_metrics = test[TARGET].to_numpy(dtype=float)
        metrics = evaluate(y_true_for_metrics, y_pred)
    else:
        metrics = evaluate(y_test, y_pred)

    if model_kind == "mlp":
        model = model.cpu()

    return {
        "name": name,
        "model": model,
        "model_kind": model_kind,
        "features": features,
        "fill_values": fill_values,
        "target": TARGET,
        "within_pct": WITHIN_PCT,
        "xgb_params": XGB_PARAMS,
        "metrics": metrics,
        "training_time_sec": float(train_time),
        "inference_time_sec": float(infer_time),
        "overall_accuracy_pct": float(metrics["within_15_pct"]),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }


def get_model(name, path, features, fill_values, train, test, retrain, model_kind: str = "xgb") -> dict:
    """Load a cached model artifact if present, otherwise train and save it."""
    if os.path.exists(path) and not retrain:
        print(f"  [{name}] loading cached model -> {path}")
        return joblib.load(path)

    print(f"  [{name}] training...")
    artifact = train_model(name, features, fill_values, train, test, model_kind=model_kind)
    joblib.dump(artifact, path)
    print(f"  [{name}] saved -> {path}  "
          f"(MAE {artifact['metrics']['mae']:.2f}, "
          f"within15 {artifact['metrics']['within_15_pct']:.1f}%)")
    return artifact


# ---------------------------------------------------------------------------
# 3. Reporting
# ---------------------------------------------------------------------------

def report_comparison(baseline_art: dict, graph_art: dict, graphsage_art: dict, graphsage_lgbm_art: dict, graphsage_mlp_art: dict, residual_art: dict) -> pd.DataFrame:
    b, g, s, l, m, r = baseline_art["metrics"], graph_art["metrics"], graphsage_art["metrics"], graphsage_lgbm_art["metrics"], graphsage_mlp_art["metrics"], residual_art["metrics"]
    comparison = pd.DataFrame(
        [
            {"model": "baseline", "n_features": len(baseline_art["features"]), "training_time_sec": baseline_art["training_time_sec"], "inference_time_sec": baseline_art["inference_time_sec"], "overall_accuracy_pct": baseline_art["overall_accuracy_pct"], **b},
            {"model": "graph_enhanced", "n_features": len(graph_art["features"]), "training_time_sec": graph_art["training_time_sec"], "inference_time_sec": graph_art["inference_time_sec"], "overall_accuracy_pct": graph_art["overall_accuracy_pct"], **g},
            {"model": "graphsage_xgb", "n_features": len(graphsage_art["features"]), "training_time_sec": graphsage_art["training_time_sec"], "inference_time_sec": graphsage_art["inference_time_sec"], "overall_accuracy_pct": graphsage_art["overall_accuracy_pct"], **s},
            {"model": "graphsage_lgbm", "n_features": len(graphsage_lgbm_art["features"]), "training_time_sec": graphsage_lgbm_art["training_time_sec"], "inference_time_sec": graphsage_lgbm_art["inference_time_sec"], "overall_accuracy_pct": graphsage_lgbm_art["overall_accuracy_pct"], **l},
            {"model": "graphsage_mlp", "n_features": len(graphsage_mlp_art["features"]), "training_time_sec": graphsage_mlp_art["training_time_sec"], "inference_time_sec": graphsage_mlp_art["inference_time_sec"], "overall_accuracy_pct": graphsage_mlp_art["overall_accuracy_pct"], **m},
            {"model": "graphsage_residual", "n_features": len(residual_art["features"]), "training_time_sec": residual_art["training_time_sec"], "inference_time_sec": residual_art["inference_time_sec"], "overall_accuracy_pct": residual_art["overall_accuracy_pct"], **r},
        ]
    )

    mae_impr_graph = (b["mae"] - g["mae"]) / b["mae"] * 100
    mae_impr_sage = (b["mae"] - s["mae"]) / b["mae"] * 100
    mae_impr_lgbm = (b["mae"] - l["mae"]) / b["mae"] * 100
    mae_impr_mlp = (b["mae"] - m["mae"]) / b["mae"] * 100
    mae_impr_residual = (b["mae"] - r["mae"]) / b["mae"] * 100
    within_impr_graph = g["within_15_pct"] - b["within_15_pct"]
    within_impr_sage = s["within_15_pct"] - b["within_15_pct"]
    within_impr_lgbm = l["within_15_pct"] - b["within_15_pct"]
    within_impr_mlp = m["within_15_pct"] - b["within_15_pct"]
    within_impr_residual = r["within_15_pct"] - b["within_15_pct"]

    best_name = min([("baseline", b["mae"]), ("graph_enhanced", g["mae"]), ("graphsage_xgb", s["mae"]), ("graphsage_lgbm", l["mae"]), ("graphsage_mlp", m["mae"]), ("graphsage_residual", r["mae"])], key=lambda x: x[1])[0]

    print("\n" + "=" * 96)
    print("MODEL COMPARISON  (evaluated on held-out test set)")
    print("=" * 96)
    print(comparison.to_string(index=False))
    print("-" * 96)
    print(f"Baseline -> graph_enhanced : MAE {mae_impr_graph:+.1f}% | within15 {within_impr_graph:+.1f} pts")
    print(f"Baseline -> graphsage_xgb  : MAE {mae_impr_sage:+.1f}% | within15 {within_impr_sage:+.1f} pts")
    print(f"Baseline -> graphsage_lgbm : MAE {mae_impr_lgbm:+.1f}% | within15 {within_impr_lgbm:+.1f} pts")
    print(f"Baseline -> graphsage_mlp  : MAE {mae_impr_mlp:+.1f}% | within15 {within_impr_mlp:+.1f} pts")
    print(f"Baseline -> graphsage_resid : MAE {mae_impr_residual:+.1f}% | within15 {within_impr_residual:+.1f} pts")
    print(f"Best ETA model so far      : {best_name}")
    print("=" * 96)

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


def plot_graphsage_feature_importance(path: str = "outputs/graphsage_feature_importance.png") -> None:
    """Create a multi-panel feature-importance plot for GraphSAGE-related ETA models."""
    artifacts = {
        "graphsage_xgb": joblib.load(GRAPHSAGE_MODEL_PATH),
        "graphsage_lgbm": joblib.load(GRAPHSAGE_LGBM_MODEL_PATH),
        "graphsage_mlp": joblib.load(GRAPHSAGE_MLP_MODEL_PATH),
        "graphsage_residual": joblib.load(GRAPHSAGE_RESIDUAL_MODEL_PATH),
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.ravel()

    for ax, (name, artifact) in zip(axes, artifacts.items()):
        features = artifact["features"]
        model = artifact["model"]
        fill_values = artifact.get("fill_values", {})

        if hasattr(model, "feature_importances_"):
            importances = np.asarray(model.feature_importances_, dtype=float)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            model.eval()
            x0 = torch.tensor([fill_values.get(f, 0.0) for f in features], dtype=torch.float32, device=device).unsqueeze(0)
            x0.requires_grad_(True)
            with torch.enable_grad():
                y = model(x0)
                grads = torch.autograd.grad(y.sum(), x0)[0]
            importances = grads.abs().detach().cpu().numpy().reshape(-1)
            model = model.cpu()

        order = np.argsort(importances)[::-1][:10]
        ax.barh([features[i] for i in order], importances[order], color="steelblue")
        ax.invert_yaxis()
        ax.set_title(f"{name} - top 10 feature importance")
        ax.set_xlabel("Relative importance")

    fig.suptitle("GraphSAGE-related ETA model feature importance", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"GraphSAGE feature-importance plot saved -> {path}")


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

    both_cached = (
        os.path.exists(BASELINE_MODEL_PATH)
        and os.path.exists(GRAPH_MODEL_PATH)
        and os.path.exists(GRAPHSAGE_MODEL_PATH)
        and os.path.exists(GRAPHSAGE_LGBM_MODEL_PATH)
        and os.path.exists(GRAPHSAGE_MLP_MODEL_PATH)
    )

    if both_cached and not args.retrain:
        # Fast path: skip the 55 MB CSV load entirely and reuse saved models.
        print("Cached models found - loading (pass --retrain to rebuild).")
        baseline_art = joblib.load(BASELINE_MODEL_PATH)
        graph_art = joblib.load(GRAPH_MODEL_PATH)
        graphsage_art = joblib.load(GRAPHSAGE_MODEL_PATH)
        graphsage_lgbm_art = joblib.load(GRAPHSAGE_LGBM_MODEL_PATH)
        graphsage_mlp_art = joblib.load(GRAPHSAGE_MLP_MODEL_PATH)
        residual_art = joblib.load(GRAPHSAGE_RESIDUAL_MODEL_PATH)
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
        graphsage_art = get_model(
            "graphsage_xgb", GRAPHSAGE_MODEL_PATH,
            GRAPHSAGE_FEATURES, {**baseline_fill, **emb_fill}, train, test, args.retrain,
            model_kind="xgb",
        )
        graphsage_lgbm_art = get_model(
            "graphsage_lgbm", GRAPHSAGE_LGBM_MODEL_PATH,
            GRAPHSAGE_FEATURES, {**baseline_fill, **emb_fill}, train, test, args.retrain,
            model_kind="lgbm",
        )
        graphsage_mlp_art = get_model(
            "graphsage_mlp", GRAPHSAGE_MLP_MODEL_PATH,
            GRAPHSAGE_FEATURES, {**baseline_fill, **emb_fill}, train, test, args.retrain,
            model_kind="mlp",
        )
        residual_art = get_model(
            "graphsage_residual", GRAPHSAGE_RESIDUAL_MODEL_PATH,
            RESIDUAL_FEATURES, graph_fill, train, test, args.retrain,
            model_kind="residual",
        )

    report_comparison(baseline_art, graph_art, graphsage_art, graphsage_lgbm_art, graphsage_mlp_art, residual_art)
    plot_feature_importance(graph_art, IMPORTANCE_PLOT_PATH)
    plot_graphsage_feature_importance()

    print("\nDone!\n")
    return baseline_art, graph_art


if __name__ == "__main__":
    baseline_art, graph_art = main()
