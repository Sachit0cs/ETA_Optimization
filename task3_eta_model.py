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

from graphsage_eta_model import compute_graphsage_embeddings
from node2vec_eta_model_fixed import compute_node2vec_embeddings, N2V_DIM


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH = "delivery_data.csv"
NODE_METRICS_PATH = "outputs/node_metrics.csv"
OUTPUT_DIR = "outputs"
MODEL_DIR = "models"
GRAPH_PICKLE = os.path.join(OUTPUT_DIR, "logistics_graph.pkl")
EMBEDDINGS_PATH = os.path.join(OUTPUT_DIR, "node_emb_graphsage.csv")
N2V_EMBEDDINGS_PATH = os.path.join(OUTPUT_DIR, "node_emb_node2vec.csv")

TARGET = "actual_time"
WITHIN_PCT = 0.15         # business accuracy threshold: within 15% of actual
RANDOM_STATE = 42
# EDA finding: trip actual_time is heavily right-skewed (skew=3.37), so squared
# error over-weights the long tail. Train direct-ETA models on log1p(actual_time)
# and invert at prediction time. (The residual model is exempt -- it already
# models actual-osrm.) See outputs/eda/hyperparameter_signals.csv.
LOG_TARGET = True

BASELINE_MODEL_PATH = os.path.join(MODEL_DIR, "baseline_eta_xgb.joblib")
GRAPH_MODEL_PATH = os.path.join(MODEL_DIR, "graph_eta_xgb.joblib")
GRAPHSAGE_MODEL_PATH = os.path.join(MODEL_DIR, "graphsage_eta_xgb.joblib")
GRAPHSAGE_LGBM_MODEL_PATH = os.path.join(MODEL_DIR, "graphsage_eta_lgbm.joblib")
GRAPHSAGE_MLP_MODEL_PATH = os.path.join(MODEL_DIR, "graphsage_eta_mlp.joblib")
GRAPHSAGE_RESIDUAL_MODEL_PATH = os.path.join(MODEL_DIR, "graphsage_residual_xgb.joblib")
NODE2VEC_MODEL_PATH = os.path.join(MODEL_DIR, "node2vec_eta_xgb.joblib")
NODE2VEC_ENHANCED_MODEL_PATH = os.path.join(MODEL_DIR, "node2vec_enhanced_eta_xgb.joblib")
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
# EDA flagged several near-constant GraphSAGE dims at EMB_DIM=8, but reducing to 4
# was tested and HURT downstream (graphsage_xgb collapsed to the baseline -- the few
# useful dims got dropped too), so we keep 8. See outputs/eda/hyperparameter_signals.csv.
EMB_DIM = 8
SRC_EMB_COLS = [f"src_emb_{i}" for i in range(EMB_DIM)]
DST_EMB_COLS = [f"dst_emb_{i}" for i in range(EMB_DIM)]
GRAPH_FEATURES = BASELINE_FEATURES + GRAPH_METRIC_COLS + SRC_EMB_COLS + DST_EMB_COLS
GRAPHSAGE_FEATURES = BASELINE_FEATURES + SRC_EMB_COLS + DST_EMB_COLS
RESIDUAL_FEATURES = BASELINE_FEATURES + GRAPH_METRIC_COLS + SRC_EMB_COLS + DST_EMB_COLS
# node2vec embeddings (computed from the SAME training-only graph pickle by
# node2vec_eta_model_fixed.compute_node2vec_embeddings). The two node2vec models
# mirror the GraphSAGE pair so the embedding methods compare like-for-like:
#   node2vec_xgb      <-> graphsage_xgb   (baseline + embeddings only)
#   node2vec_enhanced <-> graph_enhanced  (baseline + hub metrics + embeddings)
SRC_N2V_COLS = [f"src_n2v_{i}" for i in range(N2V_DIM)]
DST_N2V_COLS = [f"dst_n2v_{i}" for i in range(N2V_DIM)]
NODE2VEC_FEATURES = BASELINE_FEATURES + SRC_N2V_COLS + DST_N2V_COLS
NODE2VEC_ENHANCED_FEATURES = BASELINE_FEATURES + GRAPH_METRIC_COLS + SRC_N2V_COLS + DST_N2V_COLS


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


def aggregate_to_trips(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse segment/scan-level rows to ONE row per trip.

    The raw data is segment-level: actual_time / osrm_time are cumulative within
    each OD leg, so training and scoring on raw rows double-counts long trips and
    makes the "% within 15%" metric per-row instead of per-trip as the brief
    requires. We rebuild trip-level totals from the per-hop segment_* columns
    (matching the LSTM pipeline): target = sum(segment_actual_time); the OSRM
    features are the corresponding segment sums. Graph features are taken from the
    trip's origin (first) and final destination (last) facility.
    """
    df = df.sort_values(["trip_uuid", "od_start_time"])
    trips = (
        df.groupby("trip_uuid", sort=False)
        .agg(
            actual_time=("segment_actual_time", "sum"),
            osrm_time=("segment_osrm_time", "sum"),
            osrm_distance=("segment_osrm_distance", "sum"),
            source_center=("source_center", "first"),
            destination_center=("destination_center", "last"),
            route_type=("route_type", "first"),
            trip_creation_time=("trip_creation_time", "first"),
            data=("data", "first"),
        )
        .reset_index()
    )
    print(f"  aggregated {len(df):,} segment rows -> {len(trips):,} trips")
    return trips


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
    # log1p target for direct-ETA tree models. The tiny MLP is exempt: log-space
    # made its predictions unstable (post-expm1 blow-up), so it keeps the raw target.
    use_log = LOG_TARGET and not use_residual and model_kind != "mlp"

    if use_residual:
        y_train = train[TARGET] - train["osrm_time"]
    elif use_log:
        y_train = np.log1p(train[TARGET])
    else:
        y_train = train[TARGET]

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
        torch.manual_seed(RANDOM_STATE)  # deterministic weight init / training
        np.random.seed(RANDOM_STATE)
        X_train_np = X_train.to_numpy(dtype=np.float32)
        X_test_np = X_test.to_numpy(dtype=np.float32)
        y_train_np = np.asarray(y_train, dtype=np.float32)

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

    y_pred = np.asarray(y_pred, dtype=float)
    if use_residual:
        y_pred = test["osrm_time"].to_numpy(dtype=float) + y_pred
    elif use_log:
        # Clip in log space before inverting so a poorly-behaved learner (e.g. the
        # MLP) can't produce an expm1 overflow. Bound at ~3x the max training trip.
        hi = float(np.log1p(train[TARGET].max())) + 1.0
        y_pred = np.expm1(np.clip(y_pred, 0.0, hi))  # back to minutes
    metrics = evaluate(test[TARGET].to_numpy(dtype=float), y_pred)

    if model_kind == "mlp":
        model = model.cpu()

    return {
        "name": name,
        "model": model,
        "model_kind": model_kind,
        "features": features,
        "fill_values": fill_values,
        "target": TARGET,
        "target_transform": "log1p" if use_log else ("residual" if use_residual else "identity"),
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

def report_comparison(artifacts: list) -> pd.DataFrame:
    """Single source of truth for benchmark results. The first artifact must be
    the baseline; every model is evaluated with the same evaluate() on the same
    trip-level test split, so the rows are directly comparable."""
    comparison = pd.DataFrame(
        [
            {
                "model": art["name"],
                "n_features": len(art["features"]),
                "training_time_sec": art["training_time_sec"],
                "inference_time_sec": art["inference_time_sec"],
                "overall_accuracy_pct": art["overall_accuracy_pct"],
                **art["metrics"],
            }
            for art in artifacts
        ]
    )

    b = artifacts[0]["metrics"]
    best_name = min(artifacts, key=lambda a: a["metrics"]["mae"])["name"]

    print("\n" + "=" * 96)
    print("MODEL COMPARISON  (evaluated on held-out test set)")
    print("=" * 96)
    print(comparison.to_string(index=False))
    print("-" * 96)
    for art in artifacts[1:]:
        m = art["metrics"]
        mae_impr = (b["mae"] - m["mae"]) / b["mae"] * 100
        within_impr = m["within_15_pct"] - b["within_15_pct"]
        print(f"Baseline -> {art['name']:<18} : MAE {mae_impr:+.1f}% | within15 {within_impr:+.1f} pts")
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

    model_paths = [
        BASELINE_MODEL_PATH,
        GRAPH_MODEL_PATH,
        GRAPHSAGE_MODEL_PATH,
        GRAPHSAGE_LGBM_MODEL_PATH,
        GRAPHSAGE_MLP_MODEL_PATH,
        GRAPHSAGE_RESIDUAL_MODEL_PATH,
        NODE2VEC_MODEL_PATH,
        NODE2VEC_ENHANCED_MODEL_PATH,
    ]
    all_cached = all(os.path.exists(p) for p in model_paths)

    if all_cached and not args.retrain:
        # Fast path: skip the 55 MB CSV load entirely and reuse saved models.
        print("Cached models found - loading (pass --retrain to rebuild).")
        artifacts = [joblib.load(p) for p in model_paths]
    else:
        print("Loading data...")
        df, node_metrics = load_data()
        # Collapse segment-level rows to one row per trip (per-trip ETA target)
        df = aggregate_to_trips(df)
        # Merge per-node graph metrics
        df = merge_graph_features(df, node_metrics)

        # Ensure GraphSAGE-like embeddings exist (recompute when retraining or missing)
        G = None
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

        # node2vec embeddings from the SAME training-only graph pickle. Saved to
        # outputs/ so eda.py's node2vec section auto-activates.
        n2v_df = None
        if not args.retrain and os.path.exists(N2V_EMBEDDINGS_PATH):
            n2v_df = pd.read_csv(N2V_EMBEDDINGS_PATH)
            print(f"  loaded existing node2vec embeddings -> {N2V_EMBEDDINGS_PATH}")
        else:
            if G is None:
                G = load_graph()
            n2v_df, _ = compute_node2vec_embeddings(G, N2V_DIM)
            if n2v_df is not None:
                n2v_df.to_csv(N2V_EMBEDDINGS_PATH, index=False)
                print(f"  node2vec embeddings saved -> {N2V_EMBEDDINGS_PATH}")
        have_n2v = n2v_df is not None  # gensim missing -> skip the node2vec pair

        # Merge embeddings into trip rows (src_/dst_ prefixed columns)
        df = merge_embeddings(df, emb_df)
        if have_n2v:
            df = merge_embeddings(df, n2v_df)
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
        n2v_fill = {c: 0.0 for c in (SRC_N2V_COLS + DST_N2V_COLS)}
        graph_fill = {**baseline_fill, **graph_medians, **emb_fill}

        print("\nTraining / loading models...")
        artifacts = [
            get_model(
                "baseline", BASELINE_MODEL_PATH,
                BASELINE_FEATURES, baseline_fill, train, test, args.retrain,
            ),
            get_model(
                "graph_enhanced", GRAPH_MODEL_PATH,
                GRAPH_FEATURES, graph_fill, train, test, args.retrain,
            ),
            get_model(
                "graphsage_xgb", GRAPHSAGE_MODEL_PATH,
                GRAPHSAGE_FEATURES, {**baseline_fill, **emb_fill}, train, test, args.retrain,
                model_kind="xgb",
            ),
            get_model(
                "graphsage_lgbm", GRAPHSAGE_LGBM_MODEL_PATH,
                GRAPHSAGE_FEATURES, {**baseline_fill, **emb_fill}, train, test, args.retrain,
                model_kind="lgbm",
            ),
            get_model(
                "graphsage_mlp", GRAPHSAGE_MLP_MODEL_PATH,
                GRAPHSAGE_FEATURES, {**baseline_fill, **emb_fill}, train, test, args.retrain,
                model_kind="mlp",
            ),
            get_model(
                "graphsage_residual", GRAPHSAGE_RESIDUAL_MODEL_PATH,
                RESIDUAL_FEATURES, graph_fill, train, test, args.retrain,
                model_kind="residual",
            ),
        ]
        if have_n2v:
            artifacts.append(get_model(
                "node2vec_xgb", NODE2VEC_MODEL_PATH,
                NODE2VEC_FEATURES, {**baseline_fill, **n2v_fill}, train, test, args.retrain,
                model_kind="xgb",
            ))
            artifacts.append(get_model(
                "node2vec_enhanced", NODE2VEC_ENHANCED_MODEL_PATH,
                NODE2VEC_ENHANCED_FEATURES, {**baseline_fill, **graph_medians, **n2v_fill},
                train, test, args.retrain,
                model_kind="xgb",
            ))

    report_comparison(artifacts)
    plot_feature_importance(artifacts[1], IMPORTANCE_PLOT_PATH)  # graph_enhanced
    plot_graphsage_feature_importance()

    print("\nDone!\n")
    return artifacts


if __name__ == "__main__":
    artifacts = main()
