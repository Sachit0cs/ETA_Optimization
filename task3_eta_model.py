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


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH = "delivery_data.csv"
NODE_METRICS_PATH = "outputs/node_metrics.csv"
OUTPUT_DIR = "outputs"
MODEL_DIR = "models"

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
GRAPH_FEATURES = BASELINE_FEATURES + GRAPH_METRIC_COLS


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
        df = merge_graph_features(df, node_metrics)
        df = engineer_features(df)

        print("\nSplitting train/test...")
        train, test = split_train_test(df)

        # Fill strategy: baseline cols -> 0; graph metrics -> training median so
        # unmatched hubs don't get distorting zeros. Stored in the artifact so
        # prediction-time filling stays identical.
        graph_medians = train[GRAPH_METRIC_COLS].median().to_dict()
        baseline_fill = {f: 0 for f in BASELINE_FEATURES}
        graph_fill = {**baseline_fill, **graph_medians}

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
