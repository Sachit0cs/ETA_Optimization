

import argparse
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH = "delivery_data.csv"
NODE_METRICS_PATH = "outputs/node_metrics.csv"
OUTPUT_DIR = "outputs"
MODEL_DIR = "models"

LSTM_PATH = os.path.join(MODEL_DIR, "lstm_eta.pt")
XGB_BASELINE_PATH = os.path.join(MODEL_DIR, "lstm_trip_xgb_baseline.joblib")
COMPARISON_PATH = os.path.join(OUTPUT_DIR, "lstm_model_comparison.csv")
CURVE_PATH = os.path.join(OUTPUT_DIR, "lstm_training_curve.png")

WITHIN_PCT = 0.15
SEED = 42

# Per-scan features fed to the LSTM at each timestep (NO actual-time leakage).
SEQ_FEATURES = [
    "segment_osrm_time", "segment_osrm_distance",
    "is_ftl", "hour_of_day", "day_of_week",
    "src_betweenness", "src_in_degree", "src_out_degree",
    "src_avg_delay", "src_bottleneck_score",
    "dst_betweenness", "dst_in_degree", "dst_out_degree",
    "dst_avg_delay", "dst_bottleneck_score",
]

# LSTM hyperparameters
HIDDEN = 64
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 64
LR = 1e-3
MAX_EPOCHS = 40
PATIENCE = 6          # early-stopping patience on validation MAE
VAL_FRACTION = 0.1    # slice of TRAIN trips held out for early stopping

DEVICE = torch.device("cpu")  # CPU build


# ---------------------------------------------------------------------------
# 1. Data: build per-trip sequences
# ---------------------------------------------------------------------------

def load_and_merge() -> pd.DataFrame:
    if not os.path.exists(NODE_METRICS_PATH):
        raise FileNotFoundError(
            f"{NODE_METRICS_PATH} not found. Run build_graph.py first - it "
            "produces the per-hub graph metrics this model depends on."
        )
    df = pd.read_csv(DATA_PATH)
    node_metrics = pd.read_csv(NODE_METRICS_PATH)

    # Attach source/destination hub graph metrics to every scan row.
    keep = ["center", "betweenness", "in_degree", "out_degree",
            "avg_incoming_delay_factor", "bottleneck_score"]
    nm = node_metrics[keep].rename(columns={"avg_incoming_delay_factor": "avg_delay"})
    for side, join_key in [("src", "source_center"), ("dst", "destination_center")]:
        prefixed = {c: f"{side}_{c}" for c in nm.columns if c != "center"}
        df = df.merge(
            nm.rename(columns=prefixed),
            left_on=join_key, right_on="center", how="left",
        ).drop(columns="center")

    # Per-scan time features from the leg's departure time.
    df["od_start_time"] = pd.to_datetime(df["od_start_time"], errors="coerce")
    df["hour_of_day"] = df["od_start_time"].dt.hour
    df["day_of_week"] = df["od_start_time"].dt.dayofweek
    df["is_ftl"] = (df["route_type"] == "FTL").astype(int)

    # Scan order within a trip: leg order (od_start_time), then cumulative
    # actual_time which rises monotonically inside each leg. mergesort = stable.
    df = df.sort_values(["trip_uuid", "od_start_time", "actual_time"],
                        kind="mergesort").reset_index(drop=True)
    return df


def build_sequences(df: pd.DataFrame):
    """Return dict: trip_uuid -> (feature_array[T, F], total_actual, split)."""
    feat = df[SEQ_FEATURES].to_numpy(dtype=np.float32)
    target_inc = df["segment_actual_time"].to_numpy(dtype=np.float32)
    trips = df["trip_uuid"].to_numpy()
    splits = df["data"].to_numpy()

    sequences, targets, split_of = {}, {}, {}
    # Contiguous trip blocks (df is sorted by trip_uuid).
    _, start_idx = np.unique(trips, return_index=True)
    bounds = np.append(np.sort(start_idx), len(trips))
    for a, b in zip(bounds[:-1], bounds[1:]):
        tid = trips[a]
        sequences[tid] = feat[a:b]
        targets[tid] = float(target_inc[a:b].sum())
        split_of[tid] = splits[a]
    return sequences, targets, split_of


def split_trip_ids(targets, split_of):
    train_ids = [t for t in targets if split_of[t] == "training"]
    test_ids = [t for t in targets if split_of[t] == "test"]
    rng = np.random.default_rng(SEED)
    rng.shuffle(train_ids)
    n_val = int(len(train_ids) * VAL_FRACTION)
    val_ids, fit_ids = train_ids[:n_val], train_ids[n_val:]
    return fit_ids, val_ids, test_ids


# ---------------------------------------------------------------------------
# 2. Torch dataset / model
# ---------------------------------------------------------------------------

class TripSeqDataset(Dataset):
    """Sequences are pre-scaled; target stored as log1p(total_actual)."""

    def __init__(self, ids, sequences, targets, mean, std):
        self.ids = ids
        self.X = [(sequences[t] - mean) / std for t in ids]
        self.y = np.log1p(np.array([targets[t] for t in ids], dtype=np.float32))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i])


def collate(batch):
    seqs, ys = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    padded = pad_sequence(seqs, batch_first=True)  # [B, T, F]
    return padded, lengths, torch.stack(ys)


class LSTMRegressor(nn.Module):
    def __init__(self, n_features, hidden=HIDDEN, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1),
        )

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.head(h[-1]).squeeze(-1)  # final-layer hidden state -> scalar


# ---------------------------------------------------------------------------
# 3. Train / evaluate
# ---------------------------------------------------------------------------

def evaluate_minutes(y_true, y_pred) -> dict:
    """Inputs are in real minutes."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    mask = y_true > 0
    within = np.mean(np.abs(y_pred[mask] - y_true[mask]) / y_true[mask] <= WITHIN_PCT) * 100
    return {"mae": float(mean_absolute_error(y_true, y_pred)), "within_15_pct": float(within)}


def predict_minutes(model, loader) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for x, lengths, _ in loader:
            out = model(x.to(DEVICE), lengths)
            preds.append(np.expm1(out.cpu().numpy()))  # invert log1p -> minutes
    return np.concatenate(preds)


def train_lstm(fit_ds, val_ds, val_true_minutes, n_features):
    torch.manual_seed(SEED)
    model = LSTMRegressor(n_features).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.SmoothL1Loss()  # Huber: robust to long-tail trips

    fit_loader = DataLoader(fit_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=collate)

    history = {"train_loss": [], "val_mae": []}
    best_mae, best_state, stale = float("inf"), None, 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running = 0.0
        for x, lengths, y in fit_loader:
            opt.zero_grad()
            loss = loss_fn(model(x.to(DEVICE), lengths), y.to(DEVICE))
            loss.backward()
            opt.step()
            running += loss.item() * len(y)
        train_loss = running / len(fit_ds)

        val_mae = evaluate_minutes(val_true_minutes, predict_minutes(model, val_loader))["mae"]
        history["train_loss"].append(train_loss)
        history["val_mae"].append(val_mae)
        print(f"  epoch {epoch:02d}  train_loss={train_loss:.4f}  val_MAE={val_mae:.2f} min")

        if val_mae < best_mae - 1e-4:
            best_mae, best_state, stale = val_mae, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"  early stop (no val improvement for {PATIENCE} epochs)")
                break

    model.load_state_dict(best_state)
    return model, history, best_mae


# ---------------------------------------------------------------------------
# 4. Baselines
# ---------------------------------------------------------------------------

def osrm_sum_baseline(df, ids_set):
    g = (df[df["trip_uuid"].isin(ids_set)]
         .groupby("trip_uuid")
         .agg(total_osrm=("segment_osrm_time", "sum"),
              total_actual=("segment_actual_time", "sum")))
    return evaluate_minutes(g["total_actual"], g["total_osrm"])


def flat_trip_table(df):
    """One row per trip: summed OSRM + averaged hub metrics (no sequence)."""
    agg = {
        "segment_osrm_time": ("segment_osrm_time", "sum"),
        "segment_osrm_distance": ("segment_osrm_distance", "sum"),
        "n_seg": ("segment_osrm_time", "size"),
        "is_ftl": ("is_ftl", "max"),
        "hour_of_day": ("hour_of_day", "first"),
        "day_of_week": ("day_of_week", "first"),
        "total_actual": ("segment_actual_time", "sum"),
        "data": ("data", "first"),
    }
    for c in SEQ_FEATURES:
        if c.startswith(("src_", "dst_")):
            agg[c] = (c, "mean")
    return df.groupby("trip_uuid").agg(**agg).reset_index()


def train_flat_xgb(df, fit_ids, val_ids, test_ids, retrain):
    if os.path.exists(XGB_BASELINE_PATH) and not retrain:
        print("  [flat-xgb] loading cached baseline")
        art = joblib.load(XGB_BASELINE_PATH)
    else:
        print("  [flat-xgb] training trip-level XGBoost baseline...")
        flat = flat_trip_table(df)
        features = [c for c in flat.columns
                    if c not in ("trip_uuid", "total_actual", "data")]
        tr = flat[flat["trip_uuid"].isin(set(fit_ids) | set(val_ids))]
        model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                            random_state=SEED, verbosity=0)
        model.fit(tr[features], np.log1p(tr["total_actual"]))
        art = {"model": model, "features": features}
        joblib.dump(art, XGB_BASELINE_PATH)
        print(f"  [flat-xgb] saved -> {XGB_BASELINE_PATH}")

    flat = flat_trip_table(df)
    te = flat[flat["trip_uuid"].isin(set(test_ids))]
    pred = np.expm1(art["model"].predict(te[art["features"]]))
    return evaluate_minutes(te["total_actual"], pred)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train/benchmark the LSTM time-series ETA model.")
    p.add_argument("--retrain", action="store_true",
                   help="Force retraining even if a cached LSTM exists in ./models/.")
    p.add_argument("--epochs", type=int, default=MAX_EPOCHS,
                   help=f"Max training epochs (default {MAX_EPOCHS}).")
    return p.parse_args()


def main():
    global MAX_EPOCHS
    args = parse_args()
    MAX_EPOCHS = args.epochs
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading & merging data...")
    df = load_and_merge()
    sequences, targets, split_of = build_sequences(df)
    fit_ids, val_ids, test_ids = split_trip_ids(targets, split_of)
    print(f"  trips -> fit: {len(fit_ids):,} | val: {len(val_ids):,} | test: {len(test_ids):,}")

    n_features = len(SEQ_FEATURES)

    # ---- LSTM: train fresh, or load cached checkpoint --------------------
    if os.path.exists(LSTM_PATH) and not args.retrain:
        print("\nCached LSTM found - loading (pass --retrain to rebuild).")
        ckpt = torch.load(LSTM_PATH, weights_only=False)
        mean, std = ckpt["scaler_mean"], ckpt["scaler_std"]
        model = LSTMRegressor(n_features).to(DEVICE)
        model.load_state_dict(ckpt["state_dict"])
        lstm_metrics = ckpt["metrics"]
        history = ckpt.get("history", {"train_loss": [], "val_mae": []})
    else:
        # Feature scaler fit on FIT timesteps only (no val/test leakage).
        fit_stack = np.concatenate([sequences[t] for t in fit_ids], axis=0)
        mean = fit_stack.mean(axis=0)
        std = fit_stack.std(axis=0)
        std[std == 0] = 1.0

        fit_ds = TripSeqDataset(fit_ids, sequences, targets, mean, std)
        val_ds = TripSeqDataset(val_ids, sequences, targets, mean, std)
        val_true = [targets[t] for t in val_ids]

        print("\nTraining LSTM...")
        model, history, best_val = train_lstm(fit_ds, val_ds, val_true, n_features)

        test_ds = TripSeqDataset(test_ids, sequences, targets, mean, std)
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate)
        lstm_metrics = evaluate_minutes([targets[t] for t in test_ids],
                                        predict_minutes(model, test_loader))

        torch.save({
            "state_dict": model.state_dict(),
            "config": {"n_features": n_features, "hidden": HIDDEN,
                       "num_layers": NUM_LAYERS, "dropout": DROPOUT},
            "seq_features": SEQ_FEATURES,
            "scaler_mean": mean, "scaler_std": std,
            "target_transform": "log1p",
            "within_pct": WITHIN_PCT,
            "metrics": lstm_metrics,
            "history": history,
            "best_val_mae": float(best_val),
            "n_train": len(fit_ids), "n_val": len(val_ids), "n_test": len(test_ids),
            "trained_at": datetime.now().isoformat(timespec="seconds"),
        }, LSTM_PATH)
        print(f"\nLSTM saved -> {LSTM_PATH}  "
              f"(MAE {lstm_metrics['mae']:.2f}, within15 {lstm_metrics['within_15_pct']:.1f}%)")

    # ---- Baselines (always evaluated on the same test trips) -------------
    print("\nBaselines...")
    osrm_metrics = osrm_sum_baseline(df, set(test_ids))
    xgb_metrics = train_flat_xgb(df, fit_ids, val_ids, test_ids, args.retrain)

    # ---- Report ----------------------------------------------------------
    comparison = pd.DataFrame([
        {"model": "OSRM_sum (naive)", **osrm_metrics},
        {"model": "flat_XGBoost", **xgb_metrics},
        {"model": "LSTM_sequence", **lstm_metrics},
    ])
    print("\n" + "=" * 64)
    print("TRIP-LEVEL ETA COMPARISON  (held-out test trips, minutes)")
    print("=" * 64)
    print(comparison.to_string(index=False))
    print("-" * 64)
    mae_vs_osrm = (osrm_metrics["mae"] - lstm_metrics["mae"]) / osrm_metrics["mae"] * 100
    mae_vs_xgb = (xgb_metrics["mae"] - lstm_metrics["mae"]) / xgb_metrics["mae"] * 100
    print(f"LSTM vs OSRM-sum : MAE {mae_vs_osrm:+.1f}%  | "
          f"within15 {lstm_metrics['within_15_pct'] - osrm_metrics['within_15_pct']:+.1f} pts")
    print(f"LSTM vs flat-XGB : MAE {mae_vs_xgb:+.1f}%  | "
          f"within15 {lstm_metrics['within_15_pct'] - xgb_metrics['within_15_pct']:+.1f} pts")
    print("=" * 64)
    comparison.to_csv(COMPARISON_PATH, index=False)
    print(f"Comparison saved -> {COMPARISON_PATH}")

    # ---- Training curve --------------------------------------------------
    if history["train_loss"]:
        fig, ax1 = plt.subplots(figsize=(8, 5))
        epochs = range(1, len(history["train_loss"]) + 1)
        ax1.plot(epochs, history["train_loss"], "o-", color="steelblue", label="train loss (Huber)")
        ax1.set_xlabel("epoch"); ax1.set_ylabel("train loss", color="steelblue")
        ax2 = ax1.twinx()
        ax2.plot(epochs, history["val_mae"], "s-", color="darkorange", label="val MAE (min)")
        ax2.set_ylabel("val MAE (min)", color="darkorange")
        plt.title("LSTM Training Curve")
        fig.tight_layout()
        plt.savefig(CURVE_PATH, dpi=150)
        plt.close()
        print(f"Training curve saved -> {CURVE_PATH}")

    print("\nDone!\n")
    return model, comparison


if __name__ == "__main__":
    model, comparison = main()
