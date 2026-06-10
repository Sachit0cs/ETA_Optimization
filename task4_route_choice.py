"""
Task 4 -- FTL vs Carting decision framework (per TASK4_PLAN.md).

Why a T-learner: only 14 / 2,481 corridors were ever run on BOTH route types,
so route choice cannot be answered by lookup. Instead two counterfactual time
models (one per route type) predict what every dispatch would take under FTL
and under Carting, two p90 quantile models price the tail risk, and a
parametric cost model turns the time/risk gap into a recommendation.

Route type is confounded with distance (Carting median 31 km, FTL 73 km), so
any recommendation outside the recommended route type's observed distance
support (p5-p95 of training legs) is flagged `extrapolated`, not hidden.

Cost numbers are explicit, configurable ASSUMPTIONS (no cost column exists in
the data); the per-dispatch break-even value-of-time makes the framework
robust to whatever real rates the ops team plugs in.

Outputs
  models/route_choice.joblib                      bundle: 4 models + params
  outputs/route_choice_features.csv               every test leg scored
  outputs/route_choice_recommendations.csv        switch opportunities summary
  outputs/visualizations/07_route_choice_tradeoff.png
  outputs/visualizations/08_route_choice_by_distance.png

Run:  python task4_route_choice.py [--retrain]
"""

import argparse
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless-safe: write PNGs without a display
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


def evaluate(y_true, y_pred) -> dict:
    """MAE + % within 15% of actual. Mirrors task3_eta_model.evaluate exactly;
    inlined (not imported) so this module stays importable without pulling the
    Task-3 torch/lightgbm dependency chain into the dashboard process."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true > 0
    within = np.mean(np.abs(y_pred[mask] - y_true[mask]) / y_true[mask] <= 0.15) * 100
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "within_15_pct": float(within),
    }


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH = "delivery_data.csv"
NODE_METRICS_PATH = "outputs/node_metrics.csv"
OUTPUT_DIR = "outputs"
MODEL_DIR = "models"
VIZ_DIR = os.path.join(OUTPUT_DIR, "visualizations")

BUNDLE_PATH = os.path.join(MODEL_DIR, "route_choice.joblib")
FEATURES_CSV = os.path.join(OUTPUT_DIR, "route_choice_features.csv")
RECS_CSV = os.path.join(OUTPUT_DIR, "route_choice_recommendations.csv")
PLOT_TRADEOFF = os.path.join(VIZ_DIR, "07_route_choice_tradeoff.png")
PLOT_BY_DIST = os.path.join(VIZ_DIR, "08_route_choice_by_distance.png")

RANDOM_STATE = 42
XGB_PARAMS = dict(            # same conventions as task3_eta_model.XGB_PARAMS
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=RANDOM_STATE,
    verbosity=0,
)

# Leg-level features shared by all four models. No is_ftl -- route type is
# separated by model (T-learner). Graph position comes from the training-only
# node_metrics.csv, so no test-set delay information leaks in.
FEATURES = [
    "osrm_time", "osrm_distance", "hour_of_day", "day_of_week",
    "src_betweenness", "src_bottleneck_score", "src_avg_delay",
    "src_in_degree", "src_out_degree", "dst_bottleneck_score",
]
TARGET = "actual_time"

# --- Cost model: ILLUSTRATIVE assumptions, override per call or in the app ---
# cost_cart = CART_PER_KM * dist          (shared-truck consignment rate)
# cost_ftl  = FTL_FIXED + FTL_PER_KM * dist  (dedicated truck: fixed + lower /km)
# Break-even distance with the defaults: 2500/(45-30) ~ 167 km, matching the
# long-haul=FTL reality in the data.
DEFAULT_PARAMS = dict(
    ftl_fixed=2500.0,        # Rs per FTL dispatch
    ftl_per_km=30.0,         # Rs/km, FTL
    cart_per_km=45.0,        # Rs/km, Carting
    value_of_time=20.0,      # Rs per delivery-minute saved (SLA exposure proxy)
    risk_weight=0.5,         # weight of tail (p90) minutes vs expected minutes
    risk_gain=0.5,           # max value-of-time uplift at the riskiest source hub
)

DISTANCE_BANDS = [0, 25, 50, 100, 250, 500, np.inf]
DISTANCE_LABELS = ["0-25km", "25-50km", "50-100km", "100-250km", "250-500km", "500km+"]


# ---------------------------------------------------------------------------
# 1. Data prep -- one row per OD-leg dispatch
# ---------------------------------------------------------------------------

def load_legs() -> pd.DataFrame:
    """Leg-level rows (is_cutoff=False = the departure scan per OD leg; mirrors
    build_graph.extract_leg_records but keeps BOTH train and test legs)."""
    df = pd.read_csv(DATA_PATH)

    if df["is_cutoff"].dtype == object:
        df["is_cutoff"] = df["is_cutoff"].str.strip().str.lower().map(
            {"true": True, "false": False}
        )
    legs = df[~df["is_cutoff"]].copy()

    legs["od_start_time"] = pd.to_datetime(legs["od_start_time"], errors="coerce")
    legs["hour_of_day"] = legs["od_start_time"].dt.hour
    legs["day_of_week"] = legs["od_start_time"].dt.dayofweek
    legs["tod_bucket"] = pd.cut(
        legs["hour_of_day"],
        bins=[0, 6, 12, 18, 24],
        labels=["night", "morning", "afternoon", "evening"],
        right=False,
    )
    legs["is_ftl"] = (legs["route_type"] == "FTL").astype(int)
    legs["factor"] = legs["actual_time"] / legs["osrm_time"]

    # Graph position of the source hub (+ destination risk), training-only metrics.
    nm = pd.read_csv(NODE_METRICS_PATH).rename(
        columns={"avg_incoming_delay_factor": "avg_delay"}
    )
    src_cols = ["betweenness", "bottleneck_score", "avg_delay", "in_degree", "out_degree"]
    legs = legs.merge(
        nm[["center"] + src_cols].rename(columns={c: f"src_{c}" for c in src_cols}),
        left_on="source_center", right_on="center", how="left",
    ).drop(columns="center")
    legs = legs.merge(
        nm[["center", "bottleneck_score"]].rename(
            columns={"bottleneck_score": "dst_bottleneck_score"}),
        left_on="destination_center", right_on="center", how="left",
    ).drop(columns="center")

    print(f"  legs: {len(legs):,} | FTL {int(legs['is_ftl'].sum()):,} "
          f"| Carting {int((1 - legs['is_ftl']).sum()):,}")
    return legs


def split_train_test(legs: pd.DataFrame):
    train = legs[legs["data"] == "training"].copy()
    test = legs[legs["data"] == "test"].copy()
    print(f"  train: {len(train):,} legs | test: {len(test):,} legs")
    return train, test


# ---------------------------------------------------------------------------
# 2. T-learner time models + p90 tail models
# ---------------------------------------------------------------------------

def _design(frame: pd.DataFrame, fill: dict) -> pd.DataFrame:
    return frame[FEATURES].copy().fillna(fill)


def _fit_time_model(train_rt: pd.DataFrame, fill: dict, quantile: float | None):
    """One XGBoost on log1p(actual_time) for a single route type.
    quantile=None -> squared error (expected time); else reg:quantileerror."""
    X = _design(train_rt, fill)
    y = np.log1p(train_rt[TARGET])
    if quantile is None:
        model = XGBRegressor(**XGB_PARAMS)
    else:
        # Quantiles are monotone-invariant, so the p90 fitted in log space maps
        # back to the p90 in minutes after expm1.
        try:
            model = XGBRegressor(
                **XGB_PARAMS, objective="reg:quantileerror", quantile_alpha=quantile,
            )
            model.fit(X, y)
            return model
        except Exception as exc:  # older xgboost: residual-spread fallback
            print(f"    quantile objective unavailable ({exc}); using residual-spread fallback")
            model = XGBRegressor(**XGB_PARAMS)
            model.fit(X, y)
            resid = y - model.predict(X)
            model._p90_offset = float(np.quantile(resid, quantile))
            return model
    model.fit(X, y)
    return model


def _predict_minutes(model, frame: pd.DataFrame, fill: dict, log_hi: float) -> np.ndarray:
    pred = model.predict(_design(frame, fill)).astype(float)
    pred += getattr(model, "_p90_offset", 0.0)
    # Same log-space clip as task3 so one bad leaf can't expm1-overflow.
    return np.expm1(np.clip(pred, 0.0, log_hi))


def train_bundle(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Fit the 4 models, validate each on held-out legs of its own route type,
    and bundle everything needed to score a new dispatch."""
    graph_cols = [c for c in FEATURES if c.startswith(("src_", "dst_"))]
    fill = {c: 0.0 for c in FEATURES}
    fill.update(train[graph_cols].median().to_dict())  # unseen hubs -> training median

    log_hi = float(np.log1p(train[TARGET].max())) + 1.0

    models, support = {}, {}
    for rt in ["FTL", "Carting"]:
        sub = train[train["route_type"] == rt]
        print(f"  [{rt}] {len(sub):,} training legs")
        models[f"time_{rt}"] = _fit_time_model(sub, fill, quantile=None)
        models[f"p90_{rt}"] = _fit_time_model(sub, fill, quantile=0.9)
        # Common-support band: counterfactuals outside it are extrapolations.
        support[rt] = (
            float(sub["osrm_distance"].quantile(0.05)),
            float(sub["osrm_distance"].quantile(0.95)),
        )

        held = test[test["route_type"] == rt]
        m = evaluate(
            held[TARGET].to_numpy(dtype=float),
            _predict_minutes(models[f"time_{rt}"], held, fill, log_hi),
        )
        print(f"    held-out ({len(held):,} legs): MAE {m['mae']:.2f} min "
              f"| within15 {m['within_15_pct']:.1f}%")

    # Scale for the bottleneck risk multiplier (p95 caps outlier hubs).
    bn_p95 = float(train["src_bottleneck_score"].quantile(0.95))

    return {
        "models": models,
        "features": FEATURES,
        "fill_values": fill,
        "log_hi": log_hi,
        "distance_support": support,
        "bottleneck_p95": bn_p95 if bn_p95 > 0 else 1.0,
        "default_params": dict(DEFAULT_PARAMS),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------------------------
# 3. Scoring + decision rule (kept separate so the app can re-decide cheaply)
# ---------------------------------------------------------------------------

def score_times(legs: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Counterfactual predictions for every dispatch under BOTH route types."""
    out = legs.copy()
    fill, log_hi = bundle["fill_values"], bundle["log_hi"]
    out["pred_time_ftl"] = _predict_minutes(bundle["models"]["time_FTL"], out, fill, log_hi)
    out["pred_time_cart"] = _predict_minutes(bundle["models"]["time_Carting"], out, fill, log_hi)
    out["pred_p90_ftl"] = _predict_minutes(bundle["models"]["p90_FTL"], out, fill, log_hi)
    out["pred_p90_cart"] = _predict_minutes(bundle["models"]["p90_Carting"], out, fill, log_hi)
    return out


def apply_decision(scored: pd.DataFrame, bundle: dict, **overrides) -> pd.DataFrame:
    """Cost model + decision rule. Pass value_of_time=..., risk_weight=... etc.
    to override the documented defaults (this is what the dashboard exposes)."""
    p = {**bundle["default_params"], **overrides}
    out = scored.copy()

    dist = out["osrm_distance"].fillna(0).to_numpy(dtype=float)
    out["cost_cart"] = p["cart_per_km"] * dist
    out["cost_ftl"] = p["ftl_fixed"] + p["ftl_per_km"] * dist
    out["delta_cost"] = out["cost_ftl"] - out["cost_cart"]          # extra Rs for FTL

    out["delta_time_min"] = out["pred_time_cart"] - out["pred_time_ftl"]   # min FTL saves
    out["delta_risk_min"] = out["pred_p90_cart"] - out["pred_p90_ftl"]     # tail min FTL saves

    # Risky source hub -> reliability is worth more -> effective VoT rises.
    bn = (out["src_bottleneck_score"].fillna(0) / bundle["bottleneck_p95"]).clip(0, 1)
    out["vot_effective"] = p["value_of_time"] * (1.0 + p["risk_gain"] * bn)

    weighted_saving = out["delta_time_min"] + p["risk_weight"] * out["delta_risk_min"]
    out["benefit_ftl"] = out["vot_effective"] * weighted_saving
    out["recommended_route"] = np.where(
        out["benefit_ftl"] > out["delta_cost"], "FTL", "Carting"
    )

    # Rs/min at which FTL becomes worth it (inf when FTL saves no time at all).
    out["break_even_vot"] = np.where(
        weighted_saving > 1e-9, out["delta_cost"] / weighted_saving, np.inf
    )

    out["switch"] = out["recommended_route"] != out["route_type"]
    lo_hi = bundle["distance_support"]
    rec_lo = out["recommended_route"].map({rt: lo_hi[rt][0] for rt in lo_hi})
    rec_hi = out["recommended_route"].map({rt: lo_hi[rt][1] for rt in lo_hi})
    out["extrapolated"] = (out["osrm_distance"] < rec_lo) | (out["osrm_distance"] > rec_hi)

    # Direction-aware framing: positive = following the recommendation saves
    # minutes / costs extra rupees relative to the dispatch's CURRENT route.
    rec_is_ftl = out["recommended_route"] == "FTL"
    out["switch_minutes_saved"] = np.where(
        rec_is_ftl, out["delta_time_min"], -out["delta_time_min"]
    )
    out["switch_extra_cost"] = np.where(
        rec_is_ftl, out["delta_cost"], -out["delta_cost"]
    )
    return out


def recommend(legs: pd.DataFrame, bundle: dict, **overrides) -> pd.DataFrame:
    """One-call API: counterfactual times + costs + recommendation per dispatch."""
    return apply_decision(score_times(legs, bundle), bundle, **overrides)


# ---------------------------------------------------------------------------
# 4. Validation: do the few co-observed corridors agree with the model?
# ---------------------------------------------------------------------------

def counterfactual_sanity_check(legs_scored: pd.DataFrame) -> None:
    """On corridors observed under BOTH route types, the model's predicted
    faster route should usually match the observed faster route."""
    obs = (
        legs_scored.groupby(["source_center", "destination_center", "route_type"],
                            observed=True)[TARGET]
        .median().unstack("route_type")
    )
    overlap = obs.dropna(subset=["FTL", "Carting"])
    if overlap.empty:
        print("  no overlap corridors found -- skipping")
        return

    pred = (
        legs_scored.groupby(["source_center", "destination_center"], observed=True)[
            ["pred_time_ftl", "pred_time_cart"]
        ].median()
    )
    joined = overlap.join(pred, how="inner")
    obs_faster = joined["FTL"] < joined["Carting"]
    pred_faster = joined["pred_time_ftl"] < joined["pred_time_cart"]
    agree = float((obs_faster == pred_faster).mean()) * 100
    print(f"  {len(joined)} corridors ran both route types | "
          f"predicted faster route matches observed: {agree:.0f}%")


# ---------------------------------------------------------------------------
# 5. Aggregation + visualisation
# ---------------------------------------------------------------------------

def aggregate_recommendations(scored: pd.DataFrame) -> pd.DataFrame:
    scored = scored.copy()
    scored["distance_band"] = pd.cut(
        scored["osrm_distance"], bins=DISTANCE_BANDS, labels=DISTANCE_LABELS
    )
    def _switch_mean(col):
        def inner(s):
            sw = scored.loc[s.index, "switch"]
            return scored.loc[s.index[sw], col].mean() if sw.any() else np.nan
        return inner

    agg = (
        scored.groupby(["distance_band", "route_type"], observed=True)
        .agg(
            n_dispatches=("recommended_route", "size"),
            pct_recommend_ftl=("recommended_route", lambda s: (s == "FTL").mean() * 100),
            n_switch=("switch", "sum"),
            avg_minutes_saved_if_switch=("switch", _switch_mean("switch_minutes_saved")),
            avg_extra_cost_if_switch=("switch", _switch_mean("switch_extra_cost")),
            avg_break_even_vot=("break_even_vot",
                                lambda s: (lambda f: f.median() if len(f) else np.nan)(
                                    s.replace(np.inf, np.nan).dropna())),
            pct_extrapolated=("extrapolated", lambda s: s.mean() * 100),
        )
        .reset_index()
    )
    return agg


def plot_tradeoff(scored: pd.DataFrame, params: dict, path: str) -> None:
    """Delta-time vs delta-cost scatter with the (average) break-even line."""
    fig, ax = plt.subplots(figsize=(10, 7))
    sample = scored.sample(min(len(scored), 6000), random_state=RANDOM_STATE)
    colors = sample["recommended_route"].map({"FTL": "#d1495b", "Carting": "#1f7a8c"})
    ax.scatter(sample["delta_time_min"], sample["delta_cost"],
               c=colors, s=10, alpha=0.45, linewidths=0)

    xs = np.linspace(sample["delta_time_min"].min(), sample["delta_time_min"].max(), 50)
    ax.plot(xs, params["value_of_time"] * xs, color="#333", lw=1.5, ls="--",
            label=f"break-even @ Rs{params['value_of_time']:.0f}/min (risk excluded)")
    ax.axhline(0, color="#999", lw=0.8)
    ax.axvline(0, color="#999", lw=0.8)
    ax.set_xlabel("Minutes FTL saves vs Carting (predicted)")
    ax.set_ylabel("Extra cost of FTL vs Carting (Rs)")
    ax.set_title("FTL vs Carting: time-cost trade-off per dispatch (test legs)")
    handles = [
        plt.Line2D([], [], marker="o", ls="", color="#d1495b", label="recommend FTL"),
        plt.Line2D([], [], marker="o", ls="", color="#1f7a8c", label="recommend Carting"),
        plt.Line2D([], [], color="#333", ls="--",
                   label=f"break-even @ Rs{params['value_of_time']:.0f}/min"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  plot saved -> {path}")


def plot_by_distance(scored: pd.DataFrame, bundle: dict, path: str) -> None:
    """Recommended-route share by distance band, with the common-support bands."""
    scored = scored.copy()
    scored["distance_band"] = pd.cut(
        scored["osrm_distance"], bins=DISTANCE_BANDS, labels=DISTANCE_LABELS
    )
    share = (
        scored.groupby("distance_band", observed=True)["recommended_route"]
        .value_counts(normalize=True).unstack().fillna(0) * 100
    )

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1]}
    )
    share.plot(kind="bar", stacked=True, ax=ax1,
               color={"FTL": "#d1495b", "Carting": "#1f7a8c"}, width=0.7)
    ax1.set_ylabel("% of dispatches")
    ax1.set_xlabel("")
    ax1.set_title("Recommended route type by distance band (test legs)")
    ax1.legend(title="", frameon=False)
    ax1.tick_params(axis="x", rotation=0)

    # Observed distance support per route type (log scale; p5-p95 band).
    for i, rt in enumerate(["Carting", "FTL"]):
        lo, hi = bundle["distance_support"][rt]
        color = "#1f7a8c" if rt == "Carting" else "#d1495b"
        ax2.barh(i, hi - lo, left=lo, height=0.5, color=color, alpha=0.6)
        ax2.text(hi, i, f" {rt}: {lo:.0f}-{hi:.0f} km", va="center", fontsize=9)
    ax2.set_xscale("log")
    ax2.set_yticks([])
    ax2.set_xlabel("Observed training distance support, p5-p95 (km, log scale)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  plot saved -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task 4: FTL vs Carting decision framework.")
    p.add_argument("--retrain", action="store_true",
                   help="Force retraining even if models/route_choice.joblib exists.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(VIZ_DIR, exist_ok=True)

    print("Loading leg-level data...")
    legs = load_legs()
    train, test = split_train_test(legs)

    if os.path.exists(BUNDLE_PATH) and not args.retrain:
        print(f"Loading cached bundle -> {BUNDLE_PATH} (pass --retrain to rebuild)")
        bundle = joblib.load(BUNDLE_PATH)
    else:
        print("Training T-learner time models + p90 tail models...")
        bundle = train_bundle(train, test)
        joblib.dump(bundle, BUNDLE_PATH)
        print(f"  bundle saved -> {BUNDLE_PATH}")

    print("\nCounterfactual sanity check (corridors observed on both route types):")
    counterfactual_sanity_check(score_times(legs, bundle))

    print("\nScoring test legs with default cost/decision parameters...")
    scored = recommend(test, bundle)
    keep = [
        "trip_uuid", "source_center", "source_name", "destination_center",
        "destination_name", "data", "route_type", "tod_bucket",
        "osrm_distance", "osrm_time", "actual_time",
        "hour_of_day", "day_of_week",
        "src_betweenness", "src_bottleneck_score", "src_avg_delay",
        "src_in_degree", "src_out_degree", "dst_bottleneck_score",
        "pred_time_ftl", "pred_time_cart", "pred_p90_ftl", "pred_p90_cart",
        "delta_time_min", "delta_risk_min", "cost_ftl", "cost_cart", "delta_cost",
        "vot_effective", "benefit_ftl", "break_even_vot",
        "recommended_route", "switch", "extrapolated",
    ]
    scored[[c for c in keep if c in scored.columns]].to_csv(FEATURES_CSV, index=False)
    print(f"  scored legs saved -> {FEATURES_CSV}")

    agg = aggregate_recommendations(scored)
    agg.to_csv(RECS_CSV, index=False)
    print(f"  switch-opportunity summary saved -> {RECS_CSV}")

    plot_tradeoff(scored, bundle["default_params"], PLOT_TRADEOFF)
    plot_by_distance(scored, bundle, PLOT_BY_DIST)

    # Console summary for the Task-5 memo. Headline numbers use only switches
    # INSIDE the recommended route's distance support -- extrapolated counter-
    # factuals are counted but never averaged into the savings claims.
    print("\n" + "=" * 72)
    print("SWITCH OPPORTUNITIES (test legs, default parameters)")
    print("=" * 72)
    for current, target in [("Carting", "FTL"), ("FTL", "Carting")]:
        now = scored[scored["route_type"] == current]
        sw = now[now["switch"]]
        trusted = sw[~sw["extrapolated"]]
        print(f"{current} -> {target} : {len(sw):,} of {len(now):,} dispatches "
              f"({len(sw) / max(len(now), 1) * 100:.1f}%) | "
              f"trusted (inside support): {len(trusted):,}")
        if len(trusted):
            extra = trusted["switch_extra_cost"].mean()
            cost_txt = (f"avg extra cost Rs{extra:,.0f}" if extra >= 0
                        else f"avg cost SAVED Rs{-extra:,.0f}")
            print(f"   trusted switches: avg minutes saved "
                  f"{trusted['switch_minutes_saved'].mean():.1f} | {cost_txt}")
    print(f"Overall inside common support: {100 - scored['extrapolated'].mean() * 100:.0f}%")
    print("=" * 72)
    print("\nDone!\n")
    return bundle, scored


if __name__ == "__main__":
    bundle, scored = main()
