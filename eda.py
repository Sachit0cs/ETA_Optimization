"""
Exploratory Data Analysis for the Delhivery graph-ETA project.

Run:
    python eda.py                      # all sections
    python eda.py --sections target_delay,temporal,corridor_edge
    python eda.py --list               # list registered sections + variables

Outputs:
    outputs/eda/EDA_REPORT.md          # narrative report with findings + tables
    outputs/eda/plots/*.png            # every figure
    outputs/eda/hyperparameter_signals.csv   # machine-readable signals that may
                                             # justify changing model/graph config

Extensibility
-------------
Each analysis is a `@section(...)`-decorated function (see eda_utils). To add EDA
for node2vec embeddings, the Task-4 FTL/Carting framework, or the Task-5
hub-impact variables, just write another decorated function -- the runner picks
it up automatically. Stubs for those three are included at the bottom, disabled
until the upstream artifacts exist.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import eda_utils as E
from eda_utils import section

DATA_PATH = "delivery_data.csv"
NODE_METRICS_PATH = "outputs/node_metrics.csv"
CORRIDOR_PATH = "outputs/corridor_audit.csv"
EMB_PATH = "outputs/node_emb_graphsage.csv"


# ===========================================================================
# THE VARIABLE LIST  --  what we run EDA on, grouped by role.
# Each future task (node2vec / task4 / task5) appends its own group here.
# ===========================================================================

VARIABLE_GROUPS: dict[str, dict] = {
    "target_delay": {
        "desc": "Delivery time + delay-ratio targets the models must predict.",
        "vars": ["actual_time (trip)", "factor (= actual/osrm)", "segment_factor",
                 "segment_actual_time"],
    },
    "osrm_features": {
        "desc": "OSRM routing-engine estimates -- the main predictive signal.",
        "vars": ["osrm_time (trip)", "osrm_distance (trip)",
                 "segment_osrm_time", "segment_osrm_distance"],
    },
    "distance": {
        "desc": "Distance variables and OSRM-vs-actual distance agreement.",
        "vars": ["actual_distance_to_destination", "osrm_distance"],
    },
    "categorical": {
        "desc": "Route type and scan-type flags.",
        "vars": ["route_type", "is_ftl", "is_cutoff"],
    },
    "temporal": {
        "desc": "Time-of-day / day-of-week / seasonality of trips and delays.",
        "vars": ["trip_creation_time -> hour_of_day, day_of_week",
                 "od_start_time -> tod_bucket"],
    },
    "trip_structure": {
        "desc": "Multi-leg structure of a trip (graph path length).",
        "vars": ["segments_per_trip", "legs_per_trip"],
    },
    "graph_structural": {
        "desc": "Node-level graph metrics used as model features (node_metrics.csv).",
        "vars": ["betweenness", "in_degree", "out_degree",
                 "in_degree_weighted", "out_degree_weighted",
                 "clustering", "avg_incoming_delay_factor", "bottleneck_score"],
    },
    "corridor_edge": {
        "desc": "Edge/corridor metrics driving the bottleneck audit (corridor_audit.csv).",
        "vars": ["median_factor", "total_trips", "pct_delayed",
                 "median_osrm_dist_km", "is_sparse", "is_chronically_delayed"],
    },
    "relationships": {
        "desc": "Bivariate structure: actual~osrm, delay~distance, delay~graph position, "
                "and the model-feature correlation matrix.",
        "vars": ["actual_time~osrm_time", "factor~osrm_distance",
                 "factor~src_bottleneck_score", "feature correlation matrix"],
    },
    "embeddings": {
        "desc": "Learned node embeddings used as features (GraphSAGE now; node2vec later).",
        "vars": ["emb_0 .. emb_7"],
    },
}


# ---------------------------------------------------------------------------
# Data loading (self-contained: no xgboost/torch import, so EDA runs anywhere)
# ---------------------------------------------------------------------------

def load_context() -> E.EDAContext:
    print("Loading delivery_data.csv ...")
    raw = pd.read_csv(DATA_PATH)
    for col in ["trip_creation_time", "od_start_time", "od_end_time"]:
        if col in raw.columns:
            raw[col] = pd.to_datetime(raw[col], errors="coerce")

    trips = build_trip_level(raw)

    nodes = pd.read_csv(NODE_METRICS_PATH) if os.path.exists(NODE_METRICS_PATH) else None
    corridors = pd.read_csv(CORRIDOR_PATH) if os.path.exists(CORRIDOR_PATH) else None
    emb = pd.read_csv(EMB_PATH) if os.path.exists(EMB_PATH) else None

    # Attach source-hub graph position to trips for relationship EDA.
    if nodes is not None:
        keep = nodes[["center", "betweenness", "bottleneck_score",
                      "avg_incoming_delay_factor"]].rename(
            columns={"betweenness": "src_betweenness",
                     "bottleneck_score": "src_bottleneck_score",
                     "avg_incoming_delay_factor": "src_avg_delay"})
        trips = trips.merge(keep, left_on="source_center", right_on="center",
                            how="left").drop(columns="center")

    print(f"  raw rows: {len(raw):,} | trips: {len(trips):,} | "
          f"nodes: {0 if nodes is None else len(nodes)} | "
          f"corridors: {0 if corridors is None else len(corridors)}")
    return E.EDAContext(raw=raw, trips=trips, nodes=nodes, corridors=corridors,
                        embeddings=emb)


def build_trip_level(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per trip (mirrors task3.aggregate_to_trips, kept dependency-free)."""
    df = raw.sort_values(["trip_uuid", "od_start_time"])
    g = df.groupby("trip_uuid", sort=False)
    trips = g.agg(
        actual_time=("segment_actual_time", "sum"),
        osrm_time=("segment_osrm_time", "sum"),
        osrm_distance=("segment_osrm_distance", "sum"),
        actual_distance_to_destination=("actual_distance_to_destination", "max"),
        source_center=("source_center", "first"),
        destination_center=("destination_center", "last"),
        route_type=("route_type", "first"),
        trip_creation_time=("trip_creation_time", "first"),
        data=("data", "first"),
        segments_per_trip=("trip_uuid", "size"),
        legs_per_trip=("is_cutoff", lambda s: int((~s.astype(bool)).sum())),
    ).reset_index()

    trips["factor"] = trips["actual_time"] / trips["osrm_time"].replace(0, np.nan)
    trips["is_ftl"] = (trips["route_type"] == "FTL").astype(int)
    trips["hour_of_day"] = trips["trip_creation_time"].dt.hour
    trips["day_of_week"] = trips["trip_creation_time"].dt.dayofweek
    return trips


# Collects (signal, value, note, suggested_action) rows that may justify a
# hyperparameter / config change. Written to CSV at the end.
SIGNALS: list[dict] = []


def signal(name: str, value, note: str, action: str) -> None:
    SIGNALS.append({"signal": name, "value": value, "note": note,
                    "suggested_action": action})


# ===========================================================================
# SECTIONS
# ===========================================================================

@section("overview", "Dataset shape, missingness, train/test balance", order=1)
def s_overview(ctx: E.EDAContext) -> None:
    ctx.heading("1. Dataset Overview")
    raw, trips = ctx.raw, ctx.trips
    ctx.add_finding(f"Segment-level rows: {len(raw):,}; distinct trips: {trips['trip_uuid'].nunique():,}.")
    if "data" in raw.columns:
        split = trips["data"].value_counts()
        ctx.add_finding("Trip split -> " + ", ".join(f"{k}: {v:,}" for k, v in split.items()))
    ctx.add_finding(f"Facilities (nodes): {pd.unique(raw[['source_center','destination_center']].values.ravel()).size:,}.")

    miss = (raw.isna().mean() * 100).round(2)
    miss = miss[miss > 0].sort_values(ascending=False)
    if len(miss):
        mt = pd.DataFrame({"column": miss.index, "missing_%": miss.values})
        ctx.add_table(mt, "Columns with missing values (segment-level)")


@section("target_delay", "Delivery-time and delay-ratio target distributions", order=10)
def s_target(ctx: E.EDAContext) -> None:
    ctx.heading("2. Target & Delay Variables")
    trips, raw = ctx.trips, ctx.raw

    summ = E.numeric_summary(trips, ["actual_time", "factor"])
    summ2 = E.numeric_summary(raw, ["segment_actual_time", "segment_factor", "factor"])
    ctx.add_table(pd.concat([summ, summ2], ignore_index=True), "Target/delay summary (trip- and segment-level)")

    ctx.add_plot(E.plot_distribution(trips["actual_time"].rename("actual_time"),
                 "target_actual_time_hist.png", "Trip actual_time (minutes)"), "Trip ETA target -- raw")
    ctx.add_plot(E.plot_distribution(trips["actual_time"].rename("actual_time"),
                 "target_actual_time_log.png", "Trip actual_time -- log1p", logx=True), "Trip ETA target -- log1p")
    ctx.add_plot(E.plot_distribution(trips["factor"].rename("factor"),
                 "target_factor_hist.png", "Trip delay factor (actual/osrm)"), "Delay factor")

    skew = float(pd.to_numeric(trips["actual_time"], errors="coerce").skew())
    med_factor = float(pd.to_numeric(trips["factor"], errors="coerce").median())
    ctx.add_finding(f"Trip `actual_time` is heavily right-skewed (skew={skew:.2f}; "
                    f"median {trips['actual_time'].median():.0f} vs max {trips['actual_time'].max():.0f} min).")
    ctx.add_finding(f"Median trip delay factor = {med_factor:.2f} -> OSRM systematically under-estimates ETA "
                    f"by ~{(med_factor-1)*100:.0f}% on a typical trip.")
    if skew > 2:
        signal("target_skew", round(skew, 2),
               "actual_time is heavily right-skewed; squared-error loss over-weights the long tail.",
               "Train models on log1p(actual_time) (or use an L1/Tweedie objective) and invert for scoring.")


@section("osrm_features", "OSRM routing-engine feature distributions", order=20)
def s_osrm(ctx: E.EDAContext) -> None:
    ctx.heading("3. OSRM Features")
    ctx.add_table(E.numeric_summary(ctx.trips, ["osrm_time", "osrm_distance"]),
                  "Trip-level OSRM features")
    ctx.add_table(E.numeric_summary(ctx.raw, ["segment_osrm_time", "segment_osrm_distance"]),
                  "Segment-level OSRM features")
    ctx.add_plot(E.plot_distribution(ctx.trips["osrm_time"].rename("osrm_time"),
                 "osrm_time_hist.png", "Trip osrm_time (minutes)"), "OSRM time")
    ctx.add_plot(E.plot_distribution(ctx.trips["osrm_distance"].rename("osrm_distance"),
                 "osrm_distance_hist.png", "Trip osrm_distance (km)"), "OSRM distance")
    corr = float(ctx.trips[["osrm_time", "actual_time"]].corr().iloc[0, 1])
    ctx.add_finding(f"Trip osrm_time correlates {corr:.3f} with actual_time -- a strong single predictor, "
                    "so the graph features must add lift beyond OSRM to be worthwhile.")


@section("distance", "Distance variables and OSRM-vs-actual distance agreement", order=30)
def s_distance(ctx: E.EDAContext) -> None:
    ctx.heading("4. Distance")
    ctx.add_table(E.numeric_summary(ctx.trips, ["osrm_distance", "actual_distance_to_destination"]),
                  "Distance summary (trip-level)")
    ctx.add_plot(E.plot_scatter(ctx.trips, "osrm_distance", "actual_distance_to_destination",
                 "dist_osrm_vs_actual.png", "OSRM vs actual distance (trip)"),
                 "OSRM vs actual distance")


@section("categorical", "Route type and scan-flag breakdowns", order=40)
def s_categorical(ctx: E.EDAContext) -> None:
    ctx.heading("5. Categorical Variables")
    ctx.add_table(E.categorical_summary(ctx.trips["route_type"]), "route_type (trip-level)")
    ctx.add_table(E.categorical_summary(ctx.raw["is_cutoff"]), "is_cutoff (segment-level)")
    ctx.add_plot(E.plot_box_by_group(ctx.trips, "factor", "route_type",
                 "factor_by_route_type.png", "Delay factor by route type"),
                 "Delay factor by route type")
    grp = ctx.trips.groupby("route_type")["factor"].median()
    ctx.add_finding("Median delay factor by route type -> " +
                    ", ".join(f"{k}: {v:.2f}" for k, v in grp.items()) +
                    " (informs the Task-4 FTL-vs-Carting trade-off).")


@section("temporal", "Time-of-day / day-of-week patterns in volume and delay", order=50)
def s_temporal(ctx: E.EDAContext) -> None:
    ctx.heading("6. Temporal Patterns")
    trips = ctx.trips
    vol = trips["hour_of_day"].value_counts().sort_index()
    ctx.add_plot(E.plot_bar(vol.index, vol.values, "vol_by_hour.png",
                 "Trip volume by creation hour", "hour_of_day", "trips"), "Volume by hour")
    delay_by_hour = trips.groupby("hour_of_day")["factor"].median()
    ctx.add_plot(E.plot_bar(delay_by_hour.index, delay_by_hour.values, "delay_by_hour.png",
                 "Median delay factor by hour", "hour_of_day", "median factor"), "Delay by hour")
    dow = trips.groupby("day_of_week")["factor"].median()
    ctx.add_plot(E.plot_bar(dow.index, dow.values, "delay_by_dow.png",
                 "Median delay factor by day of week", "day_of_week (0=Mon)", "median factor"),
                 "Delay by day of week")
    spread = float(delay_by_hour.max() - delay_by_hour.min())
    ctx.add_finding(f"Delay factor varies by hour-of-day (median range {spread:.2f}) -> "
                    "time-of-day stratification of edge weights is justified.")


@section("trip_structure", "Multi-leg structure of trips", order=60)
def s_structure(ctx: E.EDAContext) -> None:
    ctx.heading("7. Trip Structure")
    ctx.add_table(E.numeric_summary(ctx.trips, ["segments_per_trip", "legs_per_trip"]),
                  "Segments / legs per trip")
    ctx.add_plot(E.plot_distribution(ctx.trips["legs_per_trip"].rename("legs_per_trip"),
                 "legs_per_trip_hist.png", "Legs (hops) per trip", bins=30), "Legs per trip")
    multi = float((ctx.trips["legs_per_trip"] > 1).mean() * 100)
    ctx.add_finding(f"{multi:.1f}% of trips are multi-leg (hub-and-spoke) -> the graph path matters; "
                    "single-leg trips are the simplest baseline case.")


@section("graph_structural", "Node-level graph metric distributions", order=70)
def s_graph(ctx: E.EDAContext) -> None:
    ctx.heading("8. Graph Structural Metrics (node-level)")
    if ctx.nodes is None:
        ctx.add_note("_node_metrics.csv not found -- run build_graph.py first._")
        return
    cols = ["betweenness", "in_degree", "out_degree", "in_degree_weighted",
            "out_degree_weighted", "clustering", "avg_incoming_delay_factor", "bottleneck_score"]
    ctx.add_table(E.numeric_summary(ctx.nodes, cols), "Node-metric summary")
    for c in ["betweenness", "in_degree", "bottleneck_score"]:
        ctx.add_plot(E.plot_distribution(ctx.nodes[c].rename(c), f"node_{c}_hist.png",
                     f"Node {c}", bins=40), f"Node {c}")
    bw_skew = float(pd.to_numeric(ctx.nodes["betweenness"], errors="coerce").skew())
    ctx.add_finding(f"Betweenness is highly skewed (skew={bw_skew:.1f}) -> a few hub facilities dominate "
                    "routing, consistent with a hub-and-spoke chokepoint structure.")


@section("corridor_edge", "Edge/corridor metrics and bottleneck thresholds", order=80)
def s_corridor(ctx: E.EDAContext) -> None:
    ctx.heading("9. Corridor / Edge Metrics")
    if ctx.corridors is None:
        ctx.add_note("_corridor_audit.csv not found -- run build_graph.py first._")
        return
    cdf = ctx.corridors
    ctx.add_table(E.numeric_summary(cdf, ["median_factor", "total_trips", "pct_delayed",
                                          "median_osrm_dist_km"]), "Corridor metric summary")
    ctx.add_plot(E.plot_distribution(cdf["total_trips"].rename("total_trips"),
                 "corridor_trips_hist.png", "Trips per corridor", bins=40, logx=True),
                 "Trips per corridor (log1p)")

    # SPARSE_THRESHOLD diagnostic
    for thr in [2, 3, 5, 10]:
        share = float((cdf["total_trips"] < thr).mean() * 100)
        ctx.add_finding(f"Corridors with < {thr} trips: {share:.1f}%.")
    sparse5 = float((cdf["total_trips"] < 5).mean() * 100)
    signal("sparse_share_lt5", round(sparse5, 1),
           f"{sparse5:.1f}% of corridors have <5 trips; their raw medians are unreliable.",
           "Keep SPARSE_THRESHOLD=5 with the route-type-median fallback (already applied).")

    # DELAY_THRESHOLD (=1.2) diagnostic -- the brief FIXES this at >20%.
    if "is_chronically_delayed" in cdf.columns:
        share_delayed = float(cdf["is_chronically_delayed"].mean() * 100)
        ctx.add_finding(f"{share_delayed:.1f}% of corridors exceed the >20% (factor>1.2) threshold the brief "
                        "defines for 'chronically delayed'. The threshold is fixed by the problem statement, "
                        "but the near-universal breach is itself the headline insight: OSRM under-estimates "
                        "almost everywhere, so prioritise by severity x volume, not the binary flag.")


@section("relationships", "Bivariate relationships + model-feature correlation matrix", order=90)
def s_relationships(ctx: E.EDAContext) -> None:
    ctx.heading("10. Relationships")
    trips = ctx.trips
    ctx.add_plot(E.plot_scatter(trips, "osrm_time", "actual_time",
                 "rel_osrm_vs_actual.png", "actual_time vs osrm_time (trip)"),
                 "The core gap OSRM misses")
    if "src_bottleneck_score" in trips.columns:
        ctx.add_plot(E.plot_scatter(trips, "src_bottleneck_score", "factor",
                     "rel_bottleneck_vs_factor.png", "delay factor vs source bottleneck score"),
                     "Does graph position predict delay?")
    cols = ["actual_time", "osrm_time", "osrm_distance", "is_ftl", "hour_of_day",
            "day_of_week", "legs_per_trip", "src_betweenness", "src_bottleneck_score", "src_avg_delay"]
    res = E.plot_corr_heatmap(trips, cols, "rel_corr_heatmap.png", "Trip-level feature correlations")
    path, corr = res if isinstance(res, tuple) else (res, None)
    ctx.add_plot(path, "Feature correlation matrix")
    if corr is not None and "actual_time" in corr.columns:
        top = corr["actual_time"].drop("actual_time").abs().sort_values(ascending=False).head(5)
        ctx.add_finding("Top correlates of actual_time -> " +
                        ", ".join(f"{k} ({corr['actual_time'][k]:.2f})" for k in top.index) + ".")


@section("embeddings", "Learned node-embedding distributions (GraphSAGE)", order=95)
def s_embeddings(ctx: E.EDAContext) -> None:
    ctx.heading("11. Node Embeddings (GraphSAGE)")
    if ctx.embeddings is None:
        ctx.add_note("_node_emb_graphsage.csv not found -- run task3 --retrain first._")
        return
    emb_cols = [c for c in ctx.embeddings.columns if c.startswith("emb_")]
    ctx.add_table(E.numeric_summary(ctx.embeddings, emb_cols), "Embedding dimension summary")
    # How many dims are effectively dead (near-zero variance)?
    var = ctx.embeddings[emb_cols].var()
    dead = int((var < 1e-6).sum())
    ctx.add_finding(f"GraphSAGE embedding: {len(emb_cols)} dims, {dead} near-constant. "
                    f"EMB_DIM={len(emb_cols)} looks {'over-sized' if dead else 'reasonable'}.")
    if dead >= len(emb_cols) // 2:
        signal("dead_embeddings", dead,
               f"{dead}/{len(emb_cols)} embedding dims are near-constant.",
               "Reduce EMB_DIM (e.g. to 4) to cut noise features.")


# ===========================================================================
# EXTENSION STUBS  --  enable when the upstream artifacts land.
# Pattern: copy a section above, point it at the new data, and it auto-registers.
# ===========================================================================

@section("node2vec", "node2vec embedding EDA (enable once node2vec is added)", order=110)
def s_node2vec(ctx: E.EDAContext) -> None:
    path = "outputs/node_emb_node2vec.csv"
    if not os.path.exists(path):
        return  # silently skip until node2vec embeddings exist
    ctx.heading("12. Node Embeddings (node2vec)")
    emb = pd.read_csv(path)
    emb_cols = [c for c in emb.columns if c.startswith("emb_")]
    ctx.add_table(E.numeric_summary(emb, emb_cols), "node2vec embedding summary")
    ctx.add_finding("node2vec embeddings present; compare downstream lift vs GraphSAGE in task3.")


@section("task4_route_choice", "FTL-vs-Carting feature EDA (Task 4)", order=120)
def s_task4(ctx: E.EDAContext) -> None:
    path = "outputs/route_choice_features.csv"
    if not os.path.exists(path):
        return  # enable when the Task-4 framework writes its feature table
    ctx.heading("13. FTL vs Carting -- Route-Choice Features (Task 4)")
    df = pd.read_csv(path)
    ctx.add_table(E.numeric_summary(df, [c for c in df.columns if df[c].dtype != object]),
                  "Route-choice feature summary")


@section("task5_hub_impact", "Hub SLA-breach / revenue-impact EDA (Task 5)", order=130)
def s_task5(ctx: E.EDAContext) -> None:
    path = "outputs/hub_impact.csv"
    if not os.path.exists(path):
        return  # enable when the Task-5 strategy-memo pipeline writes hub impact
    ctx.heading("14. Hub SLA-Breach & Revenue Impact (Task 5)")
    df = pd.read_csv(path)
    ctx.add_table(E.numeric_summary(df, [c for c in df.columns if df[c].dtype != object]),
                  "Hub-impact feature summary")


# ===========================================================================
# Runner
# ===========================================================================

def write_report(ctx: E.EDAContext, ran: list[str]) -> None:
    E.ensure_dirs()
    header = [
        "# Delhivery Graph-ETA -- Exploratory Data Analysis",
        "",
        "_Auto-generated by `eda.py`. Plots in `plots/`._",
        "",
        "## Variables analysed",
        "",
    ]
    for name, meta in VARIABLE_GROUPS.items():
        header.append(f"- **{name}** -- {meta['desc']}")
        header.append(f"  - vars: {', '.join(meta['vars'])}")
    header += ["", f"Sections executed this run: {', '.join(ran)}", ""]

    body = ctx.report_markdown()

    if SIGNALS:
        sig_df = pd.DataFrame(SIGNALS)
        sig_df.to_csv(os.path.join(E.EDA_DIR, "hyperparameter_signals.csv"), index=False)
        body += "\n\n## Hyperparameter / config signals\n\n" + E.df_to_md(sig_df) + "\n"

    with open(E.REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n" + body + "\n")
    print(f"\nReport written -> {E.REPORT_PATH}")
    if SIGNALS:
        print(f"Hyperparameter signals -> {os.path.join(E.EDA_DIR, 'hyperparameter_signals.csv')}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run EDA for the Delhivery graph-ETA project.")
    p.add_argument("--sections", default=None,
                   help="Comma-separated section names to run (default: all).")
    p.add_argument("--list", action="store_true", help="List sections + variables and exit.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.list:
        print("Variable groups:")
        for name, meta in VARIABLE_GROUPS.items():
            print(f"  {name}: {meta['desc']}\n     {', '.join(meta['vars'])}")
        print("\nRegistered sections:")
        for name, desc, _ in E.get_sections():
            print(f"  {name:20s} {desc}")
        return

    only = args.sections.split(",") if args.sections else None
    ctx = load_context()

    ran = []
    for name, desc, fn in E.get_sections(only):
        print(f"\n[section] {name} -- {desc}")
        fn(ctx)
        ran.append(name)

    write_report(ctx, ran)
    print("\nEDA done.")


if __name__ == "__main__":
    main()
