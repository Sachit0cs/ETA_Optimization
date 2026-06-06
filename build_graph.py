"""
Graph construction pipeline for Delhivery logistics network.

Builds a directed weighted graph:
  - Nodes  = facilities (source_center / destination_center IDs)
  - Edges  = corridors (directed source → destination)
  - Weight = median delay factor (actual_time / osrm_time) per corridor

Edge weight lookup table is stratified by (route_type × time_of_day bucket)
so the ML pipeline can retrieve the right weight at prediction time.

Outputs (saved to ./outputs/):
  - logistics_graph.pkl        NetworkX DiGraph
  - edge_weight_table.csv      Per (src, dst, route_type, tod_bucket) stats
  - edge_rt_aggregated.csv     Per (src, dst, route_type) aggregate
  - node_metrics.csv           Betweenness, degree, bottleneck score per node
  - corridor_audit.csv         All edges with delay flag
  - graph_summary.txt          Human-readable summary
"""

import os
import pickle

import numpy as np
import pandas as pd
import networkx as nx

DATA_PATH = "delivery_data.csv"
OUTPUT_DIR = "outputs"
SPARSE_THRESHOLD = 5      # corridors with fewer trips use route_type fallback
DELAY_THRESHOLD = 1.2     # factor > this = chronically delayed


# ---------------------------------------------------------------------------
# 1. Load & preprocess
# ---------------------------------------------------------------------------

def load_and_preprocess(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Leakage guard: build the graph and ALL edge/node aggregates from TRAINING
    # trips only. Otherwise test-set delay information leaks into the graph
    # features later used to predict the test set.
    if "data" in df.columns:
        n_before = len(df)
        df = df[df["data"] == "training"].copy()
        print(f"  filtered to training rows: {len(df):,} / {n_before:,}")

    for col in ["od_start_time", "od_end_time", "trip_creation_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # 6-hour time-of-day buckets based on leg start
    df["hour"] = df["od_start_time"].dt.hour
    df["tod_bucket"] = pd.cut(
        df["hour"],
        bins=[0, 6, 12, 18, 24],
        labels=["night", "morning", "afternoon", "evening"],
        right=False,
    )

    # Normalise is_cutoff to bool in case it is stored as string
    if df["is_cutoff"].dtype == object:
        df["is_cutoff"] = df["is_cutoff"].str.strip().str.lower().map(
            {"true": True, "false": False}
        )

    return df


# ---------------------------------------------------------------------------
# 2. Extract one departure record per OD leg
# ---------------------------------------------------------------------------

def extract_leg_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    is_cutoff=False rows are the departure scan for each OD leg.
    actual_time and osrm_time at departure = full remaining time for the leg,
    so factor = actual_time / osrm_time is the per-leg delay ratio.
    """
    departures = df[~df["is_cutoff"]].copy()
    return departures


# ---------------------------------------------------------------------------
# 3. Build edge weight lookup tables
# ---------------------------------------------------------------------------

def build_edge_weight_table(departures: pd.DataFrame):
    """
    Returns:
      edge_stats     – per (src, dst, route_type, tod_bucket)
      edge_rt_agg    – per (src, dst, route_type)  [primary edge table]
      rt_medians     – route_type → median factor  [sparse fallback]
      global_median  – float                       [last-resort fallback]
    """
    rt_medians = departures.groupby("route_type")["factor"].median().to_dict()
    global_median = float(departures["factor"].median())

    # Fine-grained: route_type × time-of-day
    edge_stats = (
        departures
        .groupby(
            ["source_center", "destination_center", "route_type", "tod_bucket"],
            observed=True,
        )
        .agg(
            median_factor=("factor", "median"),
            trip_count=("trip_uuid", "nunique"),
            median_osrm_dist=("osrm_distance", "median"),
            median_osrm_time=("osrm_time", "median"),
            pct_delayed=("factor", lambda x: (x > DELAY_THRESHOLD).mean()),
        )
        .reset_index()
    )

    # Coarser: route_type only (used as primary edge attribute)
    edge_rt_agg = (
        departures
        .groupby(["source_center", "destination_center", "route_type"])
        .agg(
            overall_median_factor=("factor", "median"),
            total_trips=("trip_uuid", "nunique"),
            median_osrm_dist=("osrm_distance", "median"),
            median_osrm_time=("osrm_time", "median"),
            pct_delayed=("factor", lambda x: (x > DELAY_THRESHOLD).mean()),
        )
        .reset_index()
    )

    return edge_stats, edge_rt_agg, rt_medians, global_median


# ---------------------------------------------------------------------------
# 4. Construct NetworkX DiGraph
# ---------------------------------------------------------------------------

def build_graph(
    edge_rt_agg: pd.DataFrame,
    edge_stats: pd.DataFrame,
    df: pd.DataFrame,
    rt_medians: dict,
    global_median: float,
) -> nx.DiGraph:
    G = nx.DiGraph()

    # Build facility ID → name lookup from both source and destination columns
    # Use str() to guard against NaN values in name columns
    node_meta: dict = {}
    for _, row in df[["source_center", "source_name"]].drop_duplicates().iterrows():
        node_meta[row["source_center"]] = (
            str(row["source_name"]) if pd.notna(row["source_name"]) else str(row["source_center"])
        )
    for _, row in df[["destination_center", "destination_name"]].drop_duplicates().iterrows():
        if row["destination_center"] not in node_meta:
            node_meta[row["destination_center"]] = (
                str(row["destination_name"]) if pd.notna(row["destination_name"]) else str(row["destination_center"])
            )

    all_centers = set(df["source_center"].unique()) | set(df["destination_center"].unique())
    for center in all_centers:
        G.add_node(center, name=node_meta.get(center, str(center)))

    # Add edges grouped by route_type; aggregate across route types onto one edge
    for (src, dst), grp in edge_rt_agg.groupby(["source_center", "destination_center"]):
        total_trips = int(grp["total_trips"].sum())
        # Overall median factor weighted by trip count across route types
        weighted_factor = float(
            np.average(grp["overall_median_factor"], weights=grp["total_trips"])
        )
        # Sparse corridors have too few trips for a trustworthy median, so a single
        # outlier trip (e.g. factor=50) would otherwise top the bottleneck ranking.
        # Fall back to the trip-weighted route-type median instead.
        if total_trips < SPARSE_THRESHOLD:
            weighted_factor = float(
                np.average(
                    [rt_medians.get(rt, global_median) for rt in grp["route_type"]],
                    weights=grp["total_trips"],
                )
            )
        median_dist = float(grp["median_osrm_dist"].median())
        pct_delayed = float(grp["pct_delayed"].mean())

        # Per-route-type breakdown stored as nested dict
        rt_breakdown = {
            row["route_type"]: {
                "overall_median_factor": row["overall_median_factor"],
                "total_trips": int(row["total_trips"]),
                "median_osrm_dist": row["median_osrm_dist"],
                "pct_delayed": row["pct_delayed"],
            }
            for _, row in grp.iterrows()
        }

        G.add_edge(
            src, dst,
            weight=weighted_factor,       # scalar weight for graph algorithms
            total_trips=total_trips,
            median_osrm_dist=median_dist,
            pct_delayed=pct_delayed,
            is_sparse=(total_trips < SPARSE_THRESHOLD),
            route_types=rt_breakdown,
            tod_lookup={},                # filled in next loop
        )

    # Attach time-of-day lookup table to each existing edge
    for _, row in edge_stats.iterrows():
        src, dst = row["source_center"], row["destination_center"]
        if pd.isna(row["tod_bucket"]):
            continue  # skip legs with an unparseable start time (phantom "nan" bucket)
        if G.has_edge(src, dst):
            key = (row["route_type"], str(row["tod_bucket"]))
            G[src][dst]["tod_lookup"][key] = {
                "median_factor": row["median_factor"],
                "trip_count": int(row["trip_count"]),
                "pct_delayed": row["pct_delayed"],
            }

    return G


# ---------------------------------------------------------------------------
# 5. Graph metrics
# ---------------------------------------------------------------------------

def compute_graph_metrics(G: nx.DiGraph) -> pd.DataFrame:
    print("  betweenness centrality...")
    # Topological betweenness (unweighted). NetworkX treats edge weight as a
    # DISTANCE, so passing the delay factor would make high-delay corridors look
    # "far" and be avoided by shortest paths -- inverting the meaning of a
    # chokepoint. Delay severity is captured separately via avg_incoming_delay_factor.
    betweenness = nx.betweenness_centrality(G, weight=None, normalized=True)

    in_degree   = dict(G.in_degree())
    out_degree  = dict(G.out_degree())
    in_deg_w    = dict(G.in_degree(weight="weight"))
    out_deg_w   = dict(G.out_degree(weight="weight"))

    print("  clustering coefficient...")
    clustering  = nx.clustering(G.to_undirected())

    # Average delay factor of all incoming edges (proxy for hub congestion)
    avg_in_delay = {}
    for node in G.nodes():
        preds = list(G.predecessors(node))
        avg_in_delay[node] = (
            float(np.mean([G[p][node]["weight"] for p in preds]))
            if preds else 0.0
        )

    records = []
    for n in G.nodes():
        records.append(
            {
                "center": n,
                "name": G.nodes[n].get("name", ""),
                "betweenness": betweenness[n],
                "in_degree": in_degree[n],
                "out_degree": out_degree[n],
                "in_degree_weighted": in_deg_w[n],
                "out_degree_weighted": out_deg_w[n],
                "clustering": clustering[n],
                "avg_incoming_delay_factor": avg_in_delay[n],
            }
        )

    metrics = pd.DataFrame(records)
    # Bottleneck score: nodes with high betweenness AND high incoming delay
    metrics["bottleneck_score"] = (
        metrics["betweenness"] * metrics["avg_incoming_delay_factor"]
    )
    metrics = metrics.sort_values("bottleneck_score", ascending=False).reset_index(drop=True)
    return metrics


# ---------------------------------------------------------------------------
# 6. Corridor audit
# ---------------------------------------------------------------------------

def identify_delayed_corridors(G: nx.DiGraph) -> pd.DataFrame:
    rows = []
    for src, dst, data in G.edges(data=True):
        rows.append(
            {
                "source_center": src,
                "destination_center": dst,
                "source_name": G.nodes[src].get("name", ""),
                "destination_name": G.nodes[dst].get("name", ""),
                "median_factor": data["weight"],
                "total_trips": data["total_trips"],
                "median_osrm_dist_km": data.get("median_osrm_dist", np.nan),
                "pct_delayed": data.get("pct_delayed", np.nan),
                "is_sparse": data.get("is_sparse", False),
                "is_chronically_delayed": data["weight"] > DELAY_THRESHOLD,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("median_factor", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data...")
    df = load_and_preprocess(DATA_PATH)
    print(f"  {len(df):,} rows | {df['trip_uuid'].nunique():,} trips | "
          f"{df['source_center'].nunique()} source facilities")

    print("\nExtracting departure records (is_cutoff=False, one per OD leg)...")
    departures = extract_leg_records(df)
    n_corridors = departures.groupby(["source_center", "destination_center"]).ngroups
    print(f"  {len(departures):,} departure rows | {n_corridors} unique corridors")

    print("\nBuilding edge weight lookup table...")
    edge_stats, edge_rt_agg, rt_medians, global_median = build_edge_weight_table(departures)
    print(f"  Fine-grained rows (route_type × tod): {len(edge_stats)}")
    edge_stats.to_csv(f"{OUTPUT_DIR}/edge_weight_table.csv", index=False)
    edge_rt_agg.to_csv(f"{OUTPUT_DIR}/edge_rt_aggregated.csv", index=False)

    print("\nConstructing directed graph...")
    G = build_graph(edge_rt_agg, edge_stats, df, rt_medians, global_median)
    print(f"  Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")
    sparse_count = sum(1 for _, _, d in G.edges(data=True) if d.get("is_sparse"))
    print(f"  Sparse corridors (<{SPARSE_THRESHOLD} trips): {sparse_count}")

    print("\nComputing graph metrics...")
    metrics = compute_graph_metrics(G)
    metrics.to_csv(f"{OUTPUT_DIR}/node_metrics.csv", index=False)

    print("\nTop 10 bottleneck hubs:")
    display_cols = ["name", "betweenness", "in_degree", "out_degree",
                    "avg_incoming_delay_factor", "bottleneck_score"]
    print(metrics.head(10)[display_cols].to_string(index=False))

    print("\nIdentifying chronically delayed corridors (factor > 1.2)...")
    corridors = identify_delayed_corridors(G)
    corridors.to_csv(f"{OUTPUT_DIR}/corridor_audit.csv", index=False)
    delayed = corridors[corridors["is_chronically_delayed"]]
    print(f"  {len(delayed)} / {len(corridors)} corridors delayed "
          f"({len(delayed)/len(corridors)*100:.1f}%)")

    print("\nSaving graph pickle...")
    with open(f"{OUTPUT_DIR}/logistics_graph.pkl", "wb") as f:
        pickle.dump(G, f)

    # Human-readable summary
    with open(f"{OUTPUT_DIR}/graph_summary.txt", "w", encoding="utf-8") as f:
        f.write("Graph Summary\n" + "=" * 50 + "\n")
        f.write(f"Nodes (facilities)          : {G.number_of_nodes()}\n")
        f.write(f"Edges (corridors)            : {G.number_of_edges()}\n")
        f.write(f"Sparse corridors             : {sparse_count}\n")
        f.write(f"Chronically delayed (>20%)   : {len(delayed)}\n")
        f.write(f"Global median factor         : {global_median:.3f}\n")
        f.write(f"\nRoute-type median factors:\n")
        for rt, med in rt_medians.items():
            f.write(f"  {rt}: {med:.3f}\n")
        f.write(f"\nTop 5 Bottleneck Hubs:\n")
        for i, row in metrics.head(5).iterrows():
            f.write(
                f"  {i+1}. {row['name']}\n"
                f"     betweenness={row['betweenness']:.4f}  "
                f"score={row['bottleneck_score']:.4f}  "
                f"in_deg={int(row['in_degree'])}  "
                f"avg_delay={row['avg_incoming_delay_factor']:.3f}\n"
            )

    print(f"\nAll outputs saved to ./{OUTPUT_DIR}/")
    print("Done!\n")
    return G, metrics, corridors


if __name__ == "__main__":
    G, metrics, corridors = main()
