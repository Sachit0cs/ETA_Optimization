"""
Task 5 (data layer) -- per-hub SLA-breach contribution and revenue at risk.

Turns the graph + delivery data into the numbers the strategy memo and the
operations dashboard need:
  * how many SLA breaches each hub is responsible for (source-hub attribution),
  * what an upgrade could recover (excess-over-network-median breach rate), and
  * the rupee value at risk under an explicit, configurable assumption.

SLA definition. 95% of legs exceed RAW OSRM by >20% -- OSRM under-predicts
everywhere (median factor ~1.9), so raw-OSRM breaches cannot differentiate
hubs. The promise an ops team would actually quote is a CALIBRATED ETA:
promised = osrm_time x (route-type median delay factor). A breach is a
dispatch that runs >20% over that calibrated promise (same 1.2 threshold as
the corridor audit, applied to the calibrated baseline).

This is a descriptive operational audit over ALL legs (train + test) -- there
is no model evaluation here, so train/test leakage rules don't apply. Hub
structure (betweenness, bottleneck score) still comes from the training-only
node_metrics.csv for consistency with Tasks 2-3.

Output: outputs/hub_impact.csv  (auto-activates the Task-5 section in eda.py)
Run:    python task5_hub_impact.py
"""

import os

import numpy as np
import pandas as pd

from task4_route_choice import load_legs

OUTPUT_PATH = os.path.join("outputs", "hub_impact.csv")

DELAY_THRESHOLD = 1.2        # same ">20% over OSRM" rule as build_graph.py
REVENUE_PER_BREACH = 500.0   # Rs cost per late dispatch (penalty/refund/CX proxy)
MIN_LEGS = 30                # ignore hubs with too few dispatches to rank fairly


def compute_hub_impact() -> pd.DataFrame:
    legs = load_legs()
    # Calibrated promise: what a route-type-aware ETA would quote (see docstring).
    rt_median_factor = legs.groupby("route_type")["factor"].transform("median")
    legs["promised_time"] = legs["osrm_time"] * rt_median_factor
    legs["breach"] = legs["actual_time"] > DELAY_THRESHOLD * legs["promised_time"]
    total_breaches = int(legs["breach"].sum())
    print(f"  network: {total_breaches:,} breached dispatches "
          f"({legs['breach'].mean() * 100:.1f}% of {len(legs):,} legs, "
          f"vs calibrated promise +{(DELAY_THRESHOLD - 1) * 100:.0f}%)")

    out = (
        legs.groupby("source_center")
        .agg(
            name=("source_name", "first"),
            legs_out=("breach", "size"),
            breaches=("breach", "sum"),
        )
        .reset_index()
        .rename(columns={"source_center": "center"})
    )
    inbound = (
        legs.groupby("destination_center")["breach"]
        .agg(["size", "sum"])
        .rename(columns={"size": "legs_in", "sum": "breaches_in"})
        .reset_index()
        .rename(columns={"destination_center": "center"})
    )
    out = out.merge(inbound, on="center", how="left").fillna({"legs_in": 0, "breaches_in": 0})

    out["breach_rate"] = out["breaches"] / out["legs_out"]
    out["network_breach_share_pct"] = out["breaches"] / total_breaches * 100

    # Upgrade model: a fixed hub still breaches at the network median rate of
    # comparable hubs (>= MIN_LEGS dispatches); only the EXCESS is recoverable.
    eligible = out[out["legs_out"] >= MIN_LEGS]
    median_rate = float(eligible["breach_rate"].median())
    out["recoverable_breaches"] = (
        (out["breach_rate"] - median_rate).clip(lower=0) * out["legs_out"]
    ).round(0)
    out["revenue_at_risk_rs"] = out["breaches"] * REVENUE_PER_BREACH
    out["revenue_recoverable_rs"] = out["recoverable_breaches"] * REVENUE_PER_BREACH

    # Graph position (training-only metrics; missing for hubs unseen in training).
    nm = pd.read_csv("outputs/node_metrics.csv")[
        ["center", "betweenness", "in_degree", "out_degree",
         "avg_incoming_delay_factor", "bottleneck_score", "clustering"]
    ]
    out = out.merge(nm, on="center", how="left")

    # Rank by breach contribution among hubs with a fair sample size.
    out["rankable"] = out["legs_out"] >= MIN_LEGS
    out = out.sort_values(
        ["rankable", "breaches"], ascending=[False, False]
    ).reset_index(drop=True)
    out.insert(0, "rank", np.where(out["rankable"], np.arange(1, len(out) + 1), 0))

    print(f"  median breach rate (hubs with >= {MIN_LEGS} legs): {median_rate * 100:.1f}%")
    return out


def main():
    print("Computing per-hub SLA-breach contribution...")
    impact = compute_hub_impact()
    impact.to_csv(OUTPUT_PATH, index=False)
    print(f"  saved -> {OUTPUT_PATH}")

    top = impact[impact["rankable"]].head(5)
    print("\nTOP 5 BOTTLENECK HUBS BY SLA-BREACH CONTRIBUTION")
    print("=" * 88)
    for _, r in top.iterrows():
        print(f"  #{int(r['rank'])} {r['name']:<42} "
              f"breaches {int(r['breaches']):>5,} ({r['network_breach_share_pct']:.1f}% of network) | "
              f"rate {r['breach_rate'] * 100:.0f}% | recoverable {int(r['recoverable_breaches']):,}")
    top3 = impact[impact["rankable"]].head(3)
    total = impact["breaches"].sum()
    rec3 = top3["recoverable_breaches"].sum()
    print("-" * 88)
    print(f"Upgrading the top 3 hubs recovers ~{int(rec3):,} late dispatches "
          f"({rec3 / total * 100:.1f}% of all breaches) "
          f"~= Rs{rec3 * REVENUE_PER_BREACH:,.0f} at Rs{REVENUE_PER_BREACH:.0f}/breach")
    print("=" * 88)


if __name__ == "__main__":
    main()
