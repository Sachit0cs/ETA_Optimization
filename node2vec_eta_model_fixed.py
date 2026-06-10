"""
node2vec embeddings for the logistics graph -- LEAKAGE-FIXED rewrite.

This module replaced the deleted outputs/node2vec_eta_model.ipynb. The notebook
trained its own baseline / graph / node2vec XGBoost models on raw segment rows
and reported a "graph advantage" that was inflated by data leakage. The
benchmarking now lives in ONE place -- task3_eta_model.py -- which imports
compute_node2vec_embeddings() from here and evaluates node2vec side by side
with the baseline and GraphSAGE models (same trip-level test split, same
metrics, single results file: outputs/model_comparison.csv).

------------------------------------------------------------------------------
BUGS / DATA LEAKAGE THAT WERE IN THE NOTEBOOK (kept for the audit trail)
------------------------------------------------------------------------------
1. TRAIN/TEST CONTAMINATION (the main leak).
   The notebook read pre-computed node_metrics.csv / edge_weight_table.csv and
   merged them onto every row -- but those aggregates were built over the FULL
   dataset (training + test). Proof: the notebook printed node IND000000ACB
   with in_degree=45/out_degree=49, while the training-only file produced by
   build_graph.py has 41/48. Test-row outcomes were baked into the features
   used to predict the test set.
   FIX: build_graph.py now builds every graph aggregate from TRAINING rows
   only; the node2vec walks here run on that training-only graph.

2. TARGET-DERIVED FEATURES USED AS PREDICTORS.
   The notebook's feature list included src/dst avg_incoming_delay_factor and
   bottleneck_score while predicting delay_factor = actual_time/osrm_time --
   dst_avg_delay is literally an average of the target. The unified pipeline
   computes those metrics from training rows only, which makes them legitimate
   target-encoding features rather than leaks.

3. node2vec graph built from a multi-row edge table -> silent weight overwrite.
   The notebook looped edge_weight_table.csv with G.add_edge(...); that table
   has one row per (src, dst, route_type, tod_bucket), so repeated add_edge
   calls kept only the LAST bucket's median_factor as the edge weight.
   FIX: walks run on build_graph's aggregated DiGraph (one weight per corridor).

4. INCONSISTENT / DIVIDE-BY-ZERO "within 15%" metric.
   The notebook's cells disagreed on masking actual_time==0, making models
   non-comparable. FIX: task3_eta_model.evaluate() masks non-positive actuals
   identically for every model.

------------------------------------------------------------------------------
Run standalone to (re)generate embeddings only:
    python node2vec_eta_model_fixed.py
Run the full benchmark (all models, one comparison CSV):
    python task3_eta_model.py --retrain
"""

import os
import pickle

import numpy as np
import pandas as pd

# Fallback graph construction when outputs/logistics_graph.pkl is absent.
from build_graph import (
    load_and_preprocess,     # reads delivery_data.csv AND filters to data=="training"
    extract_leg_records,
    build_edge_weight_table,
    build_graph,
)

DATA_PATH = "delivery_data.csv"
GRAPH_PICKLE = os.path.join("outputs", "logistics_graph.pkl")
N2V_EMBEDDINGS_PATH = os.path.join("outputs", "node_emb_node2vec.csv")
RANDOM_STATE = 42
N2V_DIM = 32


# ---------------------------------------------------------------------------
# Training-only graph (loads build_graph.py's pickle, rebuilds if missing)
# ---------------------------------------------------------------------------

def load_training_graph():
    """The directed corridor graph built from TRAINING rows only. Prefers the
    pickle written by build_graph.py so walks run on exactly the same graph the
    node metrics came from."""
    if os.path.exists(GRAPH_PICKLE):
        with open(GRAPH_PICKLE, "rb") as f:
            return pickle.load(f)
    print(f"{GRAPH_PICKLE} not found -- rebuilding training-only graph...")
    df_train = load_and_preprocess(DATA_PATH)
    departures = extract_leg_records(df_train)
    edge_stats, edge_rt_agg, rt_medians, global_median = build_edge_weight_table(departures)
    return build_graph(edge_rt_agg, edge_stats, df_train, rt_medians, global_median)


# ---------------------------------------------------------------------------
# node2vec embeddings from the TRAINING graph (optional gensim dependency)
# ---------------------------------------------------------------------------

def _node2vec_walks(G, num_walks: int, walk_length: int, p: float, q: float, rng):
    """Generate node2vec biased second-order random walks on the weighted graph.

    Implemented in-process (networkx + numpy) instead of the `node2vec` PyPI
    package, which pins numpy<2.0 and cannot be installed alongside this env's
    numpy 2.x. Walks run on the undirected view so sparse one-way corridors
    don't dead-end immediately; edge `weight` (training-only median delay
    factor) biases transition probability. p=q=1 reproduces the notebook's
    default (weighted DeepWalk)."""
    UG = G.to_undirected()
    # Precompute neighbours, weights and a neighbour-set per node.
    adj = {}
    for n in UG.nodes():
        nbrs = list(UG.neighbors(n))
        w = np.array([float(UG[n][nb].get("weight", 1.0)) for nb in nbrs], dtype=float)
        adj[n] = (nbrs, w, set(nbrs))

    nodes = [n for n in UG.nodes() if adj[n][0]]   # skip isolated nodes
    walks = []
    for _ in range(num_walks):
        rng.shuffle(nodes)
        for start in nodes:
            walk = [start]
            while len(walk) < walk_length:
                cur = walk[-1]
                nbrs, w, _ = adj[cur]
                if not nbrs:
                    break
                if len(walk) == 1:
                    probs = w
                else:
                    prev = walk[-2]
                    prev_set = adj[prev][2]
                    probs = w.copy()
                    for i, nb in enumerate(nbrs):
                        if nb == prev:
                            probs[i] = w[i] / p          # return to previous
                        elif nb not in prev_set:
                            probs[i] = w[i] / q          # explore (distance 2)
                        # else distance 1 -> keep w[i]
                probs = probs / probs.sum()
                walk.append(nbrs[rng.choice(len(nbrs), p=probs)])
            walks.append([str(n) for n in walk])
    return walks


def compute_node2vec_embeddings(G, dim: int = N2V_DIM, num_walks: int = 10,
                                walk_length: int = 30, p: float = 1.0, q: float = 1.0):
    """node2vec on the aggregated training graph. Single weight per corridor
    (fix #3), training-only edges (fix #1). Returns (emb_df, cols) or
    (None, []) if gensim is unavailable."""
    try:
        from gensim.models import Word2Vec
    except ImportError:
        print("\ngensim not installed -- skipping node2vec model.")
        print("  install with: pip install gensim")
        return None, []

    print(f"\nGenerating node2vec walks on the training graph "
          f"({num_walks} walks x len {walk_length})...")
    rng = np.random.default_rng(RANDOM_STATE)
    walks = _node2vec_walks(G, num_walks, walk_length, p, q, rng)
    print(f"  {len(walks):,} walks -> training skip-gram (dim={dim})...")

    w2v = Word2Vec(
        walks,
        vector_size=dim,
        window=10,
        min_count=1,
        sg=1,                 # skip-gram, as in node2vec
        workers=1,            # workers=1 + seed -> reproducible
        seed=RANDOM_STATE,
        epochs=5,
    )

    rows = []
    for node in G.nodes():
        key = str(node)
        if key in w2v.wv:
            rows.append([node] + list(w2v.wv[key]))
    cols = [f"n2v_{i}" for i in range(dim)]
    emb_df = pd.DataFrame(rows, columns=["center"] + cols)
    print(f"  embeddings: {emb_df.shape[0]} nodes x {dim} dims")
    return emb_df, cols


# ---------------------------------------------------------------------------
# Main: embeddings only. Benchmarking lives in task3_eta_model.py.
# ---------------------------------------------------------------------------

def main():
    G = load_training_graph()
    print(f"Training graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    emb_df, _ = compute_node2vec_embeddings(G, N2V_DIM)
    if emb_df is None:
        return
    os.makedirs(os.path.dirname(N2V_EMBEDDINGS_PATH), exist_ok=True)
    emb_df.to_csv(N2V_EMBEDDINGS_PATH, index=False)
    print(f"Saved -> {N2V_EMBEDDINGS_PATH}")
    print("Run `python task3_eta_model.py --retrain` to benchmark all models "
          "into outputs/model_comparison.csv.")


if __name__ == "__main__":
    main()
