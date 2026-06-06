"""
Graph visualization for Delhivery logistics network.

Reads outputs produced by build_graph.py and generates:
  01_full_network.png        Full network, nodes colored by betweenness centrality
  02_bottleneck_hubs.png     Top-10 bottleneck hubs highlighted
  03_delayed_corridors.png   Horizontal bar chart of most-delayed corridors
  04_degree_distributions.png In/out-degree & betweenness histograms
  05_top_hubs_subgraph.png   Subgraph view of top-5 hubs + their neighbors
  06_interactive_network.html Interactive pyvis HTML (requires pyvis)

Run build_graph.py first to generate the required input files.
"""

import os
import pickle

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")           # non-interactive backend, safe for all environments
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize

OUTPUT_DIR = "outputs"
VIZ_DIR    = os.path.join(OUTPUT_DIR, "visualizations")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_artifacts():
    with open(f"{OUTPUT_DIR}/logistics_graph.pkl", "rb") as f:
        G = pickle.load(f)
    metrics  = pd.read_csv(f"{OUTPUT_DIR}/node_metrics.csv")
    corridors = pd.read_csv(f"{OUTPUT_DIR}/corridor_audit.csv")
    return G, metrics, corridors


def compute_layout(G: nx.DiGraph, seed: int = 42) -> dict:
    """Spring layout; k adjusted so nodes are not too packed."""
    k = 2.5 / np.sqrt(max(G.number_of_nodes(), 1))
    return nx.spring_layout(G, k=k, seed=seed, iterations=60)


# ---------------------------------------------------------------------------
# Plot 1 – Full network
# ---------------------------------------------------------------------------

def plot_full_network(G, metrics, pos, save_path):
    fig, ax = plt.subplots(figsize=(20, 15))

    bw_map  = dict(zip(metrics["center"], metrics["betweenness"]))
    node_list = list(G.nodes())
    node_bw   = [bw_map.get(n, 0.0) for n in node_list]
    max_bw    = max(node_bw) if max(node_bw) > 0 else 1.0

    norm_bw   = Normalize(vmin=0, vmax=max_bw)
    cmap_node = cm.YlOrRd
    node_colors = [cmap_node(norm_bw(b)) for b in node_bw]
    node_sizes  = [80 + 2500 * norm_bw(b) for b in node_bw]

    edge_list    = list(G.edges())
    edge_weights = [G[u][v]["weight"] for u, v in edge_list]
    w_min, w_max = min(edge_weights), max(edge_weights)
    norm_ew      = Normalize(vmin=w_min, vmax=w_max)
    cmap_edge    = cm.RdYlGn_r
    edge_colors  = [cmap_edge(norm_ew(w)) for w in edge_weights]
    edge_widths  = [0.3 + 1.5 * norm_ew(w) for w in edge_weights]

    nx.draw_networkx_edges(
        G, pos, edgelist=edge_list,
        edge_color=edge_colors, width=edge_widths,
        arrows=True, arrowsize=6,
        connectionstyle="arc3,rad=0.08",
        ax=ax, alpha=0.55,
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=node_list,
        node_color=node_colors, node_size=node_sizes,
        ax=ax,
    )

    # Label only the top-15 nodes by betweenness to avoid clutter
    top15 = metrics.nlargest(15, "betweenness")["center"].tolist()
    labels = {n: str(G.nodes[n].get("name", n))[:14] for n in top15}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=6, ax=ax)

    sm = cm.ScalarMappable(cmap=cmap_node, norm=norm_bw)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("Betweenness Centrality", fontsize=10)

    sm2 = cm.ScalarMappable(cmap=cmap_edge, norm=norm_ew)
    sm2.set_array([])
    cb2 = plt.colorbar(sm2, ax=ax, fraction=0.025, pad=0.04)
    cb2.set_label("Median Delay Factor (edge)", fontsize=10)

    ax.set_title(
        "Delhivery Logistics Network\n"
        "Node colour = Betweenness Centrality  |  Edge colour = Delay Factor",
        fontsize=14, fontweight="bold", pad=15,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Plot 2 – Bottleneck hubs
# ---------------------------------------------------------------------------

def plot_bottleneck_hubs(G, metrics, pos, save_path, top_n: int = 10):
    fig, ax = plt.subplots(figsize=(20, 15))

    bs_map   = dict(zip(metrics["center"], metrics["bottleneck_score"]))
    top_hubs = set(metrics.nlargest(top_n, "bottleneck_score")["center"].tolist())
    max_bs   = max(bs_map.values()) if bs_map and max(bs_map.values()) > 0 else 1.0

    bg_nodes = [n for n in G.nodes() if n not in top_hubs]
    nx.draw_networkx_nodes(
        G, pos, nodelist=bg_nodes,
        node_color="#cccccc", node_size=60, ax=ax, alpha=0.4,
    )

    # Size hubs by the same metric they are ranked on (bottleneck_score), so the
    # top-ranked hub renders largest in a plot titled "bottleneck hubs".
    hub_sizes = [800 + 4000 * bs_map.get(n, 0) / max_bs for n in top_hubs]
    nx.draw_networkx_nodes(
        G, pos, nodelist=list(top_hubs),
        node_color="#d73027", node_size=hub_sizes, ax=ax, alpha=0.9,
    )

    edge_list   = list(G.edges())
    edge_colors = [
        "#d73027" if G[u][v]["weight"] > 1.2 else "#999999"
        for u, v in edge_list
    ]
    edge_widths = [
        1.8 if G[u][v]["weight"] > 1.2 else 0.3
        for u, v in edge_list
    ]
    nx.draw_networkx_edges(
        G, pos, edgelist=edge_list,
        edge_color=edge_colors, width=edge_widths,
        arrows=True, arrowsize=7,
        connectionstyle="arc3,rad=0.08",
        ax=ax, alpha=0.65,
    )

    hub_labels = {n: str(G.nodes[n].get("name", n))[:16] for n in top_hubs}
    nx.draw_networkx_labels(
        G, pos, labels=hub_labels,
        font_size=7, font_weight="bold", ax=ax,
    )

    legend_handles = [
        mpatches.Patch(color="#d73027", label=f"Top {top_n} Bottleneck Hubs"),
        mpatches.Patch(color="#cccccc", label="Other Facilities"),
        mpatches.Patch(color="#d73027", alpha=0.5,
                       label="Chronically Delayed Corridor (factor > 1.2)"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=10)
    ax.set_title(
        f"Top {top_n} Bottleneck Hubs & Chronic Delay Corridors",
        fontsize=14, fontweight="bold",
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Plot 3 – Delayed corridors bar chart
# ---------------------------------------------------------------------------

def plot_delayed_corridors(corridors: pd.DataFrame, save_path, top_n: int = 30):
    reliable = corridors[corridors["total_trips"] >= 5]
    top      = reliable.nlargest(top_n, "median_factor").reset_index(drop=True)

    labels = [
        f"{str(r['source_name'])[:12]} → {str(r['destination_name'])[:12]}"
        for _, r in top.iterrows()
    ]
    colors = [
        "#d73027" if f > 1.5 else "#fc8d59" if f > 1.2 else "#fee08b"
        for f in top["median_factor"]
    ]

    fig, ax = plt.subplots(figsize=(14, max(8, top_n * 0.35)))
    bars = ax.barh(range(len(top)), top["median_factor"], color=colors, edgecolor="white")
    ax.axvline(x=1.2, color="#333333", linestyle="--", linewidth=1.5,
               label="20% delay threshold (1.2×)")
    ax.axvline(x=1.0, color="#888888", linestyle=":",  linewidth=1.0,
               label="OSRM baseline (1.0×)")

    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Median Delay Factor  (actual_time / osrm_time)", fontsize=11)
    ax.set_title(
        f"Top {top_n} Most Delayed Corridors  (≥5 trips only)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10)

    for bar, val in zip(bars, top["median_factor"]):
        ax.text(
            val + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}", va="center", fontsize=7,
        )

    legend_extra = [
        mpatches.Patch(color="#d73027", label="Severe (>1.5×)"),
        mpatches.Patch(color="#fc8d59", label="Delayed (1.2–1.5×)"),
        mpatches.Patch(color="#fee08b", label="Mild (<1.2×)"),
    ]
    ax.legend(handles=legend_extra + ax.get_legend_handles_labels()[0], fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Plot 4 – Degree & betweenness distributions
# ---------------------------------------------------------------------------

def plot_degree_distributions(G, metrics, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    in_degs  = [d for _, d in G.in_degree()]
    out_degs = [d for _, d in G.out_degree()]

    axes[0].hist(in_degs,  bins=30, color="#4c72b0", edgecolor="white")
    axes[0].set_xlabel("In-Degree")
    axes[0].set_ylabel("Number of Facilities")
    axes[0].set_title("In-Degree Distribution")

    axes[1].hist(out_degs, bins=30, color="#c44e52", edgecolor="white")
    axes[1].set_xlabel("Out-Degree")
    axes[1].set_ylabel("Number of Facilities")
    axes[1].set_title("Out-Degree Distribution")

    axes[2].hist(metrics["betweenness"], bins=40, color="#8172b2", edgecolor="white")
    p95 = metrics["betweenness"].quantile(0.95)
    axes[2].axvline(p95, color="red", linestyle="--", label=f"95th percentile ({p95:.4f})")
    axes[2].set_xlabel("Betweenness Centrality")
    axes[2].set_ylabel("Number of Facilities")
    axes[2].set_title("Betweenness Distribution\n(right tail = bottleneck hubs)")
    axes[2].legend(fontsize=9)

    plt.suptitle(
        "Graph Structural Metrics — Distribution Across All Facilities",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Plot 5 – Top-5 hubs subgraph
# ---------------------------------------------------------------------------

def plot_top_hubs_subgraph(G, metrics, save_path, top_n: int = 5):
    top_hubs = metrics.nlargest(top_n, "bottleneck_score")["center"].tolist()

    # 1-hop neighbourhood
    subgraph_nodes = set(top_hubs)
    for hub in top_hubs:
        subgraph_nodes.update(G.predecessors(hub))
        subgraph_nodes.update(G.successors(hub))

    SG     = G.subgraph(subgraph_nodes).copy()
    k_sub  = 3.0 / np.sqrt(max(len(SG), 1))
    sub_pos = nx.spring_layout(SG, k=k_sub, seed=42, iterations=80)

    fig, ax = plt.subplots(figsize=(17, 13))

    hub_set    = set(top_hubs)
    other_nodes = [n for n in SG.nodes() if n not in hub_set]
    hub_nodes   = [n for n in SG.nodes() if n in hub_set]

    nx.draw_networkx_nodes(
        SG, sub_pos, nodelist=other_nodes,
        node_color="#aec7e8", node_size=350, ax=ax,
    )
    nx.draw_networkx_nodes(
        SG, sub_pos, nodelist=hub_nodes,
        node_color="#d73027", node_size=1200, ax=ax,
    )

    sg_edge_list = list(SG.edges())
    sg_colors = [
        "#d73027" if SG[u][v]["weight"] > 1.2 else "#aec7e8"
        for u, v in sg_edge_list
    ]
    sg_widths = [
        2.0 if SG[u][v]["weight"] > 1.2 else 0.8
        for u, v in sg_edge_list
    ]
    nx.draw_networkx_edges(
        SG, sub_pos, edgelist=sg_edge_list,
        edge_color=sg_colors, width=sg_widths,
        arrows=True, arrowsize=14,
        connectionstyle="arc3,rad=0.1", ax=ax,
    )

    all_labels = {n: str(G.nodes[n].get("name", n))[:16] for n in SG.nodes()}
    nx.draw_networkx_labels(SG, sub_pos, labels=all_labels, font_size=8, ax=ax)

    legend_handles = [
        mpatches.Patch(color="#d73027", label=f"Top {top_n} Bottleneck Hubs"),
        mpatches.Patch(color="#aec7e8", label="Immediate Neighbours"),
        mpatches.Patch(color="#d73027", alpha=0.5, label="Delayed Edge (factor > 1.2)"),
    ]
    ax.legend(handles=legend_handles, fontsize=10)
    ax.set_title(
        f"Subgraph: Top {top_n} Bottleneck Hubs + Immediate Neighbours",
        fontsize=13, fontweight="bold",
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Plot 6 – Interactive HTML (pyvis)
# ---------------------------------------------------------------------------

def create_interactive_html(G, metrics, save_path):
    try:
        from pyvis.network import Network
    except ImportError:
        print("  pyvis not installed — skipping interactive HTML.")
        print("  Install with: pip install pyvis")
        return

    net = Network(
        height="820px", width="100%",
        directed=True,
        bgcolor="#1a1a2e",
        font_color="white",
        notebook=False,
    )
    net.set_options("""{
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -3500,
          "centralGravity": 0.25,
          "springLength": 180,
          "springConstant": 0.04
        },
        "minVelocity": 0.75
      },
      "edges": { "smooth": { "type": "dynamic" } }
    }""")

    bw_map  = dict(zip(metrics["center"], metrics["betweenness"]))
    bs_map  = dict(zip(metrics["center"], metrics["bottleneck_score"]))
    max_bw  = max(bw_map.values()) + 1e-9
    top10   = set(metrics.nlargest(10, "bottleneck_score")["center"].tolist())

    for node in G.nodes():
        b    = bw_map.get(node, 0.0)
        name = str(G.nodes[node].get("name", node))
        bs   = bs_map.get(node, 0.0)
        size = 8 + 45 * (b / max_bw)
        color = "#ff3333" if node in top10 else "#4a90d9"
        title = (
            f"<b>{name}</b><br>"
            f"Betweenness: {b:.4f}<br>"
            f"Bottleneck Score: {bs:.4f}<br>"
            f"In-degree: {G.in_degree(node)} | Out-degree: {G.out_degree(node)}"
        )
        net.add_node(str(node), label=name[:14], size=size, color=color, title=title)

    for src, dst, data in G.edges(data=True):
        w     = data["weight"]
        color = "#ff4444" if w > 1.5 else "#ff9933" if w > 1.2 else "#33cc55"
        width = 0.5 + 3.0 * min(max(w - 0.5, 0), 2.0)
        title = (
            f"Factor: {w:.2f}<br>"
            f"Trips: {data['total_trips']}<br>"
            f"% Delayed: {data.get('pct_delayed', 0) * 100:.1f}%"
        )
        net.add_edge(str(src), str(dst), color=color, title=title, width=width)

    net.save_graph(save_path)
    print(f"  Saved interactive HTML: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(VIZ_DIR, exist_ok=True)

    print("Loading graph and metrics from outputs/...")
    G, metrics, corridors = load_artifacts()
    print(f"  Graph: {G.number_of_nodes()} nodes | {G.number_of_edges()} edges")

    print("\nComputing spring layout (may take ~20-40 s)...")
    pos = compute_layout(G)

    print("\n[1/6] Full network plot...")
    plot_full_network(G, metrics, pos, f"{VIZ_DIR}/01_full_network.png")

    print("[2/6] Bottleneck hubs plot...")
    plot_bottleneck_hubs(G, metrics, pos, f"{VIZ_DIR}/02_bottleneck_hubs.png")

    print("[3/6] Delayed corridors bar chart...")
    plot_delayed_corridors(corridors, f"{VIZ_DIR}/03_delayed_corridors.png")

    print("[4/6] Degree & betweenness distributions...")
    plot_degree_distributions(G, metrics, f"{VIZ_DIR}/04_degree_distributions.png")

    print("[5/6] Top-5 hubs subgraph...")
    plot_top_hubs_subgraph(G, metrics, f"{VIZ_DIR}/05_top_hubs_subgraph.png")

    print("[6/6] Interactive HTML visualization...")
    create_interactive_html(G, metrics, f"{VIZ_DIR}/06_interactive_network.html")

    print(f"\nAll visualizations saved to ./{VIZ_DIR}/")


if __name__ == "__main__":
    main()
