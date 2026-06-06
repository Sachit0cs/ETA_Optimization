import os
from typing import List

import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim


MODEL_DIR = "models"
EMBEDDINGS_PATH = os.path.join("outputs", "node_emb_graphsage.csv")
GRAPHSAGE_MODEL_PATH = os.path.join(MODEL_DIR, "graphsage_eta.pt")
EMB_DIM = 8


class GraphSAGE(nn.Module):
    """Simple 1-layer GraphSAGE-style encoder for node embeddings."""

    def __init__(self, in_dim: int, emb_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim * 2, 128)
        self.fc2 = nn.Linear(128, emb_dim)
        self.reg = nn.Linear(emb_dim, 1)

    def forward(self, x, neigh_mean):
        h = torch.cat([x, neigh_mean], dim=1)
        h = torch.relu(self.fc1(h))
        emb = torch.relu(self.fc2(h))
        out = self.reg(emb)
        return emb, out


def compute_graphsage_embeddings(
    G: nx.DiGraph,
    node_metrics: pd.DataFrame,
    emb_dim: int = EMB_DIM,
    embeddings_path: str = EMBEDDINGS_PATH,
    model_path: str = GRAPHSAGE_MODEL_PATH,
) -> pd.DataFrame:
    """Train a small GraphSAGE model and save its embeddings plus model artifact."""
    # Reproducibility: seed before any random weight initialisation.
    np.random.seed(42)
    torch.manual_seed(42)

    # Structural features only. The regression target (avg_incoming_delay_factor)
    # and bottleneck_score (which is derived from it) are deliberately EXCLUDED
    # from the inputs to avoid target leakage -- previously the encoder was asked
    # to predict one of its own input features and trivially learned identity.
    feat_cols = [
        "betweenness",
        "in_degree",
        "out_degree",
    ]
    target_col = "avg_incoming_delay_factor"
    nm = node_metrics.set_index("center")

    nodes = list(G.nodes())
    idx_map = {n: i for i, n in enumerate(nodes)}

    X = np.vstack(
        [nm.loc[n, feat_cols].values if n in nm.index else np.zeros(len(feat_cols), dtype=float) for n in nodes]
    ).astype(float)
    # Standardise features so large-magnitude columns (degree counts) don't
    # dominate the mean-aggregation and the linear layers over betweenness (~[0,1]).
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    X = (X - mu) / sigma
    y = np.array(
        [nm.loc[n, target_col] if n in nm.index else 0.0 for n in nodes],
        dtype=float,
    )

    nbrs: List[List[int]] = []
    for n in nodes:
        neigh = set(G.predecessors(n)) | set(G.successors(n))
        nbrs.append([idx_map[nb] for nb in neigh if nb in idx_map])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(1)

    # Features are fixed, so neighbour means are constant across epochs -- compute once.
    neigh_means = []
    for nbr_idx in nbrs:
        if nbr_idx:
            neigh_means.append(X_t[nbr_idx].mean(dim=0, keepdim=True))
        else:
            neigh_means.append(torch.zeros(1, X_t.size(1), device=device))
    neigh_means_t = torch.cat(neigh_means, dim=0)

    model = GraphSAGE(X.shape[1], emb_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    epochs = 100
    for ep in range(epochs):
        model.train()
        embeddings, preds = model(X_t, neigh_means_t)
        loss = loss_fn(preds, y_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (ep + 1) % 25 == 0:
            print(f"    [GraphSAGE] epoch {ep + 1}/{epochs} loss={loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        embeddings, _ = model(X_t, neigh_means_t)

    emb_np = embeddings.cpu().numpy()
    emb_df = pd.DataFrame(emb_np, columns=[f"emb_{i}" for i in range(emb_np.shape[1])])
    emb_df["center"] = nodes
    emb_df = emb_df[["center"] + [c for c in emb_df.columns if c != "center"]]

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(embeddings_path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "emb_dim": emb_dim,
            "feature_cols": feat_cols,
        },
        model_path,
    )
    emb_df.to_csv(embeddings_path, index=False)

    print(f"  Trained GraphSAGE model saved -> {model_path}")
    print(f"  Trained GraphSAGE embeddings saved -> {embeddings_path} (dim={emb_np.shape[1]})")
    return emb_df
