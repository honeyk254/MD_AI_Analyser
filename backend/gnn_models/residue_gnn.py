from __future__ import annotations

"""
Graph Neural Network models for residue-level protein analysis.

Implements a GAT + GCN hybrid architecture for learning residue importance,
detecting allosteric pathways, and identifying communication hubs from
dynamic residue interaction graphs.
"""

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.trajectory_utils import (
    collect_ca_positions,
    compute_fluctuations,
    select_ca_atoms,
)
from ..utils.ml_feature_utils import set_global_seed

logger = logging.getLogger("md_ai_analyzer")


# ── GNN Model (module-level) ────────────────────────────────────

class ResidueGNN(nn.Module):
    """Graph Attention + Graph Convolution hybrid for residue importance.

    Architecture
    ------------
    * Two GAT layers (multi-head attention, 4 heads each)
    * One GCN layer producing final embeddings
    * A linear head that maps embeddings to scalar importance scores

    Parameters
    ----------
    in_features : int
        Dimension of per-node input features.
    hidden : int
        Hidden layer width (default 64).
    out : int
        Embedding output dimension (default 32).
    """

    def __init__(self, in_features: int, hidden: int = 64, out: int = 32) -> None:
        from torch_geometric.nn import GATConv, GCNConv

        super().__init__()
        self.conv1 = GATConv(in_features, hidden, heads=4, concat=False)
        self.conv2 = GATConv(hidden, hidden, heads=4, concat=False)
        self.conv3 = GCNConv(hidden, out)
        self.importance_head = nn.Linear(out, 1)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, Any, Any]:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape ``(n_nodes, in_features)``.
        edge_index : torch.Tensor
            Graph connectivity in COO format ``(2, n_edges)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, Any, Any]
            ``(embeddings, importance, attn_weights_layer1, attn_weights_layer2)``
        """
        h1, attn1 = self.conv1(x, edge_index, return_attention_weights=True)
        h1 = self.bn1(F.elu(h1))
        h1 = F.dropout(h1, p=0.1, training=self.training)

        h2, attn2 = self.conv2(h1, edge_index, return_attention_weights=True)
        h2 = self.bn2(F.elu(h2))
        h2 = F.dropout(h2, p=0.1, training=self.training)

        embeddings = self.conv3(h2, edge_index)
        importance = self.importance_head(embeddings).squeeze(-1)

        return embeddings, importance, attn1, attn2


# ── Main analysis entry point ───────────────────────────────────

def run_gnn_analysis(universe: Any, **kwargs: Any) -> dict[str, Any]:
    """Run GNN-based analysis on dynamic residue interaction graphs.

    The pipeline:

    1. Builds per-frame residue contact graphs from Calpha positions.
    2. Computes node features (RMSF, average contacts, displacement
       fluctuations).
    3. Trains a GAT+GCN hybrid to predict per-residue RMSF in a
       self-supervised fashion.
    4. Extracts learned attention weights and embeddings to rank residues.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        Loaded trajectory universe.
    **kwargs : Any
        Reserved for future options.

    Returns
    -------
    dict[str, Any]
        Keys:

        - ``resids`` -- residue IDs
        - ``residue_importance`` -- GNN-learned importance scores per residue
        - ``embeddings_2d`` -- 2-D PCA projection of GNN residue embeddings
        - ``top_residues`` -- most important residues identified by GNN
        - ``attention_interactions`` -- top GAT attention-weighted interactions
        - ``community_assignments`` -- GNN-based community detection
        - ``n_edges`` -- number of undirected edges in the contact graph
        - ``model_type`` -- string label for the architecture
        - ``training_losses`` -- per-epoch training loss curve
        - ``reconstruction_error`` -- final reconstruction MSE
    """
    try:
        from torch_geometric.data import Data  # noqa: F401
        from torch_geometric.nn import GCNConv, GATConv, SAGEConv  # noqa: F401
        from sklearn.decomposition import PCA
        from MDAnalysis.lib.distances import distance_array

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("GNN analysis starting on device=%s", device)

        # ── Select CA atoms ──────────────────────────────────────
        ca = select_ca_atoms(universe)
        n_res = len(ca)
        resids: list[int] = ca.resids.tolist()

        # ── Compute node features ────────────────────────────────
        # Feature 1: Per-residue RMSF
        from MDAnalysis.analysis.rms import RMSF as MDA_RMSF

        rmsf_calc = MDA_RMSF(ca).run()
        rmsf_values: np.ndarray = rmsf_calc.results.rmsf

        # Feature 2: Average contact count per residue
        contact_counts = np.zeros(n_res)
        positions_all = collect_ca_positions(universe, atoms=ca)
        n_frames = positions_all.shape[0]

        for frame_idx in range(n_frames):
            dists = distance_array(positions_all[frame_idx], positions_all[frame_idx])
            contacts = (dists < 8.0).sum(axis=1) - 1  # exclude self
            contact_counts += contacts

        contact_counts /= n_frames

        # Feature 3: Position fluctuation magnitude (from shared util)
        fluctuations: np.ndarray = compute_fluctuations(positions_all)

        # Build node feature matrix [n_res, n_features]
        node_features = np.column_stack([
            rmsf_values / (rmsf_values.max() + 1e-8),
            contact_counts / (contact_counts.max() + 1e-8),
            fluctuations / (fluctuations.max() + 1e-8),
        ]).astype(np.float32)

        # ── Build average contact graph (vectorised) ─────────────
        logger.info("Building average contact graph (%d frames, %d residues)", n_frames, n_res)

        avg_dists = np.zeros((n_res, n_res), dtype=np.float64)
        for frame_idx in range(n_frames):
            avg_dists += distance_array(
                positions_all[frame_idx], positions_all[frame_idx]
            )
        avg_dists /= n_frames

        # Vectorised edge construction: find pairs (i, j) with j >= i+3
        # and avg distance < 10 A
        ii, jj = np.where(avg_dists < 10.0)
        mask = jj >= ii + 3
        ii = ii[mask]
        jj = jj[mask]

        # Build symmetric edge lists and weights
        edge_src = np.concatenate([ii, jj])
        edge_dst = np.concatenate([jj, ii])
        dists_selected = avg_dists[ii, jj]
        weights = 1.0 / (dists_selected + 1e-3)
        edge_weights = np.concatenate([weights, weights])

        if len(edge_src) < 2:
            logger.warning("Too few edges (%d) for GNN analysis", len(edge_src))
            return {"error": "Too few edges for GNN analysis"}

        edge_index = torch.tensor(
            np.stack([edge_src, edge_dst], axis=0), dtype=torch.long
        ).to(device)
        edge_weight = torch.tensor(edge_weights, dtype=torch.float32).to(device)
        x = torch.tensor(node_features, dtype=torch.float32).to(device)

        # ── Train with self-supervised objectives ────────────────
        set_global_seed(42)

        model = ResidueGNN(in_features=node_features.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

        # Target: predict RMSF from graph structure (self-supervised)
        target = torch.tensor(
            rmsf_values / (rmsf_values.max() + 1e-8), dtype=torch.float32
        ).to(device)

        training_losses: list[float] = []
        model.train()
        for epoch in range(200):
            optimizer.zero_grad()
            embeddings, importance, _, _ = model(x, edge_index)
            loss = F.mse_loss(importance, target)

            # Graph regularisation: connected nodes should have similar embeddings
            src_emb = embeddings[edge_index[0]]
            dst_emb = embeddings[edge_index[1]]
            reg_loss = 0.01 * (edge_weight.unsqueeze(1) * (src_emb - dst_emb) ** 2).mean()
            total_loss = loss + reg_loss
            total_loss.backward()
            optimizer.step()

            loss_val = round(float(total_loss.item()), 6)
            training_losses.append(loss_val)

            if epoch % 50 == 0 or epoch == 199:
                logger.info(
                    "GNN epoch %3d/200 | loss=%.6f (mse=%.6f, reg=%.6f)",
                    epoch, loss_val, float(loss.item()), float(reg_loss.item()),
                )

        # ── Extract results ──────────────────────────────────────
        model.eval()
        with torch.no_grad():
            embeddings, importance, attn1, attn2 = model(x, edge_index)

        recon_error: float = float(F.mse_loss(importance, target).item())
        importance_scores: np.ndarray = importance.cpu().numpy()
        embeddings_np: np.ndarray = embeddings.cpu().numpy()

        # 2-D projection of embeddings
        if embeddings_np.shape[0] > 2:
            pca = PCA(n_components=2)
            emb_2d: np.ndarray = pca.fit_transform(embeddings_np)
        else:
            emb_2d = embeddings_np[:, :2]

        # Top residues by importance
        sorted_idx = np.argsort(importance_scores)[::-1]
        top_residues: list[dict[str, Any]] = []
        for idx in sorted_idx[:20]:
            top_residues.append({
                "resid": int(resids[idx]),
                "importance": round(float(importance_scores[idx]), 4),
                "rmsf": round(float(rmsf_values[idx]), 3),
                "contacts": round(float(contact_counts[idx]), 1),
            })

        # Extract attention weights for key interactions
        attn_edge_index: np.ndarray = attn2[0].cpu().numpy()
        attn_weights: np.ndarray = attn2[1].cpu().numpy()

        top_attn: list[dict[str, Any]] = []
        if (
            attn_weights.size > 0
            and attn_edge_index.ndim == 2
            and attn_edge_index.shape[0] == 2
        ):
            if attn_weights.ndim > 1:
                attn_mean = attn_weights.mean(axis=1)
            else:
                attn_mean = attn_weights
            n_attn_edges = min(len(attn_mean), attn_edge_index.shape[1])
            top_attn_idx = np.argsort(attn_mean[:n_attn_edges])[::-1][:50]
            for idx in top_attn_idx:
                src = int(attn_edge_index[0, idx])
                dst = int(attn_edge_index[1, idx])
                if src < n_res and dst < n_res:
                    top_attn.append({
                        "res_i": int(resids[src]),
                        "res_j": int(resids[dst]),
                        "attention": round(float(attn_mean[idx]), 4),
                    })

        # Community detection from embeddings
        from sklearn.cluster import KMeans

        n_comm = min(5, n_res // 5)
        communities: dict[int, list[int]] = {}
        if n_comm >= 2:
            km = KMeans(n_clusters=n_comm, n_init=10, random_state=42)
            comm_labels = km.fit_predict(embeddings_np)
            for i, label in enumerate(comm_labels):
                communities.setdefault(int(label), []).append(int(resids[i]))

        logger.info(
            "GNN analysis complete: %d residues, %d edges, recon_error=%.6f",
            n_res, len(ii), recon_error,
        )

        return {
            "resids": resids,
            "residue_importance": importance_scores.tolist(),
            "embeddings_2d": emb_2d.tolist(),
            "top_residues": top_residues,
            "attention_interactions": top_attn,
            "community_assignments": communities,
            "n_edges": len(ii),
            "model_type": "GAT+GCN hybrid",
            "training_losses": training_losses,
            "reconstruction_error": round(recon_error, 6),
        }

    except ImportError as e:
        logger.error("GNN import error: %s", e)
        return {"error": f"PyTorch Geometric not installed: {e}"}
    except Exception as e:
        logger.error("GNN analysis failed: %s", e, exc_info=True)
        return {"error": str(e)}
