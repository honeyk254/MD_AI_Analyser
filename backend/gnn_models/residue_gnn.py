"""
Graph Neural Network models for residue-level protein analysis.
Implements GCN, GAT, and GraphSAGE for learning residue importance,
detecting allosteric pathways, and identifying communication hubs.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def run_gnn_analysis(universe, **kwargs):
    """
    Run GNN-based analysis on dynamic residue interaction graphs.

    1. Builds per-frame residue contact graphs
    2. Computes node features (RMSF, contacts, SS, SASA)
    3. Trains a GNN to predict residue importance
    4. Uses learned attention/embeddings to identify key residues

    Returns dict with:
        - residue_importance: GNN-learned importance scores per residue
        - attention_weights: GAT attention weights (key interactions)
        - embeddings_2d: 2D projection of GNN residue embeddings
        - top_residues: most important residues identified by GNN
        - community_assignments: GNN-based community detection
    """
    try:
        from torch_geometric.data import Data
        from torch_geometric.nn import GCNConv, GATConv, SAGEConv
        from torch_geometric.nn import global_mean_pool
        from sklearn.decomposition import PCA

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ca = universe.select_atoms("protein and name CA")
        if len(ca) == 0:
            return {"error": "No CA atoms found"}

        n_res = len(ca)
        resids = ca.resids.tolist()

        # ── Compute node features ──────────────────────────────
        # Feature 1: Per-residue RMSF
        from MDAnalysis.analysis.rms import RMSF as MDA_RMSF
        rmsf_calc = MDA_RMSF(ca).run()
        rmsf_values = rmsf_calc.results.rmsf

        # Feature 2: Average contact count per residue
        from MDAnalysis.lib.distances import distance_array
        contact_counts = np.zeros(n_res)
        positions_all = []
        n_frames = 0

        for ts in universe.trajectory:
            positions_all.append(ca.positions.copy())
            dists = distance_array(ca.positions, ca.positions)
            contacts = (dists < 8.0).sum(axis=1) - 1  # exclude self
            contact_counts += contacts
            n_frames += 1

        contact_counts /= n_frames
        positions_all = np.array(positions_all)

        # Feature 3: Position fluctuation magnitude
        mean_pos = positions_all.mean(axis=0)
        fluctuations = np.sqrt(np.mean(np.sum((positions_all - mean_pos)**2, axis=2), axis=0))

        # Build node feature matrix [n_res, n_features]
        node_features = np.column_stack([
            rmsf_values / (rmsf_values.max() + 1e-8),
            contact_counts / (contact_counts.max() + 1e-8),
            fluctuations / (fluctuations.max() + 1e-8),
        ]).astype(np.float32)

        # ── Build average contact graph ────────────────────────
        avg_dists = np.zeros((n_res, n_res))
        for pos in positions_all:
            avg_dists += distance_array(pos, pos)
        avg_dists /= n_frames

        # Edges: contacts within 10Å (excluding sequential neighbors ±2)
        edge_src, edge_dst = [], []
        edge_weights = []
        for i in range(n_res):
            for j in range(i + 3, n_res):
                if avg_dists[i, j] < 10.0:
                    edge_src.extend([i, j])
                    edge_dst.extend([j, i])
                    w = 1.0 / (avg_dists[i, j] + 1e-3)
                    edge_weights.extend([w, w])

        if len(edge_src) < 2:
            return {"error": "Too few edges for GNN analysis"}

        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long).to(device)
        edge_weight = torch.tensor(edge_weights, dtype=torch.float32).to(device)
        x = torch.tensor(node_features, dtype=torch.float32).to(device)

        # ── Define GNN Model ──────────────────────────────────
        class ResidueGNN(nn.Module):
            def __init__(self, in_features, hidden=64, out=32):
                super().__init__()
                self.conv1 = GATConv(in_features, hidden, heads=4, concat=False)
                self.conv2 = GATConv(hidden, hidden, heads=4, concat=False)
                self.conv3 = GCNConv(hidden, out)
                self.importance_head = nn.Linear(out, 1)
                self.bn1 = nn.BatchNorm1d(hidden)
                self.bn2 = nn.BatchNorm1d(hidden)

            def forward(self, x, edge_index):
                h1, attn1 = self.conv1(x, edge_index, return_attention_weights=True)
                h1 = self.bn1(F.elu(h1))
                h1 = F.dropout(h1, p=0.1, training=self.training)

                h2, attn2 = self.conv2(h1, edge_index, return_attention_weights=True)
                h2 = self.bn2(F.elu(h2))
                h2 = F.dropout(h2, p=0.1, training=self.training)

                embeddings = self.conv3(h2, edge_index)
                importance = self.importance_head(embeddings).squeeze(-1)

                return embeddings, importance, attn1, attn2

        # ── Train with self-supervised objectives ──────────────
        model = ResidueGNN(in_features=node_features.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

        # Target: predict RMSF from graph structure (self-supervised)
        target = torch.tensor(rmsf_values / (rmsf_values.max() + 1e-8),
                             dtype=torch.float32).to(device)

        training_losses = []  # item 53: track training loss
        model.train()
        for epoch in range(200):
            optimizer.zero_grad()
            embeddings, importance, _, _ = model(x, edge_index)
            loss = F.mse_loss(importance, target)
            # Graph regularization: connected nodes should have similar embeddings
            src_emb = embeddings[edge_index[0]]
            dst_emb = embeddings[edge_index[1]]
            reg_loss = 0.01 * (edge_weight.unsqueeze(1) * (src_emb - dst_emb)**2).mean()
            total_loss = loss + reg_loss
            total_loss.backward()
            optimizer.step()
            training_losses.append(round(float(total_loss.item()), 6))

        # ── Extract results ───────────────────────────────────
        model.eval()
        with torch.no_grad():
            embeddings, importance, attn1, attn2 = model(x, edge_index)
        # Reconstruction quality metric (item 55)
        recon_error = float(F.mse_loss(importance, target).item())
        importance_scores = importance.cpu().numpy()
        embeddings_np = embeddings.cpu().numpy()

        # 2D projection of embeddings
        if embeddings_np.shape[0] > 2:
            pca = PCA(n_components=2)
            emb_2d = pca.fit_transform(embeddings_np)
        else:
            emb_2d = embeddings_np[:, :2]

        # Top residues by importance
        sorted_idx = np.argsort(importance_scores)[::-1]
        top_residues = []
        for idx in sorted_idx[:20]:
            top_residues.append({
                "resid": int(resids[idx]),
                "importance": round(float(importance_scores[idx]), 4),
                "rmsf": round(float(rmsf_values[idx]), 3),
                "contacts": round(float(contact_counts[idx]), 1),
            })

        # Extract attention weights for key interactions (item 56: shape validation)
        attn_edge_index = attn2[0].cpu().numpy()
        attn_weights = attn2[1].cpu().numpy()

        top_attn = []
        if attn_weights.size > 0 and attn_edge_index.ndim == 2 and attn_edge_index.shape[0] == 2:
            if attn_weights.ndim > 1:
                attn_mean = attn_weights.mean(axis=1)
            else:
                attn_mean = attn_weights
            # Validate shapes match
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
        if n_comm >= 2:
            km = KMeans(n_clusters=n_comm, n_init=10, random_state=42)
            comm_labels = km.fit_predict(embeddings_np)
            communities = {}
            for i, label in enumerate(comm_labels):
                communities.setdefault(int(label), []).append(int(resids[i]))
        else:
            communities = {}

        return {
            "resids": resids,
            "residue_importance": importance_scores.tolist(),
            "embeddings_2d": emb_2d.tolist(),
            "top_residues": top_residues,
            "attention_interactions": top_attn,
            "community_assignments": communities,
            "n_edges": len(edge_src) // 2,
            "model_type": "GAT+GCN hybrid",
            "training_losses": training_losses,
            "reconstruction_error": round(recon_error, 6),
        }

    except ImportError as e:
        return {"error": f"PyTorch Geometric not installed: {e}"}
    except Exception as e:
        return {"error": str(e)}
