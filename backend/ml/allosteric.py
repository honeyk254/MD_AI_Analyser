from __future__ import annotations

"""Allosteric pathway detection via residue interaction-network analysis.

Builds a graph whose edges encode significant dynamical cross-correlations
between C-alpha atoms and applies centrality metrics, community detection,
and shortest-path analysis to identify putative allosteric communication
pathways.
"""

import logging
from typing import Any, Dict, List

import numpy as np
import networkx as nx

from ..utils.trajectory_utils import (
    select_ca_atoms,
    collect_ca_positions,
    compute_dccm_from_positions,
)

logger = logging.getLogger("md_ai_analyzer")

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_allosteric_pathways(
    universe: Any,
    correlation_threshold: float = 0.5,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Build a residue interaction network from the DCCM and detect allosteric pathways.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        Loaded trajectory universe.
    correlation_threshold : float
        Minimum absolute correlation for an edge to be included in the
        network graph.
    **kwargs
        Additional keyword arguments (unused; accepted for orchestrator
        compatibility).

    Returns
    -------
    dict
        Keys:
        - ``resids`` : residue ID list
        - ``betweenness_centrality`` : per-residue betweenness centrality
        - ``closeness_centrality`` : per-residue closeness centrality
        - ``hub_residues`` : high-centrality communication hub residues
        - ``communities`` : greedy-modularity communities
        - ``shortest_paths`` : top shortest paths between distant correlated
          residues
        - ``network_edges`` : list of significant edges
        - ``n_edges`` : total edge count in the network
        - ``n_communities`` : number of communities detected
        - ``modularity_score`` : modularity of greedy communities
        - ``louvain_communities`` : Louvain community assignments
        - ``louvain_modularity`` : modularity of Louvain partition
    """
    try:
        ca = select_ca_atoms(universe, fallback="protein")
        n_res = len(ca)
        resids: List[int] = ca.resids.tolist()

        # ── DCCM via shared utility (avoids redundant recomputation) ─
        positions = collect_ca_positions(universe, atoms=ca)
        dccm_norm = compute_dccm_from_positions(positions)
        mean_pos = positions.mean(axis=0)

        # ── Build graph ──────────────────────────────────────────
        G = nx.Graph()
        for i in range(n_res):
            G.add_node(i, resid=int(resids[i]))

        # Vectorised edge extraction: upper-triangle beyond sequence
        # separation of 5 residues with |correlation| > threshold
        abs_dccm = np.abs(dccm_norm)
        row_idx, col_idx = np.triu_indices(n_res, k=5)
        strong_mask = abs_dccm[row_idx, col_idx] > correlation_threshold

        edge_rows = row_idx[strong_mask]
        edge_cols = col_idx[strong_mask]
        edge_list: List[Dict[str, Any]] = []
        for i, j in zip(edge_rows, edge_cols):
            corr_val = float(abs_dccm[i, j])
            G.add_edge(int(i), int(j), weight=corr_val, distance=1.0 - corr_val)
            edge_list.append(
                {
                    "res_i": int(resids[i]),
                    "res_j": int(resids[j]),
                    "correlation": round(float(dccm_norm[i, j]), 3),
                }
            )

        if G.number_of_edges() == 0:
            logger.warning("No edges above correlation threshold %.2f.", correlation_threshold)
            return {"error": "No significant correlations found", "hub_residues": []}

        # ── Centrality metrics ───────────────────────────────────
        betweenness: Dict[int, float] = nx.betweenness_centrality(G, weight="distance")
        closeness: Dict[int, float] = nx.closeness_centrality(G, distance="distance")

        bc_values = [round(betweenness.get(i, 0.0), 4) for i in range(n_res)]
        cc_values = [round(closeness.get(i, 0.0), 4) for i in range(n_res)]

        # ── Hub residues (top betweenness centrality) ────────────
        bc_arr = np.array(list(betweenness.values()))
        hub_threshold = float(bc_arr.mean() + bc_arr.std())
        hubs: List[Dict[str, Any]] = []
        for node, bc in sorted(betweenness.items(), key=lambda x: -x[1]):
            if bc > hub_threshold:
                hubs.append(
                    {
                        "resid": int(resids[node]),
                        "betweenness": round(bc, 4),
                        "closeness": round(closeness.get(node, 0.0), 4),
                        "degree": G.degree(node),
                    }
                )

        # ── Community detection (greedy modularity) ──────────────
        community_list: List[Dict[str, Any]] = []
        modularity_score: float = 0.0
        try:
            communities = list(nx.community.greedy_modularity_communities(G))
            modularity_score = float(nx.community.modularity(G, communities))
            for idx, comm in enumerate(communities):
                comm_resids = sorted([int(resids[n]) for n in comm])
                community_list.append(
                    {"id": idx, "residues": comm_resids, "size": len(comm)}
                )
        except Exception as exc:
            logger.warning("Greedy-modularity community detection failed: %s", exc)

        # ── Louvain community detection (alternative) ────────────
        louvain_communities: List[Dict[str, Any]] = []
        louvain_modularity: float = 0.0
        try:
            louvain_comms = nx.community.louvain_communities(
                G, weight="weight", seed=42
            )
            louvain_modularity = float(nx.community.modularity(G, louvain_comms))
            for idx, comm in enumerate(louvain_comms):
                louvain_communities.append(
                    {
                        "id": idx,
                        "residues": sorted([int(resids[n]) for n in comm]),
                        "size": len(comm),
                    }
                )
        except Exception as exc:
            logger.warning("Louvain community detection failed: %s", exc)

        # ── Shortest paths between distant correlated pairs ──────
        shortest_paths: List[Dict[str, Any]] = []
        far_row, far_col = np.triu_indices(n_res, k=15)
        far_strong = np.abs(dccm_norm[far_row, far_col]) > 0.6
        for i, j in zip(far_row[far_strong], far_col[far_strong]):
            if G.has_node(int(i)) and G.has_node(int(j)):
                try:
                    path = nx.shortest_path(G, int(i), int(j), weight="distance")
                    path_resids = [int(resids[n]) for n in path]
                    shortest_paths.append(
                        {
                            "from_resid": int(resids[i]),
                            "to_resid": int(resids[j]),
                            "correlation": round(float(dccm_norm[i, j]), 3),
                            "path": path_resids,
                            "path_length": len(path),
                        }
                    )
                except nx.NetworkXNoPath:
                    pass

        shortest_paths.sort(key=lambda x: -abs(x["correlation"]))

        logger.info(
            "Allosteric network: %d edges, %d communities, %d hubs.",
            G.number_of_edges(),
            len(community_list),
            len(hubs),
        )

        return {
            "resids": resids,
            "betweenness_centrality": bc_values,
            "closeness_centrality": cc_values,
            "hub_residues": hubs[:20],
            "communities": community_list,
            "shortest_paths": shortest_paths[:30],
            "network_edges": edge_list[:200],
            "n_edges": G.number_of_edges(),
            "n_communities": len(community_list),
            "modularity_score": round(modularity_score, 4),
            "louvain_communities": louvain_communities,
            "louvain_modularity": round(louvain_modularity, 4),
        }

    except Exception as e:
        logger.exception("Allosteric pathway detection failed: %s", e)
        return {"error": str(e)}
