from __future__ import annotations

"""Dynamic Network Analysis with Time Resolution.

Splits the trajectory into time windows, computes DCCM-based correlation
networks per window, and tracks how communities and hub residues evolve
over the course of the simulation.
"""

import logging
from collections import Counter
from typing import Any, Dict, List, Tuple

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


def compute_dynamic_network(
    universe: Any,
    n_windows: int = 5,
    correlation_threshold: float = 0.5,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Time-windowed correlation network analysis.

    Splits the trajectory into *n_windows* windows, computes the DCCM and
    correlation network per window, and tracks how communities and hubs
    evolve.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        Loaded trajectory universe.
    n_windows : int
        Number of time windows to split the trajectory into.
    correlation_threshold : float
        Minimum absolute correlation to include an edge in the per-window
        network.
    **kwargs
        Additional keyword arguments (unused; accepted for orchestrator
        compatibility).

    Returns
    -------
    dict
        Keys:
        - ``resids`` : residue IDs
        - ``n_windows`` : number of time windows used
        - ``window_times`` : start/end time for each window
        - ``community_evolution`` : per-window community assignments
        - ``hub_evolution`` : per-window hub residues
        - ``edge_persistence`` : fraction of windows each edge is present
        - ``community_stability`` : per-residue community stability score
    """
    try:
        ca = select_ca_atoms(universe, fallback="protein")
        n_res = len(ca)
        resids: List[int] = ca.resids.tolist()
        n_frames = len(universe.trajectory)

        if n_frames < n_windows * 5:
            n_windows = max(2, n_frames // 5)

        frames_per_window = n_frames // n_windows

        # ── Collect all positions via shared utility ──────────────
        all_positions = collect_ca_positions(universe, atoms=ca)
        # Collect timestamps
        all_times: List[float] = [float(universe.trajectory[i].time) for i in range(n_frames)]

        community_evolution: List[List[int]] = []
        hub_evolution: List[List[Dict[str, Any]]] = []
        window_times: List[Dict[str, float]] = []
        all_edges: Dict[Tuple[int, int], int] = {}

        for w in range(n_windows):
            start_frame = w * frames_per_window
            end_frame = min((w + 1) * frames_per_window, n_frames)
            window_pos = all_positions[start_frame:end_frame]

            t_start = all_times[start_frame]
            t_end = all_times[min(end_frame - 1, n_frames - 1)]
            window_times.append({"start_ps": round(t_start, 1), "end_ps": round(t_end, 1)})

            # ── DCCM for this window via shared utility ──────────
            dccm_norm = compute_dccm_from_positions(window_pos)

            # ── Build graph (vectorised edge extraction) ─────────
            G = nx.Graph()
            for i in range(n_res):
                G.add_node(i, resid=int(resids[i]))

            abs_dccm = np.abs(dccm_norm)
            row_idx, col_idx = np.triu_indices(n_res, k=5)
            strong_mask = abs_dccm[row_idx, col_idx] > correlation_threshold

            for i, j in zip(row_idx[strong_mask], col_idx[strong_mask]):
                corr_val = float(abs_dccm[i, j])
                G.add_edge(int(i), int(j), weight=corr_val)
                edge_key = (int(min(i, j)), int(max(i, j)))
                all_edges[edge_key] = all_edges.get(edge_key, 0) + 1

            # ── Community detection ──────────────────────────────
            try:
                communities = list(nx.community.greedy_modularity_communities(G))
                comm_assignment = np.zeros(n_res, dtype=int)
                for c_idx, comm in enumerate(communities):
                    for node in comm:
                        comm_assignment[node] = c_idx
                community_evolution.append(comm_assignment.tolist())
            except Exception as exc:
                logger.debug("Community detection failed for window %d: %s", w, exc)
                community_evolution.append([0] * n_res)

            # ── Hub detection ────────────────────────────────────
            if G.number_of_edges() > 0:
                bc: Dict[int, float] = nx.betweenness_centrality(G)
                bc_arr = np.array(list(bc.values()))
                bc_threshold = float(bc_arr.mean() + bc_arr.std())
                window_hubs: List[Dict[str, Any]] = []
                for node, b in sorted(bc.items(), key=lambda x: -x[1]):
                    if b > bc_threshold:
                        window_hubs.append(
                            {
                                "resid": int(resids[node]),
                                "betweenness": round(float(b), 4),
                            }
                        )
                hub_evolution.append(window_hubs[:10])
            else:
                hub_evolution.append([])

        # ── Edge persistence ─────────────────────────────────────
        edge_persistence: List[Dict[str, Any]] = []
        for (i, j), count in sorted(all_edges.items(), key=lambda x: -x[1]):
            persistence = count / n_windows
            if persistence >= 0.3:
                edge_persistence.append(
                    {
                        "resid_i": int(resids[i]),
                        "resid_j": int(resids[j]),
                        "persistence": round(persistence, 2),
                        "n_windows_present": count,
                    }
                )
            if len(edge_persistence) >= 50:
                break

        # ── Community stability ──────────────────────────────────
        community_stability: List[float]
        if len(community_evolution) >= 2:
            # Vectorise: build matrix (n_windows, n_res) and compute mode fraction
            ce_arr = np.array(community_evolution)  # (n_windows, n_res)
            community_stability = []
            for r in range(n_res):
                assignments = ce_arr[:, r].tolist()
                counts = Counter(assignments)
                most_common_frac = counts.most_common(1)[0][1] / n_windows
                community_stability.append(round(most_common_frac, 3))
        else:
            community_stability = [1.0] * n_res

        logger.info(
            "Dynamic network analysis: %d windows, %d persistent edges.",
            n_windows,
            len(edge_persistence),
        )

        return {
            "resids": resids,
            "n_windows": n_windows,
            "window_times": window_times,
            "community_evolution": community_evolution,
            "hub_evolution": hub_evolution,
            "edge_persistence": edge_persistence,
            "community_stability": community_stability,
        }

    except Exception as e:
        logger.exception("Dynamic network analysis failed: %s", e)
        return {"error": str(e)}
