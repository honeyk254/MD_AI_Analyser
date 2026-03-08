"""
Dynamic Network Analysis with Time Resolution.
Splits the trajectory into time windows and computes correlation networks
per window to track how allosteric networks evolve over time.
"""
import numpy as np
import networkx as nx


def compute_dynamic_network(universe, n_windows=5, correlation_threshold=0.5, **kwargs):
    """
    Time-windowed correlation network analysis.

    Splits the trajectory into N windows, computes DCCM + community detection
    per window, and tracks how communities and hubs evolve.

    Returns dict with:
        - resids: residue IDs
        - n_windows: number of time windows
        - window_times: start/end time for each window
        - community_evolution: per-window community assignments per residue
        - hub_evolution: per-window hub residues
        - edge_persistence: fraction of windows each edge is present
        - community_stability: per-residue community stability score
    """
    try:
        ca = universe.select_atoms("protein and name CA")
        if len(ca) == 0:
            return {"error": "No CA atoms found"}

        n_res = len(ca)
        resids = ca.resids.tolist()
        n_frames = len(universe.trajectory)

        if n_frames < n_windows * 5:
            n_windows = max(2, n_frames // 5)

        frames_per_window = n_frames // n_windows

        # Collect all positions
        all_positions = []
        all_times = []
        for ts in universe.trajectory:
            all_positions.append(ca.positions.copy())
            all_times.append(float(ts.time))
        all_positions = np.array(all_positions)

        community_evolution = []
        hub_evolution = []
        window_times = []
        all_edges = {}  # (i,j) -> count of windows with this edge

        for w in range(n_windows):
            start_frame = w * frames_per_window
            end_frame = min((w + 1) * frames_per_window, n_frames)
            window_pos = all_positions[start_frame:end_frame]

            t_start = all_times[start_frame]
            t_end = all_times[min(end_frame - 1, n_frames - 1)]
            window_times.append({"start_ps": round(t_start, 1), "end_ps": round(t_end, 1)})

            # Compute DCCM for this window
            mean_pos = window_pos.mean(axis=0)
            delta = window_pos - mean_pos
            n_w_frames = len(window_pos)

            dccm = np.zeros((n_res, n_res))
            for i in range(n_res):
                for j in range(i, n_res):
                    cij = np.mean(np.sum(delta[:, i, :] * delta[:, j, :], axis=1))
                    dccm[i, j] = cij
                    dccm[j, i] = cij

            diag = np.sqrt(np.diag(dccm))
            diag[diag == 0] = 1e-10
            dccm_norm = dccm / np.outer(diag, diag)

            # Build graph
            G = nx.Graph()
            for i in range(n_res):
                G.add_node(i, resid=int(resids[i]))

            for i in range(n_res):
                for j in range(i + 5, n_res):
                    corr = abs(dccm_norm[i, j])
                    if corr > correlation_threshold:
                        G.add_edge(i, j, weight=corr)
                        edge_key = (min(i, j), max(i, j))
                        all_edges[edge_key] = all_edges.get(edge_key, 0) + 1

            # Community detection
            try:
                communities = list(nx.community.greedy_modularity_communities(G))
                comm_assignment = np.zeros(n_res, dtype=int)
                for c_idx, comm in enumerate(communities):
                    for node in comm:
                        comm_assignment[node] = c_idx
                community_evolution.append(comm_assignment.tolist())
            except Exception:
                community_evolution.append([0] * n_res)

            # Hub detection
            if G.number_of_edges() > 0:
                bc = nx.betweenness_centrality(G)
                bc_threshold = np.mean(list(bc.values())) + np.std(list(bc.values()))
                window_hubs = []
                for node, b in sorted(bc.items(), key=lambda x: -x[1]):
                    if b > bc_threshold:
                        window_hubs.append({
                            "resid": int(resids[node]),
                            "betweenness": round(float(b), 4),
                        })
                hub_evolution.append(window_hubs[:10])
            else:
                hub_evolution.append([])

        # Edge persistence: fraction of windows each edge appears in
        edge_persistence = []
        for (i, j), count in sorted(all_edges.items(), key=lambda x: -x[1]):
            persistence = count / n_windows
            if persistence >= 0.3:
                edge_persistence.append({
                    "resid_i": int(resids[i]),
                    "resid_j": int(resids[j]),
                    "persistence": round(persistence, 2),
                    "n_windows_present": count,
                })
            if len(edge_persistence) >= 50:
                break

        # Community stability: how often does each residue stay in the same community?
        community_stability = []
        if len(community_evolution) >= 2:
            for r in range(n_res):
                assignments = [community_evolution[w][r] for w in range(n_windows)]
                # Fraction of windows in the most common community
                from collections import Counter
                counts = Counter(assignments)
                most_common_frac = counts.most_common(1)[0][1] / n_windows
                community_stability.append(round(most_common_frac, 3))
        else:
            community_stability = [1.0] * n_res

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
        return {"error": str(e)}
