"""
Allosteric pathway detection via residue interaction network analysis.
Uses graph centrality metrics and shortest path analysis on the DCCM graph.
"""
import numpy as np
import networkx as nx
from MDAnalysis.lib.distances import distance_array


def detect_allosteric_pathways(universe, correlation_threshold=0.5, **kwargs):
    """
    Build residue interaction network from DCCM and detect allosteric communication pathways.

    Returns dict with:
        - betweenness_centrality: per-residue betweenness centrality
        - closeness_centrality: per-residue closeness centrality
        - hub_residues: high-centrality communication hub residues
        - communities: detected residue communities
        - shortest_paths: top shortest paths between distant correlated residues
        - network_edges: list of significant edges
    """
    try:
        ca = universe.select_atoms("protein and name CA")
        if len(ca) == 0:
            return {"error": "No CA atoms found"}

        n_res = len(ca)
        resids = ca.resids.tolist()

        # Compute DCCM
        positions = []
        for ts in universe.trajectory:
            positions.append(ca.positions.copy())
        positions = np.array(positions)

        mean_pos = positions.mean(axis=0)
        delta = positions - mean_pos

        dccm = np.zeros((n_res, n_res))
        for i in range(n_res):
            for j in range(i, n_res):
                cij = np.mean(np.sum(delta[:, i, :] * delta[:, j, :], axis=1))
                dccm[i, j] = cij
                dccm[j, i] = cij

        diag = np.sqrt(np.diag(dccm))
        diag[diag == 0] = 1e-10
        dccm_norm = dccm / np.outer(diag, diag)

        # Build graph: edge weight = |correlation|, edge distance = 1 - |corr|
        G = nx.Graph()
        for i in range(n_res):
            G.add_node(i, resid=int(resids[i]))

        edge_list = []
        for i in range(n_res):
            for j in range(i + 5, n_res):  # skip sequential neighbors
                corr = abs(dccm_norm[i, j])
                if corr > correlation_threshold:
                    G.add_edge(i, j, weight=corr, distance=1.0 - corr)
                    edge_list.append({
                        "res_i": int(resids[i]),
                        "res_j": int(resids[j]),
                        "correlation": round(float(dccm_norm[i, j]), 3),
                    })

        if G.number_of_edges() == 0:
            return {"error": "No significant correlations found", "hub_residues": []}

        # Centrality metrics
        betweenness = nx.betweenness_centrality(G, weight='distance')
        closeness = nx.closeness_centrality(G, distance='distance')

        bc_values = [round(betweenness.get(i, 0), 4) for i in range(n_res)]
        cc_values = [round(closeness.get(i, 0), 4) for i in range(n_res)]

        # Hub residues (top betweenness centrality)
        hub_threshold = np.mean(list(betweenness.values())) + np.std(list(betweenness.values()))
        hubs = []
        for node, bc in sorted(betweenness.items(), key=lambda x: -x[1]):
            if bc > hub_threshold:
                hubs.append({
                    "resid": int(resids[node]),
                    "betweenness": round(bc, 4),
                    "closeness": round(closeness.get(node, 0), 4),
                    "degree": G.degree(node),
                })

        # Community detection — enhanced with modularity optimization (item 51)
        try:
            communities = list(nx.community.greedy_modularity_communities(G))
            community_list = []
            modularity_score = nx.community.modularity(G, communities)
            for idx, comm in enumerate(communities):
                comm_resids = sorted([int(resids[n]) for n in comm])
                community_list.append({
                    "id": idx,
                    "residues": comm_resids,
                    "size": len(comm),
                })
        except Exception:
            community_list = []
            modularity_score = 0.0

        # Louvain community detection (alternative)
        louvain_communities = []
        try:
            louvain_comms = nx.community.louvain_communities(G, weight='weight', seed=42)
            louvain_modularity = nx.community.modularity(G, louvain_comms)
            for idx, comm in enumerate(louvain_comms):
                louvain_communities.append({
                    "id": idx,
                    "residues": sorted([int(resids[n]) for n in comm]),
                    "size": len(comm),
                })
        except Exception:
            louvain_modularity = 0.0

        # Shortest paths between distant correlated pairs
        # Get average spatial positions
        mean_positions = mean_pos
        shortest_paths = []
        for i in range(n_res):
            for j in range(i + 15, n_res):  # far in sequence
                if abs(dccm_norm[i, j]) > 0.6 and G.has_node(i) and G.has_node(j):
                    try:
                        path = nx.shortest_path(G, i, j, weight='distance')
                        path_resids = [int(resids[n]) for n in path]
                        shortest_paths.append({
                            "from_resid": int(resids[i]),
                            "to_resid": int(resids[j]),
                            "correlation": round(float(dccm_norm[i, j]), 3),
                            "path": path_resids,
                            "path_length": len(path),
                        })
                    except nx.NetworkXNoPath:
                        pass

        shortest_paths.sort(key=lambda x: -abs(x["correlation"]))

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
            "modularity_score": round(float(modularity_score), 4),
            "louvain_communities": louvain_communities,
            "louvain_modularity": round(float(louvain_modularity), 4),
        }

    except Exception as e:
        return {"error": str(e)}
