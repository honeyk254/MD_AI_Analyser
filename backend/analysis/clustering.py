"""
Conformational clustering.
Groups trajectory frames into structural clusters.
"""
import numpy as np
from sklearn.cluster import KMeans


def cluster_conformations(universe, **kwargs):
    """
    Cluster trajectory conformations using coordinates from PCA space.
    Falls back to Cα RMSD-based coordinates if PCA not available.

    Returns dict with:
        - labels: cluster assignment per frame
        - n_clusters: number of clusters
        - populations: fraction of frames in each cluster
        - centroids: centroid frame index per cluster
        - silhouette_score: clustering quality metric
    """
    try:
        from sklearn.metrics import silhouette_score as sk_silhouette

        ca = universe.select_atoms("protein and name CA")
        if len(ca) == 0:
            ca = universe.select_atoms("all")

        # Collect flattened coords
        coords = []
        for ts in universe.trajectory:
            coords.append(ca.positions.flatten().copy())
        coords = np.array(coords)

        # Reduce dimensionality first with PCA
        from sklearn.decomposition import PCA
        n_comp = min(10, coords.shape[0] - 1, coords.shape[1])
        if n_comp < 2:
            return {"error": "Not enough frames for clustering"}

        pca = PCA(n_components=n_comp)
        reduced = pca.fit_transform(coords)

        # Determine optimal k using elbow / silhouette
        best_k = 2
        best_score = -1
        max_k = min(10, len(reduced) // 5)
        max_k = max(max_k, 2)

        scores = {}
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(reduced)
            if len(set(labels)) < 2:
                continue
            score = sk_silhouette(reduced, labels)
            scores[k] = float(score)
            if score > best_score:
                best_score = score
                best_k = k

        # Final clustering
        km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
        labels = km.fit_predict(reduced)

        # Populations
        unique, counts = np.unique(labels, return_counts=True)
        populations = {int(u): round(float(c) / len(labels), 3) for u, c in zip(unique, counts)}

        # Find centroid frame (closest to cluster center)
        centroids = {}
        for cl in unique:
            mask = labels == cl
            center = km.cluster_centers_[cl]
            dists = np.linalg.norm(reduced[mask] - center, axis=1)
            frame_indices = np.where(mask)[0]
            centroids[int(cl)] = int(frame_indices[np.argmin(dists)])

        # Transition sequence
        transitions = []
        for i in range(1, len(labels)):
            if labels[i] != labels[i-1]:
                transitions.append({
                    "frame": i,
                    "from_cluster": int(labels[i-1]),
                    "to_cluster": int(labels[i]),
                })

        return {
            "labels": labels.tolist(),
            "n_clusters": best_k,
            "populations": populations,
            "centroids": centroids,
            "silhouette_score": best_score,
            "silhouette_scores_by_k": scores,
            "n_transitions": len(transitions),
            "transitions": transitions[:100],
        }

    except Exception as e:
        return {"error": str(e), "labels": [], "n_clusters": 0}
