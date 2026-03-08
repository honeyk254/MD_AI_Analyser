"""Conformational clustering.

Groups trajectory frames into structural clusters using K-Means in
PCA-reduced coordinate space, with automatic selection of the optimal
cluster count via silhouette analysis.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import MDAnalysis as mda
from sklearn.cluster import KMeans

from ..utils.trajectory_utils import select_ca_atoms, collect_ca_coords_flat
from ..utils.ml_feature_utils import pca_reduce, find_optimal_k

logger = logging.getLogger("md_ai_analyzer")


def cluster_conformations(
    universe: mda.Universe,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Cluster trajectory conformations in PCA-reduced C-alpha space.

    Parameters
    ----------
    universe : mda.Universe
        MDAnalysis Universe with a loaded trajectory.
    **kwargs : Any
        Ignored; accepted for orchestrator compatibility.

    Returns
    -------
    dict[str, Any]
        Keys:

        * ``labels`` -- cluster assignment per frame.
        * ``n_clusters`` -- optimal number of clusters.
        * ``populations`` -- fraction of frames in each cluster.
        * ``centroids`` -- representative frame index per cluster.
        * ``silhouette_score`` -- best silhouette score.
        * ``silhouette_scores_by_k`` -- silhouette scores for each *k*.
        * ``n_transitions`` -- number of cluster-to-cluster transitions.
        * ``transitions`` -- first 100 transition events.
    """
    try:
        ca: mda.AtomGroup = select_ca_atoms(universe)

        # Collect flattened coordinates via shared utility
        coords: np.ndarray = collect_ca_coords_flat(universe, atoms=ca)

        # PCA reduction via shared utility
        try:
            reduced, _pca_model = pca_reduce(coords, n_components=10)
        except ValueError:
            logger.error("Not enough frames for PCA reduction in clustering")
            return {"error": "Not enough frames for clustering"}

        if reduced.shape[1] < 2:
            logger.error("PCA produced fewer than 2 components")
            return {"error": "Not enough frames for clustering"}

        # Determine optimal k via shared utility
        best_k, scores = find_optimal_k(reduced, k_min=2, random_state=42)

        # Final clustering with the optimal k
        km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
        labels: np.ndarray = km.fit_predict(reduced)

        # Populations (vectorised)
        unique, counts = np.unique(labels, return_counts=True)
        populations: dict[int, float] = {
            int(u): round(float(c) / len(labels), 3)
            for u, c in zip(unique, counts)
        }

        # Centroid frame: closest frame to each cluster centre (vectorised)
        centroids: dict[int, int] = {}
        for cl in unique:
            mask = labels == cl
            center = km.cluster_centers_[cl]
            dists = np.linalg.norm(reduced[mask] - center, axis=1)
            frame_indices = np.where(mask)[0]
            centroids[int(cl)] = int(frame_indices[np.argmin(dists)])

        # Transition detection (vectorised diff)
        label_diffs = np.diff(labels)
        transition_frames = np.where(label_diffs != 0)[0] + 1
        transitions = [
            {
                "frame": int(f),
                "from_cluster": int(labels[f - 1]),
                "to_cluster": int(labels[f]),
            }
            for f in transition_frames[:100]
        ]

        best_score: float = scores.get(best_k, -1.0)

        logger.info(
            "Clustering: k=%d, silhouette=%.3f, %d transitions",
            best_k,
            best_score,
            len(transition_frames),
        )

        return {
            "labels": labels.tolist(),
            "n_clusters": best_k,
            "populations": populations,
            "centroids": centroids,
            "silhouette_score": best_score,
            "silhouette_scores_by_k": scores,
            "n_transitions": int(len(transition_frames)),
            "transitions": transitions,
        }

    except Exception as e:
        logger.exception("Clustering computation failed")
        return {"error": str(e), "labels": [], "n_clusters": 0}
