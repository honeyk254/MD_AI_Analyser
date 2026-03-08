from __future__ import annotations

"""Dimensionality reduction for conformational ensemble visualisation.

Applies PCA, UMAP, and t-SNE to C-alpha coordinate data, producing both
2-D and 3-D projections suitable for interactive scatter plots.
"""

import logging
from typing import Any, Dict

import numpy as np

from ..utils.trajectory_utils import (
    select_ca_atoms,
    collect_ca_coords_flat,
)
from ..utils.ml_feature_utils import (
    pca_reduce,
    standardise_features,
    set_global_seed,
)

logger = logging.getLogger("md_ai_analyzer")

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_dimensionality_reduction(
    universe: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Apply multiple dimensionality-reduction methods for visualisation.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        Loaded trajectory universe.
    **kwargs
        Additional keyword arguments (unused; accepted for orchestrator
        compatibility).

    Returns
    -------
    dict
        Keys:
        - ``pca_2d`` : 2-D PCA projections (list of [x, y])
        - ``pca_variance`` : explained variance ratios for 2-D PCA
        - ``pca_3d`` : 3-D PCA projections (list of [x, y, z])
        - ``pca_3d_variance`` : explained variance ratios for 3-D PCA
        - ``umap_2d`` : 2-D UMAP projections
        - ``umap_3d`` : 3-D UMAP projections
        - ``tsne_2d`` : 2-D t-SNE projections
        - ``tsne_3d`` : 3-D t-SNE projections
    """
    set_global_seed(42)
    try:
        # ── Coordinate collection ────────────────────────────────
        ca = select_ca_atoms(universe)
        coords = collect_ca_coords_flat(universe, atoms=ca)

        if len(coords) < 5:
            logger.warning("Only %d frames; need >= 5 for dimensionality reduction.", len(coords))
            return {"error": "Too few frames"}

        coords_scaled, _, _ = standardise_features(coords)

        # Pre-reduce to speed up UMAP / t-SNE
        reduced, _pca_pre = pca_reduce(coords_scaled, n_components=20)

        results: Dict[str, Any] = {}

        # ── PCA 2-D ──────────────────────────────────────────────
        pca_2d_proj, pca_2d_model = pca_reduce(coords_scaled, n_components=2)
        results["pca_2d"] = pca_2d_proj.tolist()
        results["pca_variance"] = pca_2d_model.explained_variance_ratio_.tolist()

        # ── PCA 3-D ──────────────────────────────────────────────
        if min(coords_scaled.shape[0] - 1, coords_scaled.shape[1]) >= 3:
            pca_3d_proj, pca_3d_model = pca_reduce(coords_scaled, n_components=3)
            results["pca_3d"] = pca_3d_proj.tolist()
            results["pca_3d_variance"] = pca_3d_model.explained_variance_ratio_.tolist()
        else:
            results["pca_3d"] = []
            results["pca_3d_variance"] = []

        # ── UMAP ─────────────────────────────────────────────────
        try:
            import umap

            n_neighbours = min(15, len(reduced) - 1)
            reducer_2d = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbours,
                min_dist=0.1,
                random_state=42,
                metric="euclidean",
            )
            umap_2d_proj: np.ndarray = reducer_2d.fit_transform(reduced)
            results["umap_2d"] = umap_2d_proj.tolist()
            logger.info("UMAP 2-D projection complete.")

            # 3-D variant
            try:
                reducer_3d = umap.UMAP(
                    n_components=3,
                    n_neighbors=n_neighbours,
                    min_dist=0.1,
                    random_state=42,
                    metric="euclidean",
                )
                umap_3d_proj: np.ndarray = reducer_3d.fit_transform(reduced)
                results["umap_3d"] = umap_3d_proj.tolist()
            except Exception as exc:
                logger.warning("UMAP 3-D failed: %s", exc)
                results["umap_3d"] = []
        except ImportError:
            logger.warning("umap-learn package not installed; skipping UMAP.")
            results["umap_2d"] = []
            results["umap_3d"] = []

        # ── t-SNE ────────────────────────────────────────────────
        try:
            from sklearn.manifold import TSNE

            perplexity = min(30, len(reduced) - 1)
            if perplexity >= 2:
                tsne_2d = TSNE(
                    n_components=2,
                    perplexity=perplexity,
                    random_state=42,
                    n_iter=1000,
                )
                tsne_2d_proj: np.ndarray = tsne_2d.fit_transform(reduced)
                results["tsne_2d"] = tsne_2d_proj.tolist()
                logger.info("t-SNE 2-D projection complete.")

                # 3-D variant
                try:
                    tsne_3d = TSNE(
                        n_components=3,
                        perplexity=perplexity,
                        random_state=42,
                        n_iter=1000,
                    )
                    tsne_3d_proj: np.ndarray = tsne_3d.fit_transform(reduced)
                    results["tsne_3d"] = tsne_3d_proj.tolist()
                except Exception as exc:
                    logger.warning("t-SNE 3-D failed: %s", exc)
                    results["tsne_3d"] = []
            else:
                logger.warning("Perplexity < 2 (%d); skipping t-SNE.", perplexity)
                results["tsne_2d"] = []
                results["tsne_3d"] = []
        except Exception as exc:
            logger.warning("t-SNE failed: %s", exc)
            results["tsne_2d"] = []
            results["tsne_3d"] = []

        return results

    except Exception as e:
        logger.exception("Dimensionality reduction failed: %s", e)
        return {"error": str(e)}
