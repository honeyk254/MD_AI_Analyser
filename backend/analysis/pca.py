"""Principal Component Analysis of protein backbone dynamics.

Extracts dominant modes of motion from C-alpha coordinates using PCA,
reporting explained variance, cumulative variance, and per-residue
contributions to the leading principal components.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import MDAnalysis as mda

from ..utils.trajectory_utils import (
    select_ca_atoms,
    collect_ca_coords_flat,
    residue_contributions_from_eigenvector,
)
from ..utils.ml_feature_utils import pca_reduce

logger = logging.getLogger("md_ai_analyzer")


def compute_pca(
    universe: mda.Universe,
    n_components: int = 10,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Perform PCA on C-alpha atom coordinates across the trajectory.

    Parameters
    ----------
    universe : mda.Universe
        MDAnalysis Universe with a loaded trajectory.
    n_components : int
        Maximum number of principal components to retain.
    **kwargs : Any
        Ignored; accepted for orchestrator compatibility.

    Returns
    -------
    dict[str, Any]
        Keys:

        * ``projections`` -- per-frame projections onto PCs (list of lists).
        * ``explained_variance`` -- explained variance ratio per component.
        * ``cumulative_variance`` -- cumulative explained variance.
        * ``n_components`` -- actual number of components returned.
        * ``mean_structure`` -- average structure as list of [x, y, z] per
          residue.
        * ``resids`` -- residue IDs.
        * ``pc1_residue_contributions`` -- per-residue contribution to PC1.
        * ``pc2_residue_contributions`` -- per-residue contribution to PC2.
        * ``pc3_residue_contributions`` -- per-residue contribution to PC3.
    """
    try:
        ca: mda.AtomGroup = select_ca_atoms(universe)
        n_atoms: int = len(ca)

        # Collect flattened coordinates using shared utility
        coords: np.ndarray = collect_ca_coords_flat(universe, atoms=ca)

        # PCA via shared utility (handles clamping of n_components)
        projections, pca_model = pca_reduce(coords, n_components=n_components)
        n_comp: int = pca_model.n_components_

        # Mean structure reshaped to (n_atoms, 3)
        mean_coords: np.ndarray = pca_model.mean_.reshape(-1, 3)

        # Per-residue contributions (vectorised via shared utility)
        def _contrib(comp_idx: int) -> list[float]:
            return residue_contributions_from_eigenvector(
                pca_model.components_[comp_idx], n_atoms
            ).tolist()

        resids: list[int] = ca.resids.tolist()

        logger.info(
            "PCA computed: %d components, cumulative variance %.2f%%",
            n_comp,
            float(np.sum(pca_model.explained_variance_ratio_) * 100),
        )

        return {
            "projections": projections.tolist(),
            "explained_variance": pca_model.explained_variance_ratio_.tolist(),
            "cumulative_variance": np.cumsum(
                pca_model.explained_variance_ratio_
            ).tolist(),
            "n_components": n_comp,
            "mean_structure": mean_coords.tolist(),
            "resids": resids,
            "pc1_residue_contributions": _contrib(0) if n_comp > 0 else [],
            "pc2_residue_contributions": _contrib(1) if n_comp > 1 else [],
            "pc3_residue_contributions": _contrib(2) if n_comp > 2 else [],
        }

    except Exception as e:
        logger.exception("PCA computation failed")
        return {"error": str(e), "projections": [], "explained_variance": []}
