"""Dynamic Cross-Correlation Matrix (DCCM).

Reveals correlated and anti-correlated motions between C-alpha atoms by
computing the normalised cross-correlation of positional fluctuations.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import MDAnalysis as mda

from ..utils.trajectory_utils import (
    select_ca_atoms,
    collect_ca_positions,
    compute_dccm_from_positions,
)

logger = logging.getLogger("md_ai_analyzer")


def compute_dccm(
    universe: mda.Universe,
    threshold: float = 0.7,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Compute the dynamic cross-correlation matrix for C-alpha atoms.

    The normalised correlation between residues *i* and *j* is

    .. math::

        C_{ij} = \\frac{\\langle \\Delta r_i \\cdot \\Delta r_j \\rangle}
                 {\\sqrt{\\langle |\\Delta r_i|^2 \\rangle \\,
                         \\langle |\\Delta r_j|^2 \\rangle}}

    Parameters
    ----------
    universe : mda.Universe
        MDAnalysis Universe with a loaded trajectory.
    threshold : float
        Absolute correlation threshold for reporting highly
        correlated / anti-correlated pairs.
    **kwargs : Any
        Ignored; accepted for orchestrator compatibility.

    Returns
    -------
    dict[str, Any]
        Keys:

        * ``dccm`` -- 2-D correlation matrix (values in [-1, 1]).
        * ``resids`` -- residue IDs.
        * ``n_residues`` -- number of C-alpha atoms.
        * ``highly_correlated_pairs`` -- top 50 pairs with C > *threshold*.
        * ``anticorrelated_pairs`` -- top 50 pairs with C < -*threshold*.
    """
    try:
        ca: mda.AtomGroup = select_ca_atoms(universe)
        n_atoms: int = len(ca)
        resids: list[int] = ca.resids.tolist()

        # Collect positions and compute DCCM via shared utilities
        positions: np.ndarray = collect_ca_positions(universe, atoms=ca)
        dccm_normalized: np.ndarray = compute_dccm_from_positions(positions)

        # --- extract highly (anti-)correlated pairs (vectorised) --- #
        # Build upper-triangle mask excluding the first 5 diagonals
        row_idx, col_idx = np.triu_indices(n_atoms, k=5)
        corr_values = dccm_normalized[row_idx, col_idx]

        # Correlated pairs
        pos_mask = corr_values > threshold
        corr_pairs = [
            {
                "res_i": int(resids[r]),
                "res_j": int(resids[c]),
                "correlation": round(float(v), 3),
            }
            for r, c, v in zip(
                row_idx[pos_mask], col_idx[pos_mask], corr_values[pos_mask]
            )
        ]
        corr_pairs.sort(key=lambda x: -x["correlation"])

        # Anti-correlated pairs (symmetric threshold)
        neg_threshold = -threshold
        neg_mask = corr_values < neg_threshold
        anticorr_pairs = [
            {
                "res_i": int(resids[r]),
                "res_j": int(resids[c]),
                "correlation": round(float(v), 3),
            }
            for r, c, v in zip(
                row_idx[neg_mask], col_idx[neg_mask], corr_values[neg_mask]
            )
        ]
        anticorr_pairs.sort(key=lambda x: x["correlation"])

        logger.info(
            "DCCM computed: %d residues, %d correlated pairs, "
            "%d anti-correlated pairs (threshold=%.2f)",
            n_atoms,
            len(corr_pairs),
            len(anticorr_pairs),
            threshold,
        )

        return {
            "dccm": dccm_normalized.tolist(),
            "resids": resids,
            "n_residues": n_atoms,
            "highly_correlated_pairs": corr_pairs[:50],
            "anticorrelated_pairs": anticorr_pairs[:50],
        }

    except Exception as e:
        logger.exception("DCCM computation failed")
        return {"error": str(e)}
