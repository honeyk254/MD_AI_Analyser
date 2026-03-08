"""Radius of Gyration analysis.

Measures protein compactness over time by tracking the radius of gyration
across all trajectory frames.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import MDAnalysis as mda

from ..utils.trajectory_utils import select_ca_atoms

logger = logging.getLogger("md_ai_analyzer")


def compute_rg(universe: mda.Universe, **kwargs: Any) -> Dict[str, Any]:
    """Compute the radius of gyration over the trajectory.

    The radius of gyration quantifies the overall compactness of the protein.
    A trend is detected by comparing the first-quarter and last-quarter
    averages.

    Parameters
    ----------
    universe : mda.Universe
        MDAnalysis Universe with a loaded trajectory.
    **kwargs : Any
        Additional keyword arguments (unused, accepted for API consistency).

    Returns
    -------
    dict[str, Any]
        ``time``
            List of timestamps (ps).
        ``rg``
            List of Rg values per frame (Angstrom).
        ``mean_rg``
            Mean radius of gyration (Angstrom).
        ``std_rg``
            Standard deviation of Rg (Angstrom).
        ``min_rg``
            Minimum Rg value (Angstrom).
        ``max_rg``
            Maximum Rg value (Angstrom).
        ``trend``
            One of ``'compacting'``, ``'expanding'``, or ``'stable'``.
    """
    try:
        protein = select_ca_atoms(
            universe, selection="protein", fallback="all"
        )
        logger.info(
            "Computing radius of gyration for %d atoms over %d frames",
            len(protein),
            len(universe.trajectory),
        )

        n_frames = len(universe.trajectory)
        times = np.empty(n_frames, dtype=np.float64)
        rg_values = np.empty(n_frames, dtype=np.float64)

        for i, ts in enumerate(universe.trajectory):
            times[i] = ts.time
            rg_values[i] = protein.radius_of_gyration()

        mean_rg = float(np.mean(rg_values))
        std_rg = float(np.std(rg_values))

        # Detect trend via first-quarter / last-quarter comparison
        if n_frames > 10:
            quarter = n_frames // 4
            first_quarter = float(np.mean(rg_values[:quarter]))
            last_quarter = float(np.mean(rg_values[-quarter:]))
            diff = last_quarter - first_quarter
            if diff < -0.5:
                trend = "compacting"
            elif diff > 0.5:
                trend = "expanding"
            else:
                trend = "stable"
        else:
            trend = "stable"

        logger.info(
            "Rg analysis complete: mean=%.2f A, std=%.2f A, trend=%s",
            mean_rg,
            std_rg,
            trend,
        )

        return {
            "time": times.tolist(),
            "rg": rg_values.tolist(),
            "mean_rg": mean_rg,
            "std_rg": std_rg,
            "min_rg": float(np.min(rg_values)),
            "max_rg": float(np.max(rg_values)),
            "trend": trend,
        }

    except Exception as e:
        logger.error("Radius of gyration computation failed: %s", e, exc_info=True)
        return {"error": str(e)}
