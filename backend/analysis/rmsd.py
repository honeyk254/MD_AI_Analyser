"""RMSD (Root Mean Square Deviation) analysis.

Computes backbone RMSD over the trajectory relative to the first frame
and estimates the equilibration point using a rolling-standard-deviation
heuristic.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.rms import RMSD as MDA_RMSD

logger = logging.getLogger("md_ai_analyzer")


def compute_rmsd(
    universe: mda.Universe,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Compute backbone RMSD over the trajectory.

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

        * ``time`` -- list of timestamps (ps).
        * ``rmsd`` -- list of RMSD values (angstrom).
        * ``mean_rmsd`` -- arithmetic mean RMSD.
        * ``std_rmsd`` -- standard deviation of RMSD.
        * ``max_rmsd`` -- maximum RMSD.
        * ``min_rmsd`` -- minimum RMSD.
        * ``equilibration_frame`` -- estimated frame index where RMSD
          stabilises (rolling-std heuristic).

    Notes
    -----
    Equilibration detection uses a rolling window whose size is 10 % of
    the trajectory length (minimum 5 frames).  The first frame where the
    rolling standard deviation drops below half the overall mean
    difference is reported.
    """
    try:
        protein = universe.select_atoms("protein and backbone")
        if len(protein) == 0:
            logger.warning(
                "No backbone atoms found; falling back to 'all' selection"
            )
            protein = universe.select_atoms("all")

        ref = universe.copy()
        ref.trajectory[0]

        R = MDA_RMSD(universe, ref, select="backbone", ref_frame=0)
        R.run()

        times: list[float] = R.results.rmsd[:, 1].tolist()
        rmsd_values: list[float] = R.results.rmsd[:, 2].tolist()

        rmsd_arr: np.ndarray = np.asarray(rmsd_values, dtype=np.float64)

        # ----- equilibration detection via rolling std ----- #
        window: int = max(len(rmsd_arr) // 10, 5)
        equil_frame: int = 0

        if len(rmsd_arr) > window:
            running_avg = np.convolve(
                rmsd_arr, np.ones(window) / window, mode="valid"
            )
            diffs = np.abs(np.diff(running_avg))
            threshold = np.mean(diffs) * 0.5
            equil_candidates = np.where(diffs < threshold)[0]
            if len(equil_candidates) > 0:
                # Offset by half-window to map convolved index back to
                # the original frame numbering.
                equil_frame = int(equil_candidates[0]) + window // 2

        logger.info(
            "RMSD computed: %d frames, mean=%.3f A, equilibration~frame %d",
            len(rmsd_arr),
            float(np.mean(rmsd_arr)),
            equil_frame,
        )

        return {
            "time": times,
            "rmsd": rmsd_values,
            "mean_rmsd": float(np.mean(rmsd_arr)),
            "std_rmsd": float(np.std(rmsd_arr)),
            "max_rmsd": float(np.max(rmsd_arr)),
            "min_rmsd": float(np.min(rmsd_arr)),
            "equilibration_frame": equil_frame,
        }

    except Exception as e:
        logger.exception("RMSD computation failed")
        return {
            "error": str(e),
            "time": [],
            "rmsd": [],
            "mean_rmsd": 0.0,
            "std_rmsd": 0.0,
            "max_rmsd": 0.0,
            "min_rmsd": 0.0,
            "equilibration_frame": 0,
        }
