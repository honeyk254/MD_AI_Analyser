"""RMSD (Root Mean Square Deviation) analysis.

Computes backbone RMSD over the analysed frame window relative to its first
frame and estimates an equilibration point using a rolling-mean heuristic.
"""

import time
import logging
from typing import List, Optional

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.rms import RMSD as MDA_RMSD

from ...schemas.analysis_bundle import ModuleResult, MetricSummary
from ..frames import FrameWindow, window_kwargs

logger = logging.getLogger("md_ai_analyzer")

__version__ = "2.0.0"

# Fraction of the mean absolute rolling-mean change below which the RMSD trace
# is considered to have stopped drifting. Heuristic, not a statistical test —
# see the "Limitations" section of the generated report.
EQUILIBRATION_TOLERANCE = 0.5


def compute_rmsd(
    universe: mda.Universe, window: Optional[FrameWindow] = None, **kwargs
) -> ModuleResult:
    """Compute backbone RMSD over the analysed frame window."""
    start_time = time.time()

    select = "backbone"
    if len(universe.select_atoms(select)) == 0:
        logger.warning("No backbone atoms found; falling back to 'all' selection")
        select = "all"

    win = window_kwargs(universe, window)

    ref = universe.copy()
    ref.trajectory[win["start"]]

    R = MDA_RMSD(universe, ref, select=select, ref_frame=win["start"])
    R.run(**win)

    times: List[float] = R.results.rmsd[:, 1].tolist()
    rmsd_values: List[float] = R.results.rmsd[:, 2].tolist()

    rmsd_arr = np.asarray(rmsd_values, dtype=np.float64)

    # ----- equilibration detection via rolling mean ----- #
    roll: int = max(len(rmsd_arr) // 10, 5)
    equil_frame: Optional[int] = None

    if len(rmsd_arr) > roll:
        running_avg = np.convolve(rmsd_arr, np.ones(roll) / roll, mode="valid")
        diffs = np.abs(np.diff(running_avg))
        threshold = float(np.mean(diffs)) * EQUILIBRATION_TOLERANCE
        settled = np.where(diffs < threshold)[0]
        if len(settled) > 0:
            # Offset by half the rolling window to map the convolved index back
            # onto the original frame numbering.
            equil_frame = int(settled[0]) + roll // 2

    logger.info(
        "RMSD computed: %d frames, mean=%.3f A, equilibration=%s",
        len(rmsd_arr),
        float(np.mean(rmsd_arr)),
        equil_frame,
    )

    return ModuleResult(
        name="rmsd",
        version=__version__,
        runtime_seconds=time.time() - start_time,
        parameters={"select": select, **win},
        scalar_metrics={
            "backbone_rmsd": MetricSummary(
                mean=float(np.mean(rmsd_arr)),
                std=float(np.std(rmsd_arr)),
                min=float(np.min(rmsd_arr)),
                max=float(np.max(rmsd_arr)),
                unit="Angstrom",
                n_frames=len(rmsd_arr),
                time_series=rmsd_values,
            )
        },
        data={
            "time_ps": times,
            "equilibration_frame": equil_frame,
            "equilibration_method": (
                f"rolling mean of {roll} frames, |d(mean)| < "
                f"{EQUILIBRATION_TOLERANCE} x mean |d(mean)| (heuristic)"
            ),
        },
    )
