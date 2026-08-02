"""RMSD (Root Mean Square Deviation) analysis.

Computes backbone RMSD over the trajectory relative to the first frame
and estimates the equilibration point using a rolling-standard-deviation
heuristic.
"""

import time
import logging
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.rms import RMSD as MDA_RMSD

from ...schemas.analysis_bundle import ModuleResult, MetricSummary

logger = logging.getLogger("md_ai_analyzer")

__version__ = "2.0.0"


def compute_rmsd(universe: mda.Universe, **kwargs) -> ModuleResult:
    """Compute backbone RMSD over the trajectory.

    Returns a structured ModuleResult containing the time series and scalar summaries.
    """
    start_time = time.time()
    
    protein = universe.select_atoms("protein and backbone")
    if len(protein) == 0:
        logger.warning("No backbone atoms found; falling back to 'all' selection")
        protein = universe.select_atoms("all")

    # The reference is the first frame of the current universe slice
    ref = universe.copy()
    ref.trajectory[0]

    R = MDA_RMSD(universe, ref, select="backbone", ref_frame=0)
    R.run()

    times: list[float] = R.results.rmsd[:, 1].tolist()
    rmsd_values: list[float] = R.results.rmsd[:, 2].tolist()

    rmsd_arr = np.asarray(rmsd_values, dtype=np.float64)

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

    return ModuleResult(
        name="rmsd",
        version=__version__,
        runtime_seconds=time.time() - start_time,
        parameters={},
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
        }
    )
