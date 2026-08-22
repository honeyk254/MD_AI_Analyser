"""Radius of Gyration analysis.

Measures protein compactness over time by tracking the radius of gyration
across all trajectory frames.
"""

import logging
import time

import MDAnalysis as mda
import numpy as np

from ...schemas.analysis_bundle import MetricSummary, ModuleResult

logger = logging.getLogger("md_ai_analyzer")

__version__ = "2.0.0"


def compute_rg(universe: mda.Universe, **kwargs) -> ModuleResult:
    """Compute the radius of gyration over the trajectory."""
    start_time = time.time()

    protein = universe.select_atoms("protein")
    if len(protein) == 0:
        protein = universe.select_atoms("all")

    n_frames = len(universe.trajectory)
    logger.info(
        "Computing radius of gyration for %d atoms over %d frames",
        len(protein), n_frames
    )

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
        mean_rg, std_rg, trend
    )

    return ModuleResult(
        name="radius_of_gyration",
        version=__version__,
        runtime_seconds=time.time() - start_time,
        parameters={},
        scalar_metrics={
            "radius_of_gyration": MetricSummary(
                mean=mean_rg,
                std=std_rg,
                min=float(np.min(rg_values)),
                max=float(np.max(rg_values)),
                unit="Angstrom",
                n_frames=n_frames,
                time_series=rg_values.tolist(),
            )
        },
        data={
            "time_ps": times.tolist(),
            "trend": trend,
        }
    )
