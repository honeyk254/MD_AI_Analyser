"""Radius of Gyration analysis.

Measures protein compactness over the analysed frame window.
"""

import time
import logging
from typing import Optional

import numpy as np
import MDAnalysis as mda

from ...schemas.analysis_bundle import ModuleResult, MetricSummary
from ..frames import FrameWindow, iter_frames

logger = logging.getLogger("md_ai_analyzer")

__version__ = "2.0.0"

# Change in mean Rg between the first and last quarter of the window, in
# Angstrom, above which the trace is described as compacting/expanding.
TREND_THRESHOLD_A = 0.5


def compute_rg(
    universe: mda.Universe, window: Optional[FrameWindow] = None, **kwargs
) -> ModuleResult:
    """Compute the radius of gyration over the analysed frame window."""
    start_time = time.time()

    selection = "protein"
    protein = universe.select_atoms(selection)
    if len(protein) == 0:
        selection = "all"
        protein = universe.select_atoms(selection)

    frames = iter_frames(universe, window)
    n_frames = len(frames)
    logger.info(
        "Computing radius of gyration for %d atoms over %d frames",
        len(protein),
        n_frames,
    )

    times = np.empty(n_frames, dtype=np.float64)
    rg_values = np.empty(n_frames, dtype=np.float64)

    for i, ts in enumerate(frames):
        times[i] = ts.time
        rg_values[i] = protein.radius_of_gyration()

    mean_rg = float(np.mean(rg_values))
    std_rg = float(np.std(rg_values))

    trend = "stable"
    quarter = n_frames // 4
    if quarter > 0:
        diff = float(np.mean(rg_values[-quarter:]) - np.mean(rg_values[:quarter]))
        if diff < -TREND_THRESHOLD_A:
            trend = "compacting"
        elif diff > TREND_THRESHOLD_A:
            trend = "expanding"

    logger.info(
        "Rg analysis complete: mean=%.2f A, std=%.2f A, trend=%s",
        mean_rg,
        std_rg,
        trend,
    )

    return ModuleResult(
        name="radius_of_gyration",
        version=__version__,
        runtime_seconds=time.time() - start_time,
        parameters={"selection": selection},
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
            "trend_threshold_angstrom": TREND_THRESHOLD_A,
        },
    )
