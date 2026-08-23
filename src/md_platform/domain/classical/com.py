"""Center-of-mass drift analysis.

Tracks the protein center of mass frame by frame and reports the drift from
the initial position — a basic translational-stability check.
"""

import logging
import time
from typing import List

import MDAnalysis as mda
import numpy as np

from ...schemas.analysis_bundle import MetricSummary, ModuleResult

logger = logging.getLogger("md_ai_analyzer")

__version__ = "2.0.0"


def compute_com(universe: mda.Universe, **kwargs) -> ModuleResult:
    """Compute protein center-of-mass drift over the trajectory."""
    start_time = time.time()

    protein = universe.select_atoms("protein")
    if len(protein) == 0:
        raise ValueError("No protein atoms found for center-of-mass analysis")

    positions: List[np.ndarray] = []
    times: List[float] = []
    for ts in universe.trajectory:
        positions.append(protein.center_of_mass())
        times.append(float(ts.time))

    com = np.asarray(positions, dtype=np.float64)
    drift = np.linalg.norm(com - com[0], axis=1)

    logger.info(
        "COM analysis complete: final drift %.2f A over %d frames",
        float(drift[-1]), len(drift),
    )

    return ModuleResult(
        name="com",
        version=__version__,
        runtime_seconds=time.time() - start_time,
        parameters={},
        scalar_metrics={
            "com_drift": MetricSummary(
                mean=float(drift.mean()),
                std=float(drift.std()),
                min=float(drift.min()),
                max=float(drift.max()),
                unit="Angstrom",
                n_frames=len(drift),
                time_series=[float(v) for v in drift],
            )
        },
        data={
            "time_ps": times,
            "final_drift_angstrom": float(drift[-1]),
            "note": "Drift is |COM(t) - COM(0)| without removing net rotation.",
        },
    )
