"""Solvent Accessible Surface Area (SASA) analysis.

Tracks protein surface exposure over time using MDTraj's Shrake-Rupley algorithm.
"""

import time
import logging
from typing import List, Optional

import numpy as np
import MDAnalysis as mda

from ...schemas.analysis_bundle import ModuleResult, MetricSummary, PerResidueSeries
from ..frames import FrameWindow
from ..mdtraj_bridge import residue_ids, to_mdtraj

logger = logging.getLogger("md_ai_analyzer")

__version__ = "2.0.0"


def compute_sasa(
    universe: mda.Universe, window: Optional[FrameWindow] = None, **kwargs
) -> ModuleResult:
    """Compute per-residue and total SASA using MDTraj Shrake-Rupley."""
    start_time = time.time()

    import mdtraj as md

    protein = universe.select_atoms("protein")
    if len(protein) == 0:
        raise ValueError("No protein atoms found for SASA analysis")

    traj, times = to_mdtraj(universe, protein, window)
    n_frames: int = traj.n_frames
    logger.info(
        "Computing SASA for %d protein atoms over %d frames", len(protein), n_frames
    )

    # Shrake-Rupley SASA per residue: (n_frames, n_residues)
    sasa: np.ndarray = md.shrake_rupley(traj, mode="residue")

    total_sasa = sasa.sum(axis=1)
    avg_per_residue = sasa.mean(axis=0)

    # Read resids off the MDTraj topology so they line up with the SASA columns.
    resids: List[int] = residue_ids(traj.topology)

    mean_sasa = float(np.mean(avg_per_residue))
    std_sasa = float(np.std(avg_per_residue))

    buried = [
        int(resids[i]) for i in np.where(avg_per_residue < mean_sasa - std_sasa)[0]
    ]
    exposed = [
        int(resids[i]) for i in np.where(avg_per_residue > mean_sasa + std_sasa)[0]
    ]

    logger.info(
        "SASA complete: mean total=%.2f nm^2, %d buried, %d exposed residues",
        float(np.mean(total_sasa)),
        len(buried),
        len(exposed),
    )

    return ModuleResult(
        name="sasa",
        version=__version__,
        runtime_seconds=time.time() - start_time,
        parameters={"algorithm": "shrake_rupley", "mode": "residue"},
        scalar_metrics={
            "total_sasa": MetricSummary(
                mean=float(np.mean(total_sasa)),
                std=float(np.std(total_sasa)),
                min=float(np.min(total_sasa)),
                max=float(np.max(total_sasa)),
                unit="nm^2",
                n_frames=n_frames,
                time_series=total_sasa.tolist(),
            )
        },
        residue_metrics={
            "per_residue_sasa": PerResidueSeries(
                values=avg_per_residue.tolist(),
                resids=resids,
                unit="nm^2",
            )
        },
        data={
            "time_ps": times.tolist(),
            "buried_residues": buried,
            "exposed_residues": exposed,
        },
    )
