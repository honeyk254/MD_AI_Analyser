"""Backbone dihedral analysis.

Computes phi/psi torsions with MDAnalysis ``Ramachandran`` and summarizes
per-residue backbone flexibility using the circular standard deviation
(std = sqrt(-2 ln R), correct across the -180/180 wraparound).
"""

import logging
import time
from typing import List

import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis.dihedrals import Ramachandran

from ...schemas.analysis_bundle import MetricSummary, ModuleResult, PerResidueSeries

logger = logging.getLogger("md_ai_analyzer")

__version__ = "2.0.0"


def _circular_std_degrees(angles: np.ndarray) -> np.ndarray:
    """Circular std per column of an (n_frames, n_residues) angle array."""
    radians = np.radians(angles)
    mean_length = np.abs(np.exp(1j * radians).mean(axis=0))
    mean_length = np.clip(mean_length, 1e-12, 1.0)
    return np.degrees(np.sqrt(-2.0 * np.log(mean_length)))


def compute_dihedrals(universe: mda.Universe, **kwargs) -> ModuleResult:
    """Compute backbone phi/psi flexibility over the trajectory."""
    start_time = time.time()

    protein = universe.select_atoms("protein")
    if len(protein) == 0:
        raise ValueError("No protein atoms found for dihedral analysis")

    n_frames: int = len(universe.trajectory)
    logger.info("Running Ramachandran analysis over %d frames", n_frames)

    rama = Ramachandran(protein).run()
    # angles shape: (n_frames, n_residue_pairs, 2) -> [phi, psi]
    angles: np.ndarray = np.asarray(rama.angles, dtype=np.float64)
    per_frame_times: List[float] = [
        float(universe.trajectory[i].time) for i in range(min(n_frames, len(angles)))
    ]

    phi_std = _circular_std_degrees(angles[:, :, 0])
    psi_std = _circular_std_degrees(angles[:, :, 1])
    flexibility = 0.5 * (phi_std + psi_std)  # per-residue mean of phi/psi circular std
    # Each column of `angles` is one residue's phi/psi pair; index by position.
    resids = list(range(1, angles.shape[1] + 1))

    logger.info(
        "Dihedral analysis complete: mean circular std %.1f deg over %d residues",
        float(flexibility.mean()), len(flexibility),
    )

    return ModuleResult(
        name="dihedrals",
        version=__version__,
        runtime_seconds=time.time() - start_time,
        parameters={"dihedrals": ["phi", "psi"]},
        scalar_metrics={
            "mean_backbone_circular_std": MetricSummary(
                mean=float(flexibility.mean()),
                std=float(flexibility.std()),
                min=float(flexibility.min()),
                max=float(flexibility.max()),
                unit="degree",
                n_frames=n_frames,
                time_series=None,
            )
        },
        residue_metrics={
            "phi_circular_std": PerResidueSeries(
                values=[float(v) for v in phi_std],
                resids=resids,
                unit="degree",
            ),
            "psi_circular_std": PerResidueSeries(
                values=[float(v) for v in psi_std],
                resids=resids,
                unit="degree",
            ),
        },
        data={
            "time_ps": per_frame_times,
            "n_residue_pairs": int(angles.shape[1]),
            "note": "Circular standard deviation (sqrt(-2 ln R)) per residue.",
        },
    )
