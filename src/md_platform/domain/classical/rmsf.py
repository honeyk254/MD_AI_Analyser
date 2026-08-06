"""RMSF (Root Mean Square Fluctuation) analysis.

Computes per-residue RMSF of C-alpha atoms to identify flexible and rigid
regions, and groups highly flexible residues into contiguous segments.
"""

import time
import logging
from typing import Dict, List, Optional

import numpy as np
import MDAnalysis as mda

from ...schemas.analysis_bundle import ModuleResult, MetricSummary, PerResidueSeries
from ..frames import FrameWindow, iter_frames, window_kwargs

logger = logging.getLogger("md_ai_analyzer")

__version__ = "2.0.0"


def select_ca_atoms(universe: mda.Universe) -> mda.AtomGroup:
    """Select C-alpha atoms of the protein."""
    return universe.select_atoms("protein and name CA")


def collect_ca_positions(
    universe: mda.Universe,
    atoms: mda.AtomGroup,
    window: Optional[FrameWindow] = None,
    align: bool = True,
) -> np.ndarray:
    """Collect C-alpha positions over the window, optionally Kabsch-aligned.

    Alignment is applied frame-by-frame rather than by materialising the whole
    trajectory in memory, so this stays usable on long trajectories.
    """
    from MDAnalysis.analysis.align import alignto

    frames = iter_frames(universe, window)
    positions = np.empty((len(frames), len(atoms), 3), dtype=np.float64)

    ref = None
    if align:
        ref = universe.copy()
        ref.trajectory[window_kwargs(universe, window)["start"]]

    for i, _ts in enumerate(frames):
        if align:
            alignto(universe, ref, select="protein and name CA")
        positions[i] = atoms.positions
    return positions


def compute_rmsf(
    universe: mda.Universe, window: Optional[FrameWindow] = None, **kwargs
) -> ModuleResult:
    """Compute per-residue RMSF of C-alpha atoms."""
    start_time = time.time()

    ca_atoms: mda.AtomGroup = select_ca_atoms(universe)
    if len(ca_atoms) == 0:
        raise ValueError("No protein C-alpha atoms found for RMSF analysis")

    # RMSF is computed from Kabsch-aligned positions so that rigid-body
    # rotation/translation is not counted as flexibility.
    positions = collect_ca_positions(universe, atoms=ca_atoms, window=window, align=True)
    n_frames = positions.shape[0]
    mean_pos = positions.mean(axis=0)
    delta = positions - mean_pos
    rmsf_values: np.ndarray = np.sqrt(np.mean(np.sum(delta**2, axis=2), axis=0))

    resids: List[int] = ca_atoms.resids.tolist()
    resnames: List[str] = ca_atoms.resnames.tolist()

    mean_rmsf: float = float(np.mean(rmsf_values))
    std_rmsf: float = float(np.std(rmsf_values))

    rmsf_arr = np.asarray(rmsf_values, dtype=np.float64)
    resid_arr = np.asarray(resids)

    high_mask = rmsf_arr > (mean_rmsf + std_rmsf)
    low_mask = rmsf_arr < (mean_rmsf - 0.5 * std_rmsf)

    high_flex: List[int] = resid_arr[high_mask].tolist()
    low_flex: List[int] = resid_arr[low_mask].tolist()

    flexible_segments = _find_contiguous_segments(high_flex)

    logger.info(
        "RMSF computed: %d residues, mean=%.3f A, %d highly flexible, %d segments",
        len(resids),
        mean_rmsf,
        len(high_flex),
        len(flexible_segments),
    )

    return ModuleResult(
        name="rmsf",
        version=__version__,
        runtime_seconds=time.time() - start_time,
        parameters={"aligned": True, "selection": "protein and name CA"},
        scalar_metrics={
            "mean_rmsf": MetricSummary(
                mean=mean_rmsf,
                std=std_rmsf,
                min=float(np.min(rmsf_arr)),
                max=float(np.max(rmsf_arr)),
                unit="Angstrom",
                n_frames=n_frames,
            )
        },
        residue_metrics={
            "rmsf": PerResidueSeries(
                values=rmsf_values.tolist(), resids=resids, unit="Angstrom"
            )
        },
        data={
            "resnames": resnames,
            "high_flexibility_residues": high_flex,
            "low_flexibility_residues": low_flex,
            "flexible_segments": flexible_segments,
        },
    )


def _find_contiguous_segments(residues: List[int], gap: int = 2) -> List[Dict[str, int]]:
    """Identify contiguous runs in a sorted list of residue IDs."""
    if not residues:
        return []

    segments: List[Dict[str, int]] = []
    current: List[int] = [residues[0]]

    for r in residues[1:]:
        if r - current[-1] <= gap:
            current.append(r)
        else:
            if len(current) >= 3:
                segments.append(
                    {"start": current[0], "end": current[-1], "length": len(current)}
                )
            current = [r]

    if len(current) >= 3:
        segments.append({"start": current[0], "end": current[-1], "length": len(current)})

    return segments
