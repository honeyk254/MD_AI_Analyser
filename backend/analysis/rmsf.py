"""RMSF (Root Mean Square Fluctuation) analysis.

Computes per-residue RMSF of C-alpha atoms to identify flexible and rigid
regions, and groups highly flexible residues into contiguous segments.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.rms import RMSF as MDA_RMSF

from ..utils.trajectory_utils import select_ca_atoms

logger = logging.getLogger("md_ai_analyzer")


def compute_rmsf(
    universe: mda.Universe,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Compute per-residue RMSF of C-alpha atoms.

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

        * ``resids`` -- list of residue IDs.
        * ``resnames`` -- list of residue names.
        * ``rmsf`` -- list of RMSF values (angstrom).
        * ``mean_rmsf`` -- mean RMSF across all residues.
        * ``std_rmsf`` -- standard deviation of RMSF values.
        * ``high_flexibility_residues`` -- residues with RMSF > mean + 1 std.
        * ``low_flexibility_residues`` -- residues with RMSF < mean - 0.5 std.
        * ``flexible_segments`` -- contiguous runs (length >= 3) of highly
          flexible residues.
    """
    try:
        ca_atoms: mda.AtomGroup = select_ca_atoms(universe)

        rmsf_calc = MDA_RMSF(ca_atoms).run()
        rmsf_values: np.ndarray = rmsf_calc.results.rmsf

        resids: list[int] = ca_atoms.resids.tolist()
        resnames: list[str] = ca_atoms.resnames.tolist()

        mean_rmsf: float = float(np.mean(rmsf_values))
        std_rmsf: float = float(np.std(rmsf_values))

        # Vectorised threshold tests
        rmsf_arr = np.asarray(rmsf_values, dtype=np.float64)
        resid_arr = np.asarray(resids)

        high_mask = rmsf_arr > (mean_rmsf + std_rmsf)
        low_mask = rmsf_arr < (mean_rmsf - 0.5 * std_rmsf)

        high_flex: list[int] = resid_arr[high_mask].tolist()
        low_flex: list[int] = resid_arr[low_mask].tolist()

        flexible_segments = _find_contiguous_segments(high_flex)

        logger.info(
            "RMSF computed: %d residues, mean=%.3f A, %d highly flexible, "
            "%d segments",
            len(resids),
            mean_rmsf,
            len(high_flex),
            len(flexible_segments),
        )

        return {
            "resids": resids,
            "resnames": resnames,
            "rmsf": rmsf_values.tolist(),
            "mean_rmsf": mean_rmsf,
            "std_rmsf": std_rmsf,
            "high_flexibility_residues": high_flex,
            "low_flexibility_residues": low_flex,
            "flexible_segments": flexible_segments,
        }

    except Exception as e:
        logger.exception("RMSF computation failed")
        return {"error": str(e)}


def _find_contiguous_segments(
    residues: List[int],
    gap: int = 2,
) -> List[Dict[str, int]]:
    """Identify contiguous runs in a sorted list of residue IDs.

    Parameters
    ----------
    residues : list[int]
        Sorted residue IDs (e.g. highly flexible residues).
    gap : int
        Maximum gap between consecutive residues to still be considered
        contiguous.

    Returns
    -------
    list[dict[str, int]]
        Each dict has keys ``start``, ``end``, ``length`` for segments of
        length >= 3.
    """
    if not residues:
        return []

    segments: list[dict[str, int]] = []
    current: list[int] = [residues[0]]

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
        segments.append(
            {"start": current[0], "end": current[-1], "length": len(current)}
        )

    return segments
