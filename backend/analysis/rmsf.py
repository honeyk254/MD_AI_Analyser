"""
RMSF (Root Mean Square Fluctuation) analysis.
Computes per-residue RMSF to identify flexible and rigid regions.
"""
import numpy as np
from MDAnalysis.analysis.rms import RMSF as MDA_RMSF


def compute_rmsf(universe, **kwargs):
    """
    Compute per-residue RMSF of Cα atoms.

    Returns dict with:
        - resids: list of residue IDs
        - resnames: list of residue names
        - rmsf: list of RMSF values (Å)
        - mean_rmsf: float
        - high_flexibility_residues: residues with RMSF > mean + 1 std
        - low_flexibility_residues: residues with RMSF < mean - 0.5 std
    """
    try:
        ca_atoms = universe.select_atoms("protein and name CA")
    except Exception:
        ca_atoms = universe.select_atoms("all")

    if len(ca_atoms) == 0:
        return {"error": "No CA atoms found"}

    rmsf_calc = MDA_RMSF(ca_atoms).run()
    rmsf_values = rmsf_calc.results.rmsf

    resids = ca_atoms.resids.tolist()
    resnames = ca_atoms.resnames.tolist()

    mean_rmsf = float(np.mean(rmsf_values))
    std_rmsf = float(np.std(rmsf_values))

    high_flex = [int(r) for r, v in zip(resids, rmsf_values) if v > mean_rmsf + std_rmsf]
    low_flex = [int(r) for r, v in zip(resids, rmsf_values) if v < mean_rmsf - 0.5 * std_rmsf]

    # Identify contiguous flexible segments (loops)
    flexible_segments = _find_contiguous_segments(high_flex)

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


def _find_contiguous_segments(residues, gap=2):
    """Find contiguous segments from a list of residue IDs."""
    if not residues:
        return []
    segments = []
    current = [residues[0]]
    for r in residues[1:]:
        if r - current[-1] <= gap:
            current.append(r)
        else:
            if len(current) >= 3:
                segments.append({"start": current[0], "end": current[-1], "length": len(current)})
            current = [r]
    if len(current) >= 3:
        segments.append({"start": current[0], "end": current[-1], "length": len(current)})
    return segments
