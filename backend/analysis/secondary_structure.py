"""Secondary structure evolution using MDTraj DSSP.

Tracks helix, sheet, and coil content per residue over time by converting
the MDAnalysis Universe into an MDTraj trajectory and running simplified
DSSP assignment.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import MDAnalysis as mda

logger = logging.getLogger("md_ai_analyzer")


def compute_secondary_structure(
    universe: mda.Universe,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Compute secondary structure per residue over time using MDTraj DSSP.

    Parameters
    ----------
    universe : mda.Universe
        MDAnalysis Universe with a loaded trajectory.
    **kwargs : Any
        Additional keyword arguments (unused, accepted for API consistency).

    Returns
    -------
    dict[str, Any]
        ``resids``
            Residue IDs corresponding to the DSSP columns.
        ``helix_fraction``
            Per-frame helix content as a list of floats.
        ``sheet_fraction``
            Per-frame sheet content as a list of floats.
        ``coil_fraction``
            Per-frame coil content as a list of floats.
        ``per_residue_dominant_ss``
            Dominant secondary structure code (H/E/C) per residue.
        ``mean_helix``
            Mean helix fraction across all frames.
        ``mean_sheet``
            Mean sheet fraction across all frames.
        ``mean_coil``
            Mean coil fraction across all frames.
    """
    try:
        import mdtraj as md

        protein = universe.select_atoms("protein")
        if len(protein) == 0:
            logger.warning("No protein atoms found for secondary structure")
            return {"error": "No protein atoms found"}

        logger.info(
            "Computing secondary structure for %d protein atoms over %d frames",
            len(protein),
            len(universe.trajectory),
        )

        # Collect positions: convert Angstrom to nm for MDTraj
        n_frames = len(universe.trajectory)
        positions = np.empty(
            (n_frames, len(protein), 3), dtype=np.float64
        )
        for i, _ts in enumerate(universe.trajectory):
            positions[i] = protein.positions / 10.0  # Angstrom -> nm

        # Build MDTraj topology
        topology = md.Topology()
        chain = topology.add_chain()
        prev_resid: int | None = None
        residue_map: dict[int, Any] = {}
        for atom in protein:
            if atom.resid != prev_resid:
                res = topology.add_residue(atom.resname, chain)
                prev_resid = atom.resid
                residue_map[atom.resid] = res
            try:
                element = md.element.Element.getBySymbol(atom.element)
            except Exception:
                element = md.element.carbon
            topology.add_atom(atom.name, element, residue_map[atom.resid])

        traj = md.Trajectory(positions, topology)
        dssp_result: np.ndarray = md.compute_dssp(traj, simplified=True)

        resids: List[int] = sorted(set(protein.resids.tolist()))
        n_residues: int = dssp_result.shape[1] if dssp_result.ndim > 1 else 0

        # --- Vectorised per-frame fractions ---
        if n_residues > 0:
            helix_frac: List[float] = (
                (dssp_result == "H").sum(axis=1) / n_residues
            ).tolist()
            sheet_frac: List[float] = (
                (dssp_result == "E").sum(axis=1) / n_residues
            ).tolist()
            coil_frac: List[float] = (
                (dssp_result == "C").sum(axis=1) / n_residues
            ).tolist()
        else:
            helix_frac, sheet_frac, coil_frac = [], [], []

        # --- Dominant SS per residue (vectorised) ---
        dominant_ss: List[str] = []
        for r in range(n_residues):
            col = dssp_result[:, r]
            counts = {
                "H": int(np.sum(col == "H")),
                "E": int(np.sum(col == "E")),
                "C": int(np.sum(col == "C")),
            }
            dominant_ss.append(max(counts, key=counts.get))  # type: ignore[arg-type]

        logger.info(
            "Secondary structure complete: mean helix=%.2f, sheet=%.2f, coil=%.2f",
            float(np.mean(helix_frac)) if helix_frac else 0.0,
            float(np.mean(sheet_frac)) if sheet_frac else 0.0,
            float(np.mean(coil_frac)) if coil_frac else 0.0,
        )

        return {
            "resids": resids[:n_residues],
            "helix_fraction": helix_frac,
            "sheet_fraction": sheet_frac,
            "coil_fraction": coil_frac,
            "per_residue_dominant_ss": dominant_ss,
            "mean_helix": float(np.mean(helix_frac)) if helix_frac else 0,
            "mean_sheet": float(np.mean(sheet_frac)) if sheet_frac else 0,
            "mean_coil": float(np.mean(coil_frac)) if coil_frac else 0,
        }

    except Exception as e:
        logger.error(
            "Secondary structure computation failed: %s", e, exc_info=True
        )
        return {
            "error": str(e),
            "helix_fraction": [],
            "sheet_fraction": [],
            "coil_fraction": [],
        }
