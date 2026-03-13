"""Solvent Accessible Surface Area (SASA) analysis.

Tracks protein surface exposure over time using MDTraj's Shrake--Rupley
algorithm.  Identifies buried (core) and exposed (surface) residues
based on their average SASA relative to the population mean.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import MDAnalysis as mda

logger = logging.getLogger("md_ai_analyzer")


def compute_sasa(
    universe: mda.Universe,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Compute SASA over the trajectory using MDTraj Shrake--Rupley.

    Parameters
    ----------
    universe : mda.Universe
        MDAnalysis Universe with a loaded trajectory.
    **kwargs : Any
        Additional keyword arguments (unused, accepted for API consistency).

    Returns
    -------
    dict[str, Any]
        ``time``
            List of timestamps (ps).
        ``total_sasa``
            Total SASA per frame (nm squared).
        ``per_residue_sasa``
            Average per-residue SASA.
        ``resids``
            Residue IDs.
        ``mean_total_sasa``
            Mean total SASA across all frames.
        ``buried_residues``
            Residues with SASA below (mean - 1 std).
        ``exposed_residues``
            Residues with SASA above (mean + 1 std).
    """
    try:
        import mdtraj as md

        protein = universe.select_atoms("protein")
        if len(protein) == 0:
            logger.warning("No protein atoms found for SASA analysis")
            return {"error": "No protein atoms found"}

        n_frames: int = len(universe.trajectory)
        logger.info(
            "Computing SASA for %d protein atoms over %d frames",
            len(protein),
            n_frames,
        )

        # Collect positions and convert Angstrom -> nm for MDTraj
        positions = np.empty(
            (n_frames, len(protein), 3), dtype=np.float64
        )
        times = np.empty(n_frames, dtype=np.float64)
        for i, ts in enumerate(universe.trajectory):
            positions[i] = protein.positions / 10.0  # Angstrom -> nm
            times[i] = ts.time

        # Build MDTraj topology
        topology = md.Topology()
        chain = topology.add_chain()
        prev_resid: int | None = None
        res_obj = None
        for atom in protein:
            if atom.resid != prev_resid:
                res_obj = topology.add_residue(atom.resname, chain)
                prev_resid = atom.resid
            try:
                element = md.element.Element.getBySymbol(atom.element)
            except Exception:
                element = md.element.carbon
            topology.add_atom(atom.name, element, res_obj)

        # Add standard bonds so MDTraj can correctly assign atom radii
        # for the Shrake-Rupley calculation.
        topology.create_standard_bonds()

        traj = md.Trajectory(positions, topology)

        # Shrake-Rupley SASA per residue: (n_frames, n_residues)
        sasa: np.ndarray = md.shrake_rupley(traj, mode="residue")

        total_sasa: List[float] = sasa.sum(axis=1).tolist()
        avg_per_residue: np.ndarray = sasa.mean(axis=0)

        resids: List[int] = sorted(set(protein.resids.tolist()))
        n_res: int = min(len(resids), len(avg_per_residue))

        avg_trimmed = avg_per_residue[:n_res]
        mean_sasa = float(np.mean(avg_trimmed))
        std_sasa = float(np.std(avg_trimmed))

        # Vectorised buried/exposed classification
        buried: List[int] = [
            int(resids[i])
            for i in np.where(avg_trimmed < mean_sasa - std_sasa)[0]
        ]
        exposed: List[int] = [
            int(resids[i])
            for i in np.where(avg_trimmed > mean_sasa + std_sasa)[0]
        ]

        logger.info(
            "SASA complete: mean total=%.2f nm^2, %d buried, %d exposed residues",
            float(np.mean(total_sasa)),
            len(buried),
            len(exposed),
        )

        return {
            "time": times.tolist(),
            "total_sasa": total_sasa,
            "per_residue_sasa": avg_trimmed.tolist(),
            "resids": resids[:n_res],
            "mean_total_sasa": float(np.mean(total_sasa)),
            "buried_residues": buried,
            "exposed_residues": exposed,
        }

    except Exception as e:
        logger.error("SASA computation failed: %s", e, exc_info=True)
        return {"error": str(e)}
