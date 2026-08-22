"""Secondary structure evolution using MDTraj DSSP.

Tracks helix, sheet, and coil content per residue over time by converting
the MDAnalysis Universe into an MDTraj trajectory and running simplified
DSSP assignment.
"""

import logging
import time
from typing import List

import MDAnalysis as mda
import numpy as np

from ...schemas.analysis_bundle import MetricSummary, ModuleResult

logger = logging.getLogger("md_ai_analyzer")

__version__ = "2.0.0"


def compute_secondary_structure(universe: mda.Universe, **kwargs) -> ModuleResult:
    """Compute secondary structure per residue over time using MDTraj DSSP."""
    start_time = time.time()

    import mdtraj as md

    protein = universe.select_atoms("protein")
    if len(protein) == 0:
        raise ValueError("No protein atoms found for secondary structure")

    n_frames = len(universe.trajectory)
    logger.info(
        "Computing secondary structure for %d protein atoms over %d frames",
        len(protein), n_frames,
    )

    positions = np.empty((n_frames, len(protein), 3), dtype=np.float64)
    for i, _ts in enumerate(universe.trajectory):
        positions[i] = protein.positions / 10.0  # Angstrom -> nm

    topology = md.Topology()
    chain = topology.add_chain()
    prev_resid = None
    residue_map = {}
    mdtraj_atoms = []
    for atom in protein:
        if atom.resid != prev_resid:
            res = topology.add_residue(atom.resname, chain)
            prev_resid = atom.resid
            residue_map[atom.resid] = res
        try:
            element = md.element.Element.getBySymbol(atom.element)
        except Exception:
            element = md.element.carbon
        mdtraj_atoms.append(
            topology.add_atom(atom.name, element, residue_map[atom.resid])
        )

    topology.create_standard_bonds()
    traj = md.Trajectory(positions, topology)
    dssp_result: np.ndarray = md.compute_dssp(traj, simplified=True)

    resids: List[int] = sorted(set(protein.resids.tolist()))
    n_residues: int = dssp_result.shape[1] if dssp_result.ndim > 1 else 0

    if n_residues > 0:
        helix_frac = ((dssp_result == "H").sum(axis=1) / n_residues).tolist()
        sheet_frac = ((dssp_result == "E").sum(axis=1) / n_residues).tolist()
        coil_frac = ((dssp_result == "C").sum(axis=1) / n_residues).tolist()
    else:
        helix_frac, sheet_frac, coil_frac = [], [], []

    # Dominant SS per residue
    dominant_ss: List[str] = []
    for r in range(n_residues):
        col = dssp_result[:, r]
        counts = {
            "H": int(np.sum(col == "H")),
            "E": int(np.sum(col == "E")),
            "C": int(np.sum(col == "C")),
        }
        dominant_ss.append(max(counts, key=lambda k: counts[k]))

    mean_h = float(np.mean(helix_frac)) if helix_frac else 0.0
    mean_s = float(np.mean(sheet_frac)) if sheet_frac else 0.0
    mean_c = float(np.mean(coil_frac)) if coil_frac else 0.0

    logger.info(
        "Secondary structure complete: mean helix=%.2f, sheet=%.2f, coil=%.2f",
        mean_h, mean_s, mean_c,
    )

    return ModuleResult(
        name="secondary_structure",
        version=__version__,
        runtime_seconds=time.time() - start_time,
        parameters={},
        scalar_metrics={
            "helix_fraction": MetricSummary(
                mean=mean_h,
                std=float(np.std(helix_frac)) if helix_frac else 0.0,
                min=float(np.min(helix_frac)) if helix_frac else 0.0,
                max=float(np.max(helix_frac)) if helix_frac else 0.0,
                unit="fraction",
                n_frames=n_frames,
                time_series=helix_frac,
            ),
            "sheet_fraction": MetricSummary(
                mean=mean_s,
                std=float(np.std(sheet_frac)) if sheet_frac else 0.0,
                min=float(np.min(sheet_frac)) if sheet_frac else 0.0,
                max=float(np.max(sheet_frac)) if sheet_frac else 0.0,
                unit="fraction",
                n_frames=n_frames,
                time_series=sheet_frac,
            ),
            "coil_fraction": MetricSummary(
                mean=mean_c,
                std=float(np.std(coil_frac)) if coil_frac else 0.0,
                min=float(np.min(coil_frac)) if coil_frac else 0.0,
                max=float(np.max(coil_frac)) if coil_frac else 0.0,
                unit="fraction",
                n_frames=n_frames,
                time_series=coil_frac,
            ),
        },
        data={
            "resids": resids[:n_residues],
            "per_residue_dominant_ss": dominant_ss,
        }
    )
