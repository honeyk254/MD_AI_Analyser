"""Solvent Accessible Surface Area (SASA) analysis.

Tracks protein surface exposure over time using MDTraj's Shrake--Rupley algorithm.
"""

import time
import logging
import numpy as np
import MDAnalysis as mda

from ...schemas.analysis_bundle import ModuleResult, MetricSummary, PerResidueSeries

logger = logging.getLogger("md_ai_analyzer")

__version__ = "2.0.0"


def compute_sasa(universe: mda.Universe, **kwargs) -> ModuleResult:
    """Compute SASA over the trajectory using MDTraj Shrake--Rupley."""
    start_time = time.time()
    
    import mdtraj as md

    protein = universe.select_atoms("protein")
    if len(protein) == 0:
        raise ValueError("No protein atoms found for SASA analysis")

    n_frames: int = len(universe.trajectory)
    logger.info(
        "Computing SASA for %d protein atoms over %d frames",
        len(protein), n_frames,
    )

    # Collect positions and convert Angstrom -> nm for MDTraj
    positions = np.empty((n_frames, len(protein), 3), dtype=np.float64)
    times = np.empty(n_frames, dtype=np.float64)
    for i, ts in enumerate(universe.trajectory):
        positions[i] = protein.positions / 10.0  # Angstrom -> nm
        times[i] = ts.time

    # Build MDTraj topology
    topology = md.Topology()
    chain = topology.add_chain()
    prev_resid = None
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

    topology.create_standard_bonds()
    traj = md.Trajectory(positions, topology)

    # Shrake-Rupley SASA per residue: (n_frames, n_residues)
    sasa: np.ndarray = md.shrake_rupley(traj, mode="residue")

    total_sasa = sasa.sum(axis=1)
    avg_per_residue = sasa.mean(axis=0)

    resids = sorted(set(protein.resids.tolist()))
    n_res = min(len(resids), len(avg_per_residue))

    avg_trimmed = avg_per_residue[:n_res]
    mean_sasa = float(np.mean(avg_trimmed))
    std_sasa = float(np.std(avg_trimmed))

    # Vectorised buried/exposed classification
    buried = [
        int(resids[i])
        for i in np.where(avg_trimmed < mean_sasa - std_sasa)[0]
    ]
    exposed = [
        int(resids[i])
        for i in np.where(avg_trimmed > mean_sasa + std_sasa)[0]
    ]

    logger.info(
        "SASA complete: mean total=%.2f nm^2, %d buried, %d exposed residues",
        float(np.mean(total_sasa)), len(buried), len(exposed),
    )

    return ModuleResult(
        name="sasa",
        version=__version__,
        runtime_seconds=time.time() - start_time,
        parameters={},
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
                values=avg_trimmed.tolist(),
                resids=resids[:n_res],
                unit="nm^2",
            )
        },
        data={
            "time_ps": times.tolist(),
            "buried_residues": buried,
            "exposed_residues": exposed,
        }
    )
