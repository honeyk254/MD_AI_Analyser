"""MDAnalysis -> MDTraj conversion.

SASA (Shrake-Rupley) and DSSP come from MDTraj, so an ``AtomGroup`` has to be
converted into an ``mdtraj.Trajectory``. The topology is round-tripped through a
temporary PDB rather than assembled atom-by-atom: PDB parsing gives MDTraj the
chain boundaries, residue identities and elements it needs, which hand-built
topologies silently get wrong for multi-chain systems or non-contiguous resids
(and a wrong element means a wrong Shrake-Rupley radius).

Residue identifiers for per-residue outputs are read back off the MDTraj
topology, so they are positionally aligned with MDTraj's per-residue arrays by
construction.
"""

import logging
import tempfile
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

import numpy as np
import MDAnalysis as mda

from .frames import FrameWindow, iter_frames

if TYPE_CHECKING:  # pragma: no cover - typing only
    import mdtraj as md

logger = logging.getLogger("md_ai_analyzer")

ANGSTROM_TO_NM = 0.1


def topology_from_atoms(atoms: mda.AtomGroup) -> "md.Topology":
    """Build an MDTraj topology for ``atoms`` via a temporary PDB round-trip."""
    import mdtraj as md

    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_path = Path(tmpdir) / "selection.pdb"
        atoms.write(str(pdb_path))
        topology = md.load(str(pdb_path)).topology

    if topology.n_atoms != len(atoms):
        raise ValueError(
            "MDTraj topology has %d atoms but the selection has %d; the PDB "
            "round-trip did not preserve the selection"
            % (topology.n_atoms, len(atoms))
        )
    return topology


def collect_positions_nm(
    universe: mda.Universe,
    atoms: mda.AtomGroup,
    window: Optional[FrameWindow] = None,
) -> "tuple[np.ndarray, np.ndarray]":
    """Return ``(positions_nm, times_ps)`` for ``atoms`` over the frame window."""
    frames = iter_frames(universe, window)
    n_frames = len(frames)
    positions = np.empty((n_frames, len(atoms), 3), dtype=np.float64)
    times = np.empty(n_frames, dtype=np.float64)

    for i, ts in enumerate(frames):
        positions[i] = atoms.positions * ANGSTROM_TO_NM
        times[i] = ts.time
    return positions, times


def to_mdtraj(
    universe: mda.Universe,
    atoms: mda.AtomGroup,
    window: Optional[FrameWindow] = None,
) -> "tuple[md.Trajectory, np.ndarray]":
    """Convert ``atoms`` over the frame window into an MDTraj trajectory."""
    import mdtraj as md

    topology = topology_from_atoms(atoms)
    positions, times = collect_positions_nm(universe, atoms, window)
    return md.Trajectory(positions, topology), times


def residue_ids(topology: "md.Topology") -> List[int]:
    """Residue sequence numbers in MDTraj's per-residue array order."""
    return [int(res.resSeq) for res in topology.residues]


def residue_labels(topology: "md.Topology") -> List[str]:
    """Human-readable ``CHAIN:RESNAMERESID`` labels in per-residue array order."""
    labels: List[str] = []
    for res in topology.residues:
        chain = res.chain.chain_id or str(res.chain.index)
        labels.append(f"{chain}:{res.name}{res.resSeq}")
    return labels
