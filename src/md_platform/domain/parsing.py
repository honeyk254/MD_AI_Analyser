"""Parsing and metadata extraction for MD trajectories.

Responsible for safely interrogating the trajectory files and universe object
to extract deterministic metadata without guessing.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import MDAnalysis as mda

from ..schemas.analysis_bundle import TrajectoryMetadata


def parse_metadata(
    universe: mda.Universe,
    topology_file: Optional[str] = None,
    trajectory_file: Optional[str] = None,
) -> TrajectoryMetadata:
    """Extract deterministic metadata from the loaded Universe.

    If information is missing, it explicitly reports "unknown" rather than
    guessing (e.g. inferring force fields from bond lengths is forbidden).
    """
    n_frames = len(universe.trajectory)
    n_atoms = universe.atoms.n_atoms
    n_residues = universe.residues.n_residues

    # Time parsing
    try:
        # MDAnalysis trajectory dt is usually in ps
        timestep_ps = float(getattr(universe.trajectory, "dt", 0.0))
    except Exception:
        timestep_ps = 0.0

    total_time_ns = (n_frames * timestep_ps) / 1000.0

    # Format detection
    topo_ext = Path(topology_file).suffix.lower() if topology_file else ""
    traj_ext = Path(trajectory_file).suffix.lower() if trajectory_file else ""
    
    if topo_ext and traj_ext:
        original_format = f"{topo_ext}/{traj_ext}"
    elif topo_ext:
        original_format = topo_ext
    else:
        original_format = "unknown"

    # Force field / water model extraction
    force_field = "unknown — not recoverable"
    # Future enhancement: If topo_ext == ".tpr", we could shell out to `gmx dump` 
    # to parse the FF exactly, but guessing from raw atoms is strictly forbidden.

    # Box dimensions (average over trajectory, if present)
    box_dimensions = None
    if hasattr(universe.trajectory, "ts") and hasattr(universe.trajectory.ts, "dimensions"):
        try:
            boxes = []
            for ts in universe.trajectory:
                if ts.dimensions is not None:
                    # ts.dimensions is usually [A, B, C, alpha, beta, gamma]
                    boxes.append(ts.dimensions[:3])
            
            if boxes:
                avg_box = np.mean(boxes, axis=0)
                box_dimensions = [float(x) for x in avg_box]
        except Exception:
            pass

    return TrajectoryMetadata(
        n_frames_analyzed=n_frames,
        n_atoms=n_atoms,
        n_residues=n_residues,
        timestep_ps=timestep_ps,
        total_time_ns=total_time_ns,
        original_format=original_format,
        force_field=force_field,
        box_dimensions=box_dimensions,
    )
