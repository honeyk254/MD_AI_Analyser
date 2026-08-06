"""Parsing and metadata extraction for MD trajectories.

Interrogates the trajectory files deterministically. Where information is not
present in the inputs it is reported as unknown rather than guessed — inferring
a force field from geometry statistics is out of scope by design.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import MDAnalysis as mda

from ..schemas.analysis_bundle import TrajectoryMetadata
from .frames import FrameWindow, iter_frames

# Box dimensions are averaged over at most this many evenly spaced frames, so
# metadata extraction never costs a full extra trajectory pass.
BOX_SAMPLE_FRAMES = 50

FORCE_FIELD_UNKNOWN = "unknown — not recoverable"
# Formats that embed the force field/parameter set and could be parsed
# deterministically (not yet implemented — reported as unknown until it is).
FF_BEARING_SUFFIXES = {".tpr", ".prmtop", ".parm7", ".psf"}


def parse_metadata(
    universe: mda.Universe,
    topology_file: Optional[str] = None,
    trajectory_file: Optional[str] = None,
    window: Optional[FrameWindow] = None,
) -> TrajectoryMetadata:
    """Extract deterministic metadata for the analysed frame window."""
    frames = iter_frames(universe, window)
    n_frames = len(frames)
    n_atoms = universe.atoms.n_atoms
    n_residues = universe.residues.n_residues

    # MDAnalysis `dt` is the interval between stored frames (ps), not the
    # integrator timestep.
    try:
        frame_interval_ps = float(getattr(universe.trajectory, "dt", 0.0))
    except (TypeError, ValueError):
        frame_interval_ps = 0.0

    stride = max(getattr(frames, "step", 1) or 1, 1)
    sampled_interval_ps = frame_interval_ps * stride
    # n_frames samples span n_frames - 1 intervals.
    total_time_ns = (max(n_frames - 1, 0) * sampled_interval_ps) / 1000.0

    topo_ext = Path(topology_file).suffix.lower() if topology_file else ""
    traj_ext = Path(trajectory_file).suffix.lower() if trajectory_file else ""

    if topo_ext and traj_ext:
        original_format = f"{topo_ext}/{traj_ext}"
    elif topo_ext:
        original_format = topo_ext
    else:
        original_format = "unknown"

    force_field = FORCE_FIELD_UNKNOWN
    if topo_ext in FF_BEARING_SUFFIXES:
        force_field = (
            f"unknown — {topo_ext} embeds parameters but deterministic "
            "extraction is not implemented"
        )

    return TrajectoryMetadata(
        n_frames_analyzed=n_frames,
        n_atoms=n_atoms,
        n_residues=n_residues,
        timestep_ps=sampled_interval_ps,
        total_time_ns=total_time_ns,
        original_format=original_format,
        force_field=force_field,
        box_dimensions=_average_box(universe, window, n_frames),
    )


def _average_box(
    universe: mda.Universe, window: Optional[FrameWindow], n_frames: int
) -> Optional[List[float]]:
    """Average box edge lengths over a sample of frames, if the box is set."""
    if n_frames == 0:
        return None

    base = window or FrameWindow()
    resolved = base.resolve(len(universe.trajectory))
    sample_step = resolved.step * max(n_frames // BOX_SAMPLE_FRAMES, 1)
    sample = FrameWindow(
        start=resolved.start, stop=resolved.stop, step=sample_step
    )

    boxes = []
    for ts in iter_frames(universe, sample):
        if ts.dimensions is not None and np.all(np.asarray(ts.dimensions[:3]) > 0):
            # ts.dimensions is [A, B, C, alpha, beta, gamma]
            boxes.append(np.asarray(ts.dimensions[:3], dtype=np.float64))

    if not boxes:
        return None
    return [float(x) for x in np.mean(boxes, axis=0)]
