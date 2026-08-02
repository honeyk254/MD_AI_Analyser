"""Validation gates for MD trajectories.

Rejects invalid inputs early and generates initial QC flags.
"""

import MDAnalysis as mda

from ..schemas.analysis_bundle import QCFlags, QCFlag, TrajectoryMetadata


def validate_trajectory(
    universe: mda.Universe,
    metadata: TrajectoryMetadata,
    min_frames: int = 10,
) -> QCFlags:
    """Run basic sanity checks on the trajectory.
    
    Raises ValueError if the trajectory is fundamentally unprocessable.
    Returns QCFlags for non-fatal issues.
    """
    flags = []
    
    # 1. Frame count check
    sufficient_frames = metadata.n_frames_analyzed >= min_frames
    if not sufficient_frames:
        flags.append(
            QCFlag(
                check_name="sufficient_frames",
                passed=False,
                details=f"Trajectory has {metadata.n_frames_analyzed} frames, which is below the recommended minimum of {min_frames}.",
            )
        )
    else:
        flags.append(
            QCFlag(
                check_name="sufficient_frames",
                passed=True,
                details=f"Frame count ({metadata.n_frames_analyzed}) is sufficient.",
            )
        )

    # 2. Time step check
    if metadata.timestep_ps <= 0:
        flags.append(
            QCFlag(
                check_name="valid_timestep",
                passed=False,
                details="Timestep is 0 or unreadable. Time-dependent kinetics calculations will be disabled or invalid.",
            )
        )
    else:
        flags.append(
            QCFlag(
                check_name="valid_timestep",
                passed=True,
                details=f"Timestep parsed as {metadata.timestep_ps:.3f} ps.",
            )
        )

    # 3. Protein presence check
    protein = universe.select_atoms("protein")
    if len(protein) == 0:
        # We don't fail outright because they might be simulating RNA/DNA or a lipid bilayer,
        # but many classical modules (like DSSP) will fail or skip.
        flags.append(
            QCFlag(
                check_name="contains_protein",
                passed=False,
                details="No protein atoms detected. Protein-specific analyses (e.g., secondary structure) will fail.",
            )
        )
    else:
        flags.append(
            QCFlag(
                check_name="contains_protein",
                passed=True,
                details=f"Found {len(protein)} protein atoms.",
            )
        )

    return QCFlags(
        is_equilibrated=False,  # Set by downstream RMSD module later
        sufficient_frames=sufficient_frames,
        flags=flags,
    )
