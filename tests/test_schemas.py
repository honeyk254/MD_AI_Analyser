"""Tests for data contracts and schemas."""

from md_platform.schemas.analysis_bundle import TrajectoryMetadata

def test_trajectory_metadata():
    """Test that TrajectoryMetadata validates correctly."""
    meta = TrajectoryMetadata(
        n_frames_analyzed=100,
        n_atoms=5000,
        n_residues=300,
        timestep_ps=2.0,
        total_time_ns=0.2,
        original_format=".tpr/.xtc",
        force_field="unknown",
    )
    assert meta.n_frames_analyzed == 100
    assert meta.timestep_ps == 2.0
