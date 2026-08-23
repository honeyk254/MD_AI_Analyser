"""Shared fixtures for the test suite."""

import pytest

from md_platform.schemas.analysis_bundle import (
    AnalysisBundle,
    MetricSummary,
    ModuleResult,
    QCFlag,
    QCFlags,
    TrajectoryMetadata,
)
from md_platform.schemas.run_card import FileProvenance, RunCard, ToolVersions


@pytest.fixture
def make_bundle():
    """Factory for a small synthetic AnalysisBundle with known numbers."""

    def _make() -> AnalysisBundle:
        return AnalysisBundle(
            run_id="run-123",
            trajectory_metadata=TrajectoryMetadata(
                n_frames_analyzed=120,
                n_atoms=5000,
                n_residues=300,
                timestep_ps=2.0,
                total_time_ns=0.24,
                original_format=".tpr/.xtc",
                force_field="unknown",
            ),
            qc_flags=QCFlags(
                is_equilibrated=True,
                sufficient_frames=True,
                flags=[QCFlag(check_name="sufficient_frames", passed=True, details="ok")],
            ),
            modules={
                "rmsd": ModuleResult(
                    name="rmsd",
                    version="2.0.0",
                    runtime_seconds=1.0,
                    parameters={},
                    scalar_metrics={
                        "backbone_rmsd": MetricSummary(
                            mean=2.5,
                            std=0.4,
                            min=1.8,
                            max=3.1,
                            unit="Angstrom",
                            n_frames=120,
                            time_series=[2.0, 2.5, 2.7],
                        )
                    },
                    data={"time_ps": [0.0, 2.0, 4.0], "equilibration_frame": 12},
                ),
                "radius_of_gyration": ModuleResult(
                    name="radius_of_gyration",
                    version="2.0.0",
                    runtime_seconds=1.0,
                    parameters={},
                    scalar_metrics={
                        "radius_of_gyration": MetricSummary(
                            mean=18.2,
                            std=0.6,
                            min=17.8,
                            max=18.9,
                            unit="Angstrom",
                            n_frames=120,
                            time_series=[18.1, 18.2, 18.3],
                        )
                    },
                    data={"time_ps": [0.0, 2.0, 4.0], "trend": "stable"},
                ),
            },
            run_card=RunCard(
                inputs={
                    "topology": FileProvenance(
                        filename="top.tpr", sha256="abc", size_bytes=1
                    )
                },
                tools=ToolVersions(python="3.11.0", mdanalysis="2.6.0", numpy="1.24.0"),
                container_digest=None,
                parameters={},
            ),
        )

    return _make
