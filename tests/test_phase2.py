from pathlib import Path

import pytest

from md_platform.aggregation.report_summary import build_report_summary
from md_platform.llm.grounding_checker import check_grounding
from md_platform.orchestrator import AnalysisOrchestrator
from md_platform.schemas.analysis_bundle import (
    AnalysisBundle,
    MetricSummary,
    ModuleResult,
    QCFlag,
    QCFlags,
    TrajectoryMetadata,
)
from md_platform.schemas.run_card import FileProvenance, RunCard, ToolVersions
from md_platform.schemas.api import RunStatus, StatusResponse


def make_bundle() -> AnalysisBundle:
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


@pytest.mark.parametrize(
    "wrong_number",
    [31.7, 44.4, 61.2, 88.8, 1234.5],
)
def test_grounding_checker_catches_injected_errors(wrong_number: float) -> None:
    bundle = make_bundle()
    report = f"RMSD averaged 2.5 Angstrom and the compactness was {wrong_number}."

    ungrounded = check_grounding(report, bundle)

    assert str(wrong_number) in ungrounded


def test_report_summary_includes_reference_context() -> None:
    summary = build_report_summary(make_bundle())

    assert summary["trajectory"]["frames"] == 120
    assert "rmsd" in summary["modules"]
    assert "radius_of_gyration" in summary["reference_ranges"]


def test_human_review_gate_approves_run(tmp_path: Path) -> None:
    orchestrator = AnalysisOrchestrator(output_dir=tmp_path)
    bundle = make_bundle()
    run_id = bundle.run_id
    orchestrator.bundles[run_id] = bundle
    orchestrator.drafts[run_id] = "Draft report with 2.5 Angstrom RMSD."
    orchestrator.statuses[run_id] = StatusResponse(
        run_id=run_id,
        status=RunStatus.HUMAN_REVIEW,
        message="Pending review",
    )

    status = orchestrator.approve_run(run_id, "Reviewer: approved")

    assert status.status == RunStatus.COMPLETED
    assert status.reviewer_signoff == "Reviewer: approved"
    assert (tmp_path / run_id / "final_report.md").exists()
