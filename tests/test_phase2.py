"""Phase 2 tests: report summary aggregation and the human review gate.

The grounding-checker fixtures live in tests/test_llm_eval.py (Phase 5 harness).
"""

from pathlib import Path

from md_platform.aggregation.report_summary import build_report_summary
from md_platform.orchestrator import AnalysisOrchestrator
from md_platform.schemas.api import RunStatus, StatusResponse


def test_report_summary_includes_reference_context(make_bundle) -> None:
    summary = build_report_summary(make_bundle())

    assert summary["trajectory"]["frames"] == 120
    assert "rmsd" in summary["modules"]
    assert "radius_of_gyration" in summary["reference_ranges"]


def test_human_review_gate_approves_run(tmp_path: Path, make_bundle) -> None:
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
