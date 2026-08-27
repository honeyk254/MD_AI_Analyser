"""Phase 2 tests: report summary aggregation and the human review gate.

The grounding-checker fixtures live in tests/test_llm_eval.py (Phase 5 harness).
"""

from pathlib import Path

from md_platform.aggregation.report_summary import _module_takeaway, build_report_summary
from md_platform.llm.orchestrator import LLMOrchestrator
from md_platform.orchestrator import AnalysisOrchestrator
from md_platform.reporting.html_report import generate_html_report
from md_platform.schemas.api import RunStatus, StatusResponse


def test_report_summary_includes_reference_context(make_bundle) -> None:
    summary = build_report_summary(make_bundle())

    assert summary["trajectory"]["frames"] == 120
    assert "rmsd" in summary["modules"]
    assert "radius_of_gyration" in summary["reference_ranges"]


def test_fallback_report_covers_every_module(make_bundle) -> None:
    """Template mode must not collapse to a hardcoded three-module subset."""
    bundle = make_bundle()
    summary = build_report_summary(bundle)

    report = LLMOrchestrator(api_key="")._render_fallback_report(summary)

    for module_name, result in bundle.modules.items():
        takeaway = _module_takeaway(module_name, result)
        assert takeaway in report, f"module '{module_name}' missing from fallback report"
        assert f"- {module_name}: {takeaway}" in report

    # The old static boilerplate sentences that made every run look identical.
    for boilerplate in (
        "intentionally conservative",
        "No free-form claims beyond the aggregated bundle",
        "Review module-specific plots before final sign-off.",
    ):
        assert boilerplate not in report


def test_html_report_renders_qc_and_module_sections(tmp_path: Path, make_bundle) -> None:
    """The HTML page renders the deterministic summary, not just the narrative blob."""
    bundle = make_bundle()
    narrative = "# Grounded MD Report\n\n## QC Assessment\n- Equilibrated: True"

    report_path = generate_html_report(
        bundle,
        plots={},
        output_dir=tmp_path,
        narrative_report=narrative,
    )
    html = report_path.read_text(encoding="utf-8")

    assert f"<title>{bundle.run_id} — MD AI Analysis Report</title>" in html
    assert "<h2>Quality Control</h2>" in html
    assert "<h2>Module Results (2)</h2>" in html
    assert "backbone_rmsd" in html and "radius_of_gyration" in html

    # Narrative is rendered as markdown headings/lists, not a raw <pre> dump.
    assert "<h3>Grounded MD Report</h3>" in html
    assert "<h4>QC Assessment</h4>" in html
    assert "<pre>" not in html


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


def test_final_report_is_distinct_from_draft_and_carries_audit_trail(
    tmp_path: Path, make_bundle
) -> None:
    """Approval produces a self-contained document: reviewed text + review decision + audit trail."""
    orchestrator = AnalysisOrchestrator(output_dir=tmp_path)
    bundle = make_bundle()
    run_id = bundle.run_id
    draft = "Draft report with 2.5 Angstrom RMSD."
    orchestrator.bundles[run_id] = bundle
    orchestrator.drafts[run_id] = draft
    orchestrator.plots[run_id] = {"rmsd_plot": "{}"}
    orchestrator.statuses[run_id] = StatusResponse(
        run_id=run_id,
        status=RunStatus.HUMAN_REVIEW,
        message="Pending review",
    )

    orchestrator.approve_run(run_id, "Reviewer: approved")

    final_text = (tmp_path / run_id / "final_report.md").read_text(encoding="utf-8")

    assert "## Human Review" in final_text
    assert "- Reviewer sign-off: Reviewer: approved" in final_text
    assert "## Audit Trail" in final_text
    # Reviewed text embedded verbatim as a blockquote, not re-rendered or mutated.
    assert f"> {draft}" in final_text
    assert "## System" in final_text and str(bundle.trajectory_metadata.n_atoms) in final_text
    assert final_text != draft
