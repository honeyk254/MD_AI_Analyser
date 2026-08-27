"""Final (post-approval) markdown report builder.

The final document records what was actually reviewed and how it was produced:
the approved draft verbatim, the human-review decision, and the audit trail the
platform already tracks (narrative mode, grounding result, cost/latency).
"""

from datetime import datetime, timezone
from typing import List, Optional

from ..ml.schemas import MLAnalysisBundle
from ..observability import LLM_METRICS, LLMCallMetric
from ..schemas.analysis_bundle import AnalysisBundle


def generate_final_markdown(
    bundle: AnalysisBundle,
    narrative: Optional[str],
    reviewer_signoff: str,
    ml_bundle: Optional[MLAnalysisBundle] = None,
    plot_names: Optional[List[str]] = None,
) -> str:
    """Build the final markdown deliverable for an approved run."""
    meta = bundle.trajectory_metadata
    approved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    lines: List[str] = [
        "# Final MD Analysis Report",
        "",
        f"- **Run:** {bundle.run_id}",
        f"- **Bundle generated at:** {bundle.created_at.isoformat()}",
        f"- **Approved at:** {approved_at}",
        "",
        "## System",
        (
            f"- Trajectory window: {meta.n_frames_analyzed} frames "
            f"(~{meta.total_time_ns:.2f} ns at {meta.timestep_ps:g} ps/frame)"
        ),
        f"- System size: {meta.n_atoms} atoms / {meta.n_residues} residues",
        f"- Input format: {meta.original_format}; force field: {meta.force_field}",
    ]

    qc_flags = bundle.qc_flags.flags
    failed_checks = [flag for flag in qc_flags if not flag.passed]
    qc_line = (
        f"{len(qc_flags) - len(failed_checks)}/{len(qc_flags)} checks passed"
        if qc_flags
        else "no flags recorded"
    )
    lines.append(f"- Quality control: {qc_line}; equilibrated={bundle.qc_flags.is_equilibrated}")

    lines.extend(["", "## Reviewed Report"])
    lines.append(
        "_Verbatim narrative as presented at human review (blockquote-quoted)._"
    )
    lines.append("")
    if narrative:
        for narrative_line in narrative.splitlines():
            lines.append(f"> {narrative_line}".rstrip())
    else:
        lines.append("> [no narrative was attached to this run]")

    lines.extend(
        [
            "",
            "## Human Review",
            f"- Reviewer sign-off: {reviewer_signoff}",
            "- Status: completed (human_review -> completed on approval)",
            "- The section above is the exact text that was reviewed.",
        ]
    )

    lines.extend(["", "## Audit Trail"])

    metric = _latest_metric_for_run(bundle.run_id)
    if metric is not None:
        model_note = (
            f"LLM-generated ({metric.mode})"
            if metric.mode == "llm"
            else "deterministic template (no LLM key configured at generation time)"
        )
        lines.append(f"- Narrative origin: {model_note}")
        lines.append(
            f"- Grounding check: {metric.ungrounded_claims} ungrounded claim(s) detected "
            f"at generation time."
        )
        lines.append(
            f"- Generation metrics: latency {metric.latency_s:.2f}s; "
            f"tokens in/out {metric.tokens_in}/{metric.tokens_out}; cost ${metric.cost_usd:.4f}."
        )
    else:
        lines.append("- Narrative origin: unknown (no LLM call metric recorded for this run).")

    modules_summary = ", ".join(
        f"{name} v{result.version} ({result.runtime_seconds:.2f}s)"
        for name, result in bundle.modules.items()
    )
    lines.append(f"- Classical modules executed ({len(bundle.modules)}): {modules_summary}")

    if ml_bundle is None:
        lines.append("- Phase 4 ML layer: not enabled for this run.")
    else:
        gating = ml_bundle.gating
        lines.append(
            f"- Phase 4 ML layer: status={ml_bundle.status}; gate "
            f"{'passed' if gating.passed else 'BLOCKED'} "
            f"(frames {gating.observed_frames}/{gating.minimum_frames_required}, "
            f"min pair transitions {gating.observed_min_transition_count}/"
            f"{gating.minimum_transition_count_required}, lag {gating.lag_frames} frames, "
            f"CK cutoff {gating.ck_cutoff})."
        )

    plot_list = ", ".join(sorted(plot_names)) if plot_names else "none"
    lines.append(f"- Figures embedded in analysis_report.html ({len(plot_names or [])}): {plot_list}")

    return "\n".join(lines)


def _latest_metric_for_run(run_id: str) -> Optional[LLMCallMetric]:
    """Most recent LLM call metric recorded for this run, if any."""
    for call in reversed(LLM_METRICS.calls):
        if call.run_id == run_id:
            return call
    return None
