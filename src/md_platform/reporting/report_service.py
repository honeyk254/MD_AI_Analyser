"""Report service: bundle -> grounded, reviewable report.

Pipeline per report:

1. aggregate the bundle into a ``GroundedSummary`` (deterministic);
2. narrate it through the three tools (model, or template offline);
3. verify every number in the narrative against the summary;
4. park the report in the human review gate.

A report that fails grounding stays a ``draft`` and cannot be approved: the gate
is the last line, the checker is the one that actually holds it.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Optional

from ..aggregation.summarizer import summarize_bundle
from ..schemas.analysis_bundle import AnalysisBundle
from ..schemas.report import (
    GeneratedReport,
    ReportAudit,
    ReviewDecision,
    ReviewRecord,
    ReviewStatus,
)
from ..schemas.summary import SUMMARY_SCHEMA_VERSION
from ..store import RunStore
from .grounding import check_narrative
from .html_report import generate_html_report
from .narrator import Narrator, TemplateNarrator
from .plots import generate_all_plots
from .prompts import PROMPT_VERSION

logger = logging.getLogger("md_ai_analyzer")


class ReviewGateError(RuntimeError):
    """Raised when a review action is not allowed in the report's current state."""


class ReportService:
    """Generates and reviews grounded reports."""

    def __init__(
        self,
        store: RunStore,
        narrator: Optional[Narrator] = None,
        fallback_narrator: Optional[Narrator] = None,
    ):
        self.store = store
        self.narrator = narrator or TemplateNarrator()
        self.fallback_narrator = fallback_narrator or TemplateNarrator()

    def generate(self, bundle: AnalysisBundle) -> GeneratedReport:
        """Produce a grounded report for ``bundle`` and persist it."""
        started = time.time()
        summary = summarize_bundle(bundle)

        try:
            narration = self.narrator.narrate(summary)
        except Exception as exc:
            # A provider outage must not leave the platform unable to report; the
            # fallback is deterministic and the audit trail records which ran.
            logger.warning("Narrator failed (%s); falling back to template.", exc)
            narration = self.fallback_narrator.narrate(summary)

        grounding = check_narrative(narration.narrative, summary)
        if not grounding.passed:
            logger.warning(
                "Grounding failed for run %s: %d mismatched, %d unsupported claims",
                bundle.run_id,
                grounding.n_mismatched,
                grounding.n_unsupported,
            )

        report = GeneratedReport(
            run_id=bundle.run_id,
            summary=summary,
            narrative=narration.narrative,
            grounding=grounding,
            audit=ReportAudit(
                prompt_version=PROMPT_VERSION,
                summary_schema_version=SUMMARY_SCHEMA_VERSION,
                bundle_hash=summary.bundle_hash,
                generation_seconds=round(time.time() - started, 3),
                usage=narration.usage,
                tool_calls=narration.tool_calls,
            ),
            review=ReviewRecord(
                status=(
                    ReviewStatus.PENDING_REVIEW if grounding.passed else ReviewStatus.DRAFT
                )
            ),
        )

        html_path = generate_html_report(
            bundle, generate_all_plots(bundle), self.store.run_dir(bundle.run_id), report
        )
        report.html_path = str(html_path)
        self.store.write_report(report)
        return report

    def review(self, run_id: str, decision: ReviewDecision) -> GeneratedReport:
        """Record a reviewer's sign-off or rejection."""
        report = self.store.read_report(run_id)
        if report is None:
            raise ReviewGateError(f"No report exists for run {run_id}.")
        if report.review.status is ReviewStatus.APPROVED:
            raise ReviewGateError("This report has already been approved.")
        if decision.approve and not report.grounding.passed:
            raise ReviewGateError(
                "This report cannot be approved: "
                f"{report.grounding.n_mismatched} claims mismatch the analysis bundle "
                f"and {report.grounding.n_unsupported} are unsupported. Regenerate it."
            )

        report.review = ReviewRecord(
            status=ReviewStatus.APPROVED if decision.approve else ReviewStatus.REJECTED,
            reviewer=decision.reviewer,
            reviewed_at=datetime.now(timezone.utc),
            comment=decision.comment,
        )
        # The HTML carries the review banner, so it is rewritten on sign-off.
        bundle = self.store.read_bundle(run_id)
        if bundle is not None:
            report.html_path = str(
                generate_html_report(
                    bundle,
                    generate_all_plots(bundle),
                    self.store.run_dir(run_id),
                    report,
                )
            )
        self.store.write_report(report)
        logger.info(
            "Run %s review: %s by %s", run_id, report.review.status.value, decision.reviewer
        )
        return report
