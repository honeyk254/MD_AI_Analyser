"""Report schemas: narrative, grounding verdicts, audit trail and review gate."""

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from .summary import GroundedSummary

REPORT_SCHEMA_VERSION = "1.0.0"


class ReviewStatus(str, Enum):
    """A report is only ``final`` once a human has signed it off."""

    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"


class ClaimStatus(str, Enum):
    """Verdict for a single numeric claim extracted from the narrative."""

    VERIFIED = "verified"
    MISMATCH = "mismatch"
    UNSUPPORTED = "unsupported"


class NarrativeSection(BaseModel):
    """One section of the draft report."""

    heading: str
    body: str


class NarrativeReport(BaseModel):
    """The generated narrative, before grounding verification."""

    sections: List[NarrativeSection]
    generator: str = Field(
        description="Which narrator produced this: an LLM model id, or 'template'."
    )

    def text(self) -> str:
        return "\n\n".join(f"{s.heading}\n{s.body}" for s in self.sections)


class NumericClaim(BaseModel):
    """A number as written in the narrative, with its surrounding context."""

    value: float
    unit: Optional[str] = None
    section: str
    context: str = Field(description="The sentence the number appeared in.")


class ClaimCheck(BaseModel):
    """Result of verifying one numeric claim against the AnalysisBundle."""

    claim: NumericClaim
    status: ClaimStatus
    matched_metric: Optional[str] = Field(
        None, description="Summary field the claim was matched against."
    )
    expected_value: Optional[float] = None
    tolerance: Optional[float] = None
    detail: str


class GroundingResult(BaseModel):
    """Aggregate outcome of the grounding/consistency pass."""

    checker_version: str
    passed: bool = Field(
        description="True only when no claim is a mismatch or unsupported."
    )
    checks: List[ClaimCheck] = Field(default_factory=list)
    n_verified: int = 0
    n_mismatched: int = 0
    n_unsupported: int = 0

    def failures(self) -> List[ClaimCheck]:
        return [c for c in self.checks if c.status is not ClaimStatus.VERIFIED]


class LLMUsage(BaseModel):
    """Token and cost accounting for one report generation."""

    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    n_llm_calls: int = 0
    n_tool_calls: int = 0
    cost_usd: float = 0.0


class ReportAudit(BaseModel):
    """Everything needed to reconstruct how a report was produced."""

    report_schema_version: str = REPORT_SCHEMA_VERSION
    prompt_version: str
    summary_schema_version: str
    bundle_hash: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    generation_seconds: float
    usage: LLMUsage
    tool_calls: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Ordered tool name/argument pairs the narrator requested.",
    )


class ReviewRecord(BaseModel):
    """Human review gate state."""

    status: ReviewStatus = ReviewStatus.DRAFT
    reviewer: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    comment: Optional[str] = None


class ReviewDecision(BaseModel):
    """Request body for signing a report off."""

    reviewer: str = Field(min_length=1)
    approve: bool
    comment: Optional[str] = None


class GeneratedReport(BaseModel):
    """A grounded report: narrative plus the evidence it was checked against."""

    run_id: str
    summary: GroundedSummary
    narrative: NarrativeReport
    grounding: GroundingResult
    audit: ReportAudit
    review: ReviewRecord = Field(default_factory=ReviewRecord)
    html_path: Optional[str] = None

    @property
    def is_final(self) -> bool:
        return self.review.status is ReviewStatus.APPROVED and self.grounding.passed
