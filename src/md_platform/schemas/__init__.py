"""Pydantic v2 schemas — the data-contract layer of the platform.

Every downstream consumer (aggregation, reporting, LLM layer) imports from
here.  Changes to these schemas are breaking changes and require a version
bump + migration path.
"""

from .analysis_bundle import (
    AnalysisBundle,
    MetricSummary,
    ModuleResult,
    PerResidueSeries,
    QCFlag,
    QCFlags,
    TrajectoryMetadata,
)
from .api import (
    AnalysisParameters,
    AnalysisRequest,
    AnalysisResponse,
    ErrorResponse,
    RunStatus,
    StatusResponse,
    SubmitRequest,
    UploadResponse,
)
from .report import (
    ClaimCheck,
    ClaimStatus,
    GeneratedReport,
    GroundingResult,
    LLMUsage,
    NarrativeReport,
    NarrativeSection,
    NumericClaim,
    ReportAudit,
    ReviewDecision,
    ReviewRecord,
    ReviewStatus,
)
from .run_card import FileProvenance, RunCard, ToolVersions
from .summary import (
    GroundedSummary,
    MetricStatistics,
    QCSummary,
    ReferenceComparison,
    TrajectoryFacts,
)

__all__ = [
    # analysis_bundle
    "AnalysisBundle",
    "MetricSummary",
    "ModuleResult",
    "PerResidueSeries",
    "QCFlag",
    "QCFlags",
    "TrajectoryMetadata",
    # run_card
    "FileProvenance",
    "RunCard",
    "ToolVersions",
    # api
    "AnalysisParameters",
    "AnalysisRequest",
    "AnalysisResponse",
    "ErrorResponse",
    "RunStatus",
    "StatusResponse",
    "SubmitRequest",
    "UploadResponse",
    # summary
    "GroundedSummary",
    "MetricStatistics",
    "QCSummary",
    "ReferenceComparison",
    "TrajectoryFacts",
    # report
    "ClaimCheck",
    "ClaimStatus",
    "GeneratedReport",
    "GroundingResult",
    "LLMUsage",
    "NarrativeReport",
    "NarrativeSection",
    "NumericClaim",
    "ReportAudit",
    "ReviewDecision",
    "ReviewRecord",
    "ReviewStatus",
]
