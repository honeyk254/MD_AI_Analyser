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
from .run_card import FileProvenance, RunCard, ToolVersions
from .api import (
    AnalysisRequest,
    AnalysisStatus,
    ErrorResponse,
    ProgressUpdate,
    UploadResponse,
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
    "AnalysisRequest",
    "AnalysisStatus",
    "ErrorResponse",
    "ProgressUpdate",
    "UploadResponse",
]
