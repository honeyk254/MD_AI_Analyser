"""Aggregation layer: module results -> AnalysisBundle -> GroundedSummary."""

from .bundle_builder import build_bundle, tool_versions
from .provenance import canonical_hash, file_provenance
from .summarizer import bundle_hash, summarize_bundle

__all__ = [
    "build_bundle",
    "bundle_hash",
    "canonical_hash",
    "file_provenance",
    "summarize_bundle",
    "tool_versions",
]
