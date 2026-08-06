"""Grounded summary schema — the only thing the LLM layer is allowed to read.

Everything here is produced deterministically from an ``AnalysisBundle`` by
``aggregation.summarizer``. There are no per-frame arrays in this schema by
design: the reporting layer narrates pre-computed statistics and never performs
arithmetic of its own.
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

SUMMARY_SCHEMA_VERSION = "1.0.0"

TrendDirection = Literal["increasing", "decreasing", "stable"]
ReferenceVerdict = Literal["within_range", "above_range", "below_range", "no_reference"]


class MetricStatistics(BaseModel):
    """Deterministic statistics for one scalar metric of one module."""

    metric: str = Field(description="Metric key, e.g. 'backbone_rmsd'.")
    module: str = Field(description="Module that produced the metric.")
    unit: str
    mean: float
    std: float
    min: float
    max: float
    n_frames: int
    coefficient_of_variation: Optional[float] = Field(
        None, description="std / |mean|; None when mean is 0."
    )
    first_half_mean: Optional[float] = None
    second_half_mean: Optional[float] = None
    drift: Optional[float] = Field(
        None, description="second_half_mean - first_half_mean, in `unit`."
    )
    drift_percent: Optional[float] = Field(
        None, description="Drift as a percentage of the first-half mean."
    )
    trend: TrendDirection = "stable"
    changepoint_frames: List[int] = Field(
        default_factory=list,
        description="Frame indices where a binary-segmentation changepoint was found.",
    )


class ReferenceComparison(BaseModel):
    """A metric compared against a literature-derived expected range."""

    metric: str
    value: float
    unit: str
    verdict: ReferenceVerdict
    reference_low: Optional[float] = None
    reference_high: Optional[float] = None
    source: Optional[str] = Field(None, description="Literature citation for the range.")
    note: str = Field(
        description="What the comparison does and does not imply, in plain language."
    )


class QCSummary(BaseModel):
    """Rule-based quality-control view of the run."""

    is_equilibrated: bool
    sufficient_frames: bool
    equilibration_frame: Optional[int] = None
    passed_checks: List[str] = Field(default_factory=list)
    failed_checks: List[str] = Field(default_factory=list)
    details: Dict[str, str] = Field(default_factory=dict)


class TrajectoryFacts(BaseModel):
    """Flat, LLM-safe view of the trajectory metadata."""

    n_frames_analyzed: int
    n_atoms: int
    n_residues: int
    frame_interval_ps: float
    total_time_ns: float
    original_format: str
    force_field: str


class GroundedSummary(BaseModel):
    """Aggregated, deterministic input to the reporting layer."""

    schema_version: str = SUMMARY_SCHEMA_VERSION
    run_id: str
    bundle_hash: str = Field(description="SHA256 of the canonical AnalysisBundle JSON.")
    trajectory: TrajectoryFacts
    qc: QCSummary
    metrics: Dict[str, MetricStatistics] = Field(
        description="Statistics keyed by metric name."
    )
    comparisons: List[ReferenceComparison] = Field(default_factory=list)
    module_errors: Dict[str, str] = Field(
        default_factory=dict,
        description="Modules that failed, and why. Reported, never hidden.",
    )
    observations: List[str] = Field(
        default_factory=list,
        description="Deterministically generated factual observations.",
    )
