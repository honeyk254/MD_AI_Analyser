"""Schemas for the Phase 4 statistical/ML layer."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field


class FeatureSummary(BaseModel):
    """Compact description of the kinetic feature matrix."""

    selection: str
    n_frames: int
    n_features: int
    feature_names: List[str] = Field(default_factory=list)
    time_ps: List[float] = Field(default_factory=list)


class KineticEmbedding(BaseModel):
    """Projection onto a low-dimensional kinetic basis."""

    method: str
    n_components: int
    explained_variance: List[float] = Field(default_factory=list)
    component_labels: List[str] = Field(default_factory=list)
    projections: List[List[float]] = Field(default_factory=list)


class MSMSummary(BaseModel):
    """Summary of a Markov state model."""

    method: str = "msm"
    lag_frames: int
    lag_ps: float
    n_states: int
    transition_counts: List[List[int]] = Field(default_factory=list)
    transition_matrix: List[List[float]] = Field(default_factory=list)
    stationary_distribution: List[float] = Field(default_factory=list)
    implied_timescales_ps: List[float] = Field(default_factory=list)
    # [p5, p95] per implied timescale from the seeded moving-block bootstrap;
    # None when the label sequence is too short to resample.
    implied_timescales_ci_ps: Optional[List[List[float]]] = None
    ck_steps: int
    ck_deviation: float
    is_markovian: bool
    minimum_pair_transition_count: int
    state_populations: List[float] = Field(default_factory=list)


class BaselineComparison(BaseModel):
    """Comparison between PCA-based clustering and the TICA/MSM baseline."""

    method: str = "pca_vs_tica"
    state_agreement_nmi: float
    timescale_relative_error: Optional[float] = None
    pca_state_labels: List[int] = Field(default_factory=list)
    tica_state_labels: List[int] = Field(default_factory=list)
    summary: str


class MLGatingSummary(BaseModel):
    """Gate that decides whether the Phase 4 model may run."""

    enabled: bool
    passed: bool
    minimum_frames_required: int
    observed_frames: int
    minimum_transition_count_required: int
    observed_min_transition_count: int = 0
    lag_frames: int
    lag_ps: float
    n_states: int
    ck_cutoff: float
    reasons: List[str] = Field(default_factory=list)


class AnalysisCard(BaseModel):
    """Short model-card style documentation for the kinetic module."""

    title: str
    purpose: str
    literature_basis: List[str] = Field(default_factory=list)
    data_requirements: List[str] = Field(default_factory=list)
    failure_modes: List[str] = Field(default_factory=list)
    baseline_protocol: str
    version: str = "phase4-v1"


class VAMPNetAblation(BaseModel):
    """Phase 6 stretch: VAMPnet ablated against the TICA/MSM baseline.

    The plan metric is the implied-timescale agreement between the learned
    nonlinear embedding and the linear TICA baseline, reported as specific
    numbers rather than a qualitative claim.
    """

    available: bool
    vamp2_score: Optional[float] = None
    leading_timescale_ps: Optional[float] = None
    tica_leading_timescale_ps: Optional[float] = None
    timescale_relative_error: Optional[float] = None
    state_agreement_nmi: Optional[float] = None
    n_states: int
    epochs: int = 0
    summary: str


class MLAnalysisBundle(BaseModel):
    """Phase 4 outputs kept separate from the classical AnalysisBundle."""

    run_id: str
    source_run_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: str
    gating: MLGatingSummary
    feature_summary: FeatureSummary
    pca: Optional[KineticEmbedding] = None
    tica: Optional[KineticEmbedding] = None
    msm: Optional[MSMSummary] = None
    baseline_comparison: Optional[BaselineComparison] = None
    vampnet_ablation: Optional[VAMPNetAblation] = None
    analysis_card: AnalysisCard
    refusal_reason: Optional[str] = None
    notes: List[str] = Field(default_factory=list)
