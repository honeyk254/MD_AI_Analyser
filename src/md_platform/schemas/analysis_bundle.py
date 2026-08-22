"""AnalysisBundle schema — the single data contract for the entire platform.

Every classical module writes its output to this shape. The LLM reporting layer
only ever reads from this schema, never raw trajectories.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .run_card import RunCard


class MetricSummary(BaseModel):
    """A scalar metric summarising a time-series or distribution."""

    mean: float
    std: float
    min: float
    max: float
    unit: str
    n_frames: int
    time_series: Optional[List[float]] = Field(
        None, description="Raw time series (e.g. per frame). Omitted by default."
    )


class PerResidueSeries(BaseModel):
    """A per-residue metric (e.g. RMSF, average SASA)."""

    values: List[float] = Field(description="Metric value per residue.")
    resids: List[int] = Field(description="Corresponding residue IDs.")
    unit: str


class ModuleResult(BaseModel):
    """The structured output of a single analysis module."""

    name: str = Field(description="Internal name of the module (e.g. 'rmsd').")
    version: str = Field(description="Version of the module logic.")
    runtime_seconds: float
    parameters: Dict[str, Any] = Field(
        description="Specific parameters used for this module."
    )

    # Payload varies by module, but we type the common ones strictly
    scalar_metrics: Dict[str, MetricSummary] = Field(
        default_factory=dict, description="Scalar metrics like total SASA, Rg."
    )
    residue_metrics: Dict[str, PerResidueSeries] = Field(
        default_factory=dict, description="Per-residue metrics like RMSF."
    )
    # Catch-all for module-specific complex data (e.g. PCA arrays, contact maps)
    # In a full implementation, we'd union these into specific schemas.
    data: Dict[str, Any] = Field(default_factory=dict)

    error: Optional[str] = Field(None, description="Error message if the module failed.")


class TrajectoryMetadata(BaseModel):
    """Physical properties and metadata of the analyzed trajectory window."""

    n_frames_analyzed: int
    n_atoms: int
    n_residues: int
    timestep_ps: float
    total_time_ns: float
    original_format: str
    force_field: str = Field(
        default="unknown",
        description="Parsed force field, or 'unknown — not recoverable'.",
    )
    box_dimensions: Optional[List[float]] = Field(
        None, description="Average box dimensions [x, y, z] in Angstroms, if periodic."
    )


class QCFlag(BaseModel):
    """A single quality control check."""

    check_name: str
    passed: bool
    details: str


class QCFlags(BaseModel):
    """Quality control results for the trajectory."""

    is_equilibrated: bool = Field(
        description="True if an equilibration point was found and passed."
    )
    sufficient_frames: bool = Field(
        description="True if enough frames exist for meaningful statistics."
    )
    flags: List[QCFlag] = Field(default_factory=list)


class AnalysisBundle(BaseModel):
    """The complete aggregated result of an analysis run."""

    run_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    trajectory_metadata: TrajectoryMetadata
    qc_flags: QCFlags

    modules: Dict[str, ModuleResult] = Field(
        description="Module outputs keyed by module name (e.g. 'rmsd')."
    )

    run_card: RunCard = Field(
        description="Provenance and reproducibility metadata."
    )
