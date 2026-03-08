"""
Pydantic models for API request/response schemas and internal data structures.

Changes from original:
- Added ``from __future__ import annotations`` for forward-reference support.
- Added ``model_config`` with ``arbitrary_types_allowed`` for MDAnalysis compat.
- Added ``msm_lag_time`` validator.
- Improved docstrings.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, field_validator


class AnalysisStatus(str, Enum):
    """Lifecycle states for an analysis job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ── API schemas ─────────────────────────────────────────────


class UploadResponse(BaseModel):
    """Response returned after a successful file upload."""

    job_id: str
    message: str
    files: Dict[str, str]


class ErrorResponse(BaseModel):
    """Standard error envelope."""

    detail: str
    error_code: Optional[str] = None


class AnalysisRequest(BaseModel):
    """Parameters controlling the analysis pipeline.

    Every field carries a sane default so that the frontend only needs to
    send ``job_id`` plus any overrides.
    """

    job_id: str
    stride: int = 1
    run_gnn: bool = True
    run_transformer: bool = True
    run_msm: bool = True
    ligand_selection: Optional[str] = None
    reference_pdb: Optional[str] = None
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    hbond_cutoff: float = 3.5
    contact_cutoff: float = 8.0
    salt_bridge_cutoff: float = 4.0
    fel_bins: int = 50
    temperature: float = 300.0
    msm_lag_time: int = 5
    grid_spacing: float = 2.0
    correlation_threshold: float = 0.5
    vae_latent_dim: int = 2

    # ── Validators ──────────────────────────────────────────

    @field_validator("stride")
    @classmethod
    def stride_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("stride must be >= 1")
        return v

    @field_validator("hbond_cutoff")
    @classmethod
    def hbond_cutoff_range(cls, v: float) -> float:
        if not 2.0 <= v <= 5.0:
            raise ValueError("hbond_cutoff must be between 2.0 and 5.0 A")
        return v

    @field_validator("contact_cutoff")
    @classmethod
    def contact_cutoff_range(cls, v: float) -> float:
        if not 3.0 <= v <= 15.0:
            raise ValueError("contact_cutoff must be between 3.0 and 15.0 A")
        return v

    @field_validator("salt_bridge_cutoff")
    @classmethod
    def salt_bridge_cutoff_range(cls, v: float) -> float:
        if not 2.0 <= v <= 8.0:
            raise ValueError("salt_bridge_cutoff must be between 2.0 and 8.0 A")
        return v

    @field_validator("temperature")
    @classmethod
    def temperature_range(cls, v: float) -> float:
        if not 200.0 <= v <= 500.0:
            raise ValueError("temperature must be between 200 and 500 K")
        return v

    @field_validator("fel_bins")
    @classmethod
    def fel_bins_range(cls, v: int) -> int:
        if not 10 <= v <= 200:
            raise ValueError("fel_bins must be between 10 and 200")
        return v

    @field_validator("vae_latent_dim")
    @classmethod
    def vae_latent_range(cls, v: int) -> int:
        if not 1 <= v <= 32:
            raise ValueError("vae_latent_dim must be between 1 and 32")
        return v

    @field_validator("correlation_threshold")
    @classmethod
    def corr_threshold_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("correlation_threshold must be between 0 and 1")
        return v

    @field_validator("msm_lag_time")
    @classmethod
    def msm_lag_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("msm_lag_time must be >= 1")
        return v


class ProgressUpdate(BaseModel):
    """Schema for SSE progress events."""

    job_id: str
    status: AnalysisStatus
    current_module: str
    progress_percent: float
    message: str


class InsightItem(BaseModel):
    """A single biological insight produced by the inference engine."""

    type: str
    residues: List[int]
    description: str
    confidence: float
    evidence: List[str]
    category: str  # structural | dynamic | allosteric | binding | transition


# ── Internal result container ───────────────────────────────


class AnalysisResult(BaseModel):
    """Aggregated output of the full analysis pipeline.

    Every analysis module stores its results as a ``Dict[str, Any]``
    keyed by the module attribute name.  Modules that were skipped or
    errored carry ``{"error": "..."}`` or remain as empty dicts.
    """

    model_config = {"arbitrary_types_allowed": True}

    job_id: str
    status: AnalysisStatus

    # Trajectory metadata
    trajectory_info: Dict[str, Any] = {}

    # Classical analysis modules
    rmsd: Dict[str, Any] = {}
    rmsf: Dict[str, Any] = {}
    rg: Dict[str, Any] = {}
    secondary_structure: Dict[str, Any] = {}
    hbonds: Dict[str, Any] = {}
    salt_bridges: Dict[str, Any] = {}
    contacts: Dict[str, Any] = {}
    pca: Dict[str, Any] = {}
    dccm: Dict[str, Any] = {}
    clustering: Dict[str, Any] = {}
    free_energy: Dict[str, Any] = {}
    sasa: Dict[str, Any] = {}
    tica: Dict[str, Any] = {}
    water_bridges: Dict[str, Any] = {}
    energy_decomposition: Dict[str, Any] = {}
    prs: Dict[str, Any] = {}
    nma: Dict[str, Any] = {}
    entropy: Dict[str, Any] = {}
    convergence: Dict[str, Any] = {}
    binding_kinetics: Dict[str, Any] = {}

    # ML modules
    ml_states: Dict[str, Any] = {}
    msm: Dict[str, Any] = {}
    allosteric: Dict[str, Any] = {}
    domains: Dict[str, Any] = {}
    ligand: Dict[str, Any] = {}
    dimensionality: Dict[str, Any] = {}
    interaction_fingerprints: Dict[str, Any] = {}
    tunnels: Dict[str, Any] = {}
    vae: Dict[str, Any] = {}
    dynamic_network: Dict[str, Any] = {}

    # Deep learning modules
    gnn_results: Dict[str, Any] = {}
    transformer_results: Dict[str, Any] = {}

    # Biological interpretation
    biological_insights: List[Dict[str, Any]] = []

    # Plotly JSON strings keyed by plot name
    plots: Dict[str, str] = {}
