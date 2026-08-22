"""API request and response schemas.

Simplified from the original backend/models.py to remove ML/DL specific flags
for Phase 1, focusing purely on classical analysis.
"""

from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel, Field, model_validator


class RunStatus(str, Enum):
    """Lifecycle states for an analysis job."""

    PENDING = "pending"
    RUNNING = "running"
    HUMAN_REVIEW = "human_review"
    COMPLETED = "completed"
    FAILED = "failed"



class UploadResponse(BaseModel):
    """Response returned after a successful file upload."""

    job_id: str
    message: str
    files: Dict[str, str]


class ErrorResponse(BaseModel):
    """Standard error envelope."""

    detail: str
    error_code: Optional[str] = None


class StatusResponse(BaseModel):
    """Response returned for a status check."""

    run_id: str
    status: RunStatus
    message: str
    results_url: Optional[str] = None
    reviewer_signoff: Optional[str] = None


class AnalysisResponse(BaseModel):
    """Response returned after submitting an analysis."""

    run_id: str
    message: str
    status_url: str


class AnalysisRequest(BaseModel):
    """Parameters controlling the analysis pipeline.

    Classical analysis remains the default path; Phase 4 ML is opt-in.
    """

    job_id: str
    run_id: Optional[str] = None
    topology_file: str
    trajectory_file: str

    # Trajectory windowing
    stride: int = Field(default=1, ge=1)
    start_frame: Optional[int] = Field(default=None, ge=0)
    end_frame: Optional[int] = Field(default=None, ge=0)
    discard_equilibration: bool = False

    # Classical thresholds
    hbond_cutoff: float = Field(default=3.5, ge=2.0, le=5.0)
    contact_cutoff: float = Field(default=8.0, ge=3.0, le=15.0)
    salt_bridge_cutoff: float = Field(default=4.0, ge=2.0, le=8.0)
    temperature: float = Field(default=300.0, ge=200.0, le=500.0)
    fel_bins: int = Field(default=50, ge=10, le=200)

    # Phase 4 opt-in
    enable_ml: bool = False
    ml_lag_frames: int = Field(default=5, ge=1)
    ml_n_states: int = Field(default=3, ge=2, le=10)
    ml_min_frames: int = Field(default=100, ge=2)
    ml_min_transition_count: int = Field(default=10, ge=1)
    ml_ck_threshold: float = Field(default=0.15, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_frame_window(self) -> "AnalysisRequest":
        if (
            self.start_frame is not None
            and self.end_frame is not None
            and self.end_frame <= self.start_frame
        ):
            raise ValueError("end_frame must be greater than start_frame")
        return self


class ReviewRequest(BaseModel):
    """Human review sign-off for a generated report."""

    reviewer_signoff: str = Field(min_length=1)


class ProgressUpdate(BaseModel):
    """Schema for SSE progress events."""

    job_id: str
    status: RunStatus
    current_module: str
    progress_percent: float
    message: str
