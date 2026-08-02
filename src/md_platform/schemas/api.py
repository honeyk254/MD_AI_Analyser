"""API request and response schemas.

Simplified from the original backend/models.py to remove ML/DL specific flags
for Phase 1, focusing purely on classical analysis.
"""

from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class AnalysisStatus(str, Enum):
    """Lifecycle states for an analysis job."""

    PENDING = "pending"
    RUNNING = "running"
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


class AnalysisRequest(BaseModel):
    """Parameters controlling the analysis pipeline.

    Cleaned up to exclude GNN/Transformer/MSM for Phase 1.
    """

    job_id: str

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

    @model_validator(mode="after")
    def validate_frame_window(self) -> "AnalysisRequest":
        if (
            self.start_frame is not None
            and self.end_frame is not None
            and self.end_frame <= self.start_frame
        ):
            raise ValueError("end_frame must be greater than start_frame")
        return self


class ProgressUpdate(BaseModel):
    """Schema for SSE progress events."""

    job_id: str
    status: AnalysisStatus
    current_module: str
    progress_percent: float
    message: str
