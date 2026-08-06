"""Global configuration.

All limits are settings rather than constants because the deployment target is a
single small host: the demo instance runs with far tighter caps than a local
development run.
"""

from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

MB = 1024 * 1024


class Settings(BaseSettings):
    """Application settings, overridable by environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = "MD AI Platform"
    debug: bool = False

    output_dir: Path = Path("data/outputs")
    data_dir: Path = Path("data/inputs")
    demo_dir: Path = Path("data/demo")

    # ----- input caps ----- #
    max_upload_bytes: int = Field(
        default=100 * MB, description="Per-file upload cap."
    )
    max_frames: int = Field(
        default=2000,
        description="Frames analysed per run; larger windows are strided down.",
    )
    max_atoms: int = Field(
        default=250_000, description="Reject systems larger than this."
    )

    # ----- rate limiting ----- #
    rate_limit_requests: int = Field(
        default=30, description="Requests allowed per client per window."
    )
    rate_limit_window_seconds: int = Field(default=60)
    rate_limit_expensive_requests: int = Field(
        default=5,
        description="Cap for run-creating endpoints (upload, submit, report).",
    )

    # ----- reporting ----- #
    llm_model: str = "claude-3-5-haiku-20241022"
    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="When unset, reports are written by the template narrator.",
    )
    llm_max_turns: int = 8

    # ----- HTTP ----- #
    cors_origins: List[str] = Field(
        default_factory=lambda: ["*"],
        description=(
            "Allowed origins. Credentials are never allowed, so a wildcard here "
            "cannot expose an authenticated session."
        ),
    )

    @property
    def llm_enabled(self) -> bool:
        return bool(self.anthropic_api_key)


settings = Settings()
