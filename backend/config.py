"""
Application configuration and settings.

All security-sensitive values are loaded from environment variables with safe
defaults.  GPU availability is probed lazily to avoid hard failures when
PyTorch is not installed.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

# ── Logging ─────────────────────────────────────────────────
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("md_ai_analyzer")

# ── Base directories ────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
REPORTS_DIR = BASE_DIR / "reports"
FRONTEND_DIR = BASE_DIR / "frontend"

for _dir in (UPLOAD_DIR, RESULTS_DIR, REPORTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ── Supported file extensions ───────────────────────────────
TRAJECTORY_EXTENSIONS: set[str] = {".xtc", ".trr"}
TOPOLOGY_EXTENSIONS: set[str] = {".tpr"}
STRUCTURE_EXTENSIONS: set[str] = {".pdb", ".gro"}
ALL_EXTENSIONS: set[str] = TRAJECTORY_EXTENSIONS | TOPOLOGY_EXTENSIONS | STRUCTURE_EXTENSIONS

# ── GPU configuration (lazy — tolerant of missing PyTorch) ──
try:
    import torch

    GPU_AVAILABLE: bool = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if GPU_AVAILABLE else "cpu")
    GPU_NAME: str = torch.cuda.get_device_name(0) if GPU_AVAILABLE else "N/A"
except ImportError:
    logger.warning("PyTorch not installed — GPU acceleration unavailable")
    GPU_AVAILABLE = False
    DEVICE = None  # type: ignore[assignment]
    GPU_NAME = "N/A (torch not installed)"

# ── Analysis defaults ───────────────────────────────────────
DEFAULT_STRIDE: int = 1
CHUNK_SIZE: int = 1000
MAX_UPLOAD_SIZE_MB: int = int(os.environ.get("MAX_UPLOAD_SIZE_MB", "5000"))
MAX_UPLOAD_SIZE_BYTES: int = MAX_UPLOAD_SIZE_MB * 1024 * 1024
CONTACT_DISTANCE_CUTOFF: float = 8.0
HBOND_DISTANCE_CUTOFF: float = 3.5
HBOND_ANGLE_CUTOFF: int = 150
SALT_BRIDGE_CUTOFF: float = 4.0

# ── Server ──────────────────────────────────────────────────
HOST: str = os.environ.get("HOST", "0.0.0.0")
PORT: int = int(os.environ.get("PORT", "8000"))

# ── CORS ────────────────────────────────────────────────────
_cors_env = os.environ.get("CORS_ORIGINS", "")
if _cors_env:
    CORS_ORIGINS: list[str] = [o.strip() for o in _cors_env.split(",") if o.strip()]
else:
    CORS_ORIGINS = [f"http://localhost:{PORT}", "http://127.0.0.1:8000"]

# ── Rate Limiting ───────────────────────────────────────────
RATE_LIMIT_REQUESTS: int = int(os.environ.get("RATE_LIMIT_REQUESTS", "60"))
RATE_LIMIT_WINDOW_SECONDS: int = int(os.environ.get("RATE_LIMIT_WINDOW", "60"))

# ── Job cleanup ─────────────────────────────────────────────
JOB_TTL_SECONDS: int = int(os.environ.get("JOB_TTL_SECONDS", "86400"))  # 24h


def get_system_info() -> dict:
    """Return a JSON-serialisable dict of system configuration."""
    return {
        "gpu_available": GPU_AVAILABLE,
        "gpu_name": GPU_NAME,
        "device": str(DEVICE) if DEVICE is not None else "cpu",
        "upload_dir": str(UPLOAD_DIR),
        "results_dir": str(RESULTS_DIR),
        "max_upload_mb": MAX_UPLOAD_SIZE_MB,
    }
