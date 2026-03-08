"""
Application configuration and settings.
All security-sensitive values are loaded from environment variables with safe defaults.
"""
import logging
import os
import torch
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

for d in [UPLOAD_DIR, RESULTS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Supported file extensions ───────────────────────────────
TRAJECTORY_EXTENSIONS = {".xtc", ".trr"}
TOPOLOGY_EXTENSIONS = {".tpr"}
STRUCTURE_EXTENSIONS = {".pdb", ".gro"}
ALL_EXTENSIONS = TRAJECTORY_EXTENSIONS | TOPOLOGY_EXTENSIONS | STRUCTURE_EXTENSIONS

# ── GPU configuration ───────────────────────────────────────
GPU_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device("cuda" if GPU_AVAILABLE else "cpu")
GPU_NAME = torch.cuda.get_device_name(0) if GPU_AVAILABLE else "N/A"

# ── Analysis defaults ───────────────────────────────────────
DEFAULT_STRIDE = 1
CHUNK_SIZE = 1000
MAX_UPLOAD_SIZE_MB = int(os.environ.get("MAX_UPLOAD_SIZE_MB", "5000"))
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
CONTACT_DISTANCE_CUTOFF = 8.0
HBOND_DISTANCE_CUTOFF = 3.5
HBOND_ANGLE_CUTOFF = 150
SALT_BRIDGE_CUTOFF = 4.0

# ── Server ──────────────────────────────────────────────────
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))

# ── CORS ────────────────────────────────────────────────────
# Comma-separated origins, or "*" for development only
_cors_env = os.environ.get("CORS_ORIGINS", "")
if _cors_env:
    CORS_ORIGINS = [o.strip() for o in _cors_env.split(",") if o.strip()]
else:
    CORS_ORIGINS = [f"http://localhost:{PORT}", "http://127.0.0.1:8000"]

# ── Rate Limiting ───────────────────────────────────────────
RATE_LIMIT_REQUESTS = int(os.environ.get("RATE_LIMIT_REQUESTS", "60"))
RATE_LIMIT_WINDOW_SECONDS = int(os.environ.get("RATE_LIMIT_WINDOW", "60"))

# ── Job cleanup ─────────────────────────────────────────────
JOB_TTL_SECONDS = int(os.environ.get("JOB_TTL_SECONDS", "86400"))  # 24 hours


def get_system_info():
    """Return system info dict."""
    return {
        "gpu_available": GPU_AVAILABLE,
        "gpu_name": GPU_NAME,
        "device": str(DEVICE),
        "upload_dir": str(UPLOAD_DIR),
        "results_dir": str(RESULTS_DIR),
        "max_upload_mb": MAX_UPLOAD_SIZE_MB,
    }
