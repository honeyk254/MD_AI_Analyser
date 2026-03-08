"""
Application configuration and settings.
"""
import os
import torch
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
REPORTS_DIR = BASE_DIR / "reports"
FRONTEND_DIR = BASE_DIR / "frontend"

# Create directories
for d in [UPLOAD_DIR, RESULTS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Supported file extensions
TRAJECTORY_EXTENSIONS = {".xtc", ".trr"}
TOPOLOGY_EXTENSIONS = {".tpr"}
STRUCTURE_EXTENSIONS = {".pdb", ".gro"}
ALL_EXTENSIONS = TRAJECTORY_EXTENSIONS | TOPOLOGY_EXTENSIONS | STRUCTURE_EXTENSIONS

# GPU configuration
GPU_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device("cuda" if GPU_AVAILABLE else "cpu")
GPU_NAME = torch.cuda.get_device_name(0) if GPU_AVAILABLE else "N/A"

# Analysis defaults
DEFAULT_STRIDE = 1
CHUNK_SIZE = 1000  # frames per chunk for memory-efficient loading
MAX_UPLOAD_SIZE_MB = 5000  # 5 GB max upload
CONTACT_DISTANCE_CUTOFF = 8.0  # Angstroms
HBOND_DISTANCE_CUTOFF = 3.5  # Angstroms
HBOND_ANGLE_CUTOFF = 150  # degrees
SALT_BRIDGE_CUTOFF = 4.0  # Angstroms

# Server
HOST = "0.0.0.0"
PORT = 8000

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
