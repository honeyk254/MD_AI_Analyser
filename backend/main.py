"""
FastAPI main application — MD AI Analyzer.
Production-ready with security middleware, rate limiting, and structured logging.
"""
import asyncio
import json
import logging
import math
import os
import re
import shutil
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from .config import (
    UPLOAD_DIR, RESULTS_DIR, REPORTS_DIR, FRONTEND_DIR,
    TRAJECTORY_EXTENSIONS, TOPOLOGY_EXTENSIONS, STRUCTURE_EXTENSIONS,
    ALL_EXTENSIONS, MAX_UPLOAD_SIZE_BYTES, MAX_UPLOAD_SIZE_MB,
    CORS_ORIGINS, RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS,
    get_system_info, logger,
)
from .models import AnalysisStatus, UploadResponse, AnalysisRequest, ErrorResponse
from .orchestrator import orchestrator

# ── Regex for valid job IDs (8 hex chars) ───────────────────
JOB_ID_PATTERN = re.compile(r"^[0-9a-f]{8}$")


def _validate_job_id(job_id: str) -> str:
    """Validate job_id is a safe 8-char hex string."""
    if not JOB_ID_PATTERN.match(job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID format")
    return job_id


def sanitize_for_json(obj):
    """Recursively replace NaN/inf/-inf with None and convert numpy types."""
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    return obj


def _sanitize_filename(filename: str) -> str:
    """Sanitize uploaded filename to prevent path traversal and injection."""
    # Strip directory components
    filename = Path(filename).name
    # Remove any non-alphanumeric chars except dots, hyphens, underscores
    filename = re.sub(r"[^\w.\-]", "_", filename)
    # Prevent hidden files
    filename = filename.lstrip(".")
    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:255 - len(ext)] + ext
    return filename or "unnamed_file"


def _validate_file_extension(filename: str, allowed_extensions: set) -> bool:
    """Check if file extension is in the allowed set."""
    ext = Path(filename).suffix.lower()
    return ext in allowed_extensions


# ── Rate Limiting Middleware ────────────────────────────────
class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiter per client IP."""

    def __init__(self, app, max_requests: int, window_seconds: int):
        super().__init__(app)
        self.max_requests = max_requests
        self.window = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        # Clean old entries
        self._requests[client_ip] = [
            t for t in self._requests[client_ip] if now - t < self.window
        ]

        if len(self._requests[client_ip]) >= self.max_requests:
            logger.warning("Rate limit exceeded for %s", client_ip)
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please retry later."},
            )

        self._requests[client_ip].append(now)
        return await call_next(request)


# ── Request ID Middleware ──────────────────────────────────
class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a unique request ID for tracing."""

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


# ── Security Headers Middleware ────────────────────────────
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to every response."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        return response


# ── App Creation ───────────────────────────────────────────
app = FastAPI(
    title="MD AI Analyzer",
    description="AI-powered molecular dynamics trajectory analysis platform",
    version="1.0.0",
)

# Middleware (order matters — outermost first)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    RateLimitMiddleware,
    max_requests=RATE_LIMIT_REQUESTS,
    window_seconds=RATE_LIMIT_WINDOW_SECONDS,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Mount frontend static files
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>MD AI Analyzer</h1><p>Frontend not found.</p>")


@app.get("/api/health")
async def health():
    """Enhanced health check with dependency status."""
    deps = {}
    for mod_name in ("MDAnalysis", "torch", "sklearn", "plotly"):
        try:
            __import__(mod_name)
            deps[mod_name] = "available"
        except ImportError:
            deps[mod_name] = "missing"
    return {"status": "ok", "system": get_system_info(), "dependencies": deps}


@app.post("/api/upload")
async def upload_files(
    request: Request,
    trajectory: Optional[UploadFile] = File(None),
    topology: Optional[UploadFile] = File(None),
    structure: Optional[UploadFile] = File(None),
    reference: Optional[UploadFile] = File(None),
):
    """Upload MD simulation files with size, extension, and filename validation."""
    job_id = str(uuid.uuid4())[:8]
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    for label, upload in [("trajectory", trajectory), ("topology", topology),
                           ("structure", structure), ("reference", reference)]:
        if upload and upload.filename:
            # Validate extension
            if not _validate_file_extension(upload.filename, ALL_EXTENSIONS):
                shutil.rmtree(job_dir, ignore_errors=True)
                raise HTTPException(
                    status_code=400,
                    detail=f"File type not allowed for '{label}': {upload.filename}",
                )

            # Sanitize filename
            safe_name = _sanitize_filename(upload.filename)

            # Read content with size limit
            content = await upload.read()
            if len(content) > MAX_UPLOAD_SIZE_BYTES:
                shutil.rmtree(job_dir, ignore_errors=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File '{label}' exceeds {MAX_UPLOAD_SIZE_MB}MB limit",
                )

            dest = job_dir / safe_name
            with open(dest, "wb") as f:
                f.write(content)
            saved_files[label] = str(dest)
            logger.info("Saved %s file: %s (%d bytes)", label, safe_name, len(content))

    if not saved_files:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="No files uploaded")

    # Register the job
    actual_job_id = orchestrator.create_job(saved_files)
    # Rename dir to match orchestrator job id
    new_dir = UPLOAD_DIR / actual_job_id
    if job_dir != new_dir:
        if new_dir.exists():
            shutil.rmtree(new_dir)
        job_dir.rename(new_dir)
        # Update file paths
        for k in saved_files:
            saved_files[k] = saved_files[k].replace(str(job_dir), str(new_dir))
        orchestrator.jobs[actual_job_id]["files"] = saved_files

    logger.info("Upload complete: job_id=%s, files=%s", actual_job_id, list(saved_files.keys()))
    return UploadResponse(
        job_id=actual_job_id,
        message="Files uploaded successfully",
        files=saved_files,
    )


@app.post("/api/analyze")
async def start_analysis(request: AnalysisRequest):
    """Start the analysis pipeline."""
    _validate_job_id(request.job_id)
    job = orchestrator.get_job(request.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] == AnalysisStatus.RUNNING:
        raise HTTPException(status_code=409, detail="Analysis already running")

    # Run analysis in background — store task reference to prevent GC
    task = asyncio.create_task(orchestrator.run_analysis(
        request.job_id,
        stride=request.stride,
        run_gnn=request.run_gnn,
        run_transformer=request.run_transformer,
        run_msm=request.run_msm,
        ligand_selection=request.ligand_selection,
        start_frame=request.start_frame,
        end_frame=request.end_frame,
        hbond_cutoff=request.hbond_cutoff,
        contact_cutoff=request.contact_cutoff,
        salt_bridge_cutoff=request.salt_bridge_cutoff,
        fel_bins=request.fel_bins,
        temperature=request.temperature,
        msm_lag_time=request.msm_lag_time,
        grid_spacing=request.grid_spacing,
        correlation_threshold=request.correlation_threshold,
        vae_latent_dim=request.vae_latent_dim,
    ))
    orchestrator.store_task(request.job_id, task)
    logger.info("Analysis started: job_id=%s", request.job_id)

    return {"job_id": request.job_id, "status": "started"}


@app.get("/api/progress/{job_id}")
async def stream_progress(job_id: str):
    """Server-Sent Events endpoint for progress updates."""
    _validate_job_id(job_id)
    job = orchestrator.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        queue = asyncio.Queue()

        async def callback(data):
            await queue.put(data)

        orchestrator.register_progress_callback(job_id, callback)

        try:
            # Send initial status
            yield f"data: {json.dumps({'job_id': job_id, 'status': job['status'].value, 'progress_percent': job['progress'], 'current_module': job['current_module'], 'message': job['message']})}\n\n"

            while True:
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {json.dumps(data)}\n\n"
                    if data.get("status") in ("completed", "failed"):
                        break
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'heartbeat': True})}\n\n"
        finally:
            orchestrator.unregister_progress_callback(job_id, callback)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/results/{job_id}")
async def get_results(job_id: str):
    """Get analysis results."""
    _validate_job_id(job_id)
    job = orchestrator.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] == AnalysisStatus.RUNNING:
        return {"status": "running", "progress": job["progress"], "message": job["message"]}

    if job["result"]:
        return JSONResponse(content=sanitize_for_json(job["result"].model_dump()))

    return {"status": job["status"].value}


@app.get("/api/report/{job_id}")
async def get_report(job_id: str):
    """Download HTML report."""
    _validate_job_id(job_id)
    job = orchestrator.get_job(job_id)
    if not job or not job.get("result"):
        raise HTTPException(status_code=404, detail="Report not found")

    report_path = job["result"].plots.get("report_html")
    if report_path and Path(report_path).exists():
        return FileResponse(report_path, media_type="text/html", filename=f"md_report_{job_id}.html")

    raise HTTPException(status_code=404, detail="Report file not found")


@app.get("/api/csv/{job_id}")
async def get_csv(job_id: str):
    """Download CSV metrics."""
    _validate_job_id(job_id)
    job = orchestrator.get_job(job_id)
    if not job or not job.get("result"):
        raise HTTPException(status_code=404, detail="CSV not found")

    csv_path = job["result"].plots.get("csv_metrics")
    if csv_path and Path(csv_path).exists():
        return FileResponse(csv_path, media_type="text/csv", filename=f"md_metrics_{job_id}.csv")

    raise HTTPException(status_code=404, detail="CSV file not found")


@app.get("/api/structure/{job_id}")
async def get_structure(job_id: str):
    """Get PDB/GRO structure file for 3D viewer."""
    _validate_job_id(job_id)
    job = orchestrator.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    structure_file = job["files"].get("structure")
    if structure_file and Path(structure_file).exists():
        return FileResponse(structure_file)

    raise HTTPException(status_code=404, detail="Structure file not found")


@app.get("/api/pdf/{job_id}")
async def get_pdf_report(job_id: str):
    """Export analysis report as PDF."""
    _validate_job_id(job_id)
    job = orchestrator.get_job(job_id)
    if not job or not job.get("result"):
        raise HTTPException(status_code=404, detail="Results not found")

    from .visualization.report_generator import export_pdf
    job_dir = Path(job["job_dir"])
    try:
        pdf_path = export_pdf(job["result"], job_dir)
        if pdf_path and pdf_path.exists():
            return FileResponse(pdf_path, media_type="application/pdf",
                              filename=f"md_report_{job_id}.pdf")
    except Exception as e:
        logger.error("PDF generation failed for job %s: %s", job_id, e)
        raise HTTPException(status_code=500, detail="PDF generation failed")

    raise HTTPException(status_code=500, detail="PDF generation failed")
