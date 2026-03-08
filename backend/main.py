"""
FastAPI main application — MD AI Analyzer.
"""
import asyncio
import json
import math
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from .config import (
    UPLOAD_DIR, RESULTS_DIR, REPORTS_DIR, FRONTEND_DIR,
    TRAJECTORY_EXTENSIONS, TOPOLOGY_EXTENSIONS, STRUCTURE_EXTENSIONS,
    get_system_info
)
from .models import AnalysisStatus, UploadResponse, AnalysisRequest
from .orchestrator import orchestrator


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

app = FastAPI(
    title="MD AI Analyzer",
    description="AI-powered molecular dynamics trajectory analysis platform",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
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
    return {"status": "ok", "system": get_system_info()}


@app.post("/api/upload")
async def upload_files(
    trajectory: Optional[UploadFile] = File(None),
    topology: Optional[UploadFile] = File(None),
    structure: Optional[UploadFile] = File(None),
    reference: Optional[UploadFile] = File(None),
):
    """Upload MD simulation files."""
    job_id = str(uuid.uuid4())[:8]
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    for label, upload in [("trajectory", trajectory), ("topology", topology),
                           ("structure", structure), ("reference", reference)]:
        if upload and upload.filename:
            dest = job_dir / upload.filename
            with open(dest, "wb") as f:
                content = await upload.read()
                f.write(content)
            saved_files[label] = str(dest)

    if not saved_files:
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

    return UploadResponse(
        job_id=actual_job_id,
        message="Files uploaded successfully",
        files=saved_files,
    )


@app.post("/api/analyze")
async def start_analysis(request: AnalysisRequest):
    """Start the analysis pipeline."""
    job = orchestrator.get_job(request.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] == AnalysisStatus.RUNNING:
        raise HTTPException(status_code=409, detail="Analysis already running")

    # Run analysis in background
    asyncio.create_task(orchestrator.run_analysis(
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

    return {"job_id": request.job_id, "status": "started"}


@app.get("/api/progress/{job_id}")
async def stream_progress(job_id: str):
    """Server-Sent Events endpoint for progress updates."""
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
    job = orchestrator.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    structure_file = job["files"].get("structure")
    if structure_file and Path(structure_file).exists():
        return FileResponse(structure_file)

    raise HTTPException(status_code=404, detail="Structure file not found")


@app.get("/api/pdf/{job_id}")
async def get_pdf_report(job_id: str):
    """Export analysis report as PDF (item 60)."""
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
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")

    raise HTTPException(status_code=500, detail="PDF generation failed")
