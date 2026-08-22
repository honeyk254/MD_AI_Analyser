"""API Routes."""

import os
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from ..demo_inputs import ensure_demo_inputs
from ..ml.schemas import MLAnalysisBundle
from ..orchestrator import AnalysisOrchestrator
from ..schemas.analysis_bundle import AnalysisBundle
from ..schemas.api import AnalysisRequest, AnalysisResponse, ReviewRequest, StatusResponse
from .dependencies import get_orchestrator

router = APIRouter(prefix="/api/v1/analysis", tags=["analysis"])
demo_router = APIRouter(prefix="/api/v1/demo", tags=["demo"])

DEMO_INPUTS = ensure_demo_inputs(Path(os.getenv("DATA_DIR", "data/inputs")))


@router.post("/submit", response_model=AnalysisResponse)
async def submit_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    orchestrator: AnalysisOrchestrator = Depends(get_orchestrator),
):
    """Submit a new analysis run."""
    if not request.run_id:
        request.run_id = request.job_id or str(uuid.uuid4())

    # Launch in background
    background_tasks.add_task(orchestrator.run_analysis, request)

    return AnalysisResponse(
        run_id=request.run_id,
        message="Analysis submitted successfully.",
        status_url=f"/api/v1/analysis/{request.run_id}/status",
    )


@demo_router.get("/examples")
async def list_demo_examples():
    """List bundled demo trajectories."""
    return [
        {"name": item["name"], "label": item["label"]}
        for item in DEMO_INPUTS.values()
    ]


@demo_router.post("/{example_name}/submit", response_model=AnalysisResponse)
async def submit_demo_analysis(
    example_name: str,
    background_tasks: BackgroundTasks,
    orchestrator: AnalysisOrchestrator = Depends(get_orchestrator),
):
    """Submit a bundled demo trajectory with zero setup."""
    example = DEMO_INPUTS.get(example_name)
    if not example:
        raise HTTPException(status_code=404, detail="Demo example not found.")

    run_id = str(uuid.uuid4())
    request = AnalysisRequest(
        job_id=f"demo-{example_name}",
        run_id=run_id,
        topology_file=example["topology_file"],
        trajectory_file=example["trajectory_file"],
    )
    background_tasks.add_task(orchestrator.run_analysis, request)

    return AnalysisResponse(
        run_id=run_id,
        message=f"Demo analysis '{example_name}' submitted successfully.",
        status_url=f"/api/v1/analysis/{run_id}/status",
    )


@router.get("/{run_id}/status", response_model=StatusResponse)
async def get_status(
    run_id: str,
    orchestrator: AnalysisOrchestrator = Depends(get_orchestrator),
):
    """Get the status of an analysis run."""
    return orchestrator.get_status(run_id)


@router.get("/{run_id}/results", response_model=AnalysisBundle)
async def get_results(
    run_id: str,
    orchestrator: AnalysisOrchestrator = Depends(get_orchestrator),
):
    """Get the final AnalysisBundle for a completed run."""
    status = orchestrator.get_status(run_id)
    if status.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Results not available. Current status: {status.status}",
        )

    bundle = orchestrator.bundles.get(run_id)
    if not bundle:
        raise HTTPException(status_code=404, detail="Bundle not found.")

    return bundle


@router.get("/{run_id}/ml-results", response_model=MLAnalysisBundle)
async def get_ml_results(
    run_id: str,
    orchestrator: AnalysisOrchestrator = Depends(get_orchestrator),
):
    """Get the Phase 4 ML bundle for a completed run."""
    status = orchestrator.get_status(run_id)
    if status.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Results not available. Current status: {status.status}",
        )

    ml_bundle = orchestrator.ml_bundles.get(run_id)
    if not ml_bundle:
        raise HTTPException(status_code=404, detail="ML bundle not found.")

    return ml_bundle


@router.post("/{run_id}/review", response_model=StatusResponse)
async def review_analysis(
    run_id: str,
    request: ReviewRequest,
    orchestrator: AnalysisOrchestrator = Depends(get_orchestrator),
):
    """Approve a run after human review."""
    try:
        return orchestrator.approve_run(run_id, request.reviewer_signoff)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
