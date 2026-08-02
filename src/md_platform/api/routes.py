"""API Routes."""

from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
import uuid

from ..schemas.api import AnalysisRequest, AnalysisResponse, StatusResponse
from ..schemas.analysis_bundle import AnalysisBundle
from ..orchestrator import AnalysisOrchestrator
from .dependencies import get_orchestrator

router = APIRouter(prefix="/api/v1/analysis", tags=["analysis"])


@router.post("/submit", response_model=AnalysisResponse)
async def submit_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    orchestrator: AnalysisOrchestrator = Depends(get_orchestrator),
):
    """Submit a new analysis run."""
    # If run_id is not provided, generate one
    if not request.run_id:
        request.run_id = str(uuid.uuid4())
        
    # Launch in background
    background_tasks.add_task(orchestrator.run_analysis, request)
    
    return AnalysisResponse(
        run_id=request.run_id,
        message="Analysis submitted successfully.",
        status_url=f"/api/v1/analysis/{request.run_id}/status",
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
