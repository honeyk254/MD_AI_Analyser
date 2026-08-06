"""Analysis and reporting routes."""

import logging
import uuid

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from ..config import Settings
from ..orchestrator import AnalysisOrchestrator
from ..reporting.report_service import ReportService, ReviewGateError
from ..schemas.analysis_bundle import AnalysisBundle
from ..schemas.api import (
    AnalysisRequest,
    AnalysisResponse,
    RunStatus,
    StatusResponse,
    SubmitRequest,
    UploadResponse,
)
from ..schemas.report import GeneratedReport, ReviewDecision
from ..store import RunStore
from .dependencies import get_orchestrator, get_report_service, get_settings, get_store
from .uploads import save_upload

logger = logging.getLogger("md_ai_analyzer")

router = APIRouter(prefix="/api/v1/analysis", tags=["analysis"])

TOPOLOGY_ROLE = "topology"
TRAJECTORY_ROLE = "trajectory"


@router.post("/upload", response_model=UploadResponse, status_code=201)
async def upload_inputs(
    topology: UploadFile = File(description="Topology file (.pdb, .gro, .psf, ...)."),
    trajectory: UploadFile = File(description="Trajectory file (.xtc, .dcd, .pdb, ...)."),
    settings: Settings = Depends(get_settings),
    store: RunStore = Depends(get_store),
) -> UploadResponse:
    """Store input files for a new run and return its id."""
    run_id = uuid.uuid4().hex
    destination = store.input_dir(run_id)

    topology_path = await save_upload(
        topology, destination, TOPOLOGY_ROLE, settings.max_upload_bytes
    )
    trajectory_path = await save_upload(
        trajectory, destination, TRAJECTORY_ROLE, settings.max_upload_bytes
    )

    store.set_status(
        run_id, RunStatus.PENDING, "Inputs uploaded; submit the run to analyse them."
    )
    return UploadResponse(
        run_id=run_id,
        message="Inputs stored.",
        files={
            TOPOLOGY_ROLE: topology_path.name,
            TRAJECTORY_ROLE: trajectory_path.name,
        },
        submit_url="/api/v1/analysis/submit",
    )


@router.post("/submit", response_model=AnalysisResponse)
async def submit_analysis(
    request: SubmitRequest,
    background_tasks: BackgroundTasks,
    orchestrator: AnalysisOrchestrator = Depends(get_orchestrator),
    store: RunStore = Depends(get_store),
) -> AnalysisResponse:
    """Analyse the inputs previously uploaded for ``run_id``."""
    analysis_request = _resolve_inputs(request, store)
    background_tasks.add_task(orchestrator.run_analysis, analysis_request)
    return AnalysisResponse(
        run_id=request.run_id,
        message="Analysis submitted.",
        status_url=f"/api/v1/analysis/{request.run_id}/status",
    )


@router.get("/{run_id}/status", response_model=StatusResponse)
async def get_status(
    run_id: str,
    orchestrator: AnalysisOrchestrator = Depends(get_orchestrator),
) -> StatusResponse:
    """Current status of a run."""
    state = orchestrator.get_status(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Unknown run '{run_id}'.")
    return state


@router.get("/{run_id}/results", response_model=AnalysisBundle)
async def get_results(
    run_id: str,
    orchestrator: AnalysisOrchestrator = Depends(get_orchestrator),
) -> AnalysisBundle:
    """The AnalysisBundle of a completed run."""
    bundle = orchestrator.get_bundle(run_id)
    if bundle is None:
        state = orchestrator.get_status(run_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Unknown run '{run_id}'.")
        raise HTTPException(
            status_code=409,
            detail=f"Results are not available yet; run status is '{state.status.value}'.",
        )
    return bundle


@router.post("/{run_id}/report", response_model=GeneratedReport, status_code=201)
async def generate_report(
    run_id: str,
    orchestrator: AnalysisOrchestrator = Depends(get_orchestrator),
    reports: ReportService = Depends(get_report_service),
) -> GeneratedReport:
    """Generate the grounded narrative report for a completed run."""
    bundle = orchestrator.get_bundle(run_id)
    if bundle is None:
        raise HTTPException(
            status_code=409,
            detail=f"No AnalysisBundle for run '{run_id}'; the analysis must finish first.",
        )
    # Report generation is synchronous CPU work plus a network call, so it runs
    # in a worker thread rather than on the event loop.
    return await run_in_threadpool(reports.generate, bundle)


@router.get("/{run_id}/report", response_model=GeneratedReport)
async def get_report(run_id: str, store: RunStore = Depends(get_store)) -> GeneratedReport:
    """The persisted report, including grounding and review state."""
    report = store.read_report(run_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"No report for run '{run_id}'.")
    return report


@router.get("/{run_id}/report.html", response_class=FileResponse)
async def get_report_html(run_id: str, store: RunStore = Depends(get_store)) -> FileResponse:
    """The rendered HTML report with plots, narrative and provenance."""
    path = store.run_dir(run_id) / "analysis_report.html"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"No HTML report for run '{run_id}'.")
    return FileResponse(path, media_type="text/html")


@router.post("/{run_id}/review", response_model=GeneratedReport)
async def review_report(
    run_id: str,
    decision: ReviewDecision,
    reports: ReportService = Depends(get_report_service),
) -> GeneratedReport:
    """Record a human reviewer's sign-off or rejection."""
    try:
        return await run_in_threadpool(reports.review, run_id, decision)
    except ReviewGateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _resolve_inputs(request: SubmitRequest, store: RunStore) -> AnalysisRequest:
    """Map a run id to the files the server stored for it."""
    input_dir = store.input_dir(request.run_id)
    files = {path.stem: path for path in input_dir.iterdir() if path.is_file()}

    missing = [role for role in (TOPOLOGY_ROLE, TRAJECTORY_ROLE) if role not in files]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Run '{request.run_id}' has no uploaded {', '.join(missing)} file. "
                "Upload inputs before submitting."
            ),
        )

    return AnalysisRequest(
        run_id=request.run_id,
        topology_file=str(files[TOPOLOGY_ROLE]),
        trajectory_file=str(files[TRAJECTORY_ROLE]),
        **request.model_dump(exclude={"run_id"}),
    )
