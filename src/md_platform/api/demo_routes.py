"""Zero-setup demo routes.

``GET /demo`` runs the whole pipeline on a preloaded ensemble and returns the
rendered report, so the platform can be evaluated from a bare URL with no
upload, no key and no client. The run id is derived from the dataset and its
parameters, so revisiting the URL serves the cached report instead of recomputing
it — which also keeps a public demo from being turned into a CPU faucet.
"""

import hashlib
import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from starlette.concurrency import run_in_threadpool

from ..config import Settings
from ..demo import DEFAULT_DEMO_KEY, DEMO_CAVEAT, DemoDataset, available_datasets, get_dataset
from ..orchestrator import AnalysisOrchestrator
from ..reporting.report_service import ReportService
from ..schemas.api import (
    AnalysisRequest,
    DemoDatasetInfo,
    DemoDatasetList,
    DemoRunResponse,
)
from ..store import RunStore
from .dependencies import get_orchestrator, get_report_service, get_settings, get_store

logger = logging.getLogger("md_ai_analyzer")

router = APIRouter(tags=["demo"])

REPORT_FILENAME = "analysis_report.html"


def _demo_run_id(dataset: DemoDataset, stride: int) -> str:
    """Deterministic run id, so one dataset+parameters pair maps to one run."""
    digest = hashlib.sha256(f"{dataset.key}:{stride}".encode()).hexdigest()[:8]
    return f"demo-{dataset.key}-{digest}"


@router.get("/api/v1/demo/datasets", response_model=DemoDatasetList)
async def list_demo_datasets() -> DemoDatasetList:
    """Preloaded datasets available on this deployment."""
    return DemoDatasetList(
        datasets=[
            DemoDatasetInfo(
                key=dataset.key,
                name=dataset.name,
                description=dataset.description,
                n_frames=dataset.n_frames,
            )
            for dataset in available_datasets()
        ]
    )


@router.post("/api/v1/demo/run", response_model=DemoRunResponse, status_code=201)
async def run_demo(
    dataset_key: str = Query(default=DEFAULT_DEMO_KEY, alias="dataset"),
    stride: int = Query(default=1, ge=1, le=100),
    force: bool = Query(
        default=False, description="Recompute even if a cached demo report exists."
    ),
    orchestrator: AnalysisOrchestrator = Depends(get_orchestrator),
    reports: ReportService = Depends(get_report_service),
    store: RunStore = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> DemoRunResponse:
    """Analyse a preloaded dataset and generate its grounded report."""
    started = time.time()
    dataset = _load_dataset(dataset_key)
    run_id = _demo_run_id(dataset, stride)

    cached = store.read_report(run_id)
    if cached is not None and not force:
        return _response(dataset, cached, time.time() - started)

    report = await run_in_threadpool(
        _run_pipeline, orchestrator, reports, dataset, run_id, stride
    )
    logger.info("Demo run %s completed in %.1fs", run_id, time.time() - started)
    return _response(dataset, report, time.time() - started)


@router.get("/demo", response_class=FileResponse)
async def demo_report(
    dataset_key: str = Query(default=DEFAULT_DEMO_KEY, alias="dataset"),
    stride: int = Query(default=1, ge=1, le=100),
    orchestrator: AnalysisOrchestrator = Depends(get_orchestrator),
    reports: ReportService = Depends(get_report_service),
    store: RunStore = Depends(get_store),
) -> FileResponse:
    """The demo report as HTML, computing it on first visit."""
    dataset = _load_dataset(dataset_key)
    run_id = _demo_run_id(dataset, stride)
    report_path = store.run_dir(run_id) / REPORT_FILENAME

    if store.read_report(run_id) is None or not report_path.is_file():
        await run_in_threadpool(
            _run_pipeline, orchestrator, reports, dataset, run_id, stride
        )

    if not report_path.is_file():
        raise HTTPException(
            status_code=500, detail="The demo report could not be generated."
        )
    return FileResponse(report_path, media_type="text/html")


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def landing() -> HTMLResponse:
    """Minimal landing page pointing at the demo and the API docs."""
    datasets = "".join(
        f"<li><strong>{dataset.name}</strong> &mdash; {dataset.n_frames} frames</li>"
        for dataset in available_datasets()
    )
    return HTMLResponse(
        f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>MD AI Platform</title>
<style>
body {{ font-family: -apple-system, sans-serif; background:#0f0f23; color:#e0e0e0;
       max-width:760px; margin:60px auto; padding:0 20px; line-height:1.6; }}
a {{ color:#00d4ff; }} code {{ background:#1a1a3e; padding:2px 6px; border-radius:4px; }}
.cta {{ display:inline-block; margin:20px 0; padding:12px 22px; background:#00d4ff;
        color:#0f0f23; font-weight:700; border-radius:8px; text-decoration:none; }}
</style></head><body>
<h1>MD AI Platform</h1>
<p>Deterministic molecular-dynamics analysis with a grounded narrative layer:
classical analyses produce every number, and each number in the narrative is
verified against the analysis bundle before the report can be approved.</p>
<a class="cta" href="/demo">Open the demo report</a>
<p>Preloaded data:</p><ul>{datasets or "<li>none installed</li>"}</ul>
<p>{DEMO_CAVEAT}</p>
<p>API: <a href="/docs">/docs</a> &middot; health: <code>/health</code></p>
</body></html>"""
    )


def _load_dataset(dataset_key: str) -> DemoDataset:
    try:
        return get_dataset(dataset_key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _run_pipeline(
    orchestrator: AnalysisOrchestrator,
    reports: ReportService,
    dataset: DemoDataset,
    run_id: str,
    stride: int,
):
    """Analyse the dataset, then report on it, synchronously."""
    # A multi-model PDB is both topology and frame source.
    request = AnalysisRequest(
        run_id=run_id,
        topology_file=str(dataset.path),
        trajectory_file=str(dataset.path),
        stride=stride,
    )
    bundle = orchestrator.run_analysis_sync(request)
    if bundle is None:
        state = orchestrator.get_status(run_id)
        raise HTTPException(
            status_code=500,
            detail=f"Demo analysis failed: {state.message if state else 'unknown error'}",
        )
    return reports.generate(bundle)


def _response(dataset: DemoDataset, report, seconds: float) -> DemoRunResponse:
    return DemoRunResponse(
        run_id=report.run_id,
        dataset=dataset.name,
        caveat=DEMO_CAVEAT,
        report_url=f"/api/v1/analysis/{report.run_id}/report",
        report_html_url=f"/api/v1/analysis/{report.run_id}/report.html",
        review_status=report.review.status.value,
        grounding_passed=report.grounding.passed,
        n_verified_claims=report.grounding.n_verified,
        generator=report.narrative.generator,
        seconds=round(seconds, 2),
    )
