"""API Routes."""

import os
import uuid
from html import escape
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import HTMLResponse

from ..demo_inputs import ensure_demo_inputs
from ..ml.schemas import MLAnalysisBundle
from ..observability import LLM_METRICS, recent_spans
from ..orchestrator import AnalysisOrchestrator
from ..schemas.analysis_bundle import AnalysisBundle
from ..schemas.api import AnalysisRequest, AnalysisResponse, ReviewRequest, StatusResponse
from .dependencies import get_orchestrator

router = APIRouter(prefix="/api/v1/analysis", tags=["analysis"])
demo_router = APIRouter(prefix="/api/v1/demo", tags=["demo"])
metrics_router = APIRouter(prefix="/api/v1/metrics", tags=["metrics"])

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


@metrics_router.get("/llm")
async def llm_metrics():
    """LLM cost/latency aggregates and recent calls (plan: cost < $0.50/report)."""
    summary = LLM_METRICS.summary()
    return {
        "summary": summary,
        "recent_calls": [
            {
                "run_id": call.run_id,
                "mode": call.mode,
                "latency_s": round(call.latency_s, 3),
                "cost_usd": round(call.cost_usd, 6),
                "tokens_in": call.tokens_in,
                "tokens_out": call.tokens_out,
                "ungrounded_claims": call.ungrounded_claims,
            }
            for call in LLM_METRICS.calls[-20:]
        ],
        "recent_spans": recent_spans(10),
    }


@metrics_router.get("/dashboard", response_class=HTMLResponse)
async def metrics_dashboard():
    """Static cost/latency dashboard over the metrics collected above."""
    summary = LLM_METRICS.summary()
    rows = "".join(
        f"<tr><td>{escape(call.run_id)}</td><td>{escape(call.mode)}</td>"
        f"<td>{call.latency_s:.2f}s</td><td>${call.cost_usd:.4f}</td>"
        f"<td>{call.ungrounded_claims}</td></tr>"
        for call in reversed(LLM_METRICS.calls[-20:])
    )
    n = summary.get("n_reports", 0)
    mean_cost = summary.get("mean_cost_usd", 0.0)
    mean_latency = summary.get("mean_latency_s", 0.0)
    return f"""<!doctype html>
<html><head><title>MD AI Platform — LLM Metrics</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #111; color: #eee; }}
table {{ border-collapse: collapse; margin-top: 1rem; }}
th, td {{ border: 1px solid #444; padding: 0.4rem 0.8rem; text-align: left; }}
.cards span {{ display: inline-block; background: #222; border: 1px solid #555;
border-radius: 6px; padding: 0.6rem 1.2rem; margin-right: 1rem; }}
</style></head><body>
<h1>LLM Cost / Latency Dashboard</h1>
<div class="cards">
<span><b>{n}</b> reports</span>
<span>mean cost <b>${mean_cost:.4f}</b> (target &lt; $0.50)</span>
<span>mean latency <b>{mean_latency:.2f}s</b></span>
</div>
<table>
<tr><th>Run</th><th>Mode</th><th>Latency</th><th>Cost</th><th>Ungrounded</th></tr>
{rows}
</table>
</body></html>"""
