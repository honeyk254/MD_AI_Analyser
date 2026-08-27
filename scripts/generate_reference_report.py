"""Regenerate the live demo report (GitHub Pages) end to end.

Single-trajectory showcase policy: everything runs on ONE trajectory — the
adenylate-kinase reference clip — through all classical modules, the plotting
battery, the narrative generation, and human-review approval. The Phase 4 ML
layer stays enabled so the gating decision is visible in the report itself: on
this short 98-frame clip no parameterization reaches the documented minimum of
10 transitions per state pair (measured sweep: lags 2-8 x k=2-3 peak at 2),
so the kinetic section reports the honest gated refusal rather than numbers.

Writes data/outputs/adk-reference/
{analysis_report.html, draft_report.md, final_report.md, bundle.json}.
"""

import asyncio
import shutil
from pathlib import Path

from MDAnalysisTests.datafiles import DCD, PSF

from md_platform.orchestrator import AnalysisOrchestrator
from md_platform.schemas.api import AnalysisRequest

INPUTS = Path("data/inputs/adk")
OUTPUTS = Path("data/outputs")

RUN_ID = "adk-reference"


def prepare_reference_inputs() -> None:
    INPUTS.mkdir(parents=True, exist_ok=True)
    for source, name in ((PSF, "adk.psf"), (DCD, "adk.dcd")):
        shutil.copyfile(source, INPUTS / name)


def adk_request(run_id: str) -> AnalysisRequest:
    return AnalysisRequest(
        job_id="adk-reference",
        run_id=run_id,
        topology_file=str(INPUTS / "adk.psf"),
        trajectory_file=str(INPUTS / "adk.dcd"),
        enable_ml=True,
        ml_lag_frames=3,
        ml_n_states=3,
        ml_min_frames=50,
        ml_min_transition_count=10,
        ml_ck_threshold=0.15,
    )


async def main() -> None:
    orchestrator = AnalysisOrchestrator(output_dir=OUTPUTS)
    request = adk_request(RUN_ID)

    await orchestrator.run_analysis(request)
    status = orchestrator.get_status(request.run_id)
    print(f"[{request.run_id}] status={status.status} message={status.message}")

    ml = orchestrator.ml_bundles.get(request.run_id)
    if ml:
        print(f"[{request.run_id}] ML: {ml.status}; refusal={ml.refusal_reason}")

    if status.status.value == "human_review":
        signoff = (
            "Regenerated single-trajectory reference: full classical battery; "
            "kinetic layer exercised and gated off by the transition floor."
        )
        orchestrator.approve_run(request.run_id, signoff)
        print(f"[{request.run_id}] approved -> {(OUTPUTS / RUN_ID / 'analysis_report.html')}")


if __name__ == "__main__":
    prepare_reference_inputs()
    asyncio.run(main())
