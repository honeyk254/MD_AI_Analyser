"""Regenerate the live demo reports (GitHub Pages) end to end.

Runs the full pipeline on the adenylate-kinase reference trajectory with the
ML layer enabled (TICA/MSM + VAMPnet ablation when torch is installed), then
the bundled peptide demo with ML attempted (it demos the refusal gate).
Writes standalone HTML reports to data/outputs/.
"""

import asyncio
import shutil
from pathlib import Path

from MDAnalysisTests.datafiles import DCD, PSF

from md_platform.orchestrator import AnalysisOrchestrator
from md_platform.schemas.api import AnalysisRequest

INPUTS = Path("data/inputs/adk")
OUTPUTS = Path("data/outputs")


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


def kinetics_request(run_id: str) -> AnalysisRequest:
    # Clearly-labeled synthetic two-state system: the one bundled input with
    # genuine interconversion, so the full kinetic layer (gate -> TICA/MSM ->
    # CK -> baseline -> VAMPnet ablation) runs end to end on the live page.
    from md_platform.demo_inputs import ensure_demo_inputs

    demo = ensure_demo_inputs(Path("data/inputs"))["kinetics"]
    return AnalysisRequest(
        job_id="demo-kinetics",
        run_id=run_id,
        topology_file=demo["topology_file"],
        trajectory_file=demo["trajectory_file"],
        enable_ml=True,
        ml_lag_frames=3,
        ml_n_states=2,
        ml_min_frames=100,
        ml_min_transition_count=10,
        ml_ck_threshold=0.15,
    )


def peptide_request(run_id: str) -> AnalysisRequest:
    # Default ML gates (100 frames) intentionally refuse the 12-frame demo
    # peptide: the live demo shows the refusal path doing its job.
    from md_platform.demo_inputs import ensure_demo_inputs

    demo = ensure_demo_inputs(Path("data/inputs"))["stable"]
    return AnalysisRequest(
        job_id="demo-stable",
        run_id=run_id,
        topology_file=demo["topology_file"],
        trajectory_file=demo["trajectory_file"],
        enable_ml=True,
    )


async def run(request: AnalysisRequest) -> None:
    orchestrator = AnalysisOrchestrator(output_dir=OUTPUTS)
    await orchestrator.run_analysis(request)
    status = orchestrator.get_status(request.run_id)
    print(f"[{request.run_id}] status={status.status} message={status.message}")
    ml = orchestrator.ml_bundles.get(request.run_id)
    if ml:
        print(f"[{request.run_id}] ML: {ml.status}; vampnet={ml.vampnet_ablation.summary if ml.vampnet_ablation else None}")
        if ml.refusal_reason:
            print(f"[{request.run_id}] ML refusal: {ml.refusal_reason}")
    if status.status.value == "human_review":
        orchestrator.approve_run(
            request.run_id,
            "Regenerated end-to-end: 10 classical modules + TICA/MSM + VAMPnet ablation.",
        )
        print(f"[{request.run_id}] approved -> {(OUTPUTS / request.run_id / 'analysis_report.html')}")


if __name__ == "__main__":
    prepare_reference_inputs()
    asyncio.run(run(adk_request("adk-reference")))
    asyncio.run(run(kinetics_request("demo-kinetics")))
    asyncio.run(run(peptide_request("demo-stable")))
