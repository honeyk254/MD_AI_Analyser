"""Analysis Orchestrator.

Ties together parsing, validation, classical analysis, aggregation, and reporting
into a single asynchronous pipeline.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict

import MDAnalysis as mda

from .aggregation.bundle_builder import build_bundle
from .domain.classical.com import compute_com
from .domain.classical.contacts import compute_contact_map
from .domain.classical.dihedrals import compute_dihedrals
from .domain.classical.hbonds import compute_hbonds
from .domain.classical.radius_of_gyration import compute_rg
from .domain.classical.rmsd import compute_rmsd
from .domain.classical.rmsf import compute_rmsf
from .domain.classical.salt_bridges import compute_salt_bridges
from .domain.classical.sasa import compute_sasa
from .domain.classical.secondary_structure import compute_secondary_structure
from .domain.parsing import parse_metadata
from .domain.validation import validate_trajectory
from .llm.orchestrator import LLMOrchestrator
from .ml.analysis import run_phase4_ml_analysis
from .ml.schemas import MLAnalysisBundle
from .reporting.html_report import generate_html_report
from .reporting.plots import generate_all_plots
from .schemas.analysis_bundle import AnalysisBundle
from .schemas.api import AnalysisRequest, RunStatus, StatusResponse

logger = logging.getLogger("md_ai_analyzer")


class AnalysisOrchestrator:
    """Manages the lifecycle of an analysis run."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Simple in-memory status tracking for Phase 1
        self.statuses: Dict[str, StatusResponse] = {}
        self.bundles: Dict[str, AnalysisBundle] = {}
        self.ml_bundles: Dict[str, MLAnalysisBundle] = {}
        self.drafts: Dict[str, str] = {}
        self.started_at: Dict[str, float] = {}

    def get_status(self, run_id: str) -> StatusResponse:
        """Get current status of a run."""
        if run_id not in self.statuses:
            return StatusResponse(run_id=run_id, status=RunStatus.FAILED, message="Run not found.")
        return self.statuses[run_id]

    async def run_analysis(self, request: AnalysisRequest):
        """Execute the full analysis pipeline."""
        run_id = request.run_id or request.job_id
        pipeline_start = time.time()
        self.started_at[run_id] = pipeline_start

        self.statuses[run_id] = StatusResponse(
            run_id=run_id, status=RunStatus.PENDING, message="Starting pipeline..."
        )

        try:
            # 1. Parsing and Ingestion
            self.statuses[run_id].status = RunStatus.RUNNING
            self.statuses[run_id].message = "Loading trajectory..."

            universe = mda.Universe(request.topology_file, request.trajectory_file)
            metadata = parse_metadata(universe, request.topology_file, request.trajectory_file)

            # 2. Validation
            self.statuses[run_id].message = "Validating trajectory..."
            qc_flags = validate_trajectory(universe, metadata)
            if not qc_flags.sufficient_frames:
                raise ValueError("Insufficient frames to proceed with analysis.")

            # 3. Classical Analysis
            self.statuses[run_id].message = "Running classical analysis modules..."

            # We run sequentially for simplicity, but these could be async/parallelized
            # in a ThreadPoolExecutor in the future.
            module_results = {}

            modules = [
                ("rmsd", compute_rmsd),
                ("rmsf", compute_rmsf),
                ("radius_of_gyration", compute_rg),
                ("sasa", compute_sasa),
                ("hbonds", compute_hbonds),
                ("contacts", compute_contact_map),
                ("secondary_structure", compute_secondary_structure),
                ("salt_bridges", compute_salt_bridges),
                ("dihedrals", compute_dihedrals),
                ("com", compute_com),
            ]

            for mod_name, mod_func in modules:
                self.statuses[run_id].message = f"Running {mod_name}..."
                await asyncio.sleep(0)  # Yield control
                try:
                    res = mod_func(universe)
                    module_results[mod_name] = res
                except Exception as e:
                    logger.error(f"Module {mod_name} failed: {e}")
                    # Skip failed module, but continue

            # 4. Aggregation
            self.statuses[run_id].message = "Aggregating results..."
            report_start = time.time()
            inputs = {
                "topology": {"file": request.topology_file, "hash": "skipped"},
                "trajectory": {"file": request.trajectory_file, "hash": "skipped"},
            }
            bundle = build_bundle(
                run_id=run_id,
                trajectory_metadata=metadata,
                qc_flags=qc_flags,
                module_results=module_results,
                inputs=inputs,
                parameters={},
            )
            self.bundles[run_id] = bundle

            # 5. Phase 4 ML analysis (opt-in)
            ml_bundle = None
            if request.enable_ml:
                self.statuses[run_id].message = "Running Phase 4 ML analysis..."
                ml_bundle = run_phase4_ml_analysis(universe, bundle, request)
                self.ml_bundles[run_id] = ml_bundle

            # 6. Reporting
            self.statuses[run_id].message = "Generating reports..."
            run_out_dir = self.output_dir / run_id
            run_out_dir.mkdir(parents=True, exist_ok=True)
            plots = generate_all_plots(bundle, ml_bundle)

            # 7. LLM Narrative Generation
            self.statuses[run_id].message = "Generating LLM narrative..."
            llm_orchestrator = LLMOrchestrator()
            draft_report = llm_orchestrator.generate_report(bundle, ml_bundle)
            self.drafts[run_id] = draft_report

            # Save bundle JSON
            bundle_json = run_out_dir / "bundle.json"
            bundle_json.write_text(bundle.model_dump_json(indent=2))

            # Save draft report
            draft_file = run_out_dir / "draft_report.md"
            draft_file.write_text(draft_report)
            generate_html_report(
                bundle,
                plots,
                run_out_dir,
                narrative_report=draft_report,
                ml_bundle=ml_bundle,
            )

            self.statuses[run_id].status = RunStatus.HUMAN_REVIEW
            self.statuses[run_id].message = "Draft report generated. Pending human review."
            self.statuses[run_id].results_url = f"/api/v1/analysis/{run_id}/results"
            self.statuses[run_id].reviewer_signoff = None

            # Plan metrics (tracked, not gated): report <30s, review turnaround logged.
            logger.info(
                "run %s: report generation %.1fs (target <30s), pipeline total %.1fs",
                run_id, time.time() - report_start, time.time() - pipeline_start,
            )

        except Exception as e:
            logger.exception("Pipeline failed")
            self.statuses[run_id].status = RunStatus.FAILED
            self.statuses[run_id].message = str(e)

    def approve_run(self, run_id: str, reviewer_signoff: str) -> StatusResponse:
        """Approve a human-review run and mark it complete."""
        status = self.get_status(run_id)
        if status.status != RunStatus.HUMAN_REVIEW:
            raise ValueError(f"Run {run_id} is not awaiting review.")

        status.status = RunStatus.COMPLETED
        status.message = "Run approved and finalized."
        status.reviewer_signoff = reviewer_signoff
        self.statuses[run_id] = status

        started = self.started_at.get(run_id)
        if started:
            logger.info("run %s: human review turnaround %.0fs", run_id, time.time() - started)

        bundle = self.bundles.get(run_id)
        if bundle:
            ml_bundle = self.ml_bundles.get(run_id)
            run_out_dir = self.output_dir / run_id
            run_out_dir.mkdir(parents=True, exist_ok=True)
            plots = generate_all_plots(bundle, ml_bundle)
            generate_html_report(
                bundle,
                plots,
                run_out_dir,
                narrative_report=self.drafts.get(run_id),
                reviewer_signoff=reviewer_signoff,
                ml_bundle=ml_bundle,
            )
            final_report = run_out_dir / "final_report.md"
            report_text = self.drafts.get(run_id)
            if report_text:
                final_report.write_text(
                    f"{report_text}\n\n## Human Review\n{reviewer_signoff}\n"
                )

        return status
