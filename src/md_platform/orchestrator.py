"""Analysis Orchestrator.

Ties together parsing, validation, classical analysis, aggregation, and reporting
into a single asynchronous pipeline.
"""

import logging
import time
import asyncio
from pathlib import Path
from typing import Dict, Any, List

import MDAnalysis as mda

from .schemas.analysis_bundle import AnalysisBundle
from .schemas.api import AnalysisRequest, RunStatus, StatusResponse
from .domain.parsing import parse_metadata
from .domain.validation import validate_trajectory
from .domain.classical.rmsd import compute_rmsd
from .domain.classical.rmsf import compute_rmsf
from .domain.classical.radius_of_gyration import compute_rg
from .domain.classical.sasa import compute_sasa
from .domain.classical.hbonds import compute_hbonds
from .domain.classical.contacts import compute_contact_map
from .domain.classical.secondary_structure import compute_secondary_structure
from .domain.classical.salt_bridges import compute_salt_bridges
from .aggregation.bundle_builder import build_bundle
from .reporting.plots import generate_all_plots
from .reporting.html_report import generate_html_report
from .llm.orchestrator import LLMOrchestrator

logger = logging.getLogger("md_ai_analyzer")


class AnalysisOrchestrator:
    """Manages the lifecycle of an analysis run."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Simple in-memory status tracking for Phase 1
        self.statuses: Dict[str, StatusResponse] = {}
        self.bundles: Dict[str, AnalysisBundle] = {}
        self.drafts: Dict[str, str] = {}

    def get_status(self, run_id: str) -> StatusResponse:
        """Get current status of a run."""
        if run_id not in self.statuses:
            return StatusResponse(run_id=run_id, status=RunStatus.FAILED, message="Run not found.")
        return self.statuses[run_id]

    async def run_analysis(self, request: AnalysisRequest):
        """Execute the full analysis pipeline."""
        run_id = request.run_id or request.job_id
        
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
            
            # 5. Reporting
            self.statuses[run_id].message = "Generating reports..."
            run_out_dir = self.output_dir / run_id
            run_out_dir.mkdir(parents=True, exist_ok=True)
            plots = generate_all_plots(bundle)
            
            # 6. LLM Narrative Generation
            self.statuses[run_id].message = "Generating LLM narrative..."
            llm_orchestrator = LLMOrchestrator()
            draft_report = llm_orchestrator.generate_report(bundle)
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
            )
            
            self.statuses[run_id].status = RunStatus.HUMAN_REVIEW
            self.statuses[run_id].message = "Draft report generated. Pending human review."
            self.statuses[run_id].results_url = f"/api/v1/analysis/{run_id}/results"
            self.statuses[run_id].reviewer_signoff = None
            
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

        bundle = self.bundles.get(run_id)
        if bundle:
            run_out_dir = self.output_dir / run_id
            run_out_dir.mkdir(parents=True, exist_ok=True)
            plots = generate_all_plots(bundle)
            generate_html_report(
                bundle,
                plots,
                run_out_dir,
                narrative_report=self.drafts.get(run_id),
                reviewer_signoff=reviewer_signoff,
            )
            final_report = run_out_dir / "final_report.md"
            report_text = self.drafts.get(run_id)
            if report_text:
                final_report.write_text(
                    f"{report_text}\n\n## Human Review\n{reviewer_signoff}\n"
                )

        return status
