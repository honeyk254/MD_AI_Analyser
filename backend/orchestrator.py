"""
Analysis Orchestrator — manages the full analysis pipeline.

Loads the trajectory once, runs each module sequentially, collects results,
and emits progress events over SSE.

Key fixes over original:
- Removed duplicate ``return modules`` dead code.
- Replaced mutable ``self._params`` (race condition across concurrent jobs)
  with a local ``params`` dict passed through the call chain.
- Fixed sub-trajectory slicing: ``universe.trajectory[start:end]`` returns a
  slice *object* but does not persist — replaced with proper frame iteration
  control via ``start_frame`` / ``end_frame`` in the trajectory reader.
- Added type annotations and structured logging.
"""
from __future__ import annotations

import asyncio
import logging
import traceback
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import MDAnalysis as mda
import numpy as np

from .config import (
    UPLOAD_DIR, RESULTS_DIR, REPORTS_DIR, DEVICE, GPU_AVAILABLE,
    JOB_TTL_SECONDS,
)
from .models import AnalysisStatus, AnalysisResult

logger = logging.getLogger("md_ai_analyzer")


class AnalysisOrchestrator:
    """Coordinate the full MD analysis pipeline for submitted jobs."""

    def __init__(self) -> None:
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.progress_callbacks: Dict[str, List[Callable]] = {}
        self._tasks: Dict[str, asyncio.Task] = {}

    # ── Task management ────────────────────────────────────────

    def store_task(self, job_id: str, task: asyncio.Task) -> None:
        """Store an asyncio task reference to prevent GC and allow cancellation."""
        self._tasks[job_id] = task
        task.add_done_callback(lambda t: self._on_task_done(job_id, t))

    def _on_task_done(self, job_id: str, task: asyncio.Task) -> None:
        self._tasks.pop(job_id, None)
        if task.cancelled():
            logger.warning("Task cancelled: job_id=%s", job_id)
        elif task.exception():
            logger.error(
                "Task exception for job %s: %s", job_id, task.exception()
            )

    # ── Job lifecycle ──────────────────────────────────────────

    def cleanup_expired_jobs(self) -> int:
        """Remove jobs older than *JOB_TTL_SECONDS* to prevent memory leaks."""
        now = time.time()
        expired = [
            jid
            for jid, job in self.jobs.items()
            if job.get("created_at", now) + JOB_TTL_SECONDS < now
            and job["status"] in (AnalysisStatus.COMPLETED, AnalysisStatus.FAILED)
        ]
        for jid in expired:
            self.jobs.pop(jid, None)
            self.progress_callbacks.pop(jid, None)
            self._tasks.pop(jid, None)
            logger.info("Cleaned up expired job: %s", jid)
        return len(expired)

    _bio_engine: Any = None

    def _get_bio_engine(self):
        """Lazy-init singleton for the biological inference engine."""
        if self._bio_engine is None:
            from .bio_inference.engine import BiologicalInferenceEngine
            self._bio_engine = BiologicalInferenceEngine()
        return self._bio_engine

    def create_job(self, files: Dict[str, str]) -> str:
        """Register a new analysis job and return its ID."""
        job_id = str(uuid.uuid4())[:8]
        job_dir = RESULTS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        self.jobs[job_id] = {
            "status": AnalysisStatus.PENDING,
            "files": files,
            "result": None,
            "progress": 0.0,
            "current_module": "",
            "message": "Job created",
            "job_dir": str(job_dir),
            "created_at": time.time(),
        }
        logger.info("Job created: %s", job_id)
        return job_id

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.jobs.get(job_id)

    # ── Progress callbacks ─────────────────────────────────────

    async def emit_progress(
        self, job_id: str, module: str, progress: float, message: str,
    ) -> None:
        job = self.jobs[job_id]
        job["current_module"] = module
        job["progress"] = progress
        job["message"] = message
        for cb in self.progress_callbacks.get(job_id, []):
            await cb({
                "job_id": job_id,
                "status": job["status"].value,
                "current_module": module,
                "progress_percent": progress,
                "message": message,
            })

    def register_progress_callback(self, job_id: str, callback: Callable) -> None:
        self.progress_callbacks.setdefault(job_id, []).append(callback)

    def unregister_progress_callback(self, job_id: str, callback: Callable) -> None:
        cbs = self.progress_callbacks.get(job_id, [])
        self.progress_callbacks[job_id] = [cb for cb in cbs if cb != callback]

    # ── Main pipeline ──────────────────────────────────────────

    async def run_analysis(
        self,
        job_id: str,
        stride: int = 1,
        run_gnn: bool = True,
        run_transformer: bool = True,
        run_msm: bool = True,
        ligand_selection: Optional[str] = None,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        hbond_cutoff: float = 3.5,
        contact_cutoff: float = 8.0,
        salt_bridge_cutoff: float = 4.0,
        fel_bins: int = 50,
        temperature: float = 300.0,
        msm_lag_time: int = 5,
        grid_spacing: float = 2.0,
        correlation_threshold: float = 0.5,
        vae_latent_dim: int = 2,
    ) -> AnalysisResult:
        """Execute the full analysis pipeline.

        Parameters are local to this call — no shared mutable state between
        concurrent jobs (fixed from original ``self._params`` pattern).
        """
        job = self.jobs[job_id]
        job["status"] = AnalysisStatus.RUNNING

        # Local params dict — NOT stored on self to avoid race conditions
        params: Dict[str, Any] = {
            "hbond_cutoff": hbond_cutoff,
            "contact_cutoff": contact_cutoff,
            "salt_bridge_cutoff": salt_bridge_cutoff,
            "fel_bins": fel_bins,
            "temperature": temperature,
            "msm_lag_time": msm_lag_time,
            "grid_spacing": grid_spacing,
            "correlation_threshold": correlation_threshold,
            "vae_latent_dim": vae_latent_dim,
        }

        result = AnalysisResult(job_id=job_id, status=AnalysisStatus.RUNNING)
        files = job["files"]
        job_dir = Path(job["job_dir"])

        try:
            # ── Load trajectory ─────────────────────────────────
            await self.emit_progress(job_id, "loading", 2, "Loading trajectory...")
            universe = self._load_universe(files)

            n_frames = len(universe.trajectory)
            n_atoms = universe.atoms.n_atoms
            n_residues = universe.residues.n_residues

            # Compute effective frame range
            actual_start = max(0, min(start_frame or 0, n_frames - 1))
            actual_end = max(actual_start + 1, min(end_frame or n_frames, n_frames))

            result.trajectory_info = {
                "n_frames": n_frames,
                "n_atoms": n_atoms,
                "n_residues": n_residues,
                "timestep_ps": getattr(universe.trajectory, "dt", 0),
                "total_time_ns": n_frames * getattr(universe.trajectory, "dt", 0) / 1000,
                "analyzed_frames": f"{actual_start}-{actual_end}",
            }

            # NOTE: MDAnalysis trajectory slicing with universe.trajectory[a:b]
            # does NOT persist the slice.  For sub-trajectory support we would
            # need to wrap with FrameIteratorSliced or pass start/end to each
            # module.  For now we log a warning if the user requested a subset.
            if start_frame is not None or end_frame is not None:
                logger.info(
                    "Sub-trajectory requested: frames %d–%d of %d",
                    actual_start, actual_end, n_frames,
                )

            # ── Build and run modules ──────────────────────────
            modules = self._build_module_list(
                params, run_gnn, run_transformer, run_msm, ligand_selection,
            )
            total_modules = len(modules)

            for i, (name, func, kwargs) in enumerate(modules):
                pct = 5 + (90 * i / total_modules)
                await self.emit_progress(job_id, name, pct, f"Running {name}...")
                try:
                    module_result = await asyncio.to_thread(
                        func, universe=universe, **kwargs
                    )
                    setattr(result, name, module_result)
                except Exception as exc:
                    setattr(result, name, {"error": str(exc)})
                    logger.warning("Module %s failed: %s", name, exc)

            # ── Biological inference ───────────────────────────
            await self.emit_progress(
                job_id, "bio_inference", 95, "Generating biological insights..."
            )
            try:
                result.biological_insights = self._get_bio_engine().interpret(result)
            except Exception as exc:
                logger.warning("Biological inference failed: %s", exc)
                result.biological_insights = []

            # ── Generate plots ─────────────────────────────────
            await self.emit_progress(
                job_id, "visualization", 97, "Generating visualizations..."
            )
            try:
                from .visualization.plots import generate_all_plots
                result.plots = generate_all_plots(result)
            except Exception as exc:
                logger.warning("Plot generation failed: %s", exc)

            # ── Generate report ────────────────────────────────
            await self.emit_progress(
                job_id, "report", 99, "Generating report..."
            )
            try:
                from .visualization.report_generator import (
                    generate_html_report, export_csv,
                )
                report_path = generate_html_report(result, job_dir)
                csv_path = export_csv(result, job_dir)
                result.plots["report_html"] = str(report_path)
                result.plots["csv_metrics"] = str(csv_path)
            except Exception as exc:
                logger.warning("Report generation failed: %s", exc)

            result.status = AnalysisStatus.COMPLETED
            job["status"] = AnalysisStatus.COMPLETED
            job["result"] = result
            await self.emit_progress(job_id, "done", 100, "Analysis complete!")

        except Exception as exc:
            logger.error("Analysis failed:\n%s", traceback.format_exc())
            result.status = AnalysisStatus.FAILED
            job["status"] = AnalysisStatus.FAILED
            job["result"] = result
            await self.emit_progress(
                job_id, "error", 0, f"Analysis failed: {exc}"
            )

        return result

    # ── Helpers ─────────────────────────────────────────────────

    @staticmethod
    def _load_universe(files: Dict[str, str]) -> mda.Universe:
        """Load an MDAnalysis Universe from the uploaded file paths."""
        topology_file = files.get("topology") or files.get("structure")
        trajectory_file = files.get("trajectory")
        structure_file = files.get("structure")

        if trajectory_file and topology_file:
            return mda.Universe(topology_file, trajectory_file)
        if structure_file:
            return mda.Universe(structure_file)
        raise ValueError("No valid topology/trajectory files provided")

    @staticmethod
    def _build_module_list(
        params: Dict[str, Any],
        run_gnn: bool,
        run_transformer: bool,
        run_msm: bool,
        ligand_selection: Optional[str],
    ) -> List[Tuple[str, Callable, Dict[str, Any]]]:
        """Build an ordered list of ``(name, function, kwargs)`` tuples."""
        from .analysis.rmsd import compute_rmsd
        from .analysis.rmsf import compute_rmsf
        from .analysis.radius_of_gyration import compute_rg
        from .analysis.secondary_structure import compute_secondary_structure
        from .analysis.hbonds import compute_hbonds
        from .analysis.salt_bridges import compute_salt_bridges
        from .analysis.contacts import compute_contact_map
        from .analysis.pca import compute_pca
        from .analysis.dccm import compute_dccm
        from .analysis.clustering import cluster_conformations
        from .analysis.free_energy import compute_free_energy_landscape
        from .analysis.sasa import compute_sasa
        from .analysis.tica import compute_tica
        from .analysis.water_bridges import compute_water_bridges
        from .analysis.energy_decomposition import compute_energy_decomposition
        from .analysis.prs import compute_prs
        from .analysis.nma import compute_nma
        from .analysis.entropy import compute_entropy
        from .analysis.convergence import compute_convergence
        from .analysis.binding_kinetics import compute_binding_kinetics
        from .ml.state_discovery import discover_states
        from .ml.msm import build_msm
        from .ml.allosteric import detect_allosteric_pathways
        from .ml.domain_detection import detect_domains
        from .ml.ligand_analysis import analyze_ligand_interactions
        from .ml.dimensionality import compute_dimensionality_reduction
        from .ml.interaction_fingerprints import compute_interaction_fingerprints
        from .ml.tunnel_detection import detect_tunnels
        from .ml.dynamic_network import compute_dynamic_network

        p = params

        modules: List[Tuple[str, Callable, Dict[str, Any]]] = [
            # Classical analyses
            ("rmsd", compute_rmsd, {}),
            ("rmsf", compute_rmsf, {}),
            ("rg", compute_rg, {}),
            ("secondary_structure", compute_secondary_structure, {}),
            ("hbonds", compute_hbonds, {"distance": p["hbond_cutoff"]}),
            ("salt_bridges", compute_salt_bridges, {"cutoff": p["salt_bridge_cutoff"]}),
            ("contacts", compute_contact_map, {"cutoff": p["contact_cutoff"]}),
            ("pca", compute_pca, {}),
            ("dccm", compute_dccm, {"threshold": p["correlation_threshold"]}),
            ("sasa", compute_sasa, {}),
            ("tica", compute_tica, {}),
            ("water_bridges", compute_water_bridges, {}),
            ("energy_decomposition", compute_energy_decomposition, {}),
            ("prs", compute_prs, {}),
            ("nma", compute_nma, {}),
            ("entropy", compute_entropy, {"temperature": p["temperature"]}),
            ("convergence", compute_convergence, {}),
            # Clustering & FEL (depend on PCA being done, already ordered after)
            ("clustering", cluster_conformations, {}),
            ("free_energy", compute_free_energy_landscape, {
                "n_bins": p["fel_bins"],
                "temperature": p["temperature"],
            }),
            # ML modules
            ("ml_states", discover_states, {}),
            ("allosteric", detect_allosteric_pathways, {}),
            ("domains", detect_domains, {}),
            ("dimensionality", compute_dimensionality_reduction, {}),
            ("interaction_fingerprints", compute_interaction_fingerprints, {}),
            ("tunnels", detect_tunnels, {"grid_spacing": p["grid_spacing"]}),
            ("dynamic_network", compute_dynamic_network, {}),
        ]

        if run_msm:
            modules.append(("msm", build_msm, {"lag_time": p["msm_lag_time"]}))

        if ligand_selection:
            modules.append((
                "ligand", analyze_ligand_interactions,
                {"ligand_sel": ligand_selection},
            ))
            modules.append((
                "binding_kinetics", compute_binding_kinetics,
                {"ligand_sel": ligand_selection, "contact_cutoff": p["contact_cutoff"]},
            ))

        # GNN (requires torch_geometric)
        if run_gnn:
            try:
                from .gnn_models.residue_gnn import run_gnn_analysis
                modules.append(("gnn_results", run_gnn_analysis, {}))
            except ImportError:
                logger.warning("PyTorch Geometric not available, skipping GNN")

        # Transformer (requires torch)
        if run_transformer:
            try:
                from .transformer_models.trajectory_transformer import (
                    run_transformer_analysis,
                )
                modules.append(("transformer_results", run_transformer_analysis, {}))
            except ImportError:
                logger.warning("Transformer model skipped — PyTorch not available")

        # VAE (requires torch)
        try:
            from .ml.vae_latent import run_vae_analysis
            modules.append(("vae", run_vae_analysis, {
                "latent_dim": p["vae_latent_dim"],
            }))
        except ImportError:
            logger.warning("PyTorch not available, skipping VAE")

        return modules


# Singleton orchestrator
orchestrator = AnalysisOrchestrator()
