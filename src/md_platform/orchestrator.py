"""Analysis Orchestrator.

Ties parsing, validation, classical analysis, aggregation and reporting into one
pipeline. The scientific work is synchronous CPU work, so it is executed in a
worker thread rather than on the event loop — otherwise a single submission
blocks every other request, including status polling.
"""

import asyncio
import functools
import logging
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import MDAnalysis as mda

from .aggregation.bundle_builder import build_bundle
from .aggregation.provenance import file_provenance
from .config import settings
from .domain.classical.contacts import compute_contact_map
from .domain.classical.hbonds import compute_hbonds
from .domain.classical.radius_of_gyration import compute_rg
from .domain.classical.rmsd import compute_rmsd
from .domain.classical.rmsf import compute_rmsf
from .domain.classical.salt_bridges import compute_salt_bridges
from .domain.classical.sasa import compute_sasa
from .domain.classical.secondary_structure import compute_secondary_structure
from .domain.frames import FrameWindow
from .domain.parsing import parse_metadata
from .domain.validation import validate_trajectory
from .reporting.html_report import generate_html_report
from .reporting.plots import generate_all_plots
from .schemas.analysis_bundle import AnalysisBundle, ModuleResult
from .schemas.api import AnalysisRequest, RunStatus
from .store import RunStore

logger = logging.getLogger("md_ai_analyzer")

ModuleFn = Callable[..., ModuleResult]

# Module name -> (function, request attribute supplying its primary threshold).
CLASSICAL_MODULES: List[Tuple[str, ModuleFn, Optional[str]]] = [
    ("rmsd", compute_rmsd, None),
    ("rmsf", compute_rmsf, None),
    ("radius_of_gyration", compute_rg, None),
    ("sasa", compute_sasa, None),
    ("hbonds", compute_hbonds, "hbond_cutoff"),
    ("contacts", compute_contact_map, "contact_cutoff"),
    ("secondary_structure", compute_secondary_structure, None),
    ("salt_bridges", compute_salt_bridges, "salt_bridge_cutoff"),
]

# Keyword each module expects its threshold under.
THRESHOLD_KEYWORD = {
    "hbond_cutoff": "distance",
    "contact_cutoff": "cutoff",
    "salt_bridge_cutoff": "cutoff",
}


class AnalysisOrchestrator:
    """Manages the lifecycle of an analysis run."""

    def __init__(self, output_dir: Path, store: Optional[RunStore] = None):
        self.output_dir = Path(output_dir)
        self.store = store or RunStore(self.output_dir)

    def get_status(self, run_id: str):
        """Persisted status of a run, or None when the run is unknown."""
        return self.store.read_status(run_id)

    def get_bundle(self, run_id: str) -> Optional[AnalysisBundle]:
        return self.store.read_bundle(run_id)

    async def run_analysis(self, request: AnalysisRequest) -> None:
        """Run the pipeline off the event loop."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, functools.partial(self.run_analysis_sync, request))

    def run_analysis_sync(self, request: AnalysisRequest) -> Optional[AnalysisBundle]:
        """Execute the full analysis pipeline synchronously."""
        run_id = request.run_id or uuid.uuid4().hex
        window = FrameWindow(
            start=request.start_frame or 0,
            stop=request.end_frame,
            step=request.stride,
        )

        self.store.set_status(
            run_id, RunStatus.PENDING, "Starting pipeline...", progress_percent=0.0
        )
        started = time.time()

        try:
            self.store.set_status(
                run_id, RunStatus.RUNNING, "Loading trajectory...", progress_percent=5.0
            )
            universe = mda.Universe(request.topology_file, request.trajectory_file)
            self._enforce_size_caps(universe)
            window = self._cap_frames(universe, window)
            metadata = parse_metadata(
                universe, request.topology_file, request.trajectory_file, window
            )

            self.store.set_status(
                run_id,
                RunStatus.RUNNING,
                "Validating trajectory...",
                progress_percent=10.0,
            )
            qc_flags = validate_trajectory(universe, metadata)
            if not qc_flags.sufficient_frames:
                raise ValueError(
                    f"Insufficient frames to proceed: {metadata.n_frames_analyzed} "
                    "frames in the analysed window."
                )

            module_results = self._run_modules(universe, request, window, run_id)

            self.store.set_status(
                run_id, RunStatus.RUNNING, "Aggregating results...", progress_percent=85.0
            )
            bundle = build_bundle(
                run_id=run_id,
                trajectory_metadata=metadata,
                qc_flags=qc_flags,
                module_results=module_results,
                inputs={
                    "topology": file_provenance(request.topology_file),
                    "trajectory": file_provenance(request.trajectory_file),
                },
                parameters={
                    **request.module_parameters(),
                    "effective_stride": window.step,
                },
            )
            self.store.write_bundle(bundle)

            self.store.set_status(
                run_id, RunStatus.RUNNING, "Generating plots...", progress_percent=92.0
            )
            plots = generate_all_plots(bundle)
            generate_html_report(bundle, plots, self.store.run_dir(run_id))

            self.store.set_status(
                run_id,
                RunStatus.COMPLETED,
                f"Analysis complete in {time.time() - started:.1f}s.",
                progress_percent=100.0,
                results_url=f"/api/v1/analysis/{run_id}/results",
            )
            return bundle

        except Exception as exc:
            logger.exception("Pipeline failed for run %s", run_id)
            self.store.set_status(run_id, RunStatus.FAILED, str(exc))
            return None

    @staticmethod
    def _enforce_size_caps(universe: mda.Universe) -> None:
        """Reject systems this deployment is not sized for."""
        n_atoms = universe.atoms.n_atoms
        if n_atoms > settings.max_atoms:
            raise ValueError(
                f"System has {n_atoms} atoms, above the {settings.max_atoms} atom "
                "limit for this deployment."
            )

    @staticmethod
    def _cap_frames(universe: mda.Universe, window: FrameWindow) -> FrameWindow:
        """Stride the window down instead of refusing an over-long trajectory."""
        resolved = window.resolve(len(universe.trajectory))
        n_frames = len(range(resolved.start, resolved.stop or 0, resolved.step))
        if n_frames <= settings.max_frames:
            return window
        factor = -(-n_frames // settings.max_frames)
        logger.info(
            "Window holds %d frames; striding by an extra factor of %d to stay under "
            "the %d frame cap.",
            n_frames,
            factor,
            settings.max_frames,
        )
        return FrameWindow(
            start=resolved.start, stop=resolved.stop, step=resolved.step * factor
        )

    def _run_modules(
        self,
        universe: mda.Universe,
        request: AnalysisRequest,
        window: FrameWindow,
        run_id: str,
    ) -> Dict[str, ModuleResult]:
        """Run every classical module, recording failures instead of dropping them."""
        module_results: Dict[str, ModuleResult] = {}
        span = 70.0 / len(CLASSICAL_MODULES)

        for index, (name, func, threshold_attr) in enumerate(CLASSICAL_MODULES):
            self.store.set_status(
                run_id,
                RunStatus.RUNNING,
                f"Running {name}...",
                current_module=name,
                progress_percent=15.0 + index * span,
            )
            thresholds = {}
            if threshold_attr is not None:
                thresholds[THRESHOLD_KEYWORD[threshold_attr]] = getattr(
                    request, threshold_attr
                )

            started = time.time()
            try:
                module_results[name] = func(universe, window=window, **thresholds)
            except Exception as exc:
                # A failed module is reported in the bundle rather than silently
                # omitted: a partial report must be visibly partial.
                logger.error("Module %s failed: %s", name, exc)
                module_results[name] = ModuleResult(
                    name=name,
                    version="unknown",
                    runtime_seconds=time.time() - started,
                    parameters={
                        **thresholds,
                        "start_frame": window.start,
                        "end_frame": window.stop,
                        "stride": window.step,
                    },
                    error=f"{type(exc).__name__}: {exc}",
                )
        return module_results
