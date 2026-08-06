"""Run store.

Run state lives on disk, one directory per ``run_id``, so status and results
survive a restart and are shared by every worker process serving the API. This
is deliberately a filesystem store rather than Postgres: the whole Phase 2/3
target is a single-host deployment, and swapping the backend later only means
reimplementing this class.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

from .schemas.analysis_bundle import AnalysisBundle
from .schemas.api import RunStatus, StatusResponse
from .schemas.report import GeneratedReport

logger = logging.getLogger("md_ai_analyzer")

STATUS_FILE = "status.json"
BUNDLE_FILE = "bundle.json"
REPORT_FILE = "report.json"
INPUT_DIR = "inputs"


class RunStore:
    """Filesystem-backed persistence for runs, bundles and reports."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_dir(self, run_id: str) -> Path:
        path = self.output_dir / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def input_dir(self, run_id: str) -> Path:
        path = self.run_dir(run_id) / INPUT_DIR
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ----- status ----- #

    def write_status(self, status: StatusResponse) -> None:
        self._write(self.run_dir(status.run_id) / STATUS_FILE, status)

    def read_status(self, run_id: str) -> Optional[StatusResponse]:
        return self._read(self.output_dir / run_id / STATUS_FILE, StatusResponse)

    def set_status(
        self,
        run_id: str,
        status: RunStatus,
        message: str,
        current_module: Optional[str] = None,
        progress_percent: float = 0.0,
        results_url: Optional[str] = None,
    ) -> StatusResponse:
        state = StatusResponse(
            run_id=run_id,
            status=status,
            message=message,
            current_module=current_module,
            progress_percent=progress_percent,
            results_url=results_url,
        )
        self.write_status(state)
        return state

    # ----- bundle ----- #

    def write_bundle(self, bundle: AnalysisBundle) -> Path:
        path = self.run_dir(bundle.run_id) / BUNDLE_FILE
        self._write(path, bundle)
        return path

    def read_bundle(self, run_id: str) -> Optional[AnalysisBundle]:
        return self._read(self.output_dir / run_id / BUNDLE_FILE, AnalysisBundle)

    # ----- report ----- #

    def write_report(self, report: GeneratedReport) -> Path:
        path = self.run_dir(report.run_id) / REPORT_FILE
        self._write(path, report)
        return path

    def read_report(self, run_id: str) -> Optional[GeneratedReport]:
        return self._read(self.output_dir / run_id / REPORT_FILE, GeneratedReport)

    # ----- listing ----- #

    def list_runs(self) -> List[str]:
        return sorted(
            p.name for p in self.output_dir.iterdir() if (p / STATUS_FILE).exists()
        )

    # ----- helpers ----- #

    @staticmethod
    def _write(path: Path, model) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(model.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(path)

    @staticmethod
    def _read(path: Path, model_cls):
        if not path.exists():
            return None
        try:
            return model_cls.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, ValueError):
            logger.exception("Corrupt %s in %s", model_cls.__name__, path)
            return None
