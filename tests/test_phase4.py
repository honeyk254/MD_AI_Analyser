from __future__ import annotations

from dataclasses import dataclass

from md_platform.aggregation.report_summary import build_report_summary
from md_platform.llm.grounding_checker import check_grounding
from md_platform.ml.analysis import run_phase4_ml_analysis
from md_platform.schemas.analysis_bundle import (
    AnalysisBundle,
    MetricSummary,
    ModuleResult,
    QCFlag,
    QCFlags,
    TrajectoryMetadata,
)
from md_platform.schemas.api import AnalysisRequest
from md_platform.schemas.run_card import FileProvenance, RunCard, ToolVersions


@dataclass
class _FakeTS:
    time: float


@dataclass
class _FakeAtom:
    resname: str
    resid: int
    name: str


class _FakeSelection:
    def __init__(self, universe: "_FakeUniverse") -> None:
        self._universe = universe
        self._atoms = [
            _FakeAtom(resname="ALA", resid=1, name="CA"),
            _FakeAtom(resname="GLY", resid=2, name="CA"),
        ]

    @property
    def positions(self):
        return self._universe.frames[self._universe.current_frame]

    def __len__(self) -> int:
        return len(self._atoms)

    def __iter__(self):
        return iter(self._atoms)


class _FakeTrajectory:
    def __init__(self, universe: "_FakeUniverse") -> None:
        self._universe = universe

    def __iter__(self):
        for idx in range(len(self._universe.frames)):
            self._universe.current_frame = idx
            yield _FakeTS(time=float(idx * 2))


class _FakeUniverse:
    def __init__(self, frames) -> None:
        self.frames = frames
        self.current_frame = 0
        self.trajectory = _FakeTrajectory(self)

    def select_atoms(self, _selection: str):
        return _FakeSelection(self)


def _make_classical_bundle() -> AnalysisBundle:
    return AnalysisBundle(
        run_id="run-ml",
        trajectory_metadata=TrajectoryMetadata(
            n_frames_analyzed=6,
            n_atoms=2,
            n_residues=1,
            timestep_ps=2.0,
            total_time_ns=0.012,
            original_format=".xtc",
            force_field="unknown",
        ),
        qc_flags=QCFlags(
            is_equilibrated=True,
            sufficient_frames=True,
            flags=[QCFlag(check_name="sufficient_frames", passed=True, details="ok")],
        ),
        modules={
            "rmsd": ModuleResult(
                name="rmsd",
                version="2.0.0",
                runtime_seconds=1.0,
                parameters={},
                scalar_metrics={
                    "backbone_rmsd": MetricSummary(
                        mean=1.2,
                        std=0.3,
                        min=0.8,
                        max=1.8,
                        unit="Angstrom",
                        n_frames=6,
                        time_series=[1.0, 1.1, 1.2, 1.3, 1.1, 1.0],
                    )
                },
                data={"time_ps": [0, 2, 4, 6, 8, 10], "equilibration_frame": 1},
            )
        },
        run_card=RunCard(
            inputs={"topology": FileProvenance(filename="top.pdb", sha256="abc", size_bytes=1)},
            tools=ToolVersions(python="3.11.0", mdanalysis="2.6.0", numpy="1.24.0"),
            container_digest=None,
            parameters={},
        ),
    )


def _make_ml_request(**overrides) -> AnalysisRequest:
    params = dict(
        job_id="job-ml",
        run_id="run-ml",
        topology_file="top.pdb",
        trajectory_file="traj.xtc",
        enable_ml=True,
        ml_lag_frames=1,
        ml_n_states=2,
        ml_min_frames=4,
        ml_min_transition_count=1,
        ml_ck_threshold=0.6,
    )
    params.update(overrides)
    return AnalysisRequest(**params)


def test_phase4_ml_analysis_builds_bundle_and_summary() -> None:
    frames = [
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [2.1, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]],
    ]
    ml_bundle = run_phase4_ml_analysis(_FakeUniverse(frames), _make_classical_bundle(), _make_ml_request())

    assert ml_bundle.status == "completed"
    assert ml_bundle.msm is not None
    assert ml_bundle.baseline_comparison is not None

    summary = build_report_summary(_make_classical_bundle(), ml_bundle)
    assert summary["ml"]["status"] == "completed"
    assert summary["ml"]["baseline_comparison"]["state_agreement_nmi"] >= 0.0


def test_phase4_ml_analysis_refuses_insufficient_frames() -> None:
    frames = [
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]],
    ]
    request = _make_ml_request(ml_min_frames=10)
    ml_bundle = run_phase4_ml_analysis(_FakeUniverse(frames), _make_classical_bundle(), request)

    assert ml_bundle.status == "blocked"
    assert any("below the minimum" in reason for reason in ml_bundle.gating.reasons)


def test_phase4_grounding_checker_catches_bad_ml_numbers() -> None:
    frames = [
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [2.1, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]],
    ]
    ml_bundle = run_phase4_ml_analysis(_FakeUniverse(frames), _make_classical_bundle(), _make_ml_request())
    report = f"PCA/TICA agreement was 999.0 and CK deviation was {ml_bundle.msm.ck_deviation:.3f}."

    assert "999.0" in check_grounding(report, _make_classical_bundle(), ml_bundle)
