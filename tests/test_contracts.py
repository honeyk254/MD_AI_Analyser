"""Phase 5 contract tests: the AnalysisBundle is the producer/consumer contract.

Producers: the 8 classical modules assembled through build_bundle (the real
assembly path, on the reference trajectory — module-scoped fixture shared with
the regression suite). Consumers: build_report_summary, check_grounding,
generate_html_report. Wire format: the bundle must survive a JSON round-trip,
which catches numpy types leaking into the Any-typed payload fields.
"""

import json

import pytest
from MDAnalysisTests.datafiles import DCD, PSF
from test_phase4 import _ML_FRAMES, _FakeUniverse, _make_classical_bundle, _make_ml_request
from test_regression import MODULES, reference_run  # noqa: F401  (shared module-scoped fixture)

from md_platform.aggregation.bundle_builder import build_bundle
from md_platform.aggregation.report_summary import build_report_summary
from md_platform.llm.grounding_checker import check_grounding
from md_platform.ml.analysis import run_phase4_ml_analysis
from md_platform.ml.schemas import MLAnalysisBundle
from md_platform.reporting.html_report import generate_html_report
from md_platform.schemas.analysis_bundle import AnalysisBundle


@pytest.fixture
def produced_bundle(reference_run):  # noqa: F811  (imported fixture, see import above)
    return build_bundle(
        run_id="contract-test",
        trajectory_metadata=reference_run["metadata"],
        qc_flags=reference_run["qc"],
        module_results=reference_run["modules"],
        inputs={"topology": {"file": PSF}, "trajectory": {"file": DCD}},
        parameters={},
    )


def test_all_module_producers_assemble_into_valid_bundle(produced_bundle):
    assert set(produced_bundle.modules) == {name for name, _ in MODULES}
    assert all(result.error is None for result in produced_bundle.modules.values())


def test_bundle_survives_json_round_trip(produced_bundle):
    wire = json.dumps(produced_bundle.model_dump(mode="json"))
    restored = AnalysisBundle.model_validate_json(wire)
    assert restored.model_dump() == produced_bundle.model_dump()


def test_report_summary_consumer_accepts_bundle(produced_bundle):
    summary = build_report_summary(produced_bundle)
    assert summary["trajectory"]["frames"] == 98
    assert set(summary["modules"]) == set(produced_bundle.modules)


def test_grounding_consumer_accepts_bundle(produced_bundle):
    rmsd = produced_bundle.modules["rmsd"].scalar_metrics["backbone_rmsd"]
    report = f"Backbone RMSD averaged {rmsd.mean:.2f} Angstrom (max {rmsd.max:.2f})."
    assert check_grounding(report, produced_bundle) == []


def test_html_consumer_accepts_bundle(produced_bundle, tmp_path):
    report_path = generate_html_report(produced_bundle, plots={}, output_dir=tmp_path)
    assert report_path.exists() and report_path.stat().st_size > 0


def test_ml_bundle_survives_json_and_carries_analysis_card():
    ml = run_phase4_ml_analysis(
        _FakeUniverse(_ML_FRAMES), _make_classical_bundle(), _make_ml_request()
    )
    restored = MLAnalysisBundle.model_validate_json(json.dumps(ml.model_dump(mode="json")))
    assert restored.model_dump() == ml.model_dump()

    # Plan metric: 100% Analysis Card coverage. The card is a required schema
    # field, so no ML bundle can ship without one; here we check it is populated.
    card = ml.analysis_card
    for field in ("title", "purpose", "literature_basis", "data_requirements",
                  "failure_modes", "baseline_protocol"):
        assert getattr(card, field), f"Analysis Card field '{field}' is empty"
    assert MLAnalysisBundle.model_fields["analysis_card"].is_required()
