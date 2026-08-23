"""Phase 5 LLM eval harness (master plan: scored rubric, not pass/fail).

Fixture set: 4 known-correct reports (must clear the grounding checker with
zero flags — no false positives) and 5 injected-error reports (each wrong
number must be caught and named). Every case earns rubric points and the
suite asserts a 100% total, so any checker or prompt change that drops a
point fails CI with the fixture id.

Rubric: known-correct cases are worth 1 point (accepted, zero flags);
injected-error cases are worth 2 (1 for a non-empty catch, 2 only if the
injected number is the one named).
"""

import pytest
from test_phase4 import _ML_FRAMES, _FakeUniverse, _make_classical_bundle, _make_ml_request

from md_platform.llm.grounding_checker import check_grounding
from md_platform.ml.analysis import run_phase4_ml_analysis

KNOWN_CORRECT = [
    ("exact-scalar-metrics", "Backbone RMSD averaged 2.5 +/- 0.4 Angstrom (range 1.8 to 3.1) over 120 frames."),
    ("one-decimal-rounding", "Radius of gyration held near 18.2 Angstrom with a minimum of 17.8 Angstrom."),
    ("integer-rounding", "The protein stayed compact at roughly 18 Angstrom radius of gyration."),
]

# (case id, report template, injected wrong number, whether the report cites ML numbers)
INJECTED_ERRORS = [
    ("wrong-rmsd-mean", "Backbone RMSD averaged {n} Angstrom.", 31.7, False),
    ("wrong-rg-mean", "The radius of gyration settled at {n} Angstrom.", 44.4, False),
    ("wrong-rg-max", "Radius of gyration peaked at {n} Angstrom.", 61.2, False),
    ("wrong-frame-count", "This report summarizes {n} frames of simulation.", 1234.5, False),
    ("wrong-ml-agreement", "PCA/TICA state agreement was {n}.", 999.0, True),
]


@pytest.fixture(scope="module")
def ml_bundle():
    return run_phase4_ml_analysis(
        _FakeUniverse(_ML_FRAMES), _make_classical_bundle(), _make_ml_request()
    )


def _check(report: str, make_bundle, ml_bundle, uses_ml: bool) -> list:
    if uses_ml:
        return check_grounding(report, _make_classical_bundle(), ml_bundle)
    return check_grounding(report, make_bundle())


def _score_known(ungrounded: list) -> int:
    return 1 if ungrounded == [] else 0


def _score_injected(ungrounded: list, wrong: float) -> int:
    if not ungrounded:
        return 0
    return 2 if ungrounded == [str(wrong)] else 1


def _ml_known_correct_report(ml_bundle) -> str:
    return f"The MSM over 2 states showed CK deviation {ml_bundle.msm.ck_deviation:.3f}."


@pytest.mark.parametrize(("case_id", "report"), KNOWN_CORRECT)
def test_known_correct_report_clears_checker(case_id, report, make_bundle):
    assert _score_known(_check(report, make_bundle, None, False)) == 1, case_id


def test_known_correct_ml_report_clears_checker(make_bundle, ml_bundle):
    assert _score_known(_check(_ml_known_correct_report(ml_bundle), make_bundle, ml_bundle, True)) == 1


@pytest.mark.parametrize(("case_id", "template", "wrong", "uses_ml"), INJECTED_ERRORS)
def test_injected_error_is_caught_and_named(case_id, template, wrong, uses_ml, make_bundle, ml_bundle):
    ungrounded = _check(template.format(n=wrong), make_bundle, ml_bundle, uses_ml)
    assert _score_injected(ungrounded, wrong) == 2, f"{case_id}: got {ungrounded}"


def test_rubric_total_score_is_100_percent(make_bundle, ml_bundle):
    """Plan metric: 100% catch rate on injected errors, zero false positives."""
    score = 0
    maximum = 0
    for _case_id, report in KNOWN_CORRECT:
        score += _score_known(_check(report, make_bundle, None, False))
        maximum += 1
    score += _score_known(_check(_ml_known_correct_report(ml_bundle), make_bundle, ml_bundle, True))
    maximum += 1
    for _case_id, template, wrong, uses_ml in INJECTED_ERRORS:
        score += _score_injected(_check(template.format(n=wrong), make_bundle, ml_bundle, uses_ml), wrong)
        maximum += 2
    assert score == maximum
