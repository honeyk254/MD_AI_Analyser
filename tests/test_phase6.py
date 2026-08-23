"""Phase 6 tests: VAMPnet ablation, observability, and the metrics endpoints."""

import math

import pytest
from fastapi.testclient import TestClient
from test_phase4 import _ML_FRAMES, _FakeUniverse, _make_classical_bundle, _make_ml_request

from md_platform.api.app import app
from md_platform.llm.orchestrator import LLMOrchestrator
from md_platform.ml.analysis import run_phase4_ml_analysis
from md_platform.observability import LLM_METRICS, recent_spans

client = TestClient(app)


@pytest.fixture(scope="module")
def ml_bundle():
    return run_phase4_ml_analysis(
        _FakeUniverse(_ML_FRAMES), _make_classical_bundle(), _make_ml_request()
    )


def test_vampnet_ablation_reports_specific_numbers(ml_bundle):
    """Plan metric: VAMPnets vs MSM agreement on implied timescales, as numbers."""
    pytest.importorskip("torch")
    ablation = ml_bundle.vampnet_ablation
    assert ablation is not None and ablation.available, ablation.summary if ablation else "missing"
    assert ablation.vamp2_score is not None and math.isfinite(ablation.vamp2_score)
    assert ablation.state_agreement_nmi is not None and 0.0 <= ablation.state_agreement_nmi <= 1.0
    if ablation.timescale_relative_error is not None:
        assert ablation.timescale_relative_error >= 0.0
    if ablation.leading_timescale_ps is not None:
        assert ablation.leading_timescale_ps > 0.0
        assert ablation.tica_leading_timescale_ps == pytest.approx(
            ml_bundle.msm.implied_timescales_ps[0]
        )


def test_vampnet_ablation_degrades_gracefully_without_torch(ml_bundle, monkeypatch):
    """Without torch the ablation reports unavailable and never breaks the ML layer."""
    import md_platform.ml.vampnets as vampnets

    monkeypatch.setattr(vampnets, "torch", None)
    ablation = vampnets.run_vampnet_ablation(
        feature_matrix=_fake_features(), lag_frames=1, lag_ps=2.0,
        n_states=2, tica_labels=[0, 1, 0, 1], tica_leading_timescale_ps=4.0,
    )
    assert not ablation.available
    assert "torch" in ablation.summary


def _fake_features():
    import numpy as np

    return np.asarray([[0.0, 0.1], [0.0, 0.2], [1.0, 1.1], [1.0, 1.2], [0.1, 0.0], [0.2, 0.0]])


def test_report_generation_records_llm_metric_and_span(make_bundle):
    LLM_METRICS.calls.clear()
    bundle = make_bundle()
    report = LLMOrchestrator(api_key=None).generate_report(bundle)

    assert report  # fallback report rendered
    assert len(LLM_METRICS.calls) == 1
    metric = LLM_METRICS.calls[0]
    assert metric.run_id == bundle.run_id
    assert metric.mode == "fallback"
    assert metric.latency_s > 0
    assert metric.cost_usd == 0.0

    spans = recent_spans()
    llm_spans = [s for s in spans if s["name"] == "llm.generate_report"]
    assert llm_spans, "expected an llm.generate_report span (install opentelemetry-sdk)"
    assert llm_spans[0]["attributes"]["run_id"] == bundle.run_id
    assert llm_spans[0]["attributes"]["mode"] == "fallback"


def test_metrics_endpoints_serve_the_dashboard(make_bundle):
    LLM_METRICS.calls.clear()
    LLMOrchestrator(api_key=None).generate_report(make_bundle())

    payload = client.get("/api/v1/metrics/llm").json()
    assert payload["summary"]["n_reports"] >= 1
    assert payload["recent_calls"][0]["mode"] == "fallback"

    page = client.get("/api/v1/metrics/dashboard")
    assert page.status_code == 200
    assert "LLM Cost / Latency Dashboard" in page.text
    assert "run-123" in page.text  # the call recorded above is on the page
