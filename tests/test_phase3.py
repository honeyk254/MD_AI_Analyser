from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import md_platform.api.app as app_module
from md_platform.api.app import _request_buckets, app
from md_platform.api import dependencies
from md_platform.demo_inputs import ensure_demo_inputs


def test_demo_inputs_materialize(tmp_path: Path) -> None:
    examples = ensure_demo_inputs(tmp_path)

    assert set(examples) == {"stable", "flexible"}
    assert Path(examples["stable"]["topology_file"]).exists()
    assert Path(examples["stable"]["trajectory_file"]).exists()


def test_demo_examples_endpoint_lists_examples() -> None:
    client = TestClient(app)

    response = client.get("/api/v1/demo/examples")

    assert response.status_code == 200
    names = {item["name"] for item in response.json()}
    assert names == {"stable", "flexible"}


def test_request_guard_blocks_large_bodies() -> None:
    client = TestClient(app)

    response = client.post("/api/v1/demo/stable/submit", data="x" * 70000)

    assert response.status_code == 413


def test_demo_submit_rate_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeOrchestrator:
        async def run_analysis(self, request):
            self.request = request

    monkeypatch.setattr(dependencies, "ORCHESTRATOR", FakeOrchestrator())
    monkeypatch.setattr(app_module, "RATE_LIMIT_PER_MINUTE", 1)
    _request_buckets.clear()

    client = TestClient(app)
    first = client.post("/api/v1/demo/stable/submit")
    second = client.post("/api/v1/demo/stable/submit")

    assert first.status_code == 200
    assert second.status_code == 429


def test_rate_limit_uses_forwarded_ip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "RATE_LIMIT_PER_MINUTE", 1)
    _request_buckets.clear()
    client = TestClient(app)

    first = client.post(
        "/api/v1/demo/stable/submit",
        headers={"x-forwarded-for": "1.2.3.4"},
    )
    second = client.post(
        "/api/v1/demo/stable/submit",
        headers={"x-forwarded-for": "1.2.3.4"},
    )

    assert first.status_code == 200
    assert second.status_code == 429
