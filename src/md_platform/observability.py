"""Observability: OpenTelemetry tracing plus LLM cost/latency metrics.

The OTel SDK is optional at runtime (graceful no-op spans when absent); when
present, spans are kept in an in-memory exporter so the /api/v1/metrics
endpoints can surface them without an external collector. The LLM cost
metrics list works with zero dependencies beyond stdlib.
"""

from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger("md_ai_analyzer.observability")

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
except ModuleNotFoundError:  # optional dependency
    trace = None  # type: ignore[assignment]

_exporter = None


def _ensure_provider() -> None:
    global _exporter
    if trace is None or _exporter is not None:
        return
    provider = TracerProvider(resource=Resource.create({"service.name": "md-ai-platform"}))
    _exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(_exporter))
    trace.set_tracer_provider(provider)


@contextmanager
def span(name: str, attributes: Optional[Dict[str, Any]] = None) -> Iterator[Any]:
    """Start a span; a no-op context if the OTel SDK is not installed."""
    if trace is None:
        yield None
        return
    _ensure_provider()
    with trace.get_tracer(__name__).start_as_current_span(name) as current:
        for key, value in (attributes or {}).items():
            if isinstance(value, float) and math.isnan(value):
                continue  # OTel attributes must not be NaN
            current.set_attribute(key, value)
        yield current


def recent_spans(limit: int = 20) -> List[Dict[str, Any]]:
    """Recent finished spans (name, duration, attributes) for the dashboard."""
    if trace is None or _exporter is None:
        return []
    spans = []
    for finished in _exporter.get_finished_spans()[-limit:][::-1]:
        if finished.end_time is None or finished.start_time is None:
            continue  # unfinished span
        spans.append(
            {
                "name": finished.name,
                "duration_s": (finished.end_time - finished.start_time) / 1e9,
                "attributes": {str(k): v for k, v in (finished.attributes or {}).items()},
            }
        )
    return spans


@dataclass
class LLMCallMetric:
    run_id: str
    mode: str  # "llm" or "fallback"
    latency_s: float
    cost_usd: float
    tokens_in: int
    tokens_out: int
    ungrounded_claims: int


@dataclass
class MetricsStore:
    calls: List[LLMCallMetric] = field(default_factory=list)

    def record(self, metric: LLMCallMetric) -> None:
        self.calls.append(metric)
        logger.info(
            "LLM call: run=%s mode=%s latency=%.2fs cost=$%.4f (%d in / %d out tokens, target <$0.50)",
            metric.run_id, metric.mode, metric.latency_s, metric.cost_usd,
            metric.tokens_in, metric.tokens_out,
        )

    def summary(self) -> Dict[str, Any]:
        calls = self.calls
        if not calls:
            return {"n_reports": 0}
        latencies = [c.latency_s for c in calls]
        costs = [c.cost_usd for c in calls]
        return {
            "n_reports": len(calls),
            "mean_latency_s": sum(latencies) / len(latencies),
            "mean_cost_usd": sum(costs) / len(costs),
            "total_cost_usd": sum(costs),
            "catch_rate_context": {
                "reports_with_warnings": sum(1 for c in calls if c.ungrounded_claims > 0),
            },
        }


LLM_METRICS = MetricsStore()
