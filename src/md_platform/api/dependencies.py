"""API dependencies.

Single shared orchestrator, run store and report service, configured from
settings. The narrator is chosen once at startup: hosted when an API key is
configured, deterministic template otherwise, with the template also wired in as
the fallback so a provider outage degrades the narrative rather than the service.
"""

import logging
from functools import lru_cache

from ..config import Settings, settings
from ..orchestrator import AnalysisOrchestrator
from ..reporting.llm import AnthropicClient
from ..reporting.narrator import LLMNarrator, Narrator, TemplateNarrator
from ..reporting.report_service import ReportService
from ..store import RunStore

logger = logging.getLogger("md_ai_analyzer")


def get_settings() -> Settings:
    return settings


@lru_cache(maxsize=1)
def get_store() -> RunStore:
    return RunStore(output_dir=settings.output_dir)


@lru_cache(maxsize=1)
def get_orchestrator() -> AnalysisOrchestrator:
    return AnalysisOrchestrator(output_dir=settings.output_dir, store=get_store())


@lru_cache(maxsize=1)
def build_narrator() -> Narrator:
    """Hosted narrator when credentials exist, deterministic one otherwise."""
    if not settings.llm_enabled:
        logger.info("No LLM credentials configured; using the template narrator.")
        return TemplateNarrator()
    try:
        client = AnthropicClient(
            model=settings.llm_model, api_key=settings.anthropic_api_key
        )
    except RuntimeError as exc:
        logger.warning("LLM narrator unavailable (%s); using the template narrator.", exc)
        return TemplateNarrator()
    return LLMNarrator(client, max_turns=settings.llm_max_turns)


@lru_cache(maxsize=1)
def get_report_service() -> ReportService:
    return ReportService(
        store=get_store(),
        narrator=build_narrator(),
        fallback_narrator=TemplateNarrator(),
    )
