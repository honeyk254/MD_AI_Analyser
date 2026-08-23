"""LLM Orchestrator.

Manages tool-calling to the LLM (Anthropic) for narrative generation.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

try:
    import anthropic
except ModuleNotFoundError:  # pragma: no cover - optional dependency in local dev
    anthropic = None

from ..aggregation.report_summary import build_report_summary
from ..ml.schemas import MLAnalysisBundle
from ..observability import LLM_METRICS, LLMCallMetric, span
from ..schemas.analysis_bundle import AnalysisBundle
from .grounding_checker import check_grounding

logger = logging.getLogger("md_ai_analyzer.llm.orchestrator")


class LLMOrchestrator:
    """Orchestrates LLM calls to generate a scientific report."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.warning("ANTHROPIC_API_KEY not found. LLM features will be disabled or mocked.")

        # In a real app we instantiate the client here:
        # self.client = anthropic.Anthropic(api_key=self.api_key)

    def _get_metric_summary(self, bundle: AnalysisBundle, metric_name: str) -> str:
        """Tool implementation: get_metric_summary."""
        aliases = {
            "rg": "radius_of_gyration",
            "radius_of_gyration": "radius_of_gyration",
            "rmsd": "rmsd",
            "rmsf": "rmsf",
            "sasa": "sasa",
            "hbonds": "hbonds",
            "contacts": "contacts",
            "secondary_structure": "secondary_structure",
            "salt_bridges": "salt_bridges",
        }
        metric_name = aliases.get(metric_name, metric_name)
        for mod_name, mod_result in bundle.modules.items():
            if metric_name in mod_result.scalar_metrics:
                summary = mod_result.scalar_metrics[metric_name]
                return f"{mod_name}.{metric_name}: mean={summary.mean:.2f} {summary.unit}, std={summary.std:.2f}"
            if mod_name == metric_name:
                # Summarize the whole module
                lines = []
                for k, v in mod_result.scalar_metrics.items():
                    lines.append(f"{k}: mean={v.mean:.2f} {v.unit}, std={v.std:.2f}")
                return "\n".join(lines) if lines else f"No scalar metrics for {metric_name}"
        return f"Metric {metric_name} not found."

    def _get_qc_flags(self, bundle: AnalysisBundle) -> str:
        """Tool implementation: get_qc_flags."""
        flags = bundle.qc_flags
        report = [
            f"Is Equilibrated: {flags.is_equilibrated}",
            f"Sufficient Frames: {flags.sufficient_frames}"
        ]
        for f in flags.flags:
            report.append(f"{f.check_name}: {'PASSED' if f.passed else 'FAILED'} - {f.details}")
        return "\n".join(report)

    def _compare_to_reference_ranges(self, bundle: AnalysisBundle, metric_name: str) -> str:
        """Tool implementation: compare_to_reference_ranges."""
        metric_name = metric_name.lower()
        if metric_name == "rg":
            metric_name = "radius_of_gyration"

        ranges = {
            "rmsd": "Typically < 3.0 A for stable globular proteins.",
            "radius_of_gyration": "Should remain stable (std < 1.0 A).",
            "sasa": "Should plateau during equilibrium.",
        }
        ref = ranges.get(metric_name, "No literature reference range available.")

        # Fetch actual value to compare
        actual = self._get_metric_summary(bundle, metric_name)

        return f"Reference: {ref}\nActual: {actual}"

    def generate_report(
        self,
        bundle: AnalysisBundle,
        ml_bundle: Optional[MLAnalysisBundle] = None,
    ) -> str:
        """Generate a narrative report using the LLM and verify its grounding."""
        summary = build_report_summary(bundle, ml_bundle)
        started = time.perf_counter()
        tokens_in = 0
        tokens_out = 0

        with span("llm.generate_report", {"run_id": bundle.run_id}) as current:
            mode = "fallback"
            if not self.api_key or anthropic is None:
                if self.api_key:
                    logger.warning("anthropic package not installed; using fallback report generation.")
                draft_report = self._render_fallback_report(summary)
            else:
                mode = "llm"
                system_prompt = (
                    "You are a structural biology expert analyzing MD trajectories. "
                    "Write a concise scientific report using only the figures returned by the tools. "
                    "Do not invent numbers. Structure the output as QC Assessment, Structural Stability, "
                    "Biological Interpretation, Limitations, Follow-ups, and an optional Phase 4 ML section if present."
                )

                tools = [
                    {
                        "name": "get_metric_summary",
                        "description": "Get mean and std values for a metric (e.g. 'rmsd', 'rg').",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "metric_name": {"type": "string", "description": "Name of the metric or module"}
                            },
                            "required": ["metric_name"]
                        }
                    },
                    {
                        "name": "get_qc_flags",
                        "description": "Get all quality control flags.",
                        "input_schema": {
                            "type": "object",
                            "properties": {}
                        }
                    },
                    {
                        "name": "compare_to_reference_ranges",
                        "description": "Compare a metric to known literature heuristics.",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "metric_name": {"type": "string"}
                            },
                            "required": ["metric_name"]
                        }
                    }
                ]

                client = anthropic.Anthropic(api_key=self.api_key)

                try:
                    messages: List[Dict[str, Any]] = [
                        {
                            "role": "user",
                            "content": (
                                "Generate a grounded MD analysis report using this compact summary:\n"
                                f"{summary}"
                            ),
                        }
                    ]
                    response = client.messages.create(
                        model="claude-3-haiku-20240307",
                        max_tokens=1000,
                        system=system_prompt,
                        tools=tools,
                        messages=messages,
                    )

                    for _ in range(4):
                        usage = getattr(response, "usage", None)
                        if usage:
                            tokens_in += usage.input_tokens
                            tokens_out += usage.output_tokens
                        if response.stop_reason != "tool_use":
                            break

                        tool_results: List[Dict[str, Any]] = []
                        for block in response.content:
                            if block.type != "tool_use":
                                continue
                            if block.name == "get_metric_summary":
                                result = self._get_metric_summary(bundle, block.input["metric_name"])
                            elif block.name == "get_qc_flags":
                                result = self._get_qc_flags(bundle)
                            elif block.name == "compare_to_reference_ranges":
                                result = self._compare_to_reference_ranges(bundle, block.input["metric_name"])
                            else:
                                result = "Unknown tool."
                            tool_results.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": result,
                                }
                            )

                        messages.extend(
                            [
                                {"role": "assistant", "content": response.content},
                                {"role": "user", "content": tool_results},
                            ]
                        )
                        response = client.messages.create(
                            model="claude-3-haiku-20240307",
                            max_tokens=1000,
                            system=system_prompt,
                            tools=tools,
                            messages=messages,
                        )

                    usage = getattr(response, "usage", None)
                    if usage:
                        tokens_in += usage.input_tokens
                        tokens_out += usage.output_tokens
                    draft_report = response.content[0].text if response.content else ""
                except Exception as e:
                    logger.error(f"LLM generation failed: {e}")
                    mode = "fallback"
                    draft_report = self._render_fallback_report(summary)

            ungrounded_claims = check_grounding(draft_report, bundle, ml_bundle)
            if ungrounded_claims:
                logger.warning(f"Ungrounded claims found: {ungrounded_claims}")
                draft_report += f"\n\n[QC WARNING] The following numbers failed the grounding check: {', '.join(ungrounded_claims)}"

            # Plan metric (tracked, not gated): cost per report, target < $0.50.
            cost_usd = (tokens_in * 0.25 + tokens_out * 1.25) / 1_000_000 if mode == "llm" else 0.0
            latency_s = time.perf_counter() - started
            if current is not None:
                current.set_attribute("mode", mode)
                current.set_attribute("latency_s", latency_s)
                current.set_attribute("cost_usd", cost_usd)
                current.set_attribute("ungrounded_claims", len(ungrounded_claims))
            LLM_METRICS.record(
                LLMCallMetric(
                    run_id=bundle.run_id,
                    mode=mode,
                    latency_s=latency_s,
                    cost_usd=cost_usd,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    ungrounded_claims=len(ungrounded_claims),
                )
            )

            return draft_report

    def _render_fallback_report(self, summary: Dict[str, Any]) -> str:
        """Render a deterministic report when no LLM key is configured."""
        qc = summary["qc"]
        modules = summary["modules"]

        lines = [
            "# Grounded MD Report",
            "",
            "## QC Assessment",
            f"- Equilibrated: {qc['is_equilibrated']}",
            f"- Sufficient frames: {qc['sufficient_frames']}",
        ]
        for flag in qc["flags"]:
            lines.append(
                f"- {flag['check_name']}: {'PASSED' if flag['passed'] else 'FAILED'} - {flag['details']}"
            )

        lines.extend(
            [
                "",
                "## Structural Stability",
            ]
        )

        for key in ("rmsd", "radius_of_gyration", "sasa"):
            module = modules.get(key)
            if module:
                lines.append(f"- {module['takeaway']}")

        lines.extend(
            [
                "",
                "## Biological Interpretation",
                "- The report is intentionally conservative and only summarizes computed metrics.",
            ]
        )

        ml = summary.get("ml")
        if ml:
            lines.extend(
                [
                    "",
                    "## Statistical / ML Layer",
                    f"- Status: {ml['status']}",
                    f"- ML gate passed: {ml['gating']['passed']}",
                ]
            )
            if ml.get("msm"):
                msm = ml["msm"]
                lines.append(
                    f"- MSM lag {msm['lag_frames']} frames ({msm['lag_ps']:.2f} ps); CK deviation {msm['ck_deviation']:.3f}."
                )
            if ml.get("baseline_comparison"):
                baseline = ml["baseline_comparison"]
                lines.append(
                    f"- PCA/TICA state agreement: {baseline['state_agreement_nmi']:.2f} NMI."
                )
            if ml.get("vampnet_ablation"):
                lines.append(f"- {ml['vampnet_ablation']['summary']}")

        lines.extend(
            [
                "",
                "## Limitations",
                "- No free-form claims beyond the aggregated bundle.",
                "",
                "## Follow-ups",
                "- Review module-specific plots before final sign-off.",
            ]
        )
        return "\n".join(lines)
