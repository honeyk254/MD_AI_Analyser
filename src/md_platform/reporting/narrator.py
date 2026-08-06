"""Narrators: turn a GroundedSummary into narrative sections.

Two implementations share one interface:

``TemplateNarrator``  deterministic, offline, no API key. Used in CI, in local
                      development and as the fallback when no LLM is configured,
                      so the platform is never unable to produce a report.
``LLMNarrator``       hosted model driving the same three tools in a bounded
                      tool-calling loop.

Both go through the identical grounding check afterwards; neither is trusted.
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..schemas.report import LLMUsage, NarrativeReport, NarrativeSection
from ..schemas.summary import GroundedSummary, MetricStatistics
from .llm import LLMClient, parse_sections
from .prompts import SECTION_PLAN, SYSTEM_PROMPT, user_prompt
from .tools import TOOL_SPECS, ReportTools

logger = logging.getLogger("md_ai_analyzer")

# Hard ceiling on model turns per report. Reached only if the model keeps
# calling tools; prevents an unbounded spend on one report.
MAX_MODEL_TURNS = 8

TEMPLATE_GENERATOR = "template"


@dataclass
class Narration:
    """A narrative plus everything the audit trail needs about producing it."""

    narrative: NarrativeReport
    usage: LLMUsage
    tool_calls: List[Dict[str, str]] = field(default_factory=list)
    seconds: float = 0.0


class Narrator(ABC):
    """Produces narrative sections from a grounded summary."""

    @abstractmethod
    def narrate(self, summary: GroundedSummary) -> Narration:
        ...


class TemplateNarrator(Narrator):
    """Deterministic narrator: same tools, fixed prose, no model."""

    def narrate(self, summary: GroundedSummary) -> Narration:
        started = time.time()
        tools = ReportTools(summary)
        calls: List[Dict[str, str]] = []

        qc = tools.get_qc_flags()
        calls.append({"tool": "get_qc_flags", "arguments": "{}"})

        metric_stats: Dict[str, Dict[str, Any]] = {}
        comparisons: Dict[str, Dict[str, Any]] = {}
        for name in tools.metric_names():
            metric_stats[name] = tools.get_metric_summary(name)
            comparisons[name] = tools.compare_to_reference_ranges(name)
            calls.append(
                {"tool": "get_metric_summary", "arguments": json.dumps({"metric_name": name})}
            )
            calls.append(
                {
                    "tool": "compare_to_reference_ranges",
                    "arguments": json.dumps({"metric_name": name}),
                }
            )

        sections = [
            NarrativeSection(heading="Quality control", body=self._qc_body(summary, qc)),
            NarrativeSection(
                heading="Structural behaviour", body=self._behaviour_body(summary)
            ),
            NarrativeSection(
                heading="Reference comparison", body=self._reference_body(comparisons)
            ),
            NarrativeSection(
                heading="Biological interpretation",
                body=self._interpretation_body(summary),
            ),
            NarrativeSection(heading="Limitations", body=self._limitations_body(summary)),
            NarrativeSection(
                heading="Suggested follow-up", body=self._follow_up_body(summary)
            ),
        ]

        return Narration(
            narrative=NarrativeReport(sections=sections, generator=TEMPLATE_GENERATOR),
            usage=LLMUsage(model=TEMPLATE_GENERATOR, n_tool_calls=len(calls)),
            tool_calls=calls,
            seconds=time.time() - started,
        )

    # ----- section bodies ----- #

    @staticmethod
    def _qc_body(summary: GroundedSummary, qc: Dict[str, Any]) -> str:
        traj = summary.trajectory
        lines = [
            f"The analysed window covers {traj.n_frames_analyzed} frames "
            f"({traj.total_time_ns:.3f} ns at {traj.frame_interval_ps:.3f} ps per frame) "
            f"of a system with {traj.n_atoms} atoms and {traj.n_residues} residues.",
        ]
        if qc["is_equilibrated"] and qc.get("equilibration_frame") is not None:
            lines.append(
                f"The RMSD trace settles at frame {qc['equilibration_frame']}, so the "
                "run is treated as equilibrated by the heuristic detector."
            )
        else:
            lines.append(
                "No stable equilibration point was detected, so time-averaged "
                "quantities should be read as covering a still-relaxing system."
            )
        passed = qc.get("passed_checks") or []
        failed = qc.get("failed_checks") or []
        lines.append(
            f"Checks passed: {', '.join(passed) if passed else 'none'}. "
            f"Checks failed: {', '.join(failed) if failed else 'none'}."
        )
        if summary.module_errors:
            for module, error in sorted(summary.module_errors.items()):
                lines.append(f"Module {module} produced no results: {error}")
        else:
            lines.append("Every analysis module produced results.")
        return " ".join(lines)

    @staticmethod
    def _behaviour_body(summary: GroundedSummary) -> str:
        if not summary.metrics:
            return "No metrics were produced, so no structural behaviour is described."

        lines: List[str] = []
        for name, stats in sorted(summary.metrics.items()):
            lines.append(TemplateNarrator._metric_sentence(name, stats))
        return " ".join(lines)

    @staticmethod
    def _metric_sentence(name: str, stats: MetricStatistics) -> str:
        sentence = (
            f"{name} averages {stats.mean:.3f} {stats.unit} "
            f"(standard deviation {stats.std:.3f}, range {stats.min:.3f} to "
            f"{stats.max:.3f}) over {stats.n_frames} frames, and is {stats.trend} "
            "across the window"
        )
        if stats.drift is not None:
            sentence += (
                f", with a first-to-second-half change of {stats.drift:.3f} "
                f"{stats.unit}"
            )
        sentence += "."
        if stats.changepoint_frames:
            frames = ", ".join(str(f) for f in stats.changepoint_frames)
            sentence += f" Mean-shift changepoints were detected at frames {frames}."
        return sentence

    @staticmethod
    def _reference_body(comparisons: Dict[str, Dict[str, Any]]) -> str:
        if not comparisons:
            return "No metrics were available to compare against reference ranges."

        lines: List[str] = []
        for name, comparison in sorted(comparisons.items()):
            verdict = comparison.get("verdict")
            if verdict == "no_reference":
                lines.append(
                    f"{name}: no reference range is asserted. {comparison.get('note', '')}"
                )
                continue
            lines.append(
                f"{name}: the mean of {comparison['value']:.3f} "
                f"{comparison['unit']} is {verdict.replace('_', ' ')} the expected band "
                f"{comparison['reference_low']:.3f} to {comparison['reference_high']:.3f} "
                f"{comparison['unit']}. {comparison.get('note', '')}"
            )
        return " ".join(lines)

    @staticmethod
    def _interpretation_body(summary: GroundedSummary) -> str:
        drifting = [
            name for name, s in sorted(summary.metrics.items()) if s.trend != "stable"
        ]
        lines = [
            "The following is interpretation, not measurement.",
        ]
        if drifting:
            lines.append(
                "The metrics that are not stable over the window ("
                + ", ".join(drifting)
                + ") are consistent with a structure still adjusting to the simulation "
                "conditions, or with a genuine conformational transition; a single "
                "trajectory cannot distinguish the two."
            )
        else:
            lines.append(
                "All metrics are stable over the window, which is consistent with a "
                "folded structure fluctuating around one conformational state."
            )
        if summary.qc.failed_checks:
            lines.append(
                "Because some quality-control checks failed, the interpretation above "
                "should be treated as provisional."
            )
        return " ".join(lines)

    @staticmethod
    def _limitations_body(summary: GroundedSummary) -> str:
        lines = [
            "This is one trajectory, so nothing here establishes a converged "
            "population average; replicate simulations would be needed for that.",
            "Equilibration is detected with a rolling-mean heuristic rather than a "
            "statistical test.",
            "Reference bands are sanity checks for globular, folded proteins and carry "
            "the caveats listed alongside each comparison.",
        ]
        if summary.trajectory.force_field.startswith("unknown"):
            lines.append(
                "The force field could not be recovered from the inputs and is reported "
                "as unknown rather than inferred."
            )
        if summary.module_errors:
            lines.append(
                "The report is partial: "
                + ", ".join(sorted(summary.module_errors))
                + " contributed no results."
            )
        return " ".join(lines)

    @staticmethod
    def _follow_up_body(summary: GroundedSummary) -> str:
        suggestions = [
            "Run independent replicates with different initial velocities to separate "
            "sampling noise from real trends.",
            "Extend the trajectory if any metric is still drifting at the end of the "
            "analysed window.",
        ]
        if summary.metrics.get("rmsf") or summary.metrics.get("mean_rmsf"):
            suggestions.append(
                "Follow up on the most flexible segments with per-residue analysis or "
                "targeted mutagenesis in silico."
            )
        suggestions.append(
            "None of these follow-ups have been performed as part of this report."
        )
        return " ".join(suggestions)


class LLMNarrator(Narrator):
    """Hosted-model narrator driving the three tools in a bounded loop."""

    def __init__(self, client: LLMClient, max_turns: int = MAX_MODEL_TURNS):
        self.client = client
        self.max_turns = max_turns

    def narrate(self, summary: GroundedSummary) -> Narration:
        started = time.time()
        tools = ReportTools(summary)
        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": user_prompt(tools.metric_names(), summary.run_id),
            }
        ]

        recorded_calls: List[Dict[str, str]] = []
        input_tokens = output_tokens = n_calls = 0
        sections: Optional[List[Dict[str, str]]] = None

        for _turn in range(self.max_turns):
            reply = self.client.reply(SYSTEM_PROMPT, messages, TOOL_SPECS)
            n_calls += 1
            input_tokens += reply.input_tokens
            output_tokens += reply.output_tokens

            if reply.tool_calls:
                messages.append({"role": "assistant", "content": reply.raw_content})
                results = []
                for call in reply.tool_calls:
                    recorded_calls.append(
                        {"tool": call.name, "arguments": json.dumps(call.arguments)}
                    )
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": call.id,
                            "content": json.dumps(tools.dispatch(call.name, call.arguments)),
                        }
                    )
                messages.append({"role": "user", "content": results})
                continue

            sections = parse_sections(reply.text or "")
            if sections:
                break

            messages.append({"role": "assistant", "content": reply.text or ""})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Return only the JSON object with the required sections, "
                        "using the headings: "
                        + ", ".join(heading for heading, _ in SECTION_PLAN)
                    ),
                }
            )

        if not sections:
            raise RuntimeError(
                f"{self.client.model} did not return report sections within "
                f"{self.max_turns} turns."
            )

        usage = LLMUsage(
            model=self.client.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            n_llm_calls=n_calls,
            n_tool_calls=len(recorded_calls),
            cost_usd=round(self.client.cost_usd(input_tokens, output_tokens), 6),
        )
        narrative = NarrativeReport(
            sections=[NarrativeSection(**section) for section in sections],
            generator=self.client.model,
        )
        return Narration(
            narrative=narrative,
            usage=usage,
            tool_calls=recorded_calls,
            seconds=time.time() - started,
        )
