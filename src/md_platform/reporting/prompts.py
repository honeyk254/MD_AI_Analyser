"""Versioned prompts.

The prompt version is recorded in every report's audit trail, so changing any
string in this file requires bumping ``PROMPT_VERSION``.
"""

PROMPT_VERSION = "2026-08-06.1"

SECTION_PLAN = [
    (
        "Quality control",
        "State whether the run passed QC, which checks failed, whether the "
        "trajectory equilibrated and at which frame, and name any module that "
        "produced no results.",
    ),
    (
        "Structural behaviour",
        "Describe what the metrics show: stability or drift of backbone RMSD and "
        "radius of gyration, where flexibility is concentrated, how hydrogen bonds, "
        "salt bridges and secondary structure behave over time.",
    ),
    (
        "Reference comparison",
        "For each metric with a reference band, state the verdict returned by the "
        "comparison tool and the caveat that comes with it.",
    ),
    (
        "Biological interpretation",
        "Offer a cautious mechanistic reading of the observations. Mark it clearly "
        "as interpretation rather than measurement.",
    ),
    (
        "Limitations",
        "State what this analysis cannot establish: single trajectory, heuristic "
        "equilibration detection, unknown force field where applicable, and any "
        "missing module.",
    ),
    (
        "Suggested follow-up",
        "Propose concrete next simulations or analyses. Do not imply any of them "
        "have been performed.",
    ),
]

SYSTEM_PROMPT = """You are writing the narrative sections of a molecular-dynamics \
analysis report for a structural biologist.

Absolute rules:
1. Every number you write MUST come verbatim from a tool result. You may round to \
at most three significant figures, and you must not perform any other arithmetic \
- no differences, ratios, percentages, unit conversions or totals of your own.
2. If a value you want is not available from a tool, say that it is not available. \
Never estimate, interpolate or recall a typical value from the literature.
3. Report verdicts (equilibration, reference-range comparisons) exactly as the \
tools return them. Do not overrule them.
4. Keep measurement and interpretation separate, and say which is which.
5. Do not claim anything about drug binding, clinical relevance, or experimental \
validation. This is a single simulation analysed computationally.
6. Write plainly, in prose, without markdown headings inside a section body.

You have exactly three tools: get_metric_summary, get_qc_flags and \
compare_to_reference_ranges. Call them as often as you need before writing. \
Write the report only once you have the numbers you intend to cite.

When you have finished gathering data, respond with the report as a JSON object:
{"sections": [{"heading": "...", "body": "..."}]}
Use exactly these headings, in this order: %s
""" % ", ".join(
    heading for heading, _ in SECTION_PLAN
)


def user_prompt(metric_names: list, run_id: str) -> str:
    """The per-run instruction: what exists, and what to produce."""
    plan = "\n".join(f"- {heading}: {intent}" for heading, intent in SECTION_PLAN)
    metrics = ", ".join(metric_names) or "none"
    return (
        f"Run {run_id} has these metrics available: {metrics}.\n\n"
        f"Write these sections:\n{plan}\n\n"
        "Gather the numbers with the tools first, then return the JSON object."
    )
