"""Deterministic grounding checker.

Every number in a narrative is extracted and matched against the facts of the
``GroundedSummary`` the narrative was written from. A claim is:

``verified``     a fact reachable from the claim's context has this value, within
                 the tolerance implied by how precisely the claim was written;
``mismatch``     the claim names a metric, but no fact of that metric has this
                 value — the classic hallucinated-number failure;
``unsupported``  the number is traceable to no fact at all.

The checker is intentionally independent of the narrator: it re-derives its facts
from the summary, so a narrator (template or model) cannot influence its own
verdict.
"""

import re
from typing import Dict, Iterable, List, Optional, Tuple

from ..schemas.report import (
    ClaimCheck,
    ClaimStatus,
    GroundingResult,
    NarrativeReport,
    NumericClaim,
)
from ..schemas.summary import GroundedSummary

CHECKER_VERSION = "1.0.0"

# A number, optionally followed by a unit. Guarded on the left so residue labels
# (ASP45) and identifiers do not read as claims.
NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_.])(-?\d+(?:\.\d+)?)\s*"
    r"(%|Å|angstroms?|nm\^?2|nm2|ps|ns|frames?|residues?|atoms?|counts?|fractions?)?",
    re.IGNORECASE,
)

SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")

UNIT_ALIASES: Dict[str, str] = {
    "%": "percent",
    "percent": "percent",
    "å": "angstrom",
    "angstrom": "angstrom",
    "angstroms": "angstrom",
    "nm^2": "nm^2",
    "nm2": "nm^2",
    "ps": "ps",
    "ns": "ns",
    "frame": "frame",
    "frames": "frame",
    "residue": "residue",
    "residues": "residue",
    "atom": "atom",
    "atoms": "atom",
    "count": "count",
    "counts": "count",
    "fraction": "fraction",
    "fractions": "fraction",
}

# Numbers written with this many decimals imply this absolute tolerance; a claim
# of "1.85" is verified by a fact of 1.8523, but not by one of 1.9.
TOLERANCE_EPSILON = 1e-9


class Fact:
    """One numeric value the narrative is allowed to cite."""

    __slots__ = ("label", "metric", "value", "unit")

    def __init__(self, label: str, metric: Optional[str], value: float, unit: str):
        self.label = label
        self.metric = metric
        self.value = float(value)
        self.unit = normalize_unit(unit)


def normalize_unit(unit: Optional[str]) -> str:
    if not unit:
        return ""
    return UNIT_ALIASES.get(unit.strip().lower(), unit.strip().lower())


def build_facts(summary: GroundedSummary) -> List[Fact]:
    """Derive every citable number from the summary."""
    traj = summary.trajectory
    facts: List[Fact] = [
        Fact("trajectory.n_frames_analyzed", None, traj.n_frames_analyzed, "frame"),
        Fact("trajectory.n_atoms", None, traj.n_atoms, "atom"),
        Fact("trajectory.n_residues", None, traj.n_residues, "residue"),
        Fact("trajectory.frame_interval_ps", None, traj.frame_interval_ps, "ps"),
        Fact("trajectory.total_time_ns", None, traj.total_time_ns, "ns"),
    ]

    if summary.qc.equilibration_frame is not None:
        facts.append(
            Fact("qc.equilibration_frame", None, summary.qc.equilibration_frame, "frame")
        )

    for name, stats in summary.metrics.items():
        facts.extend(
            [
                Fact(f"{name}.mean", name, stats.mean, stats.unit),
                Fact(f"{name}.std", name, stats.std, stats.unit),
                Fact(f"{name}.min", name, stats.min, stats.unit),
                Fact(f"{name}.max", name, stats.max, stats.unit),
                Fact(f"{name}.n_frames", name, stats.n_frames, "frame"),
            ]
        )
        for attr, unit in (
            ("first_half_mean", stats.unit),
            ("second_half_mean", stats.unit),
            ("drift", stats.unit),
            ("coefficient_of_variation", ""),
        ):
            value = getattr(stats, attr)
            if value is not None:
                facts.append(Fact(f"{name}.{attr}", name, value, unit))
        if stats.drift_percent is not None:
            facts.append(Fact(f"{name}.drift_percent", name, stats.drift_percent, "percent"))
        for frame in stats.changepoint_frames:
            facts.append(Fact(f"{name}.changepoint", name, frame, "frame"))

    for comparison in summary.comparisons:
        facts.append(
            Fact(f"{comparison.metric}.compared_value", comparison.metric, comparison.value, comparison.unit)
        )
        for attr in ("reference_low", "reference_high"):
            value = getattr(comparison, attr)
            if value is not None:
                facts.append(
                    Fact(f"{comparison.metric}.{attr}", comparison.metric, value, comparison.unit)
                )
    return facts


def metric_aliases(metric: str) -> Tuple[str, ...]:
    """Surface forms of a metric name that may appear in prose."""
    spaced = metric.replace("_", " ")
    return tuple({metric.lower(), spaced.lower()})


def tolerance_for(literal: str) -> float:
    """Tolerance implied by the precision the claim was written with."""
    decimals = len(literal.split(".")[1]) if "." in literal else 0
    return 0.5 * (10.0**-decimals) + TOLERANCE_EPSILON


def citation_strings(summary: GroundedSummary) -> List[str]:
    """Verbatim provenance strings the summary itself supplies.

    Literature citations carry numbers ("J. Mol. Biol. 196:641 (1987)") that are
    not measurements. They are exempt only when the narrative reproduces a
    citation string that came from the summary, so an invented citation is still
    checked number by number.
    """
    strings: List[str] = []
    for comparison in summary.comparisons:
        strings.extend(part for part in (comparison.source, comparison.note) if part)
    return sorted(strings, key=len, reverse=True)


def mask_citations(text: str, citations: Iterable[str]) -> str:
    """Blank out summary-provided citation spans before extracting numbers."""
    masked = text
    for citation in citations:
        if citation in masked:
            masked = masked.replace(citation, " " * len(citation))
    return masked


def extract_claims(
    narrative: NarrativeReport, citations: Iterable[str] = ()
) -> List[Tuple[NumericClaim, str]]:
    """Extract numeric claims and the literal each was written as."""
    citations = list(citations)
    claims: List[Tuple[NumericClaim, str]] = []
    for section in narrative.sections:
        for sentence in SENTENCE_PATTERN.split(mask_citations(section.body, citations)):
            for match in NUMBER_PATTERN.finditer(sentence):
                literal, unit = match.group(1), match.group(2)
                claims.append(
                    (
                        NumericClaim(
                            value=float(literal),
                            unit=normalize_unit(unit) or None,
                            section=section.heading,
                            context=sentence.strip(),
                        ),
                        literal,
                    )
                )
    return claims


def check_narrative(
    narrative: NarrativeReport, summary: GroundedSummary
) -> GroundingResult:
    """Verify every numeric claim in ``narrative`` against ``summary``."""
    facts = build_facts(summary)
    by_metric: Dict[str, List[Fact]] = {}
    global_facts: List[Fact] = []
    for fact in facts:
        if fact.metric is None:
            global_facts.append(fact)
        else:
            by_metric.setdefault(fact.metric, []).append(fact)

    checks: List[ClaimCheck] = []
    for claim, literal in extract_claims(narrative, citation_strings(summary)):
        checks.append(
            _check_claim(claim, literal, by_metric, global_facts)
        )

    counts = {status: 0 for status in ClaimStatus}
    for check in checks:
        counts[check.status] += 1

    return GroundingResult(
        checker_version=CHECKER_VERSION,
        passed=counts[ClaimStatus.MISMATCH] == 0 and counts[ClaimStatus.UNSUPPORTED] == 0,
        checks=checks,
        n_verified=counts[ClaimStatus.VERIFIED],
        n_mismatched=counts[ClaimStatus.MISMATCH],
        n_unsupported=counts[ClaimStatus.UNSUPPORTED],
    )


def _check_claim(
    claim: NumericClaim,
    literal: str,
    by_metric: Dict[str, List[Fact]],
    global_facts: List[Fact],
) -> ClaimCheck:
    tolerance = tolerance_for(literal)
    context = claim.context.lower()

    named_metrics = [
        metric
        for metric in by_metric
        if any(alias in context for alias in metric_aliases(metric))
    ]

    candidates: List[Fact] = list(global_facts)
    for metric in named_metrics:
        candidates.extend(by_metric[metric])
    if not named_metrics:
        for metric_facts in by_metric.values():
            candidates.extend(metric_facts)

    unit_matched = [
        fact
        for fact in candidates
        if claim.unit is None or fact.unit == claim.unit or fact.unit == ""
    ]

    match = _closest(unit_matched, claim.value)
    if match is not None and abs(match.value - claim.value) <= tolerance:
        return ClaimCheck(
            claim=claim,
            status=ClaimStatus.VERIFIED,
            matched_metric=match.label,
            expected_value=match.value,
            tolerance=tolerance,
            detail=f"Matches {match.label} within +/-{tolerance:g}.",
        )

    if named_metrics:
        metric_facts = [f for metric in named_metrics for f in by_metric[metric]]
        nearest = _closest(metric_facts, claim.value)
        return ClaimCheck(
            claim=claim,
            status=ClaimStatus.MISMATCH,
            matched_metric=nearest.label if nearest else None,
            expected_value=nearest.value if nearest else None,
            tolerance=tolerance,
            detail=(
                f"No fact of {', '.join(named_metrics)} equals {claim.value:g}"
                + (f"; nearest is {nearest.label}={nearest.value:g}." if nearest else ".")
            ),
        )

    return ClaimCheck(
        claim=claim,
        status=ClaimStatus.UNSUPPORTED,
        tolerance=tolerance,
        detail=f"{claim.value:g} is not traceable to any value in the analysis bundle.",
    )


def _closest(facts: Iterable[Fact], value: float) -> Optional[Fact]:
    best: Optional[Fact] = None
    for fact in facts:
        if best is None or abs(fact.value - value) < abs(best.value - value):
            best = fact
    return best
