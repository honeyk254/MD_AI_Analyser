"""Grounding Checker.

This module provides a deterministic, non-LLM way to verify that numeric claims
in a generated report accurately reflect the values in the AnalysisBundle.
"""

import logging
import re
from typing import Any, Iterable, List, Optional

from ..aggregation.report_summary import build_report_summary
from ..ml.schemas import MLAnalysisBundle
from ..schemas.analysis_bundle import AnalysisBundle

logger = logging.getLogger("md_ai_analyzer.llm.grounding")

class GroundingError(Exception):
    pass


def extract_numbers(text: str) -> List[float]:
    """Extract all distinct numeric values from text."""
    # A +/- only counts as a sign when it does not directly follow a digit or
    # dot — otherwise the range notation "CI 3.8-4.6 ps" would be misread as
    # the negative number -4.6 and falsely flagged as ungrounded.
    # Percentages ("bootstrap 90% CI") describe the method's own parameters,
    # not trajectory measurements, so they carry no grounding obligation.
    numbers = []
    for match in re.finditer(r"(?<![\d.])[+-]?(?:\d+\.\d+|\.\d+|\d+)", text):
        try:
            number = float(match.group())
        except ValueError:
            continue
        if text[match.end():].lstrip().startswith("%"):
            continue
        numbers.append(number)
    return numbers


def extract_all_values_from_bundle(
    bundle: AnalysisBundle, ml_bundle: Optional[MLAnalysisBundle] = None
) -> List[float]:
    """Extract all numeric values from the bundle for fuzzy matching."""
    values: List[float] = []

    # Metadata
    values.append(float(bundle.trajectory_metadata.n_frames_analyzed))
    values.append(float(bundle.trajectory_metadata.n_atoms))
    values.append(float(bundle.trajectory_metadata.n_residues))
    values.append(float(bundle.trajectory_metadata.timestep_ps))
    values.append(float(bundle.trajectory_metadata.total_time_ns))

    # Modules
    for _module_name, mod_res in bundle.modules.items():
        for _metric_name, scalar in mod_res.scalar_metrics.items():
            values.extend([scalar.mean, scalar.std, scalar.min, scalar.max, float(scalar.n_frames)])
            if scalar.time_series:
                values.extend(float(v) for v in scalar.time_series)
        values.extend(_collect_numeric_values(mod_res.data))

    summary = build_report_summary(bundle, ml_bundle)
    values.extend(_collect_numeric_values(summary.get("reference_ranges", {})))
    values.extend(_collect_numeric_values(summary.get("ml", {})))

    return values


def _collect_numeric_values(value: Any) -> List[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, dict):
        items: Iterable[Any] = value.values()
    elif isinstance(value, list):
        items = value
    else:
        return []
    numbers: List[float] = []
    for item in items:
        numbers.extend(_collect_numeric_values(item))
    return numbers


def check_grounding(
    draft_report: str,
    bundle: AnalysisBundle,
    ml_bundle: Optional[MLAnalysisBundle] = None,
) -> List[str]:
    """
    Checks that every number mentioned in the report is grounded in the bundle.
    Returns a list of ungrounded numbers/claims.

    In a real implementation, this would be more sophisticated (e.g. tracking
    context or requiring specific markdown citations for numbers), but this
    serves as the deterministic safety net.
    """
    reported_numbers = extract_numbers(draft_report)
    bundle_values = extract_all_values_from_bundle(bundle, ml_bundle)

    # Allow some rounding tolerance (e.g. 2 decimal places)
    rounded_bundle_values = {round(v, 2) for v in bundle_values}
    rounded_bundle_values.update({round(v, 1) for v in bundle_values})
    rounded_bundle_values.update({round(v, 0) for v in bundle_values})

    ungrounded = []
    for num in reported_numbers:
        # We skip small integers as they might be references to counts/indexes or years
        if num.is_integer() and 0 <= num <= 10:
            continue

        # Check against multiple rounding levels
        num_variants = [round(num, 2), round(num, 1), round(num, 0)]

        is_grounded = any(variant in rounded_bundle_values for variant in num_variants)

        if not is_grounded:
            ungrounded.append(str(num))

    return ungrounded
