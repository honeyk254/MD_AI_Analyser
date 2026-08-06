"""The three reporting tools.

The narrator has read-only access to exactly these three tools and to nothing
else — no trajectory, no bundle, no per-frame arrays. Adding a fourth tool is a
deliberate architectural decision, not a convenience: every extra tool is
another surface where an ungrounded number could enter a report.
"""

from typing import Any, Callable, Dict, List

from ..schemas.summary import GroundedSummary

GET_METRIC_SUMMARY = "get_metric_summary"
GET_QC_FLAGS = "get_qc_flags"
COMPARE_TO_REFERENCE_RANGES = "compare_to_reference_ranges"

TOOL_SPECS: List[Dict[str, Any]] = [
    {
        "name": GET_METRIC_SUMMARY,
        "description": (
            "Return the precomputed statistics for one metric: mean, standard "
            "deviation, min, max, frame count, half-to-half drift, trend label and "
            "changepoint frames. Use the exact numbers returned; never recompute or "
            "estimate them."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "metric_name": {
                    "type": "string",
                    "description": "Metric key, e.g. 'backbone_rmsd'.",
                }
            },
            "required": ["metric_name"],
        },
    },
    {
        "name": GET_QC_FLAGS,
        "description": (
            "Return the quality-control state of the run: equilibration status, "
            "frame sufficiency, which checks passed or failed, and any modules that "
            "produced no results."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": COMPARE_TO_REFERENCE_RANGES,
        "description": (
            "Return the deterministic comparison of a metric's mean against its "
            "documented reference band, including the verdict, the band, the source "
            "and the caveats. Report the verdict as given; do not derive your own."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "metric_name": {
                    "type": "string",
                    "description": "Metric key, e.g. 'backbone_rmsd'.",
                }
            },
            "required": ["metric_name"],
        },
    },
]


class ReportTools:
    """Deterministic accessors over a :class:`GroundedSummary`."""

    def __init__(self, summary: GroundedSummary):
        self.summary = summary

    # ----- tool implementations ----- #

    def get_metric_summary(self, metric_name: str) -> Dict[str, Any]:
        stats = self.summary.metrics.get(metric_name)
        if stats is None:
            return {
                "error": f"Unknown metric '{metric_name}'.",
                "available_metrics": self.metric_names(),
            }
        return stats.model_dump(mode="json")

    def get_qc_flags(self) -> Dict[str, Any]:
        payload = self.summary.qc.model_dump(mode="json")
        payload["module_errors"] = dict(self.summary.module_errors)
        return payload

    def compare_to_reference_ranges(self, metric_name: str) -> Dict[str, Any]:
        for comparison in self.summary.comparisons:
            if comparison.metric == metric_name:
                return comparison.model_dump(mode="json")
        return {
            "error": f"No comparison available for '{metric_name}'.",
            "available_metrics": self.metric_names(),
        }

    # ----- dispatch ----- #

    def metric_names(self) -> List[str]:
        return sorted(self.summary.metrics)

    def dispatch(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke a tool by name, rejecting anything outside the three."""
        handlers: Dict[str, Callable[..., Dict[str, Any]]] = {
            GET_METRIC_SUMMARY: self.get_metric_summary,
            GET_QC_FLAGS: self.get_qc_flags,
            COMPARE_TO_REFERENCE_RANGES: self.compare_to_reference_ranges,
        }
        handler = handlers.get(name)
        if handler is None:
            return {"error": f"Tool '{name}' does not exist.", "tools": sorted(handlers)}
        try:
            return handler(**arguments)
        except TypeError as exc:
            return {"error": f"Invalid arguments for '{name}': {exc}"}
