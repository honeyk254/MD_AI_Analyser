"""Deterministic report summary helpers.

This keeps the LLM on a short leash: the report context is pre-aggregated in
pure Python and only exposes compact summaries, trends, and threshold checks.
"""

from typing import Any, Dict, List

from ..schemas.analysis_bundle import AnalysisBundle, ModuleResult


def build_report_summary(bundle: AnalysisBundle) -> Dict[str, Any]:
    """Build the compact report context used by the LLM and HTML report."""
    return {
        "run_id": bundle.run_id,
        "trajectory": _trajectory_summary(bundle),
        "qc": _qc_summary(bundle),
        "modules": {
            name: _module_summary(name, result) for name, result in bundle.modules.items()
        },
        "reference_ranges": _reference_ranges(bundle),
    }


def _trajectory_summary(bundle: AnalysisBundle) -> Dict[str, Any]:
    meta = bundle.trajectory_metadata
    return {
        "frames": meta.n_frames_analyzed,
        "atoms": meta.n_atoms,
        "residues": meta.n_residues,
        "timestep_ps": meta.timestep_ps,
        "total_time_ns": meta.total_time_ns,
        "format": meta.original_format,
        "force_field": meta.force_field,
    }


def _qc_summary(bundle: AnalysisBundle) -> Dict[str, Any]:
    flags = bundle.qc_flags
    return {
        "is_equilibrated": flags.is_equilibrated,
        "sufficient_frames": flags.sufficient_frames,
        "flags": [
            {
                "check_name": flag.check_name,
                "passed": flag.passed,
                "details": flag.details,
            }
            for flag in flags.flags
        ],
    }


def _module_summary(name: str, result: ModuleResult) -> Dict[str, Any]:
    scalars = {
        metric_name: {
            "mean": metric.mean,
            "std": metric.std,
            "min": metric.min,
            "max": metric.max,
            "unit": metric.unit,
            "n_frames": metric.n_frames,
        }
        for metric_name, metric in result.scalar_metrics.items()
    }

    trend = result.data.get("trend")
    return {
        "name": name,
        "version": result.version,
        "runtime_seconds": round(result.runtime_seconds, 3),
        "metrics": scalars,
        "trend": trend,
        "takeaway": _module_takeaway(name, result),
    }


def _module_takeaway(name: str, result: ModuleResult) -> str:
    if name == "rmsd" and "backbone_rmsd" in result.scalar_metrics:
        metric = result.scalar_metrics["backbone_rmsd"]
        return f"Backbone RMSD averaged {metric.mean:.2f} {metric.unit}."
    if name == "radius_of_gyration" and "radius_of_gyration" in result.scalar_metrics:
        metric = result.scalar_metrics["radius_of_gyration"]
        return f"Radius of gyration averaged {metric.mean:.2f} {metric.unit}."
    if name == "sasa" and "total_sasa" in result.scalar_metrics:
        metric = result.scalar_metrics["total_sasa"]
        return f"Total SASA averaged {metric.mean:.2f} {metric.unit}."
    if name == "hbonds" and "hbond_count" in result.scalar_metrics:
        metric = result.scalar_metrics["hbond_count"]
        return f"Hydrogen bonds averaged {metric.mean:.2f} per frame."
    if name == "contacts" and "persistent_contacts" in result.data:
        return f"Found {len(result.data['persistent_contacts'])} persistent contacts."
    if name == "secondary_structure":
        helix = result.scalar_metrics.get("helix_fraction")
        if helix:
            return f"Mean helix fraction was {helix.mean:.2f}."
    if name == "salt_bridges" and "salt_bridge_count" in result.scalar_metrics:
        metric = result.scalar_metrics["salt_bridge_count"]
        return f"Salt bridges averaged {metric.mean:.2f} per frame."
    return "Summary available."


def _reference_ranges(bundle: AnalysisBundle) -> Dict[str, str]:
    return {
        "rmsd": _reference_range_text(bundle, "rmsd"),
        "radius_of_gyration": _reference_range_text(bundle, "radius_of_gyration"),
        "sasa": _reference_range_text(bundle, "sasa"),
    }


def _reference_range_text(bundle: AnalysisBundle, metric_name: str) -> str:
    module = bundle.modules.get(metric_name)
    if not module:
        return "No reference comparison available."

    if metric_name == "rmsd" and "backbone_rmsd" in module.scalar_metrics:
        mean = module.scalar_metrics["backbone_rmsd"].mean
        return "Typically < 3.0 A for stable globular proteins." if mean < 3.0 else "May indicate larger conformational drift."
    if metric_name == "radius_of_gyration" and "radius_of_gyration" in module.scalar_metrics:
        std = module.scalar_metrics["radius_of_gyration"].std
        return "Should remain stable (std < 1.0 A)." if std < 1.0 else "Shows appreciable compactness changes."
    if metric_name == "sasa" and "total_sasa" in module.scalar_metrics:
        return "Should plateau during equilibrium."
    return "No literature reference range available."
