"""Compact summaries for the Phase 4 ML bundle."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .schemas import KineticEmbedding, MLAnalysisBundle, MSMSummary


def build_ml_summary(ml_bundle: Optional[MLAnalysisBundle]) -> Optional[Dict[str, Any]]:
    """Build a compact, report-friendly ML summary."""

    if ml_bundle is None:
        return None

    return {
        "status": ml_bundle.status,
        "refusal_reason": ml_bundle.refusal_reason,
        "gating": {
            "enabled": ml_bundle.gating.enabled,
            "passed": ml_bundle.gating.passed,
            "minimum_frames_required": ml_bundle.gating.minimum_frames_required,
            "observed_frames": ml_bundle.gating.observed_frames,
            "minimum_transition_count_required": ml_bundle.gating.minimum_transition_count_required,
            "observed_min_transition_count": ml_bundle.gating.observed_min_transition_count,
            "lag_frames": ml_bundle.gating.lag_frames,
            "lag_ps": ml_bundle.gating.lag_ps,
            "n_states": ml_bundle.gating.n_states,
            "ck_cutoff": ml_bundle.gating.ck_cutoff,
            "reasons": list(ml_bundle.gating.reasons),
        },
        "feature_summary": {
            "selection": ml_bundle.feature_summary.selection,
            "n_frames": ml_bundle.feature_summary.n_frames,
            "n_features": ml_bundle.feature_summary.n_features,
            "feature_preview": ml_bundle.feature_summary.feature_names[:8],
        },
        "pca": _embedding_summary(ml_bundle.pca),
        "tica": _embedding_summary(ml_bundle.tica),
        "msm": _msm_summary(ml_bundle.msm),
        "baseline_comparison": _baseline_summary(ml_bundle.baseline_comparison),
        "analysis_card": {
            "title": ml_bundle.analysis_card.title,
            "purpose": ml_bundle.analysis_card.purpose,
            "baseline_protocol": ml_bundle.analysis_card.baseline_protocol,
            "data_requirements": list(ml_bundle.analysis_card.data_requirements),
            "failure_modes": list(ml_bundle.analysis_card.failure_modes),
        },
        "takeaway": _ml_takeaway(ml_bundle),
    }


def _embedding_summary(embedding: Optional[KineticEmbedding]) -> Optional[Dict[str, Any]]:
    if embedding is None:
        return None
    return {
        "method": embedding.method,
        "n_components": embedding.n_components,
        "explained_variance": list(embedding.explained_variance),
        "component_labels": list(embedding.component_labels),
        "first_projection": embedding.projections[0][: min(3, len(embedding.projections[0]))]
        if embedding.projections
        else [],
    }


def _msm_summary(msm: Optional[MSMSummary]) -> Optional[Dict[str, Any]]:
    if msm is None:
        return None
    return {
        "method": msm.method,
        "lag_frames": msm.lag_frames,
        "lag_ps": msm.lag_ps,
        "n_states": msm.n_states,
        "ck_steps": msm.ck_steps,
        "ck_deviation": msm.ck_deviation,
        "is_markovian": msm.is_markovian,
        "minimum_pair_transition_count": msm.minimum_pair_transition_count,
        "stationary_distribution": list(msm.stationary_distribution),
        "implied_timescales_ps": list(msm.implied_timescales_ps[:3]),
    }


def _baseline_summary(comparison: Optional[Any]) -> Optional[Dict[str, Any]]:
    if comparison is None:
        return None
    return {
        "method": comparison.method,
        "state_agreement_nmi": comparison.state_agreement_nmi,
        "timescale_relative_error": comparison.timescale_relative_error,
        "summary": comparison.summary,
    }


def _ml_takeaway(ml_bundle: MLAnalysisBundle) -> str:
    if ml_bundle.status != "completed":
        return ml_bundle.refusal_reason or "ML analysis was not run."

    if ml_bundle.msm and ml_bundle.baseline_comparison:
        markovian = "passed" if ml_bundle.msm.is_markovian else "did not pass"
        return (
            f"TICA/MSM {markovian} CK validation at lag {ml_bundle.msm.lag_frames}; "
            f"baseline agreement was {ml_bundle.baseline_comparison.state_agreement_nmi:.2f} NMI."
        )

    return "ML summary available."
