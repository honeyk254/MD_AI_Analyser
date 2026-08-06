"""Deterministic aggregation: AnalysisBundle -> GroundedSummary.

This module is the boundary between the classical layer and the reporting
layer. Everything the narrator is ever allowed to see is computed here, in pure
Python, from the bundle — so every number in a report has exactly one origin and
can be recomputed byte-for-byte from the same bundle.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..schemas.analysis_bundle import AnalysisBundle, MetricSummary
from ..schemas.summary import (
    GroundedSummary,
    MetricStatistics,
    QCSummary,
    TrajectoryFacts,
    TrendDirection,
)
from .provenance import canonical_hash
from .reference_ranges import compare_to_reference

# A metric is called "increasing"/"decreasing" only when the half-to-half drift
# clears both a relative and a noise-relative bar, so that a metric wandering
# inside its own scatter is reported as stable.
MIN_RELATIVE_DRIFT = 0.05
MIN_DRIFT_IN_STD = 0.5

# Changepoint search: segments shorter than this are not split, and at most
# MAX_CHANGEPOINTS splits are reported per metric.
MIN_SEGMENT_FRAMES = 10
MAX_CHANGEPOINTS = 3


def bundle_hash(bundle: AnalysisBundle) -> str:
    """Stable SHA256 over the bundle's canonical JSON form."""
    return canonical_hash(bundle.model_dump(mode="json"))


def summarize_bundle(bundle: AnalysisBundle) -> GroundedSummary:
    """Aggregate a bundle into the deterministic summary used for reporting."""
    metadata = bundle.trajectory_metadata

    metrics: Dict[str, MetricStatistics] = {}
    module_errors: Dict[str, str] = {}

    for module_name, module in bundle.modules.items():
        if module.error:
            module_errors[module_name] = module.error
            continue
        for metric_name, metric in module.scalar_metrics.items():
            metrics[metric_name] = _metric_statistics(metric_name, module_name, metric)

    comparisons = [
        compare_to_reference(
            metric=name,
            value=stats.mean,
            unit=stats.unit,
            n_residues=metadata.n_residues,
        )
        for name, stats in sorted(metrics.items())
    ]

    return GroundedSummary(
        run_id=bundle.run_id,
        bundle_hash=bundle_hash(bundle),
        trajectory=TrajectoryFacts(
            n_frames_analyzed=metadata.n_frames_analyzed,
            n_atoms=metadata.n_atoms,
            n_residues=metadata.n_residues,
            frame_interval_ps=metadata.timestep_ps,
            total_time_ns=metadata.total_time_ns,
            original_format=metadata.original_format,
            force_field=metadata.force_field,
        ),
        qc=_qc_summary(bundle),
        metrics=metrics,
        comparisons=comparisons,
        module_errors=module_errors,
        observations=_observations(metrics, module_errors),
    )


def _metric_statistics(
    metric_name: str, module_name: str, metric: MetricSummary
) -> MetricStatistics:
    """Derive statistics, drift, trend and changepoints for one metric."""
    cv: Optional[float] = None
    if metric.mean != 0:
        cv = abs(metric.std / metric.mean)

    first_half: Optional[float] = None
    second_half: Optional[float] = None
    drift: Optional[float] = None
    drift_percent: Optional[float] = None
    trend: TrendDirection = "stable"
    changepoints: List[int] = []

    series = metric.time_series
    if series and len(series) >= 4:
        values = np.asarray(series, dtype=np.float64)
        half = len(values) // 2
        first_half = float(np.mean(values[:half]))
        second_half = float(np.mean(values[half:]))
        drift = second_half - first_half
        if first_half != 0:
            drift_percent = 100.0 * drift / abs(first_half)
        trend = _trend(drift, first_half, metric.std)
        changepoints = detect_changepoints(values)

    return MetricStatistics(
        metric=metric_name,
        module=module_name,
        unit=metric.unit,
        mean=metric.mean,
        std=metric.std,
        min=metric.min,
        max=metric.max,
        n_frames=metric.n_frames,
        coefficient_of_variation=cv,
        first_half_mean=first_half,
        second_half_mean=second_half,
        drift=drift,
        drift_percent=drift_percent,
        trend=trend,
        changepoint_frames=changepoints,
    )


def _trend(drift: float, baseline: float, std: float) -> TrendDirection:
    relative_bar = MIN_RELATIVE_DRIFT * abs(baseline)
    noise_bar = MIN_DRIFT_IN_STD * std
    if abs(drift) <= max(relative_bar, noise_bar):
        return "stable"
    return "increasing" if drift > 0 else "decreasing"


def detect_changepoints(
    values: np.ndarray,
    min_segment: int = MIN_SEGMENT_FRAMES,
    max_changepoints: int = MAX_CHANGEPOINTS,
) -> List[int]:
    """Binary segmentation for mean shifts in a 1-D series.

    Each candidate split is kept only when it reduces the within-segment sum of
    squares by more than a BIC-style penalty (``log(n)`` times the variance of
    the whole series), which keeps noisy traces from producing changepoints.
    Returns frame indices in ascending order.
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n < 2 * min_segment:
        return []

    variance = float(np.var(values))
    if variance == 0.0:
        return []
    penalty = math.log(n) * variance

    changepoints: List[int] = []
    segments: List[Tuple[int, int]] = [(0, n)]

    while segments and len(changepoints) < max_changepoints:
        best: Optional[Tuple[float, int, Tuple[int, int]]] = None
        for segment in segments:
            split = _best_split(values[segment[0] : segment[1]], min_segment)
            if split is None:
                continue
            offset, gain = split
            if gain <= penalty:
                continue
            if best is None or gain > best[0]:
                best = (gain, segment[0] + offset, segment)

        if best is None:
            break

        _, index, segment = best
        segments.remove(segment)
        changepoints.append(index)
        for candidate in ((segment[0], index), (index, segment[1])):
            if candidate[1] - candidate[0] >= 2 * min_segment:
                segments.append(candidate)

    return sorted(changepoints)


def _best_split(
    values: np.ndarray, min_segment: int
) -> Optional[Tuple[int, float]]:
    """Best mean-shift split of ``values``, as ``(index, sse_reduction)``."""
    n = len(values)
    if n < 2 * min_segment:
        return None

    cumsum = np.cumsum(values)
    cumsum_sq = np.cumsum(values**2)
    total_sse = float(cumsum_sq[-1] - cumsum[-1] ** 2 / n)

    idx = np.arange(min_segment, n - min_segment + 1)
    left_n = idx.astype(np.float64)
    right_n = (n - idx).astype(np.float64)
    left_sum = cumsum[idx - 1]
    right_sum = cumsum[-1] - left_sum
    left_sse = cumsum_sq[idx - 1] - left_sum**2 / left_n
    right_sse = (cumsum_sq[-1] - cumsum_sq[idx - 1]) - right_sum**2 / right_n

    combined = left_sse + right_sse
    best = int(np.argmin(combined))
    return int(idx[best]), total_sse - float(combined[best])


def _qc_summary(bundle: AnalysisBundle) -> QCSummary:
    flags = bundle.qc_flags
    return QCSummary(
        is_equilibrated=flags.is_equilibrated,
        sufficient_frames=flags.sufficient_frames,
        equilibration_frame=_equilibration_frame(bundle),
        passed_checks=[f.check_name for f in flags.flags if f.passed],
        failed_checks=[f.check_name for f in flags.flags if not f.passed],
        details={f.check_name: f.details for f in flags.flags},
    )


def _equilibration_frame(bundle: AnalysisBundle) -> Optional[int]:
    rmsd = bundle.modules.get("rmsd")
    if rmsd is None or rmsd.error:
        return None
    frame = rmsd.data.get("equilibration_frame")
    return int(frame) if isinstance(frame, (int, float)) else None


def _observations(
    metrics: Dict[str, MetricStatistics], module_errors: Dict[str, str]
) -> List[str]:
    """Factual, template-generated observations — no interpretation."""
    observations: List[str] = []
    for name, stats in sorted(metrics.items()):
        observations.append(
            f"{name}: mean {stats.mean:.3f} {stats.unit} "
            f"(std {stats.std:.3f}, range {stats.min:.3f}-{stats.max:.3f}) "
            f"over {stats.n_frames} frames; trend {stats.trend}."
        )
        if stats.changepoint_frames:
            frames = ", ".join(str(f) for f in stats.changepoint_frames)
            observations.append(f"{name}: mean-shift changepoints at frames {frames}.")
    for module, error in sorted(module_errors.items()):
        observations.append(f"{module}: no results ({error}).")
    return observations
