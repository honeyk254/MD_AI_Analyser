"""Phase 4 kinetic analysis helpers.

This module keeps the Phase 4 implementation opt-in and self-contained:
TICA/MSM results live in a separate ML bundle and are summarized back into
the existing reporting path only when explicitly enabled.
"""

from __future__ import annotations

from typing import List, Tuple

import MDAnalysis as mda
import numpy as np

from ..schemas.analysis_bundle import AnalysisBundle
from ..schemas.api import AnalysisRequest
from .schemas import (
    AnalysisCard,
    BaselineComparison,
    FeatureSummary,
    KineticEmbedding,
    MLAnalysisBundle,
    MLGatingSummary,
    MSMSummary,
)
from .vampnets import run_vampnet_ablation


def run_phase4_ml_analysis(
    universe: mda.Universe,
    classical_bundle: AnalysisBundle,
    request: AnalysisRequest,
) -> MLAnalysisBundle:
    """Run the opt-in Phase 4 kinetic analysis."""

    min_frames = max(request.ml_min_frames, 2)
    lag_frames = max(request.ml_lag_frames, 1)
    n_states = max(request.ml_n_states, 2)
    min_transitions = max(request.ml_min_transition_count, 1)

    gating_reasons: List[str] = []
    notes: List[str] = []
    try:
        feature_matrix, feature_names, frame_times_ps = _extract_features(universe)
    except Exception as exc:
        feature_matrix = np.empty((0, 0), dtype=np.float64)
        feature_names = []
        frame_times_ps = []
        gating_reasons.append(f"Unable to extract kinetic features: {exc}")
    observed_frames = feature_matrix.shape[0]
    if not request.enable_ml:
        gating_reasons.append("ML analysis is disabled for this run.")
    if observed_frames < min_frames:
        gating_reasons.append(
            f"Observed {observed_frames} frames, below the minimum of {min_frames}."
        )
    if observed_frames <= lag_frames:
        gating_reasons.append(
            f"Lag of {lag_frames} frames needs more than {observed_frames} frames."
        )
    if classical_bundle.trajectory_metadata.timestep_ps <= 0:
        gating_reasons.append("Trajectory timestep is unavailable, so kinetic timescales cannot be reported.")

    gating_passed = not gating_reasons
    pca = None
    tica = None
    msm = None
    baseline = None
    vampnet_ablation = None
    observed_min_transition_count = 0
    ck_deviation = 0.0
    status = "blocked"
    refusal_reason = "; ".join(gating_reasons) if gating_reasons else None

    try:
        if gating_passed:
            pca_scores, pca_variance = _run_pca(feature_matrix, n_components=min(3, feature_matrix.shape[1]))
            tica_scores, tica_variance = _run_tica(feature_matrix, lag_frames=lag_frames, n_components=min(3, feature_matrix.shape[1]))

            effective_states = min(n_states, max(2, len(feature_matrix) // max(lag_frames, 1)))
            pca_labels = _cluster_embeddings(pca_scores[:, : min(2, pca_scores.shape[1])], effective_states)
            tica_labels = _cluster_embeddings(tica_scores[:, : min(2, tica_scores.shape[1])], effective_states)

            pca = KineticEmbedding(
                method="pca",
                n_components=pca_scores.shape[1],
                explained_variance=[float(v) for v in pca_variance],
                component_labels=[f"PC{i+1}" for i in range(pca_scores.shape[1])],
                projections=pca_scores.tolist(),
            )
            tica = KineticEmbedding(
                method="tica",
                n_components=tica_scores.shape[1],
                explained_variance=[float(v) for v in tica_variance],
                component_labels=[f"tIC{i+1}" for i in range(tica_scores.shape[1])],
                projections=tica_scores.tolist(),
            )

            msm, observed_min_transition_count, ck_deviation, msm_notes = _build_msm(
                labels=tica_labels,
                lag_frames=lag_frames,
                lag_ps=request.ml_lag_frames * classical_bundle.trajectory_metadata.timestep_ps,
                n_states=effective_states,
                ck_threshold=request.ml_ck_threshold,
            )
            notes.extend(msm_notes)

            if observed_min_transition_count < min_transitions:
                gating_reasons.append(
                    f"Observed minimum pair transitions {observed_min_transition_count}, below the minimum of {min_transitions}."
                )
                gating_passed = False
                refusal_reason = "; ".join(gating_reasons)
                status = "blocked"
                pca = None
                tica = None
                msm = None
            else:
                baseline = _build_baseline_comparison(
                    pca_labels=pca_labels,
                    tica_labels=tica_labels,
                    pca_scores=pca_scores,
                    tica_scores=tica_scores,
                    lag_frames=lag_frames,
                    lag_ps=request.ml_lag_frames * classical_bundle.trajectory_metadata.timestep_ps,
                )
                status = "completed"
                if msm and not msm.is_markovian:
                    notes.append(
                        f"CK deviation {ck_deviation:.3f} exceeded the threshold of "
                        f"{request.ml_ck_threshold:.3f}; reported is_markovian=False."
                    )
                vampnet_ablation = run_vampnet_ablation(
                    feature_matrix=feature_matrix,
                    lag_frames=lag_frames,
                    lag_ps=request.ml_lag_frames * classical_bundle.trajectory_metadata.timestep_ps,
                    n_states=effective_states,
                    tica_labels=tica_labels,
                    tica_leading_timescale_ps=(
                        msm.implied_timescales_ps[0] if msm and msm.implied_timescales_ps else None
                    ),
                )
                if vampnet_ablation and not vampnet_ablation.available:
                    notes.append(vampnet_ablation.summary)
    except Exception as exc:
        gating_reasons.append(f"ML analysis failed: {exc}")
        gating_passed = False
        status = "blocked"
        refusal_reason = "; ".join(gating_reasons)
        pca = None
        tica = None
        msm = None
        baseline = None

    feature_summary = FeatureSummary(
        selection=_feature_selection_description(universe),
        n_frames=observed_frames,
        n_features=feature_matrix.shape[1],
        feature_names=feature_names,
        time_ps=frame_times_ps,
    )

    gating = MLGatingSummary(
        enabled=request.enable_ml,
        passed=gating_passed,
        minimum_frames_required=min_frames,
        observed_frames=observed_frames,
        minimum_transition_count_required=min_transitions,
        observed_min_transition_count=observed_min_transition_count,
        lag_frames=lag_frames,
        lag_ps=request.ml_lag_frames * classical_bundle.trajectory_metadata.timestep_ps,
        n_states=n_states,
        ck_cutoff=request.ml_ck_threshold,
        reasons=list(gating_reasons),
    )

    analysis_card = AnalysisCard(
        title="TICA + MSM Phase 4 module",
        purpose="Estimate slow collective motions and metastable kinetics from aligned trajectory features.",
        literature_basis=[
            "Pérez-Hernández et al. (TICA)",
            "Bowman, Noé, and Pande (MSM)",
            "Prinz et al. (MSM validation)",
        ],
        data_requirements=[
            f"At least {min_frames} analyzed frames",
            f"At least {min_transitions} observed transitions per state pair",
            f"Non-zero lag of {lag_frames} frames",
        ],
        failure_modes=[
            "Too few frames for stable transition statistics",
            "Too few transitions for a reliable MSM",
            "CK deviation above the declared cutoff",
        ],
        baseline_protocol="Compare PCA-clustered states against TICA-clustered states and report state/timescale agreement side-by-side.",
    )

    return MLAnalysisBundle(
        run_id=classical_bundle.run_id,
        source_run_id=classical_bundle.run_id,
        status=status,
        gating=gating,
        feature_summary=feature_summary,
        pca=pca,
        tica=tica,
        msm=msm,
        baseline_comparison=baseline,
        vampnet_ablation=vampnet_ablation,
        analysis_card=analysis_card,
        refusal_reason=refusal_reason,
        notes=notes,
    )


def _feature_selection_description(universe: mda.Universe) -> str:
    selection = universe.select_atoms("protein and name CA")
    if len(selection) == 0:
        selection = universe.select_atoms("protein and backbone")
    if len(selection) == 0:
        selection = universe.select_atoms("all")
    return f"{len(selection)} atoms selected for kinetic features"


def _extract_features(universe: mda.Universe) -> Tuple[np.ndarray, List[str], List[float]]:
    selection = universe.select_atoms("protein and name CA")
    if len(selection) == 0:
        selection = universe.select_atoms("protein and backbone")
    if len(selection) == 0:
        selection = universe.select_atoms("all")

    coordinates: List[np.ndarray] = []
    frame_times_ps: List[float] = []
    reference = None
    for ts in universe.trajectory:
        frame = np.asarray(selection.positions, dtype=np.float64)
        if reference is None:
            reference = frame.copy()
        aligned = _kabsch_align(reference, frame)
        coordinates.append(aligned.reshape(-1))
        frame_times_ps.append(float(ts.time) if ts.time is not None else float(len(frame_times_ps)))

    if not coordinates:
        raise ValueError("Unable to extract any kinetic features from the trajectory.")

    feature_matrix = np.asarray(coordinates, dtype=np.float64)
    feature_matrix -= feature_matrix.mean(axis=0, keepdims=True)
    scale = feature_matrix.std(axis=0, keepdims=True)
    scale[scale == 0] = 1.0
    feature_matrix = feature_matrix / scale

    feature_names = [f"{atom.resname}:{atom.resid}:{atom.name}" for atom in selection]
    return feature_matrix, feature_names, frame_times_ps


def _kabsch_align(reference: np.ndarray, mobile: np.ndarray) -> np.ndarray:
    ref_centered = reference - reference.mean(axis=0, keepdims=True)
    mob_centered = mobile - mobile.mean(axis=0, keepdims=True)
    covariance = mob_centered.T @ ref_centered
    v, _, wt = np.linalg.svd(covariance)
    rotation = v @ wt
    if np.linalg.det(rotation) < 0:
        wt[-1, :] *= -1
        rotation = v @ wt
    return mob_centered @ rotation


def _run_pca(features: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray]:
    centered = features - features.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components]
    scores = centered @ components.T
    explained = (singular_values[:n_components] ** 2) / max(float(len(features) - 1), 1.0)
    total = float(np.sum((singular_values ** 2) / max(float(len(features) - 1), 1.0)))
    variance_ratio = explained / total if total > 0 else np.zeros_like(explained)
    return scores, variance_ratio


def _run_tica(features: np.ndarray, lag_frames: int, n_components: int) -> Tuple[np.ndarray, np.ndarray]:
    if lag_frames <= 0 or features.shape[0] <= lag_frames:
        raise ValueError("Not enough frames to compute TICA.")

    centered = features - features.mean(axis=0, keepdims=True)
    x0 = centered[:-lag_frames]
    xt = centered[lag_frames:]
    c0 = (x0.T @ x0) / float(len(x0))
    c_tau = (x0.T @ xt) / float(len(x0))
    c_tau = 0.5 * (c_tau + c_tau.T)
    regularizer = 1e-8 * np.eye(c0.shape[0])
    cholesky = np.linalg.cholesky(c0 + regularizer)
    inv_chol = np.linalg.inv(cholesky)
    sym_op = inv_chol @ c_tau @ inv_chol.T
    sym_op = 0.5 * (sym_op + sym_op.T)
    eigenvalues, eigenvectors = np.linalg.eigh(sym_op)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.real(eigenvalues[order])
    eigenvectors = np.real(eigenvectors[:, order])
    weights = inv_chol.T @ eigenvectors[:, :n_components]
    scores = centered @ weights
    return scores, np.clip(eigenvalues[:n_components], -1.0, 1.0)


def _cluster_embeddings(embedding: np.ndarray, n_states: int) -> np.ndarray:
    if embedding.ndim != 2:
        raise ValueError("Embedding must be two-dimensional.")
    if embedding.shape[0] < n_states:
        raise ValueError("Not enough samples to cluster into the requested states.")

    centers = embedding[np.linspace(0, embedding.shape[0] - 1, n_states, dtype=int)].copy()
    labels = np.zeros(embedding.shape[0], dtype=int)
    for _ in range(50):
        distances = np.linalg.norm(embedding[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(distances, axis=1)
        new_centers = centers.copy()
        for idx in range(n_states):
            members = embedding[new_labels == idx]
            if len(members) > 0:
                new_centers[idx] = members.mean(axis=0)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        centers = new_centers
    return labels


def _build_msm(
    labels: np.ndarray,
    lag_frames: int,
    lag_ps: float,
    n_states: int,
    ck_threshold: float,
) -> Tuple[MSMSummary, int, float, List[str]]:
    counts = np.zeros((n_states, n_states), dtype=int)
    for idx in range(len(labels) - lag_frames):
        counts[labels[idx], labels[idx + lag_frames]] += 1

    transition_counts = counts.copy()
    observed_pairs = counts + counts.T
    off_diag = observed_pairs[~np.eye(n_states, dtype=bool)]
    positive_off_diag = off_diag[off_diag > 0]
    observed_min_transition_count = int(positive_off_diag.min()) if len(positive_off_diag) else 0

    row_sums = counts.sum(axis=1, keepdims=True).astype(np.float64)
    if np.any(row_sums == 0):
        row_sums[row_sums == 0] = 1.0
    transition_matrix = counts.astype(np.float64) / row_sums

    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    order = np.argsort(np.real(eigenvalues))[::-1]
    eigenvalues = np.real(eigenvalues[order])
    eigenvectors = np.real(eigenvectors[:, order])
    stationary = np.abs(eigenvectors[:, 0])
    stationary = stationary / stationary.sum()

    implied = []
    for eigenvalue in eigenvalues[1:n_states]:
        if 0 < eigenvalue < 1:
            implied.append(float(-lag_ps / np.log(eigenvalue)))

    ck_steps = 2 if len(labels) > 2 * lag_frames else 1
    direct_counts = np.zeros_like(counts, dtype=float)
    for idx in range(len(labels) - lag_frames * ck_steps):
        direct_counts[labels[idx], labels[idx + lag_frames * ck_steps]] += 1
    direct_row_sums = direct_counts.sum(axis=1, keepdims=True)
    direct_row_sums[direct_row_sums == 0] = 1.0
    direct_transition = direct_counts / direct_row_sums
    predicted_transition = np.linalg.matrix_power(transition_matrix, ck_steps)
    # float() keeps the norms concrete across numpy stub versions (2.4 vs 2.5)
    ck_deviation = float(np.linalg.norm(predicted_transition - direct_transition, ord="fro")) / max(
        float(np.linalg.norm(direct_transition, ord="fro")), 1e-12
    )
    is_markovian = ck_deviation <= ck_threshold

    notes = []
    if not is_markovian:
        notes.append(
            f"CK deviation {ck_deviation:.3f} exceeded the threshold of {ck_threshold:.3f}."
        )
    else:
        notes.append(
            f"CK deviation {ck_deviation:.3f} remained within the threshold of {ck_threshold:.3f}."
        )

    msm = MSMSummary(
        lag_frames=lag_frames,
        lag_ps=lag_ps,
        n_states=n_states,
        transition_counts=transition_counts.tolist(),
        transition_matrix=transition_matrix.tolist(),
        stationary_distribution=stationary.tolist(),
        implied_timescales_ps=implied,
        ck_steps=ck_steps,
        ck_deviation=ck_deviation,
        is_markovian=is_markovian,
        minimum_pair_transition_count=observed_min_transition_count,
        state_populations=stationary.tolist(),
    )
    return msm, observed_min_transition_count, ck_deviation, notes


def _build_baseline_comparison(
    pca_labels: np.ndarray,
    tica_labels: np.ndarray,
    pca_scores: np.ndarray,
    tica_scores: np.ndarray,
    lag_frames: int,
    lag_ps: float,
) -> BaselineComparison:
    nmi = _normalized_mutual_information(pca_labels, tica_labels)
    pca_msm = _transition_timescales(pca_labels, lag_frames, lag_ps)
    tica_msm = _transition_timescales(tica_labels, lag_frames, lag_ps)
    timescale_error = None
    if pca_msm and tica_msm:
        timescale_error = abs(pca_msm[0] - tica_msm[0]) / max(max(pca_msm[0], tica_msm[0]), 1e-12)

    summary = (
        f"PCA and TICA state assignments agree at NMI {nmi:.2f}. "
        f"Leading implied timescale relative error: {timescale_error:.2f}."
        if timescale_error is not None
        else f"PCA and TICA state assignments agree at NMI {nmi:.2f}."
    )

    return BaselineComparison(
        state_agreement_nmi=float(nmi),
        timescale_relative_error=timescale_error,
        pca_state_labels=[int(v) for v in pca_labels.tolist()],
        tica_state_labels=[int(v) for v in tica_labels.tolist()],
        summary=summary,
    )


def _transition_timescales(labels: np.ndarray, lag_frames: int, lag_ps: float) -> List[float]:
    n_states = int(labels.max()) + 1
    counts = np.zeros((n_states, n_states), dtype=float)
    for idx in range(len(labels) - lag_frames):
        counts[labels[idx], labels[idx + lag_frames]] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    transition_matrix = counts / row_sums
    eigenvalues = np.real(np.linalg.eigvals(transition_matrix))
    eigenvalues = np.sort(eigenvalues)[::-1]
    timescales = []
    for eigenvalue in eigenvalues[1:n_states]:
        if 0 < eigenvalue < 1:
            timescales.append(float(-lag_ps / np.log(eigenvalue)))
    return timescales


def _normalized_mutual_information(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    a = np.asarray(labels_a, dtype=int)
    b = np.asarray(labels_b, dtype=int)
    n = len(a)
    if n == 0 or len(b) != n:
        return 0.0
    labels_a = np.unique(a)
    labels_b = np.unique(b)
    contingency = np.zeros((len(labels_a), len(labels_b)), dtype=float)
    for i, label_a in enumerate(labels_a):
        for j, label_b in enumerate(labels_b):
            contingency[i, j] = np.sum((a == label_a) & (b == label_b))
    pxy = contingency / n
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = px @ py
        valid = pxy > 0
        mi = float(np.sum(pxy[valid] * np.log(pxy[valid] / expected[valid])))
    hx = float(-np.sum(px[px > 0] * np.log(px[px > 0])))
    hy = float(-np.sum(py[py > 0] * np.log(py[py > 0])))
    if hx == 0 or hy == 0:
        return 0.0
    return mi / float(np.sqrt(hx * hy))
