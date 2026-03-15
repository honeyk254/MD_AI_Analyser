from __future__ import annotations

"""Markov State Model (MSM) construction.

Builds an exploratory kinetic model from slow trajectory features,
computes the stationary distribution, identifies metastable states, and
derives implied timescales and mean first-passage times.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.cluster import KMeans

from ..utils.trajectory_utils import (
    select_ca_atoms,
    collect_ca_coords_flat,
)
from ..utils.ml_feature_utils import (
    pca_reduce,
    find_optimal_k,
    standardise_features,
    set_global_seed,
)

logger = logging.getLogger("md_ai_analyzer")

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_msm(
    universe: Any,
    n_states: Optional[int] = None,
    lag_time: int = 5,
    reversible: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Build a Markov State Model from trajectory clustering.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        Loaded trajectory universe.
    n_states : int, optional
        Number of microstates.  If *None*, the optimal *k* is determined
        automatically via silhouette analysis.
    lag_time : int
        Lag time (in frames) used to build the count matrix.
    reversible : bool
        Whether to enforce detailed balance (reversibility) by
        symmetrising the count matrix.
    **kwargs
        Additional keyword arguments (unused; accepted for orchestrator
        compatibility).

    Returns
    -------
    dict
        Keys:
        - ``transition_matrix`` : row-stochastic transition probabilities
        - ``stationary_distribution`` : equilibrium probability per state
        - ``implied_timescales`` : list of implied timescales
        - ``eigenvalues`` : eigenvalues of *T*
        - ``metastable_states`` : ranked list of metastable-state dicts
        - ``mfpt`` : mean first-passage time matrix
        - ``n_states`` : number of MSM states
        - ``lag_time`` : lag time used
        - ``labels`` : per-frame cluster labels
        - ``feature_projection`` : slow feature space used for clustering
        - ``implied_timescale_convergence`` : implied timescales at multiple lag times
        - ``chapman_kolmogorov`` : Chapman-Kolmogorov self-consistency test results
        - ``is_markovian`` : conservative exploratory quality flag
    """
    set_global_seed(42)
    try:
        # ── Coordinate collection & slow-feature projection ──────
        ca = select_ca_atoms(universe)
        coords = collect_ca_coords_flat(universe, atoms=ca)

        if len(coords) < 20:
            logger.warning("Only %d frames; need >= 20 for MSM.", len(coords))
            return {"error": "Not enough frames for MSM"}

        coords_scaled, _, _ = standardise_features(coords)
        reduced_5d, feature_method = _slow_feature_projection(
            coords_scaled, lag_time=lag_time, n_components=5
        )

        # ── Determine n_states via silhouette if not given ───────
        if n_states is None:
            n_states, _scores = find_optimal_k(
                reduced_5d,
                k_min=2,
                k_max=min(11, len(reduced_5d) // 5),
                random_state=42,
            )
            logger.info("Optimal k=%d selected via silhouette.", n_states)

        # ── Cluster ──────────────────────────────────────────────
        km = KMeans(n_clusters=n_states, n_init=10, random_state=42)
        labels: np.ndarray = km.fit_predict(reduced_5d)

        # ── Build transition matrix ──────────────────────────────
        T = _build_transition_matrix(labels, n_states, lag_time, reversible=reversible)

        if T is None:
            return {"error": "Could not build transition matrix"}

        # Validate stochastic matrix (rows should sum to 1)
        row_sum_check = T.sum(axis=1)
        if not np.allclose(row_sum_check, 1.0, atol=1e-8):
            deviation = float(np.max(np.abs(row_sum_check - 1.0)))
            logger.warning(
                "Transition matrix row sums deviate from 1.0 by up to %.2e; renormalising.",
                deviation,
            )
            T = T / T.sum(axis=1, keepdims=True)

        # ── Stationary distribution ──────────────────────────────
        eigenvalues_T, eigenvectors_T = np.linalg.eig(T.T)
        idx = int(np.argmin(np.abs(eigenvalues_T - 1.0)))
        pi = np.real(eigenvectors_T[:, idx])
        pi = np.abs(pi)
        pi /= pi.sum()

        # ── Implied timescales ───────────────────────────────────
        T_eigenvalues: np.ndarray = np.sort(np.real(np.linalg.eigvals(T)))[::-1]
        timescales: List[float] = []
        for ev in T_eigenvalues[1:]:
            if 0 < ev < 1:
                timescales.append(float(-lag_time / np.log(ev)))
            else:
                timescales.append(0.0)

        # ── Mean first passage times ─────────────────────────────
        mfpt = _compute_mfpt(T, n_states)

        # ── Metastable states (high self-transition probability) ─
        metastable: List[Dict[str, Any]] = []
        for i in range(n_states):
            metastable.append(
                {
                    "state": i,
                    "self_transition": round(float(T[i, i]), 3),
                    "population": round(float(pi[i]), 3),
                }
            )
        metastable.sort(key=lambda x: -x["self_transition"])

        logger.info(
            "MSM built: %d states, lag=%d, reversible=%s, top timescale=%.1f.",
            n_states,
            lag_time,
            reversible,
            timescales[0] if timescales else 0.0,
        )

        # ── Implied timescale convergence (lag-time sweep) ─────
        its_convergence = _implied_timescale_sweep(
            labels, n_states, lag_time, len(labels), reversible=reversible
        )

        # ── Chapman-Kolmogorov self-consistency test ───────────
        ck_test = _chapman_kolmogorov_test(T, lag_time, n_states, labels, reversible=reversible)
        markovianity = _assess_markovianity(ck_test, its_convergence)
        ck_test.update(markovianity)

        return {
            "transition_matrix": T.tolist(),
            "stationary_distribution": pi.tolist(),
            "implied_timescales": timescales,
            "eigenvalues": T_eigenvalues.tolist(),
            "metastable_states": metastable,
            "mfpt": mfpt.tolist(),
            "n_states": n_states,
            "lag_time": lag_time,
            "labels": labels.tolist(),
            "feature_projection": feature_method,
            "implied_timescale_convergence": its_convergence,
            "chapman_kolmogorov": ck_test,
            "is_markovian": ck_test["is_markovian"],
            "caveat": (
                "MSM quality depends on feature choice, state discretisation, lag-time "
                "selection, and sampling depth. Treat this model as exploratory unless "
                "CK consistency and implied-timescale convergence are both satisfactory."
            ),
        }

    except Exception as e:
        logger.exception("MSM construction failed: %s", e)
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _compute_mfpt(T: np.ndarray, n_states: int) -> np.ndarray:
    """Compute mean first-passage times between all state pairs.

    Parameters
    ----------
    T : np.ndarray
        Row-stochastic transition matrix of shape ``(n_states, n_states)``.
    n_states : int
        Number of states.

    Returns
    -------
    np.ndarray
        MFPT matrix of shape ``(n_states, n_states)`` where entry
        ``[i, j]`` is the expected number of steps to reach state *j*
        starting from state *i*.
    """
    mfpt = np.zeros((n_states, n_states), dtype=np.float64)
    for target in range(n_states):
        indices = [i for i in range(n_states) if i != target]
        if not indices:
            continue
        T_sub = T[np.ix_(indices, indices)]
        A = np.eye(len(indices)) - T_sub
        try:
            m = np.linalg.solve(A, np.ones(len(indices)))
            for k, idx in enumerate(indices):
                mfpt[idx, target] = max(0.0, float(m[k]))
        except np.linalg.LinAlgError:
            logger.debug("Singular matrix when computing MFPT to state %d.", target)
    return mfpt


def _slow_feature_projection(
    coords_scaled: np.ndarray,
    lag_time: int,
    n_components: int,
) -> tuple[np.ndarray, str]:
    """Project coordinates into a slow feature space for state clustering."""
    try:
        reduced = _tica_projection(coords_scaled, lag_time=lag_time, n_components=n_components)
        if reduced.shape[1] >= 1:
            return reduced[:, : min(n_components, reduced.shape[1])], "tICA"
    except Exception as exc:
        logger.warning("tICA feature projection failed, falling back to PCA: %s", exc)

    reduced, _pca_model = pca_reduce(coords_scaled, n_components=max(2, n_components))
    return reduced[:, : min(n_components, reduced.shape[1])], "PCA"


def _tica_projection(
    coords_scaled: np.ndarray,
    lag_time: int,
    n_components: int,
) -> np.ndarray:
    """Project data into the leading tICA coordinates."""
    n_frames = coords_scaled.shape[0]
    if n_frames <= lag_time + 1:
        raise ValueError("Trajectory too short for requested MSM lag time")

    X = coords_scaled - coords_scaled.mean(axis=0)
    X_t = X[: n_frames - lag_time]
    X_tau = X[lag_time:]

    C0 = (X.T @ X) / max(n_frames - 1, 1)
    Ctau = (X_t.T @ X_tau) / max(n_frames - lag_time - 1, 1)
    Ctau = 0.5 * (Ctau + Ctau.T)

    reg = 1e-6 * np.eye(C0.shape[0])
    evals, evecs = np.linalg.eigh(np.linalg.pinv(C0 + reg) @ Ctau)
    order = np.argsort(np.real(evals))[::-1]
    evecs = np.real(evecs[:, order])
    n_keep = min(n_components, evecs.shape[1])
    return X @ evecs[:, :n_keep]


def _build_transition_matrix(
    labels: np.ndarray, n_states: int, lag: int, reversible: bool = True
) -> Optional[np.ndarray]:
    """Build a transition matrix at a given lag time.

    Parameters
    ----------
    labels : np.ndarray
        Per-frame cluster labels.
    n_states : int
        Number of states.
    lag : int
        Lag time in frames.
    reversible : bool
        Whether to enforce detailed balance by symmetrising the count matrix.
    """
    n = len(labels)
    if lag >= n:
        return None
    C = np.zeros((n_states, n_states), dtype=np.float64)
    src = labels[: n - lag]
    dst = labels[lag:]
    np.add.at(C, (src, dst), 1)

    if reversible:
        C = (C + C.T) / 2.0  # enforce detailed balance

    row_sums = C.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return C / row_sums


def _implied_timescale_sweep(
    labels: np.ndarray,
    n_states: int,
    base_lag: int,
    n_frames: int,
    reversible: bool = True,
) -> List[Dict[str, Any]]:
    """Compute implied timescales at multiple lag times for convergence check.

    Returns a list of dicts with ``lag`` and ``timescales`` keys.
    """
    max_lag = min(n_frames // 4, base_lag * 10)
    lag_times = sorted(set(
        [max(1, base_lag // 2), base_lag, base_lag * 2, base_lag * 4, max_lag]
    ))
    results: List[Dict[str, Any]] = []
    for lag in lag_times:
        if lag >= n_frames - 1:
            continue
        T_lag = _build_transition_matrix(labels, n_states, lag, reversible=reversible)
        if T_lag is None:
            continue
        evals = np.sort(np.real(np.linalg.eigvals(T_lag)))[::-1]
        its: List[float] = []
        for ev in evals[1:]:
            if 0 < ev < 1:
                its.append(round(float(-lag / np.log(ev)), 2))
            else:
                its.append(0.0)
        results.append({"lag": lag, "timescales": its[:5]})
    return results


def _chapman_kolmogorov_test(
    T: np.ndarray,
    lag_time: int,
    n_states: int,
    labels: np.ndarray,
    n_steps: int = 5,
    reversible: bool = True,
) -> Dict[str, Any]:
    """Chapman-Kolmogorov self-consistency test.

    Compares T(k*tau) predicted by T(tau)^k against the directly
    estimated T(k*tau) from trajectory data.  If the MSM is Markovian,
    these should agree.

    Returns predicted and estimated diagonal elements for each step,
    plus a max-absolute-deviation metric.
    """
    predicted: List[List[float]] = []
    estimated: List[List[float]] = []
    steps = list(range(1, n_steps + 1))
    matrix_deviations: List[float] = []
    diagonal_deviations: List[float] = []

    T_power = np.eye(n_states)
    for k in steps:
        T_power = T_power @ T
        predicted.append([round(float(T_power[i, i]), 4) for i in range(n_states)])

        # Directly estimate T(k * tau) from trajectory data
        T_direct = _build_transition_matrix(labels, n_states, lag_time * k, reversible=reversible)
        if T_direct is not None:
            estimated.append([round(float(T_direct[i, i]), 4) for i in range(n_states)])
            diff = np.abs(T_power - T_direct)
            matrix_deviations.append(float(np.max(diff)))
            diagonal_deviations.append(float(np.max(np.abs(np.diag(T_power) - np.diag(T_direct)))))
        else:
            estimated.append([0.0] * n_states)
            matrix_deviations.append(float("inf"))
            diagonal_deviations.append(float("inf"))

    # Compute max absolute deviation between predicted and estimated
    max_diagonal_deviation = max(diagonal_deviations) if diagonal_deviations else float("inf")
    max_matrix_deviation = max(matrix_deviations) if matrix_deviations else float("inf")

    return {
        "steps": steps,
        "lag_time": lag_time,
        "predicted_self_transition": predicted,
        "estimated_self_transition": estimated,
        "max_deviation": round(max_diagonal_deviation, 4),
        "max_diagonal_deviation": round(max_diagonal_deviation, 4),
        "max_matrix_deviation": round(max_matrix_deviation, 4),
        "description": (
            "Chapman-Kolmogorov test: compares T(tau)^k (predicted) against "
            "directly estimated T(k*tau) (estimated). Small full-matrix and "
            "diagonal deviations, together with lag-time-stable implied "
            "timescales, support a usable Markov model."
        ),
    }


def _assess_markovianity(
    ck_test: Dict[str, Any],
    its_convergence: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Assess Markovianity using CK consistency and timescale stability."""
    max_matrix_deviation = float(ck_test.get("max_matrix_deviation", float("inf")))
    max_diagonal_deviation = float(
        ck_test.get("max_diagonal_deviation", float("inf"))
    )

    primary_timescales: List[float] = []
    for item in its_convergence:
        values = item.get("timescales", [])
        if values:
            first = float(values[0])
            if first > 0:
                primary_timescales.append(first)

    if len(primary_timescales) >= 2:
        ts_ratio = max(primary_timescales) / max(min(primary_timescales), 1e-8)
    else:
        ts_ratio = float("inf")

    timescales_stable = ts_ratio <= 1.5
    ck_consistent = max_matrix_deviation < 0.1 and max_diagonal_deviation < 0.1
    is_markovian = ck_consistent and timescales_stable

    return {
        "is_markovian": is_markovian,
        "timescale_stability_ratio": (
            round(ts_ratio, 4) if np.isfinite(ts_ratio) else None
        ),
        "timescales_stable": timescales_stable,
        "assessment": (
            "usable"
            if is_markovian
            else "exploratory"
        ),
    }
