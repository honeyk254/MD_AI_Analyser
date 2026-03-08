from __future__ import annotations

"""Markov State Model (MSM) construction.

Builds a transition matrix from trajectory clustering, computes the
stationary distribution, identifies metastable states, and derives implied
timescales and mean first-passage times.
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
    """
    set_global_seed(42)
    try:
        # ── Coordinate collection & PCA ──────────────────────────
        ca = select_ca_atoms(universe)
        coords = collect_ca_coords_flat(universe, atoms=ca)

        if len(coords) < 20:
            logger.warning("Only %d frames; need >= 20 for MSM.", len(coords))
            return {"error": "Not enough frames for MSM"}

        coords_scaled, _, _ = standardise_features(coords)
        reduced, pca_model = pca_reduce(coords_scaled, n_components=10)
        reduced_5d = reduced[:, : min(5, reduced.shape[1])]

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

        # ── Build count matrix (vectorised) ──────────────────────
        count_matrix = np.zeros((n_states, n_states), dtype=np.float64)
        src = labels[: len(labels) - lag_time]
        dst = labels[lag_time:]
        np.add.at(count_matrix, (src, dst), 1)

        # ── Row-normalise to get transition matrix ───────────────
        row_sums = count_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        T = count_matrix / row_sums

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
            "MSM built: %d states, lag=%d, top timescale=%.1f.",
            n_states,
            lag_time,
            timescales[0] if timescales else 0.0,
        )

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
