from __future__ import annotations

"""Conformational state discovery using multiple clustering algorithms.

Applies HDBSCAN, Gaussian Mixture Models (GMM), and KMeans on PCA-reduced
C-alpha coordinate space to identify distinct conformational states in
molecular-dynamics trajectories.
"""

import logging
from typing import Any, Dict

import numpy as np
from sklearn.mixture import GaussianMixture

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


def discover_states(
    universe: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Discover conformational states using multiple unsupervised methods.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        Loaded trajectory universe.
    **kwargs
        Additional keyword arguments (currently unused but accepted for
        orchestrator compatibility).

    Returns
    -------
    dict
        Keys:
        - ``hdbscan_labels`` : cluster labels from HDBSCAN
        - ``n_states_hdbscan`` : number of HDBSCAN clusters
        - ``hdbscan_noise_fraction`` : fraction of frames labelled noise
        - ``gmm_labels`` : cluster labels from GMM
        - ``gmm_probabilities`` : per-frame probabilities for each state
        - ``n_states_gmm`` : optimal number of GMM components (BIC)
        - ``bic_scores`` : BIC score per *k*
        - ``state_populations`` : fraction of frames in each GMM state
        - ``state_descriptions`` : per-state summary dict
    """
    set_global_seed(42)
    try:
        # ── Coordinate collection & PCA ──────────────────────────
        ca = select_ca_atoms(universe)
        coords = collect_ca_coords_flat(universe, atoms=ca)

        if len(coords) < 10:
            logger.warning("Only %d frames available; need >= 10 for state discovery.", len(coords))
            return {"error": "Not enough frames for state discovery"}

        coords_scaled, _, _ = standardise_features(coords)
        reduced, pca_model = pca_reduce(coords_scaled, n_components=10)

        results: Dict[str, Any] = {}

        # ── HDBSCAN ──────────────────────────────────────────────
        try:
            import hdbscan

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=max(5, len(reduced) // 20),
                min_samples=3,
                cluster_selection_method="eom",
            )
            hdbscan_labels: np.ndarray = clusterer.fit_predict(reduced[:, :3])
            n_hdbscan = int(len(set(hdbscan_labels.tolist())) - (1 if -1 in hdbscan_labels else 0))
            results["hdbscan_labels"] = hdbscan_labels.tolist()
            results["n_states_hdbscan"] = n_hdbscan
            results["hdbscan_noise_fraction"] = float(np.mean(hdbscan_labels == -1))
            logger.info("HDBSCAN found %d states (noise %.1f%%).", n_hdbscan, results["hdbscan_noise_fraction"] * 100)
        except ImportError:
            logger.warning("hdbscan package not installed; skipping HDBSCAN clustering.")
            results["hdbscan_labels"] = []
            results["n_states_hdbscan"] = 0

        # ── GMM with BIC selection ───────────────────────────────
        max_components = max(min(8, len(reduced) // 10), 2)
        bic_scores: Dict[int, float] = {}
        reduced_5d = reduced[:, : min(5, reduced.shape[1])]

        for k in range(2, max_components + 1):
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                n_init=3,
                random_state=42,
            )
            gmm.fit(reduced_5d)
            bic_scores[k] = float(gmm.bic(reduced_5d))

        best_k: int = min(bic_scores, key=bic_scores.get) if bic_scores else 2

        gmm_final = GaussianMixture(
            n_components=best_k,
            covariance_type="full",
            n_init=5,
            random_state=42,
        )
        gmm_final.fit(reduced_5d)
        gmm_labels: np.ndarray = gmm_final.predict(reduced_5d)
        gmm_probs: np.ndarray = gmm_final.predict_proba(reduced_5d)

        # State populations (vectorised)
        unique, counts = np.unique(gmm_labels, return_counts=True)
        populations: Dict[int, float] = {
            int(u): round(float(c) / len(gmm_labels), 3)
            for u, c in zip(unique, counts)
        }

        # State mean structures info (vectorised radius-of-gyration)
        state_descriptions: Dict[int, Dict[str, Any]] = {}
        for state in unique:
            mask = gmm_labels == state
            # coords is (n_frames, n_atoms*3); reshape mean to (n_atoms, 3)
            state_coords = coords[mask].mean(axis=0).reshape(-1, 3)
            centroid = state_coords.mean(axis=0)
            rg = float(np.sqrt(np.mean(np.sum((state_coords - centroid) ** 2, axis=1))))
            state_descriptions[int(state)] = {
                "population": populations[int(state)],
                "mean_rg": round(rg, 2),
                "n_frames": int(np.sum(mask)),
            }

        results.update(
            {
                "gmm_labels": gmm_labels.tolist(),
                "gmm_probabilities": gmm_probs.tolist(),
                "n_states_gmm": best_k,
                "bic_scores": bic_scores,
                "state_populations": populations,
                "state_descriptions": state_descriptions,
            }
        )

        logger.info("State discovery complete: GMM selected %d states.", best_k)
        return results

    except Exception as e:
        logger.exception("State discovery failed: %s", e)
        return {"error": str(e)}
