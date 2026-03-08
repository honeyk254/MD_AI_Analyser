"""
Conformational state discovery using multiple clustering algorithms.
HDBSCAN + GMM + KMeans on PCA/tICA space.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans


def discover_states(universe, **kwargs):
    """
    Discover conformational states using multiple unsupervised methods.

    Returns dict with:
        - hdbscan_labels: cluster labels from HDBSCAN
        - gmm_labels: cluster labels from GMM
        - gmm_probabilities: per-frame probabilities for each state
        - n_states_hdbscan: number of HDBSCAN clusters
        - n_states_gmm: optimal number of GMM components (BIC)
        - state_populations: fraction in each GMM state
        - state_descriptions: brief description of each state
    """
    try:
        ca = universe.select_atoms("protein and name CA")
        if len(ca) == 0:
            ca = universe.select_atoms("all")

        coords = []
        for ts in universe.trajectory:
            coords.append(ca.positions.flatten().copy())
        coords = np.array(coords)

        if len(coords) < 10:
            return {"error": "Not enough frames for state discovery"}

        # PCA reduction
        n_comp = min(10, coords.shape[0] - 1, coords.shape[1])
        pca = PCA(n_components=n_comp)
        reduced = pca.fit_transform(coords)

        results = {}

        # ── HDBSCAN ────────────────────────────────────────────
        try:
            import hdbscan
            clusterer = hdbscan.HDBSCAN(min_cluster_size=max(5, len(reduced) // 20),
                                         min_samples=3,
                                         cluster_selection_method='eom')
            hdbscan_labels = clusterer.fit_predict(reduced[:, :3])
            n_hdbscan = len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)
            results["hdbscan_labels"] = hdbscan_labels.tolist()
            results["n_states_hdbscan"] = n_hdbscan
            results["hdbscan_noise_fraction"] = float(np.mean(hdbscan_labels == -1))
        except ImportError:
            results["hdbscan_labels"] = []
            results["n_states_hdbscan"] = 0

        # ── GMM with BIC selection ─────────────────────────────
        max_components = min(8, len(reduced) // 10)
        max_components = max(max_components, 2)
        bic_scores = {}

        for k in range(2, max_components + 1):
            gmm = GaussianMixture(n_components=k, covariance_type='full',
                                   n_init=3, random_state=42)
            gmm.fit(reduced[:, :5])
            bic_scores[k] = float(gmm.bic(reduced[:, :5]))

        # Select k with lowest BIC
        best_k = min(bic_scores, key=bic_scores.get) if bic_scores else 2

        gmm_final = GaussianMixture(n_components=best_k, covariance_type='full',
                                      n_init=5, random_state=42)
        gmm_final.fit(reduced[:, :5])
        gmm_labels = gmm_final.predict(reduced[:, :5])
        gmm_probs = gmm_final.predict_proba(reduced[:, :5])

        # State populations
        unique, counts = np.unique(gmm_labels, return_counts=True)
        populations = {int(u): round(float(c) / len(gmm_labels), 3) for u, c in zip(unique, counts)}

        # State mean structures info
        state_descriptions = {}
        for state in unique:
            mask = gmm_labels == state
            state_coords = coords[mask].mean(axis=0).reshape(-1, 3)
            rg = np.sqrt(np.mean(np.sum((state_coords - state_coords.mean(axis=0))**2, axis=1)))
            state_descriptions[int(state)] = {
                "population": populations[int(state)],
                "mean_rg": round(float(rg), 2),
                "n_frames": int(np.sum(mask)),
            }

        results.update({
            "gmm_labels": gmm_labels.tolist(),
            "gmm_probabilities": gmm_probs.tolist(),
            "n_states_gmm": best_k,
            "bic_scores": bic_scores,
            "state_populations": populations,
            "state_descriptions": state_descriptions,
        })

        return results

    except Exception as e:
        return {"error": str(e)}
