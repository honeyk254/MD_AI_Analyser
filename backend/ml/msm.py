"""
Markov State Model (MSM) construction.
Builds transition matrix, computes stationary distribution, identifies metastable states.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def build_msm(universe, n_states=None, lag_time=5, **kwargs):
    """
    Build a Markov State Model from trajectory clustering.

    Returns dict with:
        - transition_matrix: state-to-state transition probabilities
        - stationary_distribution: equilibrium probability of each state
        - implied_timescales: list of implied timescales
        - metastable_states: indices of most probable metastable states
        - mean_first_passage_times: MFPT between states
        - n_states: number of MSM states
    """
    try:
        ca = universe.select_atoms("protein and name CA")
        if len(ca) == 0:
            ca = universe.select_atoms("all")

        coords = []
        for ts in universe.trajectory:
            coords.append(ca.positions.flatten().copy())
        coords = np.array(coords)

        if len(coords) < 20:
            return {"error": "Not enough frames for MSM"}

        # PCA reduction
        n_comp = min(10, coords.shape[0] - 1, coords.shape[1])
        pca = PCA(n_components=n_comp)
        reduced = pca.fit_transform(coords)

        # Determine n_states if not given
        if n_states is None:
            from sklearn.metrics import silhouette_score
            best_k, best_score = 3, -1
            for k in range(2, min(12, len(reduced) // 5)):
                km = KMeans(n_clusters=k, n_init=10, random_state=42)
                labels = km.fit_predict(reduced[:, :5])
                if len(set(labels)) < 2:
                    continue
                score = silhouette_score(reduced[:, :5], labels)
                if score > best_score:
                    best_score = score
                    best_k = k
            n_states = best_k

        # Cluster
        km = KMeans(n_clusters=n_states, n_init=10, random_state=42)
        labels = km.fit_predict(reduced[:, :5])

        # Build count matrix
        count_matrix = np.zeros((n_states, n_states), dtype=float)
        for t in range(len(labels) - lag_time):
            i = labels[t]
            j = labels[t + lag_time]
            count_matrix[i, j] += 1

        # Row-normalize to get transition matrix
        row_sums = count_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        T = count_matrix / row_sums

        # Stationary distribution (left eigenvector of T corresponding to eigenvalue 1)
        eigenvalues, eigenvectors = np.linalg.eig(T.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        pi = np.real(eigenvectors[:, idx])
        pi = np.abs(pi)
        pi /= pi.sum()

        # Implied timescales from eigenvalues of T
        T_eigenvalues = np.sort(np.real(np.linalg.eigvals(T)))[::-1]
        timescales = []
        for ev in T_eigenvalues[1:]:
            if 0 < ev < 1:
                timescales.append(float(-lag_time / np.log(ev)))
            else:
                timescales.append(0.0)

        # Mean first passage times
        mfpt = _compute_mfpt(T, n_states)

        # Metastable states (high self-transition probability)
        metastable = []
        for i in range(n_states):
            metastable.append({
                "state": i,
                "self_transition": round(float(T[i, i]), 3),
                "population": round(float(pi[i]), 3),
            })
        metastable.sort(key=lambda x: -x["self_transition"])

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
        return {"error": str(e)}


def _compute_mfpt(T, n_states):
    """Compute mean first passage times between states."""
    mfpt = np.zeros((n_states, n_states))
    for target in range(n_states):
        # MFPT to target state from each other state
        # Solve (I - T_reduced) m = 1, where T_reduced removes target row/col
        indices = [i for i in range(n_states) if i != target]
        if not indices:
            continue
        T_sub = T[np.ix_(indices, indices)]
        A = np.eye(len(indices)) - T_sub
        try:
            m = np.linalg.solve(A, np.ones(len(indices)))
            for k, idx in enumerate(indices):
                mfpt[idx, target] = max(0, m[k])
        except np.linalg.LinAlgError:
            pass
    return mfpt
