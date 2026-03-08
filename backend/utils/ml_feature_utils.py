"""
Shared ML feature-engineering utilities.

Consolidates the repeated PCA-reduction + clustering patterns that appear
in clustering, MSM, state_discovery, free_energy, and dimensionality modules.

Functions
---------
pca_reduce
    Reduce flattened coordinate matrix with PCA.
find_optimal_k
    Silhouette-based optimal cluster count selection.
standardise_features
    Zero-mean, unit-variance feature scaling with safety for zero-variance.
set_global_seed
    Set random seeds for NumPy, scikit-learn, and PyTorch reproducibility.
"""
import logging
from typing import Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("md_ai_analyzer")


def pca_reduce(
    data: np.ndarray,
    n_components: int = 10,
    center: bool = True,
) -> Tuple[np.ndarray, PCA]:
    """Fit PCA and return projections + fitted model.

    Parameters
    ----------
    data : np.ndarray
        Shape ``(n_samples, n_features)``.
    n_components : int
        Max components; clamped to ``min(n_components, n_samples-1, n_features)``.
    center : bool
        Whether to mean-center *data* before PCA (sklearn centres internally,
        but this flag controls whether we also return a centered copy).

    Returns
    -------
    tuple[np.ndarray, PCA]
        ``(projections, fitted_pca_model)`` where *projections* has shape
        ``(n_samples, n_comp)``.

    Raises
    ------
    ValueError
        If data is too small for any PCA components.
    """
    n_comp = min(n_components, data.shape[0] - 1, data.shape[1])
    if n_comp < 1:
        raise ValueError(
            f"Cannot fit PCA: need at least 2 samples, got shape {data.shape}"
        )
    pca = PCA(n_components=n_comp)
    projections = pca.fit_transform(data)
    return projections, pca


def find_optimal_k(
    data: np.ndarray,
    k_min: int = 2,
    k_max: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[int, dict]:
    """Determine optimal cluster count via silhouette score.

    Parameters
    ----------
    data : np.ndarray
        Feature matrix ``(n_samples, n_features)``.
    k_min : int
    k_max : int, optional
        Defaults to ``min(10, n_samples // 5)``, clamped to >= k_min.
    random_state : int

    Returns
    -------
    tuple[int, dict]
        ``(best_k, {k: silhouette_score, ...})``.
    """
    from sklearn.metrics import silhouette_score

    if k_max is None:
        k_max = min(10, len(data) // 5)
    k_max = max(k_max, k_min)

    scores: dict = {}
    best_k = k_min
    best_score = -1.0

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(data)
        if len(set(labels)) < 2:
            continue
        score = float(silhouette_score(data, labels))
        scores[k] = score
        if score > best_score:
            best_score = score
            best_k = k

    return best_k, scores


def standardise_features(
    features: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zero-mean, unit-variance normalisation with zero-variance safety.

    Parameters
    ----------
    features : np.ndarray
        Shape ``(n_samples, n_features)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(normalised, mean, std)`` where *std* has zeros replaced by 1.0.
    """
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (features - mean) / std, mean, std


def set_global_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across NumPy and PyTorch.

    Parameters
    ----------
    seed : int
    """
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
