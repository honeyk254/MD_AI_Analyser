"""
Dimensionality reduction for conformational ensemble visualization.
UMAP + t-SNE + PCA projections.
"""
import numpy as np
from sklearn.decomposition import PCA


def compute_dimensionality_reduction(universe, **kwargs):
    """
    Apply multiple dimensionality reduction methods for visualization.

    Returns dict with:
        - pca_2d: 2D PCA projections
        - pca_3d: 3D PCA projections (item 57)
        - umap_2d: 2D UMAP projections
        - umap_3d: 3D UMAP projections (item 57)
        - tsne_2d: 2D t-SNE projections
        - tsne_3d: 3D t-SNE projections (item 57)
    """
    try:
        ca = universe.select_atoms("protein and name CA")
        if len(ca) == 0:
            ca = universe.select_atoms("all")

        coords = []
        for ts in universe.trajectory:
            coords.append(ca.positions.flatten().copy())
        coords = np.array(coords)

        if len(coords) < 5:
            return {"error": "Too few frames"}

        # Pre-reduce with PCA to speed up UMAP/t-SNE
        n_pca = min(20, coords.shape[0] - 1, coords.shape[1])
        pca_pre = PCA(n_components=n_pca)
        reduced = pca_pre.fit_transform(coords)

        results = {}

        # PCA 2D
        pca_2d = PCA(n_components=2)
        pca_proj = pca_2d.fit_transform(coords)
        results["pca_2d"] = pca_proj.tolist()
        results["pca_variance"] = pca_2d.explained_variance_ratio_.tolist()

        # PCA 3D (item 57)
        if n_pca >= 3:
            pca_3d = PCA(n_components=3)
            pca_3d_proj = pca_3d.fit_transform(coords)
            results["pca_3d"] = pca_3d_proj.tolist()
            results["pca_3d_variance"] = pca_3d.explained_variance_ratio_.tolist()
        else:
            results["pca_3d"] = []

        # UMAP
        try:
            import umap
            reducer = umap.UMAP(n_components=2, n_neighbors=min(15, len(reduced) - 1),
                               min_dist=0.1, random_state=42, metric='euclidean')
            umap_proj = reducer.fit_transform(reduced)
            results["umap_2d"] = umap_proj.tolist()

            # UMAP 3D (item 57)
            try:
                reducer3d = umap.UMAP(n_components=3, n_neighbors=min(15, len(reduced) - 1),
                                     min_dist=0.1, random_state=42, metric='euclidean')
                umap_3d_proj = reducer3d.fit_transform(reduced)
                results["umap_3d"] = umap_3d_proj.tolist()
            except Exception:
                results["umap_3d"] = []
        except ImportError:
            results["umap_2d"] = []
            results["umap_3d"] = []

        # t-SNE
        try:
            from sklearn.manifold import TSNE
            perplexity = min(30, len(reduced) - 1)
            if perplexity >= 2:
                tsne = TSNE(n_components=2, perplexity=perplexity,
                           random_state=42, n_iter=1000)
                tsne_proj = tsne.fit_transform(reduced)
                results["tsne_2d"] = tsne_proj.tolist()

                # t-SNE 3D (item 57)
                try:
                    tsne3d = TSNE(n_components=3, perplexity=perplexity,
                                 random_state=42, n_iter=1000)
                    tsne_3d_proj = tsne3d.fit_transform(reduced)
                    results["tsne_3d"] = tsne_3d_proj.tolist()
                except Exception:
                    results["tsne_3d"] = []
            else:
                results["tsne_2d"] = []
                results["tsne_3d"] = []
        except Exception:
            results["tsne_2d"] = []
            results["tsne_3d"] = []

        return results

    except Exception as e:
        return {"error": str(e)}
