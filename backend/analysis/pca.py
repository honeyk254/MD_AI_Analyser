"""
Principal Component Analysis of protein backbone dynamics.
Extracts dominant modes of motion from Cα coordinates.
"""
import numpy as np
from sklearn.decomposition import PCA


def compute_pca(universe, n_components=10, **kwargs):
    """
    Perform PCA on Cα atom coordinates across the trajectory.

    Returns dict with:
        - projections: list of [PC1, PC2, PC3, ...] per frame
        - explained_variance: explained variance per component
        - cumulative_variance: cumulative explained variance
        - n_components: number of components
        - pc1_residue_contributions: per-residue contribution to PC1
        - pc2_residue_contributions: per-residue contribution to PC2
    """
    try:
        ca = universe.select_atoms("protein and name CA")
        if len(ca) == 0:
            ca = universe.select_atoms("all")

        n_atoms = len(ca)

        # Collect coordinates
        coords = []
        for ts in universe.trajectory:
            coords.append(ca.positions.flatten().copy())

        coords = np.array(coords)  # (n_frames, n_atoms*3)

        # Center
        mean_coords = coords.mean(axis=0)
        coords_centered = coords - mean_coords

        # PCA
        n_comp = min(n_components, coords_centered.shape[0], coords_centered.shape[1])
        pca = PCA(n_components=n_comp)
        projections = pca.fit_transform(coords_centered)

        # Per-residue contributions to PCs (sum of x,y,z components per residue)
        def residue_contributions(component_idx):
            eigvec = pca.components_[component_idx]
            contributions = np.zeros(n_atoms)
            for i in range(n_atoms):
                contributions[i] = np.sqrt(np.sum(eigvec[3*i:3*i+3]**2))
            return contributions.tolist()

        resids = ca.resids.tolist()

        return {
            "projections": projections.tolist(),
            "explained_variance": pca.explained_variance_ratio_.tolist(),
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
            "n_components": n_comp,
            "mean_structure": mean_coords.reshape(-1, 3).tolist(),
            "resids": resids,
            "pc1_residue_contributions": residue_contributions(0) if n_comp > 0 else [],
            "pc2_residue_contributions": residue_contributions(1) if n_comp > 1 else [],
            "pc3_residue_contributions": residue_contributions(2) if n_comp > 2 else [],
        }

    except Exception as e:
        return {"error": str(e), "projections": [], "explained_variance": []}
