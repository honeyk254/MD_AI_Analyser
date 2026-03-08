"""
Time-lagged Independent Component Analysis (tICA).
Identifies slow collective motions in the trajectory.
"""
import numpy as np


def compute_tica(universe, lag_time=10, n_components=5, **kwargs):
    """
    Perform tICA on Cα coordinates to find slow motions.

    tICA finds the linear combinations of coordinates that decorrelate most slowly,
    capturing the slowest dynamical processes.

    Returns dict with:
        - projections: tICA projections per frame
        - timescales: implied timescales of each tIC
        - eigenvalues: tICA eigenvalues
        - n_components: number of components
        - tic1_residue_contributions: per-residue contribution to tIC1
    """
    try:
        ca = universe.select_atoms("protein and name CA")
        if len(ca) == 0:
            ca = universe.select_atoms("all")

        n_atoms = len(ca)

        # Collect centered coordinates
        coords = []
        for ts in universe.trajectory:
            coords.append(ca.positions.flatten().copy())
        coords = np.array(coords)

        mean_coords = coords.mean(axis=0)
        X = coords - mean_coords

        n_frames = X.shape[0]
        if n_frames <= lag_time + 1:
            return {"error": f"Trajectory too short for lag_time={lag_time}"}

        # Covariance matrix C(0)
        C0 = (X.T @ X) / (n_frames - 1)

        # Time-lagged covariance C(tau)
        X_t = X[:n_frames - lag_time]
        X_tau = X[lag_time:]
        Ctau = (X_t.T @ X_tau) / (n_frames - lag_time - 1)

        # Symmetrize Ctau
        Ctau_sym = 0.5 * (Ctau + Ctau.T)

        # Solve generalized eigenvalue problem: Ctau v = lambda C0 v
        # Use regularization for numerical stability
        reg = 1e-6 * np.eye(C0.shape[0])
        C0_reg = C0 + reg

        try:
            from scipy.linalg import eigh
            eigenvalues, eigenvectors = eigh(Ctau_sym, C0_reg)
        except Exception:
            # Fallback: compute C0^{-1/2} Ctau C0^{-1/2} and solve standard eigenvalue problem
            from scipy.linalg import sqrtm, inv
            C0_inv_sqrt = np.real(inv(sqrtm(C0_reg)))
            M = C0_inv_sqrt @ Ctau_sym @ C0_inv_sqrt
            eigenvalues, eigenvectors_t = np.linalg.eigh(M)
            eigenvectors = C0_inv_sqrt @ eigenvectors_t

        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Take top components
        n_comp = min(n_components, len(eigenvalues))
        eigenvalues = eigenvalues[:n_comp]
        eigenvectors = eigenvectors[:, :n_comp]

        # Project
        projections = X @ eigenvectors

        # Implied timescales: tau_i = -lag_time / ln(lambda_i)
        timescales = []
        for ev in eigenvalues:
            if 0 < ev < 1:
                timescales.append(float(-lag_time / np.log(ev)))
            else:
                timescales.append(float('inf'))

        # Per-residue contributions to tIC1
        def residue_contributions(comp_idx):
            vec = eigenvectors[:, comp_idx]
            contributions = np.zeros(n_atoms)
            for i in range(n_atoms):
                contributions[i] = np.sqrt(np.sum(vec[3*i:3*i+3]**2))
            return contributions.tolist()

        resids = ca.resids.tolist()

        return {
            "projections": projections.tolist(),
            "timescales": timescales,
            "eigenvalues": eigenvalues.tolist(),
            "n_components": n_comp,
            "lag_time": lag_time,
            "resids": resids,
            "tic1_residue_contributions": residue_contributions(0) if n_comp > 0 else [],
            "tic2_residue_contributions": residue_contributions(1) if n_comp > 1 else [],
        }

    except Exception as e:
        return {"error": str(e), "projections": [], "timescales": []}
