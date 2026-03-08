"""
Normal Mode Analysis (NMA).
Computes elastic network model normal modes from the MD-average Cα structure
using an Anisotropic Network Model (ANM).
"""
import numpy as np
from scipy.linalg import eigh
from MDAnalysis.lib.distances import distance_array


def compute_nma(universe, n_modes=10, cutoff=15.0, gamma=1.0, **kwargs):
    """
    Normal Mode Analysis using Anisotropic Network Model (ANM).

    Builds the Hessian from the MD-average Cα structure and diagonalises it
    to obtain the slowest collective modes.

    Returns dict with:
        - resids: residue IDs
        - eigenvalues: eigenvalues of the first n_modes non-trivial modes
        - frequencies: mode frequencies (sqrt of eigenvalue)
        - bfactors: per-residue predicted B-factors from ANM
        - mode_collectivity: collectivity index for each mode
        - mode_shapes: per-residue displacement magnitude for each mode
        - pca_overlap: overlap between ANM modes and PCA eigenvectors (if PCA was run)
    """
    try:
        ca = universe.select_atoms("protein and name CA")
        if len(ca) == 0:
            return {"error": "No CA atoms found"}

        n_res = len(ca)
        resids = ca.resids.tolist()

        # Compute average structure
        positions = []
        for ts in universe.trajectory:
            positions.append(ca.positions.copy())
        positions = np.array(positions)
        mean_pos = positions.mean(axis=0)  # (n_res, 3)

        # Build ANM Hessian
        hessian = _build_anm_hessian(mean_pos, cutoff, gamma)

        # Diagonalise (real symmetric matrix)
        eigenvalues, eigenvectors = eigh(hessian)

        # Skip first 6 trivial modes (translations + rotations)
        nontrivial_start = 6
        evals = eigenvalues[nontrivial_start: nontrivial_start + n_modes]
        evecs = eigenvectors[:, nontrivial_start: nontrivial_start + n_modes]

        # Frequencies
        frequencies = []
        for ev in evals:
            if ev > 1e-10:
                frequencies.append(round(float(np.sqrt(ev)), 6))
            else:
                frequencies.append(0.0)

        # Per-residue B-factors: B_i = (8π²/3) * sum_k (1/λ_k) * |v_ik|²
        bfactors = np.zeros(n_res)
        for k in range(min(n_modes, len(evals))):
            if evals[k] > 1e-10:
                mode = evecs[:, k].reshape(n_res, 3)
                mode_sq = np.sum(mode ** 2, axis=1)
                bfactors += mode_sq / evals[k]
        bfactors *= (8.0 * np.pi ** 2 / 3.0)

        # Normalize B-factors
        if bfactors.max() > 0:
            bfactors_norm = bfactors / bfactors.max()
        else:
            bfactors_norm = bfactors

        # Mode shapes: per-residue displacement magnitude for each mode
        mode_shapes = []
        for k in range(min(n_modes, len(evals))):
            mode = evecs[:, k].reshape(n_res, 3)
            magnitudes = np.sqrt(np.sum(mode ** 2, axis=1))
            mode_shapes.append([round(float(m), 4) for m in magnitudes])

        # Collectivity: κ = (1/N) * exp(-sum p_i ln p_i)
        collectivity = []
        for k in range(min(n_modes, len(evals))):
            mode = evecs[:, k].reshape(n_res, 3)
            sq = np.sum(mode ** 2, axis=1)
            sq_sum = sq.sum()
            if sq_sum > 0:
                p = sq / sq_sum
                p = p[p > 1e-15]
                entropy = -np.sum(p * np.log(p))
                kappa = np.exp(entropy) / n_res
                collectivity.append(round(float(kappa), 4))
            else:
                collectivity.append(0.0)

        return {
            "resids": resids,
            "eigenvalues": [round(float(x), 6) for x in evals],
            "frequencies": frequencies,
            "bfactors": [round(float(x), 4) for x in bfactors_norm],
            "mode_collectivity": collectivity,
            "mode_shapes": mode_shapes[:5],  # first 5 modes
            "n_modes_computed": int(min(n_modes, len(evals))),
        }

    except Exception as e:
        return {"error": str(e)}


def _build_anm_hessian(coords, cutoff, gamma):
    """Build the 3Nx3N Hessian matrix for an ANM."""
    n = len(coords)
    hessian = np.zeros((3 * n, 3 * n))

    for i in range(n):
        for j in range(i + 1, n):
            diff = coords[j] - coords[i]
            dist = np.linalg.norm(diff)
            if dist < cutoff:
                # Spring constant scaled by distance
                k_ij = -gamma / (dist ** 2)
                # 3x3 super-element
                outer = np.outer(diff, diff)
                h_ij = k_ij * outer

                # Off-diagonal blocks
                hessian[3*i:3*i+3, 3*j:3*j+3] = h_ij
                hessian[3*j:3*j+3, 3*i:3*i+3] = h_ij

                # Diagonal blocks
                hessian[3*i:3*i+3, 3*i:3*i+3] -= h_ij
                hessian[3*j:3*j+3, 3*j:3*j+3] -= h_ij

    return hessian
