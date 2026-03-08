"""
Dynamic Cross-Correlation Matrix (DCCM).
Reveals correlated and anti-correlated motions between residues.
"""
import numpy as np


def compute_dccm(universe, threshold=0.7, **kwargs):
    """
    Compute the dynamic cross-correlation matrix for Cα atoms.

    C_ij = <Δr_i · Δr_j> / (sqrt(<|Δr_i|²>) * sqrt(<|Δr_j|²>))

    Returns dict with:
        - dccm: 2D correlation matrix [-1, 1]
        - resids: list of residue IDs
        - highly_correlated_pairs: pairs with |C_ij| > 0.7
        - correlated_domains: groups of residues moving together
    """
    try:
        ca = universe.select_atoms("protein and name CA")
        if len(ca) == 0:
            return {"error": "No CA atoms found"}

        n_atoms = len(ca)

        # Collect positions
        positions = []
        for ts in universe.trajectory:
            positions.append(ca.positions.copy())
        positions = np.array(positions)  # (n_frames, n_atoms, 3)

        # Mean positions
        mean_pos = positions.mean(axis=0)  # (n_atoms, 3)

        # Fluctuations
        delta = positions - mean_pos  # (n_frames, n_atoms, 3)

        # Cross-correlation matrix
        dccm = np.zeros((n_atoms, n_atoms))
        for i in range(n_atoms):
            for j in range(i, n_atoms):
                cij = np.mean(np.sum(delta[:, i, :] * delta[:, j, :], axis=1))
                dccm[i, j] = cij
                dccm[j, i] = cij

        # Normalize
        diag = np.sqrt(np.diag(dccm))
        diag[diag == 0] = 1e-10
        norm = np.outer(diag, diag)
        dccm_normalized = dccm / norm

        resids = ca.resids.tolist()

        # Find highly correlated pairs (non-sequential, |C| > 0.7)
        corr_pairs = []
        anticorr_pairs = []
        for i in range(n_atoms):
            for j in range(i + 5, n_atoms):
                c = dccm_normalized[i, j]
                if c > threshold:
                    corr_pairs.append({
                        "res_i": int(resids[i]),
                        "res_j": int(resids[j]),
                        "correlation": round(float(c), 3)
                    })
                elif c < -threshold * 0.7:
                    anticorr_pairs.append({
                        "res_i": int(resids[i]),
                        "res_j": int(resids[j]),
                        "correlation": round(float(c), 3)
                    })

        corr_pairs.sort(key=lambda x: -x["correlation"])
        anticorr_pairs.sort(key=lambda x: x["correlation"])

        return {
            "dccm": dccm_normalized.tolist(),
            "resids": resids,
            "n_residues": n_atoms,
            "highly_correlated_pairs": corr_pairs[:50],
            "anticorrelated_pairs": anticorr_pairs[:50],
        }

    except Exception as e:
        return {"error": str(e)}
