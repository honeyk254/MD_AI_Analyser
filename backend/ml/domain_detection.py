"""
Dynamic domain detection using spectral clustering of DCCM.
Identifies groups of residues that move as rigid bodies.
"""
import numpy as np
from sklearn.cluster import SpectralClustering


def detect_domains(universe, n_domains=None, **kwargs):
    """
    Detect dynamic domains using spectral clustering on the DCCM.

    Returns dict with:
        - domain_labels: per-residue domain assignment
        - n_domains: number of detected domains
        - domain_info: details for each domain (residues, mean internal corr)
        - inter_domain_correlations: correlation between domains
    """
    try:
        ca = universe.select_atoms("protein and name CA")
        if len(ca) == 0:
            return {"error": "No CA atoms found"}

        n_res = len(ca)
        resids = ca.resids.tolist()

        # Compute DCCM
        positions = []
        for ts in universe.trajectory:
            positions.append(ca.positions.copy())
        positions = np.array(positions)

        mean_pos = positions.mean(axis=0)
        delta = positions - mean_pos

        n_frames = delta.shape[0]
        dccm = np.einsum('fid,fjd->ij', delta, delta) / n_frames

        diag = np.sqrt(np.diag(dccm))
        diag[diag == 0] = 1e-10
        dccm_norm = dccm / np.outer(diag, diag)

        # Affinity matrix: use |correlation| as similarity
        affinity = np.abs(dccm_norm)
        np.fill_diagonal(affinity, 1.0)

        # Determine optimal number of domains via eigengap heuristic
        if n_domains is None:
            from scipy.linalg import eigh
            D = np.diag(affinity.sum(axis=1))
            L = D - affinity  # Laplacian
            D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-10))
            L_norm = D_inv_sqrt @ L @ D_inv_sqrt

            eigenvalues = np.sort(np.real(np.linalg.eigvalsh(L_norm)))
            # Find largest eigengap
            gaps = np.diff(eigenvalues[:min(15, len(eigenvalues))])
            if len(gaps) > 1:
                n_domains = int(np.argmax(gaps[1:]) + 2)  # +2 because we skip first gap
                n_domains = max(2, min(n_domains, 8))
            else:
                n_domains = 2

        # Spectral clustering
        sc = SpectralClustering(n_clusters=n_domains, affinity='precomputed',
                                random_state=42, n_init=10)
        domain_labels = sc.fit_predict(affinity)

        # Domain info
        domain_info = []
        for d in range(n_domains):
            mask = domain_labels == d
            members = [int(resids[i]) for i in range(n_res) if mask[i]]

            # Mean internal correlation
            internal_corrs = []
            for i in range(n_res):
                if not mask[i]:
                    continue
                for j in range(i + 1, n_res):
                    if mask[j]:
                        internal_corrs.append(dccm_norm[i, j])

            mean_internal = float(np.mean(internal_corrs)) if internal_corrs else 0

            domain_info.append({
                "domain_id": d,
                "residues": members,
                "size": len(members),
                "start_resid": min(members) if members else 0,
                "end_resid": max(members) if members else 0,
                "mean_internal_correlation": round(mean_internal, 3),
            })

        # Inter-domain correlations
        inter_domain = np.zeros((n_domains, n_domains))
        for da in range(n_domains):
            for db in range(da + 1, n_domains):
                mask_a = domain_labels == da
                mask_b = domain_labels == db
                corrs = dccm_norm[np.ix_(mask_a, mask_b)]
                inter_domain[da, db] = float(np.mean(corrs))
                inter_domain[db, da] = inter_domain[da, db]

        return {
            "domain_labels": domain_labels.tolist(),
            "resids": resids,
            "n_domains": n_domains,
            "domain_info": domain_info,
            "inter_domain_correlations": inter_domain.tolist(),
        }

    except Exception as e:
        return {"error": str(e)}
