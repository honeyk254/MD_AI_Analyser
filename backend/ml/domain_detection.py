from __future__ import annotations

"""Dynamic domain detection using spectral clustering of the DCCM.

Identifies groups of residues that move as quasi-rigid bodies by applying
spectral clustering to the absolute-value correlation matrix derived from
the Dynamic Cross-Correlation Matrix.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.cluster import SpectralClustering

from ..utils.trajectory_utils import (
    select_ca_atoms,
    collect_ca_positions,
    compute_dccm_from_positions,
)
from ..utils.ml_feature_utils import set_global_seed

logger = logging.getLogger("md_ai_analyzer")

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_domains(
    universe: Any,
    n_domains: Optional[int] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Detect dynamic domains using spectral clustering on the DCCM.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        Loaded trajectory universe.
    n_domains : int, optional
        Number of domains to partition into.  When *None* the optimal
        count is estimated via the eigengap heuristic on the normalised
        graph Laplacian.
    **kwargs
        Additional keyword arguments (unused; accepted for orchestrator
        compatibility).

    Returns
    -------
    dict
        Keys:
        - ``domain_labels`` : per-residue domain assignment
        - ``resids`` : residue IDs
        - ``n_domains`` : number of detected domains
        - ``domain_info`` : list of per-domain detail dicts
        - ``inter_domain_correlations`` : mean correlation between domains
    """
    set_global_seed(42)
    try:
        ca = select_ca_atoms(universe, fallback="protein")
        n_res = len(ca)
        resids: List[int] = ca.resids.tolist()

        # ── DCCM via shared utility ──────────────────────────────
        positions = collect_ca_positions(universe, atoms=ca)
        dccm_norm = compute_dccm_from_positions(positions)

        # ── Affinity matrix: |correlation| ───────────────────────
        affinity: np.ndarray = np.abs(dccm_norm)
        np.fill_diagonal(affinity, 1.0)

        # ── Determine optimal n_domains via eigengap heuristic ───
        if n_domains is None:
            D_diag = affinity.sum(axis=1)
            D_inv_sqrt = np.where(D_diag > 1e-10, 1.0 / np.sqrt(D_diag), 0.0)
            L = np.diag(D_diag) - affinity  # unnormalised Laplacian
            L_norm = (D_inv_sqrt[:, None] * L) * D_inv_sqrt[None, :]

            eigenvalues = np.sort(np.real(np.linalg.eigvalsh(L_norm)))
            max_check = min(15, len(eigenvalues))
            gaps = np.diff(eigenvalues[:max_check])
            if len(gaps) > 1:
                n_domains = int(np.argmax(gaps[1:]) + 2)
                n_domains = max(2, min(n_domains, 8))
            else:
                n_domains = 2
            logger.info("Eigengap heuristic selected %d domains.", n_domains)

        # ── Spectral clustering ──────────────────────────────────
        sc = SpectralClustering(
            n_clusters=n_domains,
            affinity="precomputed",
            random_state=42,
            n_init=10,
        )
        domain_labels: np.ndarray = sc.fit_predict(affinity)

        # ── Domain info (vectorised internal correlation) ────────
        domain_info: List[Dict[str, Any]] = []
        for d in range(n_domains):
            mask = domain_labels == d
            member_indices = np.where(mask)[0]
            members = [int(resids[i]) for i in member_indices]

            # Vectorised: extract upper-triangle of intra-domain block
            if len(member_indices) > 1:
                sub_matrix = dccm_norm[np.ix_(member_indices, member_indices)]
                triu_vals = sub_matrix[np.triu_indices(len(member_indices), k=1)]
                mean_internal = float(np.mean(triu_vals)) if len(triu_vals) > 0 else 0.0
            else:
                mean_internal = 0.0

            domain_info.append(
                {
                    "domain_id": d,
                    "residues": members,
                    "size": len(members),
                    "start_resid": min(members) if members else 0,
                    "end_resid": max(members) if members else 0,
                    "mean_internal_correlation": round(mean_internal, 3),
                }
            )

        # ── Inter-domain correlations (vectorised) ───────────────
        inter_domain = np.zeros((n_domains, n_domains), dtype=np.float64)
        for da in range(n_domains):
            for db in range(da + 1, n_domains):
                mask_a = domain_labels == da
                mask_b = domain_labels == db
                corrs = dccm_norm[np.ix_(np.where(mask_a)[0], np.where(mask_b)[0])]
                mean_corr = float(np.mean(corrs))
                inter_domain[da, db] = mean_corr
                inter_domain[db, da] = mean_corr

        logger.info("Domain detection complete: %d domains identified.", n_domains)

        return {
            "domain_labels": domain_labels.tolist(),
            "resids": resids,
            "n_domains": n_domains,
            "domain_info": domain_info,
            "inter_domain_correlations": inter_domain.tolist(),
        }

    except Exception as e:
        logger.exception("Domain detection failed: %s", e)
        return {"error": str(e)}
