"""Contact map and distance matrix computation.

Computes the average inter-residue contact frequency and distance matrix
from C-alpha atom positions across all trajectory frames.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array

from ..utils.trajectory_utils import select_ca_atoms

logger = logging.getLogger("md_ai_analyzer")


def compute_contact_map(
    universe: mda.Universe,
    cutoff: float = 8.0,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Compute the average contact map and distance matrix between C-alpha atoms.

    Parameters
    ----------
    universe : mda.Universe
        MDAnalysis Universe with a loaded trajectory.
    cutoff : float, optional
        Contact distance cutoff in Angstrom (default 8.0).
    **kwargs : Any
        Additional keyword arguments (unused, accepted for API consistency).

    Returns
    -------
    dict[str, Any]
        ``contact_map``
            2-D list of contact frequencies in [0, 1].
        ``avg_distance_matrix``
            2-D list of average pairwise distances (Angstrom).
        ``resids``
            List of residue IDs.
        ``n_residues``
            Number of residues.
        ``persistent_contacts``
            Top 50 persistent contacts (frequency > 0.7, sequence separation >= 5).
        ``cutoff_used``
            Contact cutoff that was applied (Angstrom).
    """
    try:
        ca = select_ca_atoms(universe)
        n_res: int = len(ca)
        logger.info(
            "Computing contact map for %d residues over %d frames (cutoff=%.1f A)",
            n_res,
            len(universe.trajectory),
            cutoff,
        )

        contact_sum = np.zeros((n_res, n_res), dtype=np.float64)
        dist_sum = np.zeros((n_res, n_res), dtype=np.float64)
        n_frames: int = 0

        for ts in universe.trajectory:
            dists = distance_array(ca.positions, ca.positions, box=ts.dimensions)
            contact_sum += (dists <= cutoff).astype(np.float64)
            dist_sum += dists
            n_frames += 1

        contact_freq = contact_sum / max(n_frames, 1)
        avg_dist = dist_sum / max(n_frames, 1)

        resids: List[int] = ca.resids.tolist()

        # --- Vectorised persistent-contact extraction ---
        # Upper triangle with sequence separation >= 5
        ii, jj = np.triu_indices(n_res, k=5)
        freq_vals = contact_freq[ii, jj]
        mask = freq_vals > 0.7
        sel_i = ii[mask]
        sel_j = jj[mask]
        sel_freq = freq_vals[mask]
        sel_dist = avg_dist[sel_i, sel_j]

        # Sort by descending frequency
        order = np.argsort(-sel_freq)
        persistent_contacts: List[Dict[str, Any]] = []
        for idx in order[:50]:
            persistent_contacts.append(
                {
                    "res_i": int(resids[sel_i[idx]]),
                    "res_j": int(resids[sel_j[idx]]),
                    "frequency": round(float(sel_freq[idx]), 3),
                    "avg_distance": round(float(sel_dist[idx]), 2),
                }
            )

        logger.info(
            "Contact map complete: %d persistent contacts (freq > 0.7)",
            len(persistent_contacts),
        )

        return {
            "contact_map": contact_freq.tolist(),
            "avg_distance_matrix": avg_dist.tolist(),
            "resids": resids,
            "n_residues": n_res,
            "persistent_contacts": persistent_contacts[:50],
            "cutoff_used": cutoff,
        }

    except Exception as e:
        logger.error("Contact map computation failed: %s", e, exc_info=True)
        return {"error": str(e)}
