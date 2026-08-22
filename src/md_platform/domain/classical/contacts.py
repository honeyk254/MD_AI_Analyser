"""Contact map and distance matrix computation.

Computes the average inter-residue contact frequency and distance matrix
from C-alpha atom positions across all trajectory frames.
"""

import logging
import time
from typing import Any, Dict, List

import MDAnalysis as mda
import numpy as np
from MDAnalysis.lib.distances import distance_array

from ...schemas.analysis_bundle import ModuleResult

logger = logging.getLogger("md_ai_analyzer")

__version__ = "2.0.0"


def select_ca_atoms(universe: mda.Universe) -> mda.AtomGroup:
    """Helper to select CA atoms securely."""
    return universe.select_atoms("protein and name CA")


def compute_contact_map(universe: mda.Universe, cutoff: float = 8.0, **kwargs) -> ModuleResult:
    """Compute the average contact map and distance matrix between C-alpha atoms."""
    start_time = time.time()

    ca = select_ca_atoms(universe)
    n_res: int = len(ca)
    if n_res == 0:
        raise ValueError("No CA atoms found for contact map.")

    n_frames: int = len(universe.trajectory)
    logger.info(
        "Computing contact map for %d residues over %d frames (cutoff=%.1f A)",
        n_res, n_frames, cutoff,
    )

    contact_sum = np.zeros((n_res, n_res), dtype=np.float64)
    dist_sum = np.zeros((n_res, n_res), dtype=np.float64)

    for ts in universe.trajectory:
        dists = distance_array(ca.positions, ca.positions, box=ts.dimensions)
        contact_sum += (dists <= cutoff).astype(np.float64)
        dist_sum += dists

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
        persistent_contacts.append({
            "res_i": int(resids[sel_i[idx]]),
            "res_j": int(resids[sel_j[idx]]),
            "frequency": round(float(sel_freq[idx]), 3),
            "avg_distance": round(float(sel_dist[idx]), 2),
        })

    logger.info(
        "Contact map complete: %d persistent contacts (freq > 0.7)",
        len(persistent_contacts),
    )

    return ModuleResult(
        name="contacts",
        version=__version__,
        runtime_seconds=time.time() - start_time,
        parameters={"cutoff": cutoff},
        data={
            "contact_map": contact_freq.tolist(),
            "avg_distance_matrix": avg_dist.tolist(),
            "resids": resids,
            "n_residues": n_res,
            "persistent_contacts": persistent_contacts[:50],
            "cutoff_used": cutoff,
        }
    )
