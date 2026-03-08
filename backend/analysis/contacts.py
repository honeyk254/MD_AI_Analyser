"""
Contact map and distance matrix computation.
Computes average inter-residue contacts and distance matrices.
"""
import numpy as np
from MDAnalysis.lib.distances import distance_array


def compute_contact_map(universe, cutoff=8.0, **kwargs):
    """
    Compute average contact map and distance matrix between Cα atoms.

    Returns dict with:
        - contact_map: 2D list of contact frequencies [0-1]
        - avg_distance_matrix: 2D list of average distances (Å)
        - resids: list of residue IDs
        - n_residues: int
    """
    try:
        ca = universe.select_atoms("protein and name CA")
        if len(ca) == 0:
            return {"error": "No CA atoms found"}

        n_res = len(ca)
        contact_sum = np.zeros((n_res, n_res), dtype=np.float64)
        dist_sum = np.zeros((n_res, n_res), dtype=np.float64)
        n_frames = 0

        for ts in universe.trajectory:
            dists = distance_array(ca.positions, ca.positions, box=ts.dimensions)
            contact_sum += (dists <= cutoff).astype(float)
            dist_sum += dists
            n_frames += 1

        contact_freq = contact_sum / n_frames
        avg_dist = dist_sum / n_frames

        resids = ca.resids.tolist()

        # Find most persistent contacts (off-diagonal, > 5 residues apart)
        persistent_contacts = []
        for i in range(n_res):
            for j in range(i + 5, n_res):
                if contact_freq[i, j] > 0.7:
                    persistent_contacts.append({
                        "res_i": int(resids[i]),
                        "res_j": int(resids[j]),
                        "frequency": round(float(contact_freq[i, j]), 3),
                        "avg_distance": round(float(avg_dist[i, j]), 2),
                    })

        persistent_contacts.sort(key=lambda x: -x["frequency"])

        return {
            "contact_map": contact_freq.tolist(),
            "avg_distance_matrix": avg_dist.tolist(),
            "resids": resids,
            "n_residues": n_res,
            "persistent_contacts": persistent_contacts[:50],
            "cutoff_used": cutoff,
        }

    except Exception as e:
        return {"error": str(e)}
