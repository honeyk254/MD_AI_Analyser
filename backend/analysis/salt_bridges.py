"""
Salt bridge detection.
Identifies charged residue pairs forming salt bridges.
"""
import numpy as np
from MDAnalysis.lib.distances import distance_array


def compute_salt_bridges(universe, cutoff=4.0, **kwargs):
    """
    Detect salt bridges between oppositely charged residues.

    Returns dict with:
        - pairs: list of salt bridge pairs with occupancy
        - time: timestamps
        - total_per_frame: salt bridge count per frame
    """
    try:
        # Positively charged: Arg (CZ), Lys (NZ), His (NE2)
        pos_sel = universe.select_atoms(
            "(resname ARG and name CZ) or (resname LYS and name NZ) or (resname HIS and name NE2)"
        )
        # Negatively charged: Asp (CG), Glu (CD)
        neg_sel = universe.select_atoms(
            "(resname ASP and name CG) or (resname GLU and name CD)"
        )

        if len(pos_sel) == 0 or len(neg_sel) == 0:
            return {"pairs": [], "time": [], "total_per_frame": [], "error": "Insufficient charged residues"}

        pair_contacts = {}
        times = []
        total_per_frame = []

        for ts in universe.trajectory:
            times.append(float(ts.time))
            dists = distance_array(pos_sel.positions, neg_sel.positions, box=ts.dimensions)
            count = 0
            for i, p_atom in enumerate(pos_sel):
                for j, n_atom in enumerate(neg_sel):
                    if dists[i, j] <= cutoff:
                        key = (f"{p_atom.resname}{p_atom.resid}", f"{n_atom.resname}{n_atom.resid}")
                        pair_contacts[key] = pair_contacts.get(key, 0) + 1
                        count += 1
            total_per_frame.append(count)

        n_frames = len(times)
        pairs = []
        for (pos, neg), count in sorted(pair_contacts.items(), key=lambda x: -x[1]):
            pairs.append({
                "positive": pos,
                "negative": neg,
                "count": int(count),
                "occupancy": round(count / n_frames, 3),
            })

        return {
            "pairs": pairs[:30],
            "time": times,
            "total_per_frame": total_per_frame,
            "mean_salt_bridges": float(np.mean(total_per_frame)) if total_per_frame else 0,
            "total_unique_pairs": len(pair_contacts),
        }

    except Exception as e:
        return {"error": str(e), "pairs": [], "time": [], "total_per_frame": []}
