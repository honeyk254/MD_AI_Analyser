"""
Hydrogen bond analysis.
Computes hydrogen bonds over the trajectory and identifies persistent H-bonds.
"""
import numpy as np
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis


def compute_hbonds(universe, distance=3.5, **kwargs):
    """
    Compute hydrogen bonds over the trajectory.

    Returns dict with:
        - time: list of timestamps
        - n_hbonds: number of H-bonds per frame
        - mean_hbonds: average H-bond count
        - persistent_hbonds: H-bonds present in >50% of frames
        - top_hbond_pairs: most frequent donor-acceptor pairs
    """
    try:
        protein = universe.select_atoms("protein")
        if len(protein) == 0:
            return {"error": "No protein atoms found"}

        hbonds = HydrogenBondAnalysis(
            universe,
            donors_sel="protein",
            acceptors_sel="protein",
            d_a_cutoff=distance,
            d_h_a_angle_cutoff=150,
        )
        hbonds.run()

        results = hbonds.results.hbonds  # [frame, donor_idx, hydrogen_idx, acceptor_idx, dist, angle]

        # Count per frame
        n_frames = len(universe.trajectory)
        hbond_counts = np.zeros(n_frames, dtype=int)
        pair_counts = {}

        for row in results:
            frame = int(row[0])
            donor_idx = int(row[1])
            acceptor_idx = int(row[3])
            hbond_counts[frame] += 1

            pair_key = (donor_idx, acceptor_idx)
            pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1

        times = [float(universe.trajectory[i].time) for i in range(n_frames)]

        # Persistent H-bonds (>50% occupancy)
        threshold = n_frames * 0.5
        persistent = []
        for (d, a), count in sorted(pair_counts.items(), key=lambda x: -x[1]):
            occupancy = count / n_frames
            if occupancy >= 0.5:
                try:
                    d_atom = universe.atoms[d]
                    a_atom = universe.atoms[a]
                    persistent.append({
                        "donor": f"{d_atom.resname}{d_atom.resid}:{d_atom.name}",
                        "acceptor": f"{a_atom.resname}{a_atom.resid}:{a_atom.name}",
                        "occupancy": round(occupancy, 3),
                    })
                except (IndexError, AttributeError):
                    pass

        # Top pairs
        top_pairs = []
        for (d, a), count in sorted(pair_counts.items(), key=lambda x: -x[1])[:20]:
            try:
                d_atom = universe.atoms[d]
                a_atom = universe.atoms[a]
                top_pairs.append({
                    "donor": f"{d_atom.resname}{d_atom.resid}:{d_atom.name}",
                    "acceptor": f"{a_atom.resname}{a_atom.resid}:{a_atom.name}",
                    "count": int(count),
                    "occupancy": round(count / n_frames, 3),
                })
            except (IndexError, AttributeError):
                pass

        return {
            "time": times,
            "n_hbonds": hbond_counts.tolist(),
            "mean_hbonds": float(np.mean(hbond_counts)),
            "std_hbonds": float(np.std(hbond_counts)),
            "persistent_hbonds": persistent[:20],
            "top_hbond_pairs": top_pairs,
            "total_unique_hbonds": len(pair_counts),
        }

    except Exception as e:
        return {"error": str(e), "time": [], "n_hbonds": []}
