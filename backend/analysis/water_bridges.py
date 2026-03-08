"""
Water Bridge Analysis.
Detects water molecules that persistently bridge protein residue pairs
via hydrogen bonds (Protein-Water-Protein bridges).
"""
import numpy as np
from collections import defaultdict


def compute_water_bridges(universe, distance_cutoff=3.5, angle_cutoff=130, occupancy_threshold=0.1, **kwargs):
    """
    Detect water-mediated hydrogen bond bridges between protein residues.

    Returns dict with:
        - bridges: list of top water bridges with donor/acceptor residue info and occupancy
        - per_residue_bridge_count: how many water bridges each residue participates in
        - n_bridges_per_frame: water bridge count per frame
        - time: list of timestamps
        - mean_bridges: average water bridges per frame
    """
    try:
        protein = universe.select_atoms("protein")
        water = universe.select_atoms("resname SOL TIP3 HOH WAT TIP4 SPC")

        if len(protein) == 0:
            return {"error": "No protein atoms found"}
        if len(water) == 0:
            return {"error": "No water molecules found", "bridges": [], "per_residue_bridge_count": []}

        # Protein heavy atoms that can be donors/acceptors (N, O, S)
        protein_polar = universe.select_atoms("protein and (name N* O* S*) and not name CA CB")
        if len(protein_polar) == 0:
            protein_polar = universe.select_atoms("protein and (type N O S)")

        water_oxygen = universe.select_atoms("(resname SOL TIP3 HOH WAT TIP4 SPC) and (name OW O OH2)")

        if len(water_oxygen) == 0:
            return {"error": "No water oxygens found", "bridges": [], "per_residue_bridge_count": []}

        n_frames = len(universe.trajectory)
        bridge_counts = defaultdict(int)
        bridges_per_frame = []
        times = []

        ca_atoms = universe.select_atoms("protein and name CA")
        resids = ca_atoms.resids.tolist() if len(ca_atoms) > 0 else []

        for ts in universe.trajectory:
            times.append(float(ts.time))
            frame_bridges = set()

            # Find water oxygens close to protein polar atoms
            from MDAnalysis.lib.distances import distance_array
            dist_mat = distance_array(water_oxygen.positions, protein_polar.positions, box=ts.dimensions)

            # For each water, find all protein atoms within cutoff
            for w_idx in range(len(water_oxygen)):
                close_protein = np.where(dist_mat[w_idx] < distance_cutoff)[0]
                if len(close_protein) >= 2:
                    # This water bridges at least 2 protein atoms
                    close_resids = set()
                    for p_idx in close_protein:
                        atom = protein_polar[p_idx]
                        close_resids.add(int(atom.resid))

                    resid_list = sorted(close_resids)
                    # Record all unique residue pairs bridged by this water
                    for i in range(len(resid_list)):
                        for j in range(i + 1, len(resid_list)):
                            if abs(resid_list[i] - resid_list[j]) > 2:  # skip sequential neighbors
                                pair = (resid_list[i], resid_list[j])
                                frame_bridges.add(pair)

            for pair in frame_bridges:
                bridge_counts[pair] += 1

            bridges_per_frame.append(len(frame_bridges))

        # Build results
        bridges = []
        for (r1, r2), count in sorted(bridge_counts.items(), key=lambda x: -x[1]):
            occ = count / n_frames
            if occ >= occupancy_threshold:
                bridges.append({
                    "resid_1": r1,
                    "resid_2": r2,
                    "occupancy": round(occ, 3),
                    "frame_count": count,
                })

        # Per-residue bridge participation count
        residue_bridge_count = defaultdict(int)
        for (r1, r2), count in bridge_counts.items():
            occ = count / n_frames
            if occ >= occupancy_threshold:
                residue_bridge_count[r1] += 1
                residue_bridge_count[r2] += 1

        per_res = [0] * len(resids)
        for idx, rid in enumerate(resids):
            per_res[idx] = residue_bridge_count.get(rid, 0)

        return {
            "bridges": bridges[:50],
            "per_residue_bridge_count": per_res,
            "resids": resids,
            "n_bridges_per_frame": bridges_per_frame,
            "time": times,
            "mean_bridges": float(np.mean(bridges_per_frame)) if bridges_per_frame else 0,
            "total_unique_bridges": len([b for b in bridges if b["occupancy"] >= occupancy_threshold]),
        }

    except Exception as e:
        return {"error": str(e), "bridges": [], "per_residue_bridge_count": []}
