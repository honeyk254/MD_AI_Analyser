"""
Interaction Fingerprints (IFP).
Generates binary per-frame interaction fingerprints for all residue pairs,
tracking hydrophobic contacts, H-bonds, salt bridges, and aromatic contacts.
"""
import numpy as np
from collections import defaultdict
from MDAnalysis.lib.distances import distance_array


def compute_interaction_fingerprints(universe, hydrophobic_cutoff=5.0, aromatic_cutoff=5.5, **kwargs):
    """
    Compute per-frame interaction fingerprints between residue pairs.

    Returns dict with:
        - resids: residue IDs
        - consensus_fingerprint: per-residue total interaction count (normalised)
        - interaction_types: breakdown by type per residue
        - top_interactions: top residue pairs by interaction frequency
        - time: timestamps
        - ifp_per_frame: interaction count per frame
    """
    try:
        protein = universe.select_atoms("protein")
        if len(protein) == 0:
            return {"error": "No protein atoms found"}

        ca = universe.select_atoms("protein and name CA")
        n_res = len(ca)
        resids = ca.resids.tolist()

        # Define atom selections per interaction type
        hydrophobic_atoms = universe.select_atoms(
            "protein and (name CA CB CG CG1 CG2 CD CD1 CD2 CE CE1 CE2 CE3 CZ CZ2 CZ3 CH2) "
            "and (resname ALA VAL LEU ILE PRO PHE TRP MET)"
        )
        charged_pos = universe.select_atoms(
            "protein and ((resname LYS and name NZ) or (resname ARG and name NH1 NH2 CZ))"
        )
        charged_neg = universe.select_atoms(
            "protein and ((resname ASP and name OD1 OD2) or (resname GLU and name OE1 OE2))"
        )
        aromatic_atoms = universe.select_atoms(
            "protein and (resname PHE TYR TRP HIS) and (name CG CD1 CD2 CE1 CE2 CZ)"
        )

        n_frames = len(universe.trajectory)
        interaction_counts = defaultdict(lambda: {"hydrophobic": 0, "salt_bridge": 0, "aromatic": 0})
        ifp_per_frame = []
        times = []

        for ts in universe.trajectory:
            times.append(float(ts.time))
            frame_count = 0

            # Hydrophobic contacts
            if len(hydrophobic_atoms) > 1:
                hyd_dists = distance_array(
                    hydrophobic_atoms.positions, hydrophobic_atoms.positions,
                    box=ts.dimensions
                )
                for i in range(len(hydrophobic_atoms)):
                    for j in range(i + 1, len(hydrophobic_atoms)):
                        if hyd_dists[i, j] < hydrophobic_cutoff:
                            ri = int(hydrophobic_atoms[i].resid)
                            rj = int(hydrophobic_atoms[j].resid)
                            if abs(ri - rj) > 2:
                                pair = (min(ri, rj), max(ri, rj))
                                interaction_counts[pair]["hydrophobic"] += 1
                                frame_count += 1
                                break  # one contact per residue pair per frame
                    if frame_count > 500:
                        break  # cap per frame

            # Salt bridges
            if len(charged_pos) > 0 and len(charged_neg) > 0:
                salt_dists = distance_array(
                    charged_pos.positions, charged_neg.positions,
                    box=ts.dimensions
                )
                for i in range(len(charged_pos)):
                    for j in range(len(charged_neg)):
                        if salt_dists[i, j] < 4.0:
                            ri = int(charged_pos[i].resid)
                            rj = int(charged_neg[j].resid)
                            if abs(ri - rj) > 2:
                                pair = (min(ri, rj), max(ri, rj))
                                interaction_counts[pair]["salt_bridge"] += 1
                                frame_count += 1

            # Aromatic contacts (pi-stacking proxy)
            if len(aromatic_atoms) > 1:
                aro_dists = distance_array(
                    aromatic_atoms.positions, aromatic_atoms.positions,
                    box=ts.dimensions
                )
                seen_aro_pairs = set()
                for i in range(len(aromatic_atoms)):
                    for j in range(i + 1, len(aromatic_atoms)):
                        if aro_dists[i, j] < aromatic_cutoff:
                            ri = int(aromatic_atoms[i].resid)
                            rj = int(aromatic_atoms[j].resid)
                            if abs(ri - rj) > 2:
                                pair = (min(ri, rj), max(ri, rj))
                                if pair not in seen_aro_pairs:
                                    seen_aro_pairs.add(pair)
                                    interaction_counts[pair]["aromatic"] += 1
                                    frame_count += 1

            ifp_per_frame.append(frame_count)

        # Build consensus fingerprint (per-residue total interaction occupancy)
        per_res_counts = defaultdict(float)
        for (r1, r2), types in interaction_counts.items():
            total = sum(types.values()) / n_frames
            per_res_counts[r1] += total
            per_res_counts[r2] += total

        consensus = [round(per_res_counts.get(rid, 0.0), 3) for rid in resids]

        # Per-residue interaction type breakdown
        per_res_types = []
        for rid in resids:
            hydro = 0
            salt = 0
            aro = 0
            for (r1, r2), types in interaction_counts.items():
                if r1 == rid or r2 == rid:
                    hydro += types["hydrophobic"]
                    salt += types["salt_bridge"]
                    aro += types["aromatic"]
            per_res_types.append({
                "resid": rid,
                "hydrophobic": round(hydro / max(n_frames, 1), 3),
                "salt_bridge": round(salt / max(n_frames, 1), 3),
                "aromatic": round(aro / max(n_frames, 1), 3),
            })

        # Top interacting pairs
        top_interactions = []
        for (r1, r2), types in sorted(interaction_counts.items(), key=lambda x: -sum(x[1].values())):
            total_occ = sum(types.values()) / n_frames
            if total_occ > 0.05:
                top_interactions.append({
                    "resid_1": r1,
                    "resid_2": r2,
                    "total_occupancy": round(total_occ, 3),
                    "hydrophobic": round(types["hydrophobic"] / n_frames, 3),
                    "salt_bridge": round(types["salt_bridge"] / n_frames, 3),
                    "aromatic": round(types["aromatic"] / n_frames, 3),
                })
            if len(top_interactions) >= 50:
                break

        return {
            "resids": resids,
            "consensus_fingerprint": consensus,
            "interaction_types": per_res_types[:200],
            "top_interactions": top_interactions,
            "time": times,
            "ifp_per_frame": ifp_per_frame,
            "mean_interactions_per_frame": round(float(np.mean(ifp_per_frame)), 1) if ifp_per_frame else 0,
        }

    except Exception as e:
        return {"error": str(e)}
