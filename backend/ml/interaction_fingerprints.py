from __future__ import annotations

"""Interaction Fingerprints (IFP).

Generates binary per-frame interaction fingerprints for all residue pairs,
tracking hydrophobic contacts, hydrogen bonds (proxy), salt bridges, and
aromatic contacts.  Uses vectorised numpy boolean indexing on precomputed
distance matrices to avoid O(n^2) Python loops.
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from MDAnalysis.lib.distances import distance_array

logger = logging.getLogger("md_ai_analyzer")

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_interaction_fingerprints(
    universe: Any,
    hydrophobic_cutoff: float = 5.0,
    aromatic_cutoff: float = 5.5,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Compute per-frame interaction fingerprints between residue pairs.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        Loaded trajectory universe.
    hydrophobic_cutoff : float
        Distance cutoff (angstroms) for hydrophobic contacts.
    aromatic_cutoff : float
        Distance cutoff (angstroms) for aromatic (pi-stacking proxy)
        contacts.
    **kwargs
        Additional keyword arguments (unused; accepted for orchestrator
        compatibility).

    Returns
    -------
    dict
        Keys:
        - ``resids`` : residue IDs
        - ``consensus_fingerprint`` : per-residue normalised interaction count
        - ``interaction_types`` : per-residue breakdown by interaction type
        - ``top_interactions`` : top residue pairs by interaction frequency
        - ``time`` : timestamps
        - ``ifp_per_frame`` : interaction count per frame
        - ``mean_interactions_per_frame`` : average interactions per frame
    """
    try:
        protein = universe.select_atoms("protein")
        if len(protein) == 0:
            logger.error("No protein atoms found for IFP analysis.")
            return {"error": "No protein atoms found"}

        ca = universe.select_atoms("protein and name CA")
        n_res = len(ca)
        resids: List[int] = ca.resids.tolist()

        # ── Atom selections per interaction type ─────────────────
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

        # Pre-compute residue IDs for atom groups (avoids per-atom lookup
        # inside the frame loop).
        hyd_resids = hydrophobic_atoms.resids if len(hydrophobic_atoms) > 0 else np.array([], dtype=int)
        aro_resids = aromatic_atoms.resids if len(aromatic_atoms) > 0 else np.array([], dtype=int)

        n_frames = len(universe.trajectory)
        interaction_counts: Dict[Tuple[int, int], Dict[str, int]] = defaultdict(
            lambda: {"hydrophobic": 0, "salt_bridge": 0, "aromatic": 0}
        )
        ifp_per_frame: List[int] = []
        times: List[float] = []

        for ts in universe.trajectory:
            times.append(float(ts.time))
            frame_count = 0

            # ── Hydrophobic contacts (vectorised) ────────────────
            if len(hydrophobic_atoms) > 1:
                hyd_dists = distance_array(
                    hydrophobic_atoms.positions,
                    hydrophobic_atoms.positions,
                    box=ts.dimensions,
                )
                n_hyd = len(hydrophobic_atoms)
                # Upper triangle: i < j
                row_idx, col_idx = np.triu_indices(n_hyd, k=1)
                close_mask = hyd_dists[row_idx, col_idx] < hydrophobic_cutoff

                close_rows = row_idx[close_mask]
                close_cols = col_idx[close_mask]
                ri_arr = hyd_resids[close_rows]
                rj_arr = hyd_resids[close_cols]

                # Filter: sequence separation > 2
                sep_mask = np.abs(ri_arr.astype(np.int64) - rj_arr.astype(np.int64)) > 2
                ri_filt = ri_arr[sep_mask]
                rj_filt = rj_arr[sep_mask]

                # Canonical pair ordering and deduplication per frame
                pairs_min = np.minimum(ri_filt, rj_filt)
                pairs_max = np.maximum(ri_filt, rj_filt)
                seen_hyd: Set[Tuple[int, int]] = set()
                for pm, px in zip(pairs_min, pairs_max):
                    pair = (int(pm), int(px))
                    if pair not in seen_hyd:
                        seen_hyd.add(pair)
                        interaction_counts[pair]["hydrophobic"] += 1
                        frame_count += 1
                    if frame_count > 500:
                        break

            # ── Salt bridges (vectorised) ────────────────────────
            if len(charged_pos) > 0 and len(charged_neg) > 0:
                salt_dists = distance_array(
                    charged_pos.positions,
                    charged_neg.positions,
                    box=ts.dimensions,
                )
                close_i, close_j = np.where(salt_dists < 4.0)
                cp_resids = charged_pos.resids
                cn_resids = charged_neg.resids
                for ci, cj in zip(close_i, close_j):
                    ri = int(cp_resids[ci])
                    rj = int(cn_resids[cj])
                    if abs(ri - rj) > 2:
                        pair = (min(ri, rj), max(ri, rj))
                        interaction_counts[pair]["salt_bridge"] += 1
                        frame_count += 1

            # ── Aromatic contacts (vectorised) ───────────────────
            if len(aromatic_atoms) > 1:
                aro_dists = distance_array(
                    aromatic_atoms.positions,
                    aromatic_atoms.positions,
                    box=ts.dimensions,
                )
                n_aro = len(aromatic_atoms)
                row_idx_a, col_idx_a = np.triu_indices(n_aro, k=1)
                close_mask_a = aro_dists[row_idx_a, col_idx_a] < aromatic_cutoff

                close_rows_a = row_idx_a[close_mask_a]
                close_cols_a = col_idx_a[close_mask_a]
                ri_aro = aro_resids[close_rows_a]
                rj_aro = aro_resids[close_cols_a]

                sep_mask_a = np.abs(ri_aro.astype(np.int64) - rj_aro.astype(np.int64)) > 2
                ri_af = ri_aro[sep_mask_a]
                rj_af = rj_aro[sep_mask_a]

                pairs_min_a = np.minimum(ri_af, rj_af)
                pairs_max_a = np.maximum(ri_af, rj_af)
                seen_aro: Set[Tuple[int, int]] = set()
                for pm, px in zip(pairs_min_a, pairs_max_a):
                    pair = (int(pm), int(px))
                    if pair not in seen_aro:
                        seen_aro.add(pair)
                        interaction_counts[pair]["aromatic"] += 1
                        frame_count += 1

            ifp_per_frame.append(frame_count)

        # ── Consensus fingerprint (per-residue total occupancy) ──
        per_res_counts: Dict[int, float] = defaultdict(float)
        for (r1, r2), types in interaction_counts.items():
            total = sum(types.values()) / max(n_frames, 1)
            per_res_counts[r1] += total
            per_res_counts[r2] += total

        consensus: List[float] = [
            round(per_res_counts.get(rid, 0.0), 3) for rid in resids
        ]

        # ── Per-residue interaction type breakdown ───────────────
        # Build lookup dicts keyed on resid for O(1) access
        per_res_hyd: Dict[int, int] = defaultdict(int)
        per_res_salt: Dict[int, int] = defaultdict(int)
        per_res_aro: Dict[int, int] = defaultdict(int)
        for (r1, r2), types in interaction_counts.items():
            per_res_hyd[r1] += types["hydrophobic"]
            per_res_hyd[r2] += types["hydrophobic"]
            per_res_salt[r1] += types["salt_bridge"]
            per_res_salt[r2] += types["salt_bridge"]
            per_res_aro[r1] += types["aromatic"]
            per_res_aro[r2] += types["aromatic"]

        nf = max(n_frames, 1)
        per_res_types: List[Dict[str, Any]] = [
            {
                "resid": rid,
                "hydrophobic": round(per_res_hyd.get(rid, 0) / nf, 3),
                "salt_bridge": round(per_res_salt.get(rid, 0) / nf, 3),
                "aromatic": round(per_res_aro.get(rid, 0) / nf, 3),
            }
            for rid in resids
        ]

        # ── Top interacting pairs ────────────────────────────────
        top_interactions: List[Dict[str, Any]] = []
        sorted_pairs = sorted(
            interaction_counts.items(), key=lambda x: -sum(x[1].values())
        )
        for (r1, r2), types in sorted_pairs:
            total_occ = sum(types.values()) / nf
            if total_occ > 0.05:
                top_interactions.append(
                    {
                        "resid_1": r1,
                        "resid_2": r2,
                        "total_occupancy": round(total_occ, 3),
                        "hydrophobic": round(types["hydrophobic"] / nf, 3),
                        "salt_bridge": round(types["salt_bridge"] / nf, 3),
                        "aromatic": round(types["aromatic"] / nf, 3),
                    }
                )
            if len(top_interactions) >= 50:
                break

        logger.info(
            "IFP analysis complete: %d unique pairs, mean %.1f interactions/frame.",
            len(interaction_counts),
            float(np.mean(ifp_per_frame)) if ifp_per_frame else 0.0,
        )

        return {
            "resids": resids,
            "consensus_fingerprint": consensus,
            "interaction_types": per_res_types[:200],
            "top_interactions": top_interactions,
            "time": times,
            "ifp_per_frame": ifp_per_frame,
            "mean_interactions_per_frame": round(
                float(np.mean(ifp_per_frame)), 1
            )
            if ifp_per_frame
            else 0,
        }

    except Exception as e:
        logger.exception("Interaction fingerprint analysis failed: %s", e)
        return {"error": str(e)}
