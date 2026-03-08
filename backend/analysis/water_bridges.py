"""Water Bridge Analysis.

Detects water molecules that persistently bridge protein residue pairs
via hydrogen bonds (Protein--Water--Protein bridges).  A water bridge is
recorded when a single water oxygen is simultaneously within the distance
cutoff of polar atoms belonging to two different non-sequential residues.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Set

import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array

logger = logging.getLogger("md_ai_analyzer")


def compute_water_bridges(
    universe: mda.Universe,
    distance_cutoff: float = 3.5,
    angle_cutoff: float = 130.0,
    occupancy_threshold: float = 0.1,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Detect water-mediated hydrogen-bond bridges between protein residues.

    For each frame, the algorithm:

    1. Computes distances from every water oxygen to every protein polar
       heavy atom.
    2. For each water oxygen that is within *distance_cutoff* of at least
       two protein polar atoms, identifies the set of bridged residue IDs.
    3. Records all unique residue pairs (separated by > 2 in sequence)
       bridged by that water.

    Parameters
    ----------
    universe : mda.Universe
        MDAnalysis Universe with a loaded trajectory.
    distance_cutoff : float, optional
        Maximum water--protein polar-atom distance in Angstrom (default 3.5).
    angle_cutoff : float, optional
        Angle cutoff in degrees (reserved for future use; default 130).
    occupancy_threshold : float, optional
        Minimum bridge occupancy to include in results (default 0.1).
    **kwargs : Any
        Additional keyword arguments (unused, accepted for API consistency).

    Returns
    -------
    dict[str, Any]
        ``bridges``
            Top 50 water bridges above the occupancy threshold.
        ``per_residue_bridge_count``
            Number of qualifying water bridges each residue participates in.
        ``resids``
            Residue IDs (from CA atoms).
        ``n_bridges_per_frame``
            Number of unique bridging events per frame.
        ``time``
            List of timestamps (ps).
        ``mean_bridges``
            Mean bridges per frame.
        ``total_unique_bridges``
            Count of unique bridge pairs above the occupancy threshold.
    """
    try:
        protein = universe.select_atoms("protein")
        water = universe.select_atoms("resname SOL TIP3 HOH WAT TIP4 SPC")

        if len(protein) == 0:
            logger.warning("No protein atoms found for water-bridge analysis")
            return {"error": "No protein atoms found"}
        if len(water) == 0:
            logger.warning("No water molecules found for water-bridge analysis")
            return {
                "error": "No water molecules found",
                "bridges": [],
                "per_residue_bridge_count": [],
            }

        # Protein heavy polar atoms (N, O, S) excluding backbone carbons
        protein_polar = universe.select_atoms(
            "protein and (name N* O* S*) and not name CA CB"
        )
        if len(protein_polar) == 0:
            protein_polar = universe.select_atoms("protein and (type N O S)")

        water_oxygen = universe.select_atoms(
            "(resname SOL TIP3 HOH WAT TIP4 SPC) and (name OW O OH2)"
        )
        if len(water_oxygen) == 0:
            logger.warning("No water oxygens found")
            return {
                "error": "No water oxygens found",
                "bridges": [],
                "per_residue_bridge_count": [],
            }

        n_frames: int = len(universe.trajectory)
        logger.info(
            "Water-bridge analysis: %d polar atoms, %d water oxygens, "
            "%d frames, cutoff=%.1f A",
            len(protein_polar),
            len(water_oxygen),
            n_frames,
            distance_cutoff,
        )

        # Pre-extract residue IDs for the polar atoms (vectorised lookup)
        polar_resids: np.ndarray = np.array(
            [int(a.resid) for a in protein_polar], dtype=np.intp
        )

        ca_atoms = universe.select_atoms("protein and name CA")
        resids: List[int] = ca_atoms.resids.tolist() if len(ca_atoms) > 0 else []

        bridge_counts: Dict[tuple, int] = defaultdict(int)
        bridges_per_frame: List[int] = []
        times = np.empty(n_frames, dtype=np.float64)

        for fi, ts in enumerate(universe.trajectory):
            times[fi] = ts.time
            frame_bridges: Set[tuple] = set()

            dist_mat = distance_array(
                water_oxygen.positions, protein_polar.positions, box=ts.dimensions
            )

            # For each water oxygen, find protein atoms within cutoff
            for w_idx in range(len(water_oxygen)):
                close_protein = np.where(dist_mat[w_idx] < distance_cutoff)[0]
                if len(close_protein) < 2:
                    continue

                # Unique residue IDs bridged by this water
                close_resid_set: set = set(polar_resids[close_protein].tolist())
                resid_list = sorted(close_resid_set)

                # Record all unique non-sequential residue pairs
                for a in range(len(resid_list)):
                    for b in range(a + 1, len(resid_list)):
                        if abs(resid_list[a] - resid_list[b]) > 2:
                            frame_bridges.add((resid_list[a], resid_list[b]))

            for pair in frame_bridges:
                bridge_counts[pair] += 1

            bridges_per_frame.append(len(frame_bridges))

        # --- Build results ---
        bridges: List[Dict[str, Any]] = []
        for (r1, r2), count in sorted(
            bridge_counts.items(), key=lambda x: -x[1]
        ):
            occ = count / n_frames
            if occ >= occupancy_threshold:
                bridges.append(
                    {
                        "resid_1": r1,
                        "resid_2": r2,
                        "occupancy": round(occ, 3),
                        "frame_count": count,
                    }
                )

        # Per-residue bridge participation
        residue_bridge_count: Dict[int, int] = defaultdict(int)
        for (r1, r2), count in bridge_counts.items():
            if count / n_frames >= occupancy_threshold:
                residue_bridge_count[r1] += 1
                residue_bridge_count[r2] += 1

        per_res: List[int] = [
            residue_bridge_count.get(rid, 0) for rid in resids
        ]

        mean_br = float(np.mean(bridges_per_frame)) if bridges_per_frame else 0.0
        logger.info(
            "Water-bridge analysis complete: mean=%.1f per frame, "
            "%d unique bridges above threshold",
            mean_br,
            len(bridges),
        )

        return {
            "bridges": bridges[:50],
            "per_residue_bridge_count": per_res,
            "resids": resids,
            "n_bridges_per_frame": bridges_per_frame,
            "time": times.tolist(),
            "mean_bridges": mean_br,
            "total_unique_bridges": len(
                [b for b in bridges if b["occupancy"] >= occupancy_threshold]
            ),
        }

    except Exception as e:
        logger.error("Water-bridge analysis failed: %s", e, exc_info=True)
        return {"error": str(e), "bridges": [], "per_residue_bridge_count": []}
