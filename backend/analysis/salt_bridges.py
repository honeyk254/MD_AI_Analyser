"""Salt bridge detection.

Identifies charged residue pairs forming salt bridges across the
trajectory using distance-based criteria between canonical charge
centres (Arg CZ, Lys NZ, His NE2, Asp CG, Glu CD).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array

logger = logging.getLogger("md_ai_analyzer")


def compute_salt_bridges(
    universe: mda.Universe,
    cutoff: float = 4.0,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Detect salt bridges between oppositely charged residues.

    Parameters
    ----------
    universe : mda.Universe
        MDAnalysis Universe with a loaded trajectory.
    cutoff : float, optional
        Distance cutoff in Angstrom for salt-bridge detection (default 4.0).
    **kwargs : Any
        Additional keyword arguments (unused, accepted for API consistency).

    Returns
    -------
    dict[str, Any]
        ``pairs``
            Top 30 salt-bridge pairs with occupancy information.
        ``time``
            List of timestamps (ps).
        ``total_per_frame``
            Number of salt bridges per frame.
        ``mean_salt_bridges``
            Mean number of salt bridges per frame.
        ``total_unique_pairs``
            Total number of unique charged-residue pairs observed.
    """
    try:
        # Positively charged: Arg (CZ), Lys (NZ), His (NE2)
        pos_sel = universe.select_atoms(
            "(resname ARG and name CZ) or "
            "(resname LYS and name NZ) or "
            "(resname HIS and name NE2)"
        )
        # Negatively charged: Asp (CG), Glu (CD)
        neg_sel = universe.select_atoms(
            "(resname ASP and name CG) or (resname GLU and name CD)"
        )

        if len(pos_sel) == 0 or len(neg_sel) == 0:
            logger.warning(
                "Insufficient charged residues for salt-bridge analysis "
                "(pos=%d, neg=%d)",
                len(pos_sel),
                len(neg_sel),
            )
            return {
                "pairs": [],
                "time": [],
                "total_per_frame": [],
                "error": "Insufficient charged residues",
            }

        logger.info(
            "Salt-bridge analysis: %d positive, %d negative centres, cutoff=%.1f A",
            len(pos_sel),
            len(neg_sel),
            cutoff,
        )

        # Pre-compute residue labels (outside the frame loop)
        pos_labels: List[str] = [
            f"{a.resname}{a.resid}" for a in pos_sel
        ]
        neg_labels: List[str] = [
            f"{a.resname}{a.resid}" for a in neg_sel
        ]

        pair_contacts: Dict[tuple, int] = {}
        n_frames = len(universe.trajectory)
        times = np.empty(n_frames, dtype=np.float64)
        total_per_frame = np.empty(n_frames, dtype=np.intp)

        for fi, ts in enumerate(universe.trajectory):
            times[fi] = ts.time
            dists = distance_array(
                pos_sel.positions, neg_sel.positions, box=ts.dimensions
            )

            # Vectorised: find all pairs within cutoff
            contacts_ij = np.argwhere(dists <= cutoff)
            total_per_frame[fi] = len(contacts_ij)

            for i, j in contacts_ij:
                key = (pos_labels[i], neg_labels[j])
                pair_contacts[key] = pair_contacts.get(key, 0) + 1

        # Build sorted pair list
        pairs: List[Dict[str, Any]] = []
        for (pos, neg), count in sorted(
            pair_contacts.items(), key=lambda x: -x[1]
        ):
            pairs.append(
                {
                    "positive": pos,
                    "negative": neg,
                    "count": int(count),
                    "occupancy": round(count / n_frames, 3),
                }
            )

        mean_sb = float(np.mean(total_per_frame))
        logger.info(
            "Salt-bridge analysis complete: mean=%.1f per frame, "
            "%d unique pairs",
            mean_sb,
            len(pair_contacts),
        )

        return {
            "pairs": pairs[:30],
            "time": times.tolist(),
            "total_per_frame": total_per_frame.tolist(),
            "mean_salt_bridges": mean_sb,
            "total_unique_pairs": len(pair_contacts),
        }

    except Exception as e:
        logger.error("Salt-bridge analysis failed: %s", e, exc_info=True)
        return {"error": str(e), "pairs": [], "time": [], "total_per_frame": []}
