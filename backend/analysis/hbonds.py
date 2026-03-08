"""Hydrogen bond analysis.

Computes hydrogen bonds over the trajectory using MDAnalysis
``HydrogenBondAnalysis`` and identifies persistent H-bonds and the most
frequent donor--acceptor pairs.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis

logger = logging.getLogger("md_ai_analyzer")


def compute_hbonds(
    universe: mda.Universe,
    distance: float = 3.5,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Compute hydrogen bonds over the trajectory.

    Parameters
    ----------
    universe : mda.Universe
        MDAnalysis Universe with a loaded trajectory.
    distance : float, optional
        Donor--acceptor distance cutoff in Angstrom (default 3.5).
    **kwargs : Any
        Additional keyword arguments (unused, accepted for API consistency).

    Returns
    -------
    dict[str, Any]
        ``time``
            List of timestamps (ps).
        ``n_hbonds``
            Number of hydrogen bonds per frame.
        ``mean_hbonds``
            Average H-bond count across all frames.
        ``std_hbonds``
            Standard deviation of H-bond count.
        ``persistent_hbonds``
            H-bonds present in >50 % of frames (up to 20).
        ``top_hbond_pairs``
            Most frequent donor--acceptor pairs (up to 20).
        ``total_unique_hbonds``
            Number of unique donor--acceptor pairs observed.
    """
    try:
        protein = universe.select_atoms("protein")
        if len(protein) == 0:
            logger.warning("No protein atoms found for H-bond analysis")
            return {"error": "No protein atoms found"}

        logger.info(
            "Running H-bond analysis with d_a_cutoff=%.1f A over %d frames",
            distance,
            len(universe.trajectory),
        )

        hbonds = HydrogenBondAnalysis(
            universe,
            donors_sel="protein",
            acceptors_sel="protein",
            d_a_cutoff=distance,
            d_h_a_angle_cutoff=150,
        )
        hbonds.run()

        # results shape: (N, 6) -> [frame, donor_idx, hydrogen_idx, acceptor_idx, dist, angle]
        results: np.ndarray = hbonds.results.hbonds
        n_frames: int = len(universe.trajectory)

        # --- Vectorised per-frame H-bond counts ---
        if len(results) > 0:
            frame_indices = results[:, 0].astype(int)
            hbond_counts = np.bincount(frame_indices, minlength=n_frames).tolist()

            # --- Vectorised pair counting ---
            donor_indices = results[:, 1].astype(int)
            acceptor_indices = results[:, 3].astype(int)
            # Build unique pair keys as (donor_idx, acceptor_idx)
            pair_keys = np.column_stack((donor_indices, acceptor_indices))
            unique_pairs, inverse, counts = np.unique(
                pair_keys, axis=0, return_inverse=True, return_counts=True
            )
            pair_counts: Dict[tuple, int] = {
                (int(p[0]), int(p[1])): int(c)
                for p, c in zip(unique_pairs, counts)
            }
        else:
            hbond_counts = np.zeros(n_frames, dtype=int).tolist()
            pair_counts = {}

        times: List[float] = [
            float(universe.trajectory[i].time) for i in range(n_frames)
        ]

        # --- Persistent H-bonds (>50 % occupancy) ---
        persistent: List[Dict[str, Any]] = []
        for (d, a), count in sorted(
            pair_counts.items(), key=lambda x: -x[1]
        ):
            occupancy = count / n_frames
            if occupancy < 0.5:
                break
            try:
                d_atom = universe.atoms[d]
                a_atom = universe.atoms[a]
                persistent.append(
                    {
                        "donor": f"{d_atom.resname}{d_atom.resid}:{d_atom.name}",
                        "acceptor": f"{a_atom.resname}{a_atom.resid}:{a_atom.name}",
                        "occupancy": round(occupancy, 3),
                    }
                )
            except (IndexError, AttributeError):
                logger.debug(
                    "Skipping atom pair (%d, %d): atom lookup failed", d, a
                )

        # --- Top donor-acceptor pairs ---
        top_pairs: List[Dict[str, Any]] = []
        for (d, a), count in sorted(
            pair_counts.items(), key=lambda x: -x[1]
        )[:20]:
            try:
                d_atom = universe.atoms[d]
                a_atom = universe.atoms[a]
                top_pairs.append(
                    {
                        "donor": f"{d_atom.resname}{d_atom.resid}:{d_atom.name}",
                        "acceptor": f"{a_atom.resname}{a_atom.resid}:{a_atom.name}",
                        "count": int(count),
                        "occupancy": round(count / n_frames, 3),
                    }
                )
            except (IndexError, AttributeError):
                logger.debug(
                    "Skipping atom pair (%d, %d): atom lookup failed", d, a
                )

        mean_hb = float(np.mean(hbond_counts))
        logger.info(
            "H-bond analysis complete: mean=%.1f, unique pairs=%d, persistent=%d",
            mean_hb,
            len(pair_counts),
            len(persistent),
        )

        return {
            "time": times,
            "n_hbonds": hbond_counts,
            "mean_hbonds": mean_hb,
            "std_hbonds": float(np.std(hbond_counts)),
            "persistent_hbonds": persistent[:20],
            "top_hbond_pairs": top_pairs,
            "total_unique_hbonds": len(pair_counts),
        }

    except Exception as e:
        logger.error("H-bond analysis failed: %s", e, exc_info=True)
        return {"error": str(e), "time": [], "n_hbonds": []}
