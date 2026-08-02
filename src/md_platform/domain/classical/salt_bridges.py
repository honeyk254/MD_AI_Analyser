"""Salt bridge detection.

Identifies charged residue pairs forming salt bridges across the
trajectory using distance-based criteria between canonical charge centres.
"""

import time
import logging
from typing import Any, Dict, List
import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array

from ...schemas.analysis_bundle import ModuleResult, MetricSummary

logger = logging.getLogger("md_ai_analyzer")

__version__ = "2.0.0"


def compute_salt_bridges(universe: mda.Universe, cutoff: float = 4.0, **kwargs) -> ModuleResult:
    """Detect salt bridges between oppositely charged residues."""
    start_time = time.time()
    
    pos_sel = universe.select_atoms(
        "(resname ARG and name CZ) or "
        "(resname LYS and name NZ) or "
        "(resname HIP HSP and name NE2)"
    )
    neg_sel = universe.select_atoms(
        "(resname ASP and name CG) or "
        "(resname GLU and name CD)"
    )

    if len(pos_sel) == 0 or len(neg_sel) == 0:
        raise ValueError(f"Insufficient charged residues for salt-bridge analysis (pos={len(pos_sel)}, neg={len(neg_sel)})")

    logger.info(
        "Salt-bridge analysis: %d positive, %d negative centres, cutoff=%.1f A",
        len(pos_sel), len(neg_sel), cutoff,
    )

    pos_labels = [f"{a.resname}{a.resid}" for a in pos_sel]
    neg_labels = [f"{a.resname}{a.resid}" for a in neg_sel]

    pair_contacts: Dict[tuple, int] = {}
    n_frames = len(universe.trajectory)
    times = np.empty(n_frames, dtype=np.float64)
    total_per_frame = np.empty(n_frames, dtype=np.intp)

    for fi, ts in enumerate(universe.trajectory):
        times[fi] = ts.time
        dists = distance_array(
            pos_sel.positions, neg_sel.positions, box=ts.dimensions
        )
        contacts_ij = np.argwhere(dists <= cutoff)
        total_per_frame[fi] = len(contacts_ij)

        for i, j in contacts_ij:
            key = (pos_labels[i], neg_labels[j])
            pair_contacts[key] = pair_contacts.get(key, 0) + 1

    pairs: List[Dict[str, Any]] = []
    for (pos, neg), count in sorted(pair_contacts.items(), key=lambda x: -x[1]):
        pairs.append({
            "positive": pos,
            "negative": neg,
            "count": int(count),
            "occupancy": round(count / n_frames, 3),
        })

    mean_sb = float(np.mean(total_per_frame))
    logger.info(
        "Salt-bridge analysis complete: mean=%.1f per frame, %d unique pairs",
        mean_sb, len(pair_contacts),
    )

    return ModuleResult(
        name="salt_bridges",
        version=__version__,
        runtime_seconds=time.time() - start_time,
        parameters={"cutoff": cutoff},
        scalar_metrics={
            "salt_bridge_count": MetricSummary(
                mean=mean_sb,
                std=float(np.std(total_per_frame)),
                min=float(np.min(total_per_frame)),
                max=float(np.max(total_per_frame)),
                unit="count",
                n_frames=n_frames,
                time_series=total_per_frame.tolist(),
            )
        },
        data={
            "time_ps": times.tolist(),
            "pairs": pairs[:30],
            "total_unique_pairs": len(pair_contacts),
        }
    )
