"""Salt bridge detection.

A salt bridge is counted when any nitrogen of a cationic side chain is within
``cutoff`` of any carboxylate/carboxylic oxygen of an anionic side chain
(Barlow & Thornton, J. Mol. Biol. 1983, who use a 4 A N-O criterion).

Using the carboxylate *oxygens* rather than the carboxylate carbon matters: the
carbon sits ~1.25 A further from the interacting nitrogen, so a carbon-based
4 A cutoff systematically under-counts genuine bridges.
"""

import time
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array

from ...schemas.analysis_bundle import ModuleResult, MetricSummary
from ..frames import FrameWindow, iter_frames

logger = logging.getLogger("md_ai_analyzer")

__version__ = "2.1.0"

# Cationic side-chain nitrogens: Arg guanidinium, Lys ammonium, and histidine
# only where the topology names it as explicitly protonated (HIP/HSP). Neutral
# HIS is left out rather than assumed charged.
CATION_SELECTION = (
    "(resname ARG and name NE NH1 NH2) or "
    "(resname LYS and name NZ) or "
    "(resname HIP HSP and name ND1 NE2)"
)
# Anionic side-chain oxygens (Asp/Glu carboxylates, including protonated
# naming variants used by CHARMM/AMBER).
ANION_SELECTION = (
    "(resname ASP ASH and name OD1 OD2) or "
    "(resname GLU GLH and name OE1 OE2)"
)


def _residue_key(atom: mda.core.groups.Atom) -> str:
    segid = getattr(atom, "segid", "") or ""
    prefix = f"{segid}:" if segid else ""
    return f"{prefix}{atom.resname}{atom.resid}"


def compute_salt_bridges(
    universe: mda.Universe,
    cutoff: float = 4.0,
    window: Optional[FrameWindow] = None,
    **kwargs,
) -> ModuleResult:
    """Detect salt bridges between oppositely charged side chains."""
    start_time = time.time()

    pos_sel = universe.select_atoms(CATION_SELECTION)
    neg_sel = universe.select_atoms(ANION_SELECTION)

    if len(pos_sel) == 0 or len(neg_sel) == 0:
        raise ValueError(
            "Insufficient charged side chains for salt-bridge analysis "
            f"(cationic atoms={len(pos_sel)}, anionic atoms={len(neg_sel)})"
        )

    logger.info(
        "Salt-bridge analysis: %d cationic N, %d anionic O atoms, cutoff=%.1f A",
        len(pos_sel),
        len(neg_sel),
        cutoff,
    )

    pos_labels = [_residue_key(a) for a in pos_sel]
    neg_labels = [_residue_key(a) for a in neg_sel]

    frames = iter_frames(universe, window)
    n_frames = len(frames)
    times = np.empty(n_frames, dtype=np.float64)
    total_per_frame = np.empty(n_frames, dtype=np.int64)
    pair_frames: Dict[Tuple[str, str], int] = {}

    for fi, ts in enumerate(frames):
        times[fi] = ts.time
        dists = distance_array(pos_sel.positions, neg_sel.positions, box=ts.dimensions)
        # A residue pair is one bridge per frame however many atom pairs are
        # within the cutoff, so collapse atom pairs onto residue pairs first.
        present = {
            (pos_labels[i], neg_labels[j]) for i, j in np.argwhere(dists <= cutoff)
        }
        total_per_frame[fi] = len(present)
        for key in present:
            pair_frames[key] = pair_frames.get(key, 0) + 1

    pairs: List[Dict[str, Any]] = [
        {
            "positive": pos,
            "negative": neg,
            "frames_present": int(count),
            "occupancy": round(count / n_frames, 3),
        }
        for (pos, neg), count in sorted(pair_frames.items(), key=lambda kv: -kv[1])
    ]

    mean_sb = float(np.mean(total_per_frame))
    logger.info(
        "Salt-bridge analysis complete: mean=%.1f per frame, %d unique residue pairs",
        mean_sb,
        len(pair_frames),
    )

    return ModuleResult(
        name="salt_bridges",
        version=__version__,
        runtime_seconds=time.time() - start_time,
        parameters={
            "cutoff": cutoff,
            "criterion": "min N-O distance <= cutoff (Barlow & Thornton 1983)",
        },
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
            "total_unique_pairs": len(pair_frames),
            "persistent_pairs": [p for p in pairs if p["occupancy"] >= 0.5][:30],
        },
    )
