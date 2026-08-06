"""Hydrogen bond analysis.

Uses MDAnalysis ``HydrogenBondAnalysis`` over the analysed frame window and
reports per-frame counts, persistent bonds and the most frequent donor-acceptor
pairs.
"""

import time
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis

from ...schemas.analysis_bundle import ModuleResult, MetricSummary
from ..frames import FrameWindow, iter_frames, window_kwargs

logger = logging.getLogger("md_ai_analyzer")

__version__ = "2.1.0"

DEFAULT_ANGLE_CUTOFF = 150.0
PERSISTENT_OCCUPANCY = 0.5

# Donors/acceptors are selected by atom name rather than left to MDAnalysis'
# guesser: the guesser needs partial charges, which trajectories written without
# their force field (a PDB, for instance) do not carry. Hydrogens are matched to
# their donor by distance, so no bond records are required either.
DONOR_SELECTION = "protein and (name N* or name O* or name S*)"
HYDROGEN_SELECTION = "protein and name H*"
ACCEPTOR_SELECTION = "protein and (name N* or name O* or name S*)"


def compute_hbonds(
    universe: mda.Universe,
    distance: float = 3.5,
    angle: float = DEFAULT_ANGLE_CUTOFF,
    window: Optional[FrameWindow] = None,
    **kwargs,
) -> ModuleResult:
    """Compute intra-protein hydrogen bonds over the analysed frame window."""
    start_time = time.time()

    protein = universe.select_atoms("protein")
    if len(protein) == 0:
        raise ValueError("No protein atoms found for H-bond analysis")

    if len(universe.select_atoms("protein and name H*")) == 0:
        raise ValueError(
            "No hydrogens in the protein selection; geometric H-bond criteria "
            "cannot be applied to a heavy-atom-only trajectory"
        )

    win = window_kwargs(universe, window)
    frames = iter_frames(universe, window)
    frame_numbers = np.array([ts.frame for ts in frames], dtype=np.int64)
    times = np.array([float(ts.time) for ts in frames], dtype=np.float64)
    n_frames = len(frame_numbers)

    logger.info(
        "Running H-bond analysis with d_a_cutoff=%.1f A, angle>=%.0f deg over %d frames",
        distance,
        angle,
        n_frames,
    )

    hbonds = HydrogenBondAnalysis(
        universe,
        donors_sel=DONOR_SELECTION,
        hydrogens_sel=HYDROGEN_SELECTION,
        acceptors_sel=ACCEPTOR_SELECTION,
        d_a_cutoff=distance,
        d_h_a_angle_cutoff=angle,
    )
    hbonds.run(**win)

    # results columns: [frame, donor_idx, hydrogen_idx, acceptor_idx, dist, angle]
    results: np.ndarray = hbonds.results.hbonds

    if len(results) > 0:
        # results carry absolute frame numbers; map them onto window positions
        # so counts line up with the analysed frames even with a stride.
        positions = np.searchsorted(frame_numbers, results[:, 0].astype(np.int64))
        hbond_counts = np.bincount(positions, minlength=n_frames)[:n_frames].tolist()

        donor_indices = results[:, 1].astype(int)
        acceptor_indices = results[:, 3].astype(int)
        unique_pairs, counts = np.unique(
            np.column_stack((donor_indices, acceptor_indices)),
            axis=0,
            return_counts=True,
        )
        pair_counts: Dict[tuple, int] = {
            (int(p[0]), int(p[1])): int(c) for p, c in zip(unique_pairs, counts)
        }
    else:
        hbond_counts = np.zeros(n_frames, dtype=int).tolist()
        pair_counts = {}

    def label(index: int) -> str:
        atom = universe.atoms[index]
        return f"{atom.resname}{atom.resid}:{atom.name}"

    ranked = sorted(pair_counts.items(), key=lambda kv: -kv[1])

    persistent: List[Dict[str, Any]] = []
    for (d, a), count in ranked:
        occupancy = count / n_frames
        if occupancy < PERSISTENT_OCCUPANCY:
            break
        persistent.append(
            {
                "donor": label(d),
                "acceptor": label(a),
                "occupancy": round(occupancy, 3),
            }
        )

    top_pairs: List[Dict[str, Any]] = [
        {
            "donor": label(d),
            "acceptor": label(a),
            "count": int(count),
            "occupancy": round(count / n_frames, 3),
        }
        for (d, a), count in ranked[:20]
    ]

    mean_hb = float(np.mean(hbond_counts))
    logger.info(
        "H-bond analysis complete: mean=%.1f, unique pairs=%d, persistent=%d",
        mean_hb,
        len(pair_counts),
        len(persistent),
    )

    return ModuleResult(
        name="hbonds",
        version=__version__,
        runtime_seconds=time.time() - start_time,
        parameters={"d_a_cutoff": distance, "d_h_a_angle_cutoff": angle},
        scalar_metrics={
            "hbond_count": MetricSummary(
                mean=mean_hb,
                std=float(np.std(hbond_counts)),
                min=float(np.min(hbond_counts)),
                max=float(np.max(hbond_counts)),
                unit="count",
                n_frames=n_frames,
                time_series=hbond_counts,
            )
        },
        data={
            "time_ps": times.tolist(),
            "persistent_hbonds": persistent[:20],
            "top_hbond_pairs": top_pairs,
            "total_unique_hbonds": len(pair_counts),
        },
    )
