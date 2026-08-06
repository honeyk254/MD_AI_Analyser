"""Secondary structure evolution using MDTraj DSSP.

Tracks helix, sheet and coil content per frame and the dominant assignment per
residue, using MDTraj's simplified (H/E/C) DSSP output.
"""

import time
import logging
from typing import List, Optional

import numpy as np
import MDAnalysis as mda

from ...schemas.analysis_bundle import ModuleResult, MetricSummary
from ..frames import FrameWindow
from ..mdtraj_bridge import residue_ids, to_mdtraj

logger = logging.getLogger("md_ai_analyzer")

__version__ = "2.0.0"

# MDTraj marks residues it cannot assign (e.g. non-protein) with "NA"; those are
# excluded from the fractions so they do not silently deflate every category.
UNASSIGNED = "NA"


def compute_secondary_structure(
    universe: mda.Universe, window: Optional[FrameWindow] = None, **kwargs
) -> ModuleResult:
    """Compute per-residue secondary structure over time using MDTraj DSSP."""
    start_time = time.time()

    import mdtraj as md

    protein = universe.select_atoms("protein")
    if len(protein) == 0:
        raise ValueError("No protein atoms found for secondary structure")

    traj, times = to_mdtraj(universe, protein, window)
    n_frames = traj.n_frames
    logger.info(
        "Computing secondary structure for %d protein atoms over %d frames",
        len(protein),
        n_frames,
    )

    dssp: np.ndarray = md.compute_dssp(traj, simplified=True)
    assigned = dssp != UNASSIGNED
    n_assigned = int(assigned[0].sum()) if n_frames else 0

    if n_assigned == 0:
        raise ValueError("DSSP assigned no residues; the selection is not a protein")

    helix_frac = ((dssp == "H").sum(axis=1) / n_assigned).tolist()
    sheet_frac = ((dssp == "E").sum(axis=1) / n_assigned).tolist()
    coil_frac = ((dssp == "C").sum(axis=1) / n_assigned).tolist()

    resids: List[int] = residue_ids(traj.topology)
    dominant_ss: List[str] = []
    for r in range(dssp.shape[1]):
        col = dssp[:, r]
        counts = {code: int(np.sum(col == code)) for code in ("H", "E", "C")}
        dominant_ss.append(max(counts, key=lambda code: counts[code]))

    def summary(values: List[float]) -> MetricSummary:
        arr = np.asarray(values, dtype=np.float64)
        return MetricSummary(
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            unit="fraction",
            n_frames=n_frames,
            time_series=values,
        )

    logger.info(
        "Secondary structure complete: mean helix=%.2f, sheet=%.2f, coil=%.2f",
        float(np.mean(helix_frac)),
        float(np.mean(sheet_frac)),
        float(np.mean(coil_frac)),
    )

    return ModuleResult(
        name="secondary_structure",
        version=__version__,
        runtime_seconds=time.time() - start_time,
        parameters={"algorithm": "mdtraj_dssp", "simplified": True},
        scalar_metrics={
            "helix_fraction": summary(helix_frac),
            "sheet_fraction": summary(sheet_frac),
            "coil_fraction": summary(coil_frac),
        },
        data={
            "time_ps": times.tolist(),
            "resids": resids,
            "per_residue_dominant_ss": dominant_ss,
            "n_residues_assigned": n_assigned,
        },
    )
