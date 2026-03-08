"""Trajectory Comparison.

Compare two trajectories (e.g. wild-type vs mutant, different conditions)
or split a single trajectory in half to assess equilibration.  Metrics
compared include radius of gyration distributions, per-residue RMSF, and
contact-map differences.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import MDAnalysis as mda

from ..utils.trajectory_utils import select_ca_atoms

logger = logging.getLogger("md_ai_analyzer")


def compare_trajectories(
    universe: mda.Universe,
    universe_ref: Optional[mda.Universe] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Compare two trajectories on structural and dynamic metrics.

    If *universe_ref* is ``None``, the single trajectory is split at its
    midpoint (first half vs second half), which is useful for
    equilibration assessment.

    Parameters
    ----------
    universe : mda.Universe
        Primary MDAnalysis Universe.
    universe_ref : mda.Universe, optional
        Reference/second Universe for two-trajectory comparison.
    **kwargs : Any
        Additional keyword arguments (unused, accepted for API consistency).

    Returns
    -------
    dict[str, Any]
        ``comparison_type``
            ``"two_trajectories"`` or ``"half_split"``.
        ``n_frames_a``
            Number of frames in set A.
        ``n_frames_b``
            Number of frames in set B.
        ``comparisons``
            Dictionary of metric-specific comparison results (``rg``,
            ``rmsf``, ``contacts``).
    """
    try:
        ca = select_ca_atoms(universe)
        n_frames: int = len(universe.trajectory)

        if n_frames < 10:
            logger.warning(
                "Only %d frames; too few for trajectory comparison", n_frames
            )
            return {"error": "Too few frames for comparison"}

        # Collect metrics for set A and set B
        if universe_ref is not None:
            ca_ref = select_ca_atoms(universe_ref)
            logger.info(
                "Comparing two trajectories: %d vs %d frames",
                n_frames,
                len(universe_ref.trajectory),
            )
            metrics_a = _collect_metrics(universe, ca)
            metrics_b = _collect_metrics(universe_ref, ca_ref)
            comparison_type = "two_trajectories"
        else:
            half = n_frames // 2
            logger.info(
                "Half-split comparison: frames 0-%d vs %d-%d",
                half - 1,
                half,
                n_frames - 1,
            )
            metrics_a = _collect_metrics(universe, ca, start=0, end=half)
            metrics_b = _collect_metrics(universe, ca, start=half, end=n_frames)
            comparison_type = "half_split"

        results: Dict[str, Any] = {
            "comparison_type": comparison_type,
            "n_frames_a": metrics_a["n_frames"],
            "n_frames_b": metrics_b["n_frames"],
            "comparisons": {},
        }

        # ── Rg comparison ────────────────────────────────────────
        if metrics_a["rg_values"] and metrics_b["rg_values"]:
            rg_a = np.asarray(metrics_a["rg_values"])
            rg_b = np.asarray(metrics_b["rg_values"])
            results["comparisons"]["rg"] = {
                "mean_a": round(float(np.mean(rg_a)), 3),
                "mean_b": round(float(np.mean(rg_b)), 3),
                "std_a": round(float(np.std(rg_a)), 3),
                "std_b": round(float(np.std(rg_b)), 3),
                "difference": round(float(np.mean(rg_b) - np.mean(rg_a)), 3),
                "ks_test": _ks_test(rg_a, rg_b),
            }

        # ── Per-residue RMSF comparison ──────────────────────────
        if (
            len(metrics_a["rmsf"]) == len(metrics_b["rmsf"])
            and len(metrics_a["rmsf"]) > 0
        ):
            rmsf_a = np.asarray(metrics_a["rmsf"])
            rmsf_b = np.asarray(metrics_b["rmsf"])
            diff = rmsf_b - rmsf_a
            significant = np.where(np.abs(diff) > np.std(diff))[0]
            results["comparisons"]["rmsf"] = {
                "mean_rmsf_a": round(float(np.mean(rmsf_a)), 3),
                "mean_rmsf_b": round(float(np.mean(rmsf_b)), 3),
                "max_diff_resid": (
                    int(metrics_a["resids"][int(np.argmax(np.abs(diff)))])
                    if len(diff) > 0
                    else None
                ),
                "max_diff_value": (
                    round(float(np.max(np.abs(diff))), 3)
                    if len(diff) > 0
                    else 0
                ),
                "n_significant_changes": int(len(significant)),
                "significant_resids": [
                    int(metrics_a["resids"][i]) for i in significant[:20]
                ],
            }

        # ── Contact-map difference ───────────────────────────────
        if (
            metrics_a["contact_map"] is not None
            and metrics_b["contact_map"] is not None
            and metrics_a["contact_map"].shape == metrics_b["contact_map"].shape
        ):
            diff_map = metrics_b["contact_map"] - metrics_a["contact_map"]
            changed = np.where(np.abs(diff_map) > 0.2)
            results["comparisons"]["contacts"] = {
                "n_changed_contacts": int(len(changed[0])),
                "mean_contact_diff": round(
                    float(np.mean(np.abs(diff_map))), 4
                ),
            }

        logger.info(
            "Trajectory comparison (%s) complete: %d metrics compared",
            comparison_type,
            len(results["comparisons"]),
        )

        return results

    except Exception as e:
        logger.error("Trajectory comparison failed: %s", e, exc_info=True)
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────


def _collect_metrics(
    universe: mda.Universe,
    ca: mda.AtomGroup,
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> Dict[str, Any]:
    """Collect basic structural metrics from a trajectory range.

    Parameters
    ----------
    universe : mda.Universe
        MDAnalysis Universe.
    ca : mda.AtomGroup
        C-alpha atom group.
    start : int, optional
        Starting frame index (inclusive).
    end : int, optional
        Ending frame index (exclusive).

    Returns
    -------
    dict[str, Any]
        ``n_frames``, ``rg_values``, ``rmsf`` (list), ``resids``,
        ``contact_map`` (ndarray or None).
    """
    rg_values: List[float] = []
    coords_list: List[np.ndarray] = []
    resids: List[int] = ca.resids.tolist()

    frames = list(universe.trajectory)
    if start is not None:
        frames = frames[start:end]

    for _ts in frames:
        rg_values.append(float(ca.radius_of_gyration()))
        coords_list.append(ca.positions.copy())

    n_frames: int = len(frames)

    # RMSF (vectorised)
    if coords_list:
        coords_arr = np.array(coords_list)
        mean_pos = coords_arr.mean(axis=0)
        rmsf: np.ndarray | list = np.sqrt(
            np.mean((coords_arr - mean_pos) ** 2, axis=0).mean(axis=1)
        )
    else:
        rmsf = []

    # Average contact map (C-alpha--C-alpha)
    contact_map: Optional[np.ndarray] = None
    if coords_list and len(ca) < 500:
        from scipy.spatial.distance import cdist

        step = max(1, len(coords_list) // 50)
        contact_maps = []
        for c in coords_list[::step]:
            dm = cdist(c, c)
            contact_maps.append((dm < 8.0).astype(np.float64))
        if contact_maps:
            contact_map = np.mean(contact_maps, axis=0)

    return {
        "n_frames": n_frames,
        "rg_values": rg_values,
        "rmsf": rmsf.tolist() if isinstance(rmsf, np.ndarray) else rmsf,
        "resids": resids,
        "contact_map": contact_map,
    }


def _ks_test(
    a: np.ndarray,
    b: np.ndarray,
) -> Dict[str, Any]:
    """Kolmogorov--Smirnov test between two distributions.

    Parameters
    ----------
    a, b : np.ndarray
        1-D sample arrays.

    Returns
    -------
    dict[str, Any]
        ``statistic`` and ``p_value``, or ``None`` if scipy is unavailable.
    """
    try:
        from scipy.stats import ks_2samp

        stat, pval = ks_2samp(a, b)
        return {
            "statistic": round(float(stat), 4),
            "p_value": round(float(pval), 6),
        }
    except ImportError:
        logger.debug("scipy not available; skipping KS test")
        return {"statistic": None, "p_value": None}
