"""Convergence Assessment.

Block averaging, autocorrelation analysis, and cosine content of
principal components to evaluate whether the MD simulation has converged.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import MDAnalysis as mda

from ..utils.trajectory_utils import select_ca_atoms, collect_ca_coords_flat
from ..utils.ml_feature_utils import pca_reduce

logger = logging.getLogger("md_ai_analyzer")


def compute_convergence(
    universe: mda.Universe,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Assess trajectory convergence using multiple diagnostics.

    Diagnostics performed:

    * **Block averaging** of RMSD and Rg time series.
    * **Autocorrelation function** of RMSD.
    * **Cosine content** of the first PCA projections (values close to 1
      indicate random-walk-like sampling).
    * **Overall convergence score** (0--1) aggregated from the above.

    Parameters
    ----------
    universe : mda.Universe
        MDAnalysis Universe with a loaded trajectory.
    **kwargs : Any
        Additional keyword arguments (unused, accepted for API consistency).

    Returns
    -------
    dict[str, Any]
        ``time``
            List of timestamps (ps).
        ``rmsd_block_averages``
            Block-averaged RMSD statistics at different block counts.
        ``rg_block_averages``
            Block-averaged Rg statistics at different block counts.
        ``autocorrelation_rmsd``
            Normalised autocorrelation function of RMSD.
        ``cosine_content``
            Cosine content for the first few PCA components.
        ``convergence_score``
            Aggregated convergence score in [0, 1].
        ``rmsd_drift``
            Relative RMSD drift between first and last 20 % of frames.
        ``rg_drift``
            Relative Rg drift between first and last 20 % of frames.
        ``recommendations``
            Human-readable suggestions based on the diagnostics.
    """
    try:
        ca = select_ca_atoms(universe)
        n_frames: int = len(universe.trajectory)

        if n_frames < 20:
            logger.warning(
                "Only %d frames available; too few for convergence assessment",
                n_frames,
            )
            return {"error": "Too few frames for convergence assessment"}

        logger.info(
            "Convergence assessment: %d CA atoms, %d frames", len(ca), n_frames
        )

        # ── Collect RMSD time series ─────────────────────────────
        from MDAnalysis.analysis.rms import RMSD as MDA_RMSD

        ref = universe.copy()
        ref.trajectory[0]
        R = MDA_RMSD(universe, ref, select="backbone", ref_frame=0)
        R.run()
        rmsd_ts: np.ndarray = R.results.rmsd[:, 2]
        times: List[float] = R.results.rmsd[:, 1].tolist()

        # ── Collect Rg time series ───────────────────────────────
        rg_ts = np.empty(n_frames, dtype=np.float64)
        for i, _ts in enumerate(universe.trajectory):
            rg_ts[i] = ca.radius_of_gyration()

        # ── Block averaging ──────────────────────────────────────
        rmsd_blocks: List[Dict[str, Any]] = _block_average(rmsd_ts)
        rg_blocks: List[Dict[str, Any]] = _block_average(rg_ts)

        # ── Autocorrelation of RMSD ──────────────────────────────
        acf_rmsd: List[Dict[str, Any]] = _autocorrelation(
            rmsd_ts, max_lag=min(n_frames // 2, 200)
        )

        # ── Cosine content of PCA ────────────────────────────────
        coords: np.ndarray = collect_ca_coords_flat(universe, atoms=ca)

        cosine_content: List[Dict[str, Any]] = []
        try:
            n_comp = min(5, coords.shape[0] - 1, coords.shape[1])
            if n_comp >= 1:
                projections, _pca = pca_reduce(coords, n_components=n_comp)
                for i in range(projections.shape[1]):
                    cc = _cosine_content_pc(projections[:, i])
                    cosine_content.append(
                        {
                            "pc": i + 1,
                            "cosine_content": round(float(cc), 4),
                            "converged": cc < 0.5,
                        }
                    )
        except Exception as exc:
            logger.debug("Cosine content PCA failed: %s", exc)

        # ── Overall convergence score ────────────────────────────
        score: float = 0.0
        n_checks: int = 0
        split = max(1, n_frames // 5)

        # Check 1: RMSD stabilisation (last 20 % vs first 20 %)
        first_mean = float(np.mean(rmsd_ts[:split]))
        last_mean = float(np.mean(rmsd_ts[-split:]))
        rmsd_drift: float = abs(last_mean - first_mean) / (last_mean + 1e-8)
        rmsd_score = max(0.0, 1.0 - rmsd_drift * 2)
        score += rmsd_score
        n_checks += 1

        # Check 2: Rg stabilisation
        rg_first = float(np.mean(rg_ts[:split]))
        rg_last = float(np.mean(rg_ts[-split:]))
        rg_drift: float = abs(rg_last - rg_first) / (rg_last + 1e-8)
        rg_score = max(0.0, 1.0 - rg_drift * 2)
        score += rg_score
        n_checks += 1

        # Check 3: Block-average SEM convergence
        # In a converged simulation the SEM should plateau as block size
        # grows.  We measure convergence by how stable the SEM is between
        # the smallest and largest block sizes: a ratio close to 1
        # indicates the SEM has plateaued (converged).
        if len(rmsd_blocks) >= 2:
            first_sem = rmsd_blocks[0]["sem"]
            last_sem = rmsd_blocks[-1]["sem"]
            if last_sem > 0 and first_sem > 0:
                sem_ratio = min(first_sem, last_sem) / max(first_sem, last_sem)
                block_score = sem_ratio  # 1.0 = perfectly stable SEM
            else:
                block_score = 0.5
            score += block_score
            n_checks += 1

        # Check 4: Cosine content
        if cosine_content:
            cc_avg = float(
                np.mean([c["cosine_content"] for c in cosine_content[:3]])
            )
            cc_score = max(0.0, 1.0 - cc_avg)
            score += cc_score
            n_checks += 1

        overall_score: float = round(score / max(n_checks, 1), 3)

        # ── Recommendations ──────────────────────────────────────
        recommendations: List[str] = []
        if rmsd_drift > 0.15:
            recommendations.append(
                "RMSD has not stabilized \u2014 consider extending simulation"
            )
        if rg_drift > 0.1:
            recommendations.append(
                "Radius of gyration is still drifting \u2014 "
                "protein may not be equilibrated"
            )
        if cosine_content and cosine_content[0]["cosine_content"] > 0.7:
            recommendations.append(
                "PC1 has high cosine content \u2014 trajectory may be "
                "sampling a random walk, not converged dynamics"
            )
        if overall_score > 0.7:
            recommendations.append(
                "Trajectory appears reasonably well-converged"
            )
        elif overall_score < 0.4:
            recommendations.append(
                "Trajectory shows poor convergence \u2014 results should "
                "be interpreted with caution"
            )

        logger.info(
            "Convergence assessment complete: score=%.3f, rmsd_drift=%.4f, "
            "rg_drift=%.4f",
            overall_score,
            rmsd_drift,
            rg_drift,
        )

        return {
            "time": times,
            "rmsd_block_averages": rmsd_blocks,
            "rg_block_averages": rg_blocks,
            "autocorrelation_rmsd": acf_rmsd,
            "cosine_content": cosine_content,
            "convergence_score": overall_score,
            "rmsd_drift": round(float(rmsd_drift), 4),
            "rg_drift": round(float(rg_drift), 4),
            "recommendations": recommendations,
        }

    except Exception as e:
        logger.error("Convergence assessment failed: %s", e, exc_info=True)
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────


def _block_average(
    data: np.ndarray,
    n_blocks_list: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    """Compute block averages with increasing block sizes.

    Parameters
    ----------
    data : np.ndarray
        1-D time series.
    n_blocks_list : sequence of int, optional
        Block counts to evaluate (default ``[2, 4, 5, 8, 10, 20]``).

    Returns
    -------
    list[dict[str, Any]]
        Each entry contains ``n_blocks``, ``block_size``, ``mean``, and
        ``sem`` (standard error of the mean).
    """
    n: int = len(data)
    if n_blocks_list is None:
        n_blocks_list = [2, 4, 5, 8, 10, 20]

    results: List[Dict[str, Any]] = []
    for nb in n_blocks_list:
        if nb > n:
            continue
        block_size = n // nb
        if block_size < 2:
            continue

        # Vectorised block-mean computation
        usable = nb * block_size
        blocks = data[:usable].reshape(nb, block_size)
        block_means = blocks.mean(axis=1)
        sem = float(np.std(block_means, ddof=1) / np.sqrt(nb))

        results.append(
            {
                "n_blocks": nb,
                "block_size": block_size,
                "mean": round(float(np.mean(block_means)), 4),
                "sem": round(sem, 6),
            }
        )
    return results


def _autocorrelation(
    data: np.ndarray,
    max_lag: int = 200,
) -> List[Dict[str, Any]]:
    """Compute the normalised autocorrelation function.

    Parameters
    ----------
    data : np.ndarray
        1-D time series.
    max_lag : int, optional
        Maximum lag to compute (default 200).

    Returns
    -------
    list[dict[str, Any]]
        Each entry contains ``lag`` and the normalised ``acf`` value.
    """
    n: int = len(data)
    mean = float(np.mean(data))
    var = float(np.var(data))
    if var == 0:
        return []

    data_centered: np.ndarray = data - mean
    max_lag = min(max_lag, n - 1)
    step = max(1, max_lag // 50)

    acf: List[Dict[str, Any]] = []
    for lag in range(0, max_lag, step):
        c = float(np.mean(data_centered[: n - lag] * data_centered[lag:]))
        acf.append(
            {
                "lag": lag,
                "acf": round(c / var, 4),
            }
        )
    return acf


def _cosine_content_pc(projection: np.ndarray) -> float:
    """Compute the cosine content of a principal-component projection.

    A value close to 1 indicates random-diffusion-like motion; close to 0
    indicates well-converged conformational sampling.

    Parameters
    ----------
    projection : np.ndarray
        1-D PCA projection time series.

    Returns
    -------
    float
        Cosine content in approximately [0, 1].
    """
    n: int = len(projection)
    if n < 2:
        return 1.0

    t = np.arange(n, dtype=np.float64)
    cos_wave = np.cos(np.pi * t / (n - 1))

    numerator = float(np.sum(projection * cos_wave)) ** 2
    denominator = float(np.sum(projection ** 2))

    if denominator == 0:
        return 1.0

    return (2.0 / n) * numerator / denominator
