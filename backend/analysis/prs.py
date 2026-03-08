"""Perturbation Response Scanning (PRS).

Predicts the structural response to perturbation at each residue using
covariance-based linear response theory.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
from scipy.linalg import pinv

import MDAnalysis as mda

from ..utils.trajectory_utils import select_ca_atoms, collect_ca_positions

logger = logging.getLogger("md_ai_analyzer")


def compute_prs(
    universe: mda.Universe,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Perturbation Response Scanning based on covariance inversion.

    For each residue *i* (the effector), a unit perturbation is applied
    along each Cartesian direction and the linear-response displacement
    of every other residue *j* (the sensor) is computed using:

    .. math::

        \\Delta \\mathbf{r} = \\mathbf{C} \\cdot \\mathbf{F}

    where **C** is the positional covariance matrix.

    Parameters
    ----------
    universe : mda.Universe
        MDAnalysis Universe with a loaded trajectory.
    **kwargs : Any
        Ignored; accepted for orchestrator compatibility.

    Returns
    -------
    dict[str, Any]
        Keys:

        * ``resids`` -- residue IDs.
        * ``effector_scores`` -- normalised per-residue effector score.
        * ``sensor_scores`` -- normalised per-residue sensor score.
        * ``response_matrix`` -- truncated PRS response matrix (max 150 x 150).
        * ``top_effectors`` -- top 20 effector residues.
        * ``top_sensors`` -- top 20 sensor residues.
        * ``top_pairs`` -- top 30 effector -> sensor pairs.

    Notes
    -----
    For proteins larger than 500 C-alpha atoms the analysis is performed
    on a stride-reduced subset to keep the covariance matrix tractable.
    """
    try:
        ca: mda.AtomGroup = select_ca_atoms(universe)
        n_res: int = len(ca)
        resids: list[int] = ca.resids.tolist()

        # Optional stride for very large proteins
        if n_res > 500:
            stride: int = max(1, n_res // 300)
            ca_subset: mda.AtomGroup = ca[::stride]
            n_res_eff: int = len(ca_subset)
            resids_eff: list[int] = ca_subset.resids.tolist()
        else:
            ca_subset = ca
            n_res_eff = n_res
            resids_eff = resids

        # Collect C-alpha positions via shared utility
        positions: np.ndarray = collect_ca_positions(universe, atoms=ca_subset)
        n_frames: int = positions.shape[0]

        if n_frames < 10:
            msg = "Too few frames for PRS analysis"
            logger.error(msg)
            return {"error": msg}

        # Flatten to (n_frames, 3*n_res_eff)
        flat: np.ndarray = positions.reshape(n_frames, n_res_eff * 3)
        mean_pos: np.ndarray = flat.mean(axis=0)
        delta: np.ndarray = flat - mean_pos

        # Covariance matrix (3N x 3N)
        cov: np.ndarray = np.cov(delta.T)

        # --- Vectorised PRS response matrix --- #
        # For each effector residue i and perturbation direction d:
        #   displacement = cov[:, 3i+d]  (column of C)
        # Response of sensor j = sum_{alpha} displacement[3j+alpha]^2
        # Total over 3 directions: sum_{d,alpha} cov[3j+alpha, 3i+d]^2
        #
        # Reshape cov to residue-block form (j, alpha, i, d):
        cov_blocks: np.ndarray = cov.reshape(n_res_eff, 3, n_res_eff, 3)

        # Sum over alpha (axis 1) and d (axis 3) of squared elements
        # Result shape: (j, i) -- transpose to get (i, j)
        response_matrix: np.ndarray = np.sum(cov_blocks ** 2, axis=(1, 3)).T

        # Average over 3 directions, then sqrt
        response_matrix /= 3.0
        response_matrix = np.sqrt(response_matrix)

        # Effector and sensor scores
        effector_scores: np.ndarray = np.mean(response_matrix, axis=1)
        sensor_scores: np.ndarray = np.mean(response_matrix, axis=0)

        # Normalise to [0, 1]
        eff_max = effector_scores.max()
        effector_norm: np.ndarray = (
            effector_scores / eff_max if eff_max > 0 else effector_scores
        )
        sens_max = sensor_scores.max()
        sensor_norm: np.ndarray = (
            sensor_scores / sens_max if sens_max > 0 else sensor_scores
        )

        # Top 20 effectors
        eff_order: np.ndarray = np.argsort(-effector_norm)
        top_effectors: list[dict[str, Any]] = [
            {"resid": int(resids_eff[i]), "score": round(float(effector_norm[i]), 4)}
            for i in eff_order[:20]
        ]

        # Top 20 sensors
        sens_order: np.ndarray = np.argsort(-sensor_norm)
        top_sensors: list[dict[str, Any]] = [
            {"resid": int(resids_eff[i]), "score": round(float(sensor_norm[i]), 4)}
            for i in sens_order[:20]
        ]

        # Top 30 effector->sensor pairs (vectorised pair extraction)
        ii, jj = np.meshgrid(
            np.arange(n_res_eff), np.arange(n_res_eff), indexing="ij"
        )
        valid_mask: np.ndarray = (ii != jj) & (np.abs(ii - jj) > 5)

        valid_resp: np.ndarray = np.where(valid_mask, response_matrix, -np.inf)
        flat_indices: np.ndarray = np.argsort(valid_resp.ravel())[::-1][:30]
        top_i, top_j = np.unravel_index(flat_indices, response_matrix.shape)

        top_pairs: list[dict[str, Any]] = [
            {
                "effector_resid": int(resids_eff[int(ti)]),
                "sensor_resid": int(resids_eff[int(tj)]),
                "response": round(float(response_matrix[ti, tj]), 4),
            }
            for ti, tj in zip(top_i, top_j)
            if response_matrix[ti, tj] > 0
        ]

        # Truncate matrix for JSON serialisation
        matrix_size: int = min(n_res_eff, 150)
        response_trunc: np.ndarray = response_matrix[:matrix_size, :matrix_size]

        logger.info(
            "PRS computed: %d residues (effective %d), "
            "top effector resid=%d, top sensor resid=%d",
            n_res,
            n_res_eff,
            top_effectors[0]["resid"] if top_effectors else -1,
            top_sensors[0]["resid"] if top_sensors else -1,
        )

        return {
            "resids": resids_eff,
            "effector_scores": [round(float(x), 4) for x in effector_norm],
            "sensor_scores": [round(float(x), 4) for x in sensor_norm],
            "response_matrix": response_trunc.tolist(),
            "top_effectors": top_effectors,
            "top_sensors": top_sensors,
            "top_pairs": top_pairs,
        }

    except Exception as e:
        logger.exception("PRS computation failed")
        return {"error": str(e)}
