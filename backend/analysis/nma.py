"""Normal Mode Analysis (NMA).

Computes elastic network model normal modes from the MD-average C-alpha
structure using an Anisotropic Network Model (ANM) with a fully
vectorised Hessian construction.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
from scipy.linalg import eigh

import MDAnalysis as mda

from ..utils.trajectory_utils import select_ca_atoms, collect_ca_positions

logger = logging.getLogger("md_ai_analyzer")


def compute_nma(
    universe: mda.Universe,
    n_modes: int = 10,
    cutoff: float = 15.0,
    gamma: float = 1.0,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Normal Mode Analysis using an Anisotropic Network Model (ANM).

    Builds the 3N x 3N Hessian from the MD-average C-alpha structure and
    diagonalises it to obtain the slowest collective modes of motion.

    Parameters
    ----------
    universe : mda.Universe
        MDAnalysis Universe with a loaded trajectory.
    n_modes : int
        Number of non-trivial modes to report (after removing the 6
        rigid-body modes).
    cutoff : float
        Distance cutoff in angstrom for ANM springs.
    gamma : float
        Uniform spring constant.
    **kwargs : Any
        Ignored; accepted for orchestrator compatibility.

    Returns
    -------
    dict[str, Any]
        Keys:

        * ``resids`` -- residue IDs.
        * ``eigenvalues`` -- eigenvalues of the first *n_modes* non-trivial
          modes.
        * ``frequencies`` -- mode frequencies (sqrt of eigenvalue).
        * ``bfactors`` -- normalised per-residue predicted B-factors.
        * ``mode_collectivity`` -- collectivity index for each mode.
        * ``mode_shapes`` -- per-residue displacement magnitude for the
          first 5 modes.
        * ``n_modes_computed`` -- number of modes actually returned.
    """
    try:
        ca: mda.AtomGroup = select_ca_atoms(universe)
        n_res: int = len(ca)
        resids: list[int] = ca.resids.tolist()

        # Average structure from trajectory
        positions: np.ndarray = collect_ca_positions(universe, atoms=ca)
        mean_pos: np.ndarray = positions.mean(axis=0)  # (n_res, 3)

        # Build vectorised ANM Hessian
        hessian: np.ndarray = _build_anm_hessian_vectorised(
            mean_pos, cutoff, gamma
        )

        # Diagonalise (real symmetric matrix)
        eigenvalues, eigenvectors = eigh(hessian)

        # Skip first 6 trivial modes (translations + rotations)
        nontrivial_start: int = 6
        evals: np.ndarray = eigenvalues[nontrivial_start: nontrivial_start + n_modes]
        evecs: np.ndarray = eigenvectors[:, nontrivial_start: nontrivial_start + n_modes]
        n_actual: int = len(evals)

        # Frequencies (vectorised)
        freq_arr: np.ndarray = np.where(
            evals > 1e-10, np.sqrt(np.maximum(evals, 0.0)), 0.0
        )
        frequencies: list[float] = [round(float(f), 6) for f in freq_arr]

        # Per-residue B-factors: B_i = (8 pi^2 / 3) * sum_k |v_ik|^2 / lambda_k
        # Reshape eigenvectors to (n_res, 3, n_modes_actual)
        evecs_reshaped: np.ndarray = evecs.reshape(n_res, 3, n_actual)
        mode_sq: np.ndarray = np.sum(evecs_reshaped ** 2, axis=1)  # (n_res, n_actual)

        # Inverse eigenvalues (mask near-zero modes)
        valid_mask: np.ndarray = evals > 1e-10
        inv_evals: np.ndarray = np.where(valid_mask, 1.0 / np.where(valid_mask, evals, 1.0), 0.0)

        bfactors: np.ndarray = (8.0 * np.pi ** 2 / 3.0) * (mode_sq @ inv_evals)
        bfactors_norm: np.ndarray = (
            bfactors / bfactors.max() if bfactors.max() > 0 else bfactors
        )

        # Mode shapes: per-residue displacement magnitude (vectorised)
        mode_magnitudes: np.ndarray = np.sqrt(mode_sq)  # (n_res, n_actual)
        mode_shapes: list[list[float]] = [
            [round(float(m), 4) for m in mode_magnitudes[:, k]]
            for k in range(min(5, n_actual))
        ]

        # Collectivity: kappa = (1/N) * exp(-sum p_i ln p_i)
        collectivity: list[float] = []
        for k in range(n_actual):
            sq = mode_sq[:, k]
            sq_sum: float = float(sq.sum())
            if sq_sum > 0:
                p = sq / sq_sum
                p_safe = p[p > 1e-15]
                entropy = -float(np.sum(p_safe * np.log(p_safe)))
                kappa = float(np.exp(entropy)) / n_res
                collectivity.append(round(kappa, 4))
            else:
                collectivity.append(0.0)

        logger.info(
            "NMA (ANM) computed: %d modes, cutoff=%.1f A, "
            "lowest freq=%.6f",
            n_actual,
            cutoff,
            frequencies[0] if frequencies else 0.0,
        )

        return {
            "resids": resids,
            "eigenvalues": [round(float(x), 6) for x in evals],
            "frequencies": frequencies,
            "bfactors": [round(float(x), 4) for x in bfactors_norm],
            "mode_collectivity": collectivity,
            "mode_shapes": mode_shapes,
            "n_modes_computed": n_actual,
        }

    except Exception as e:
        logger.exception("NMA computation failed")
        return {"error": str(e)}


# ------------------------------------------------------------------ #
#  Internal helpers                                                    #
# ------------------------------------------------------------------ #


def _build_anm_hessian_vectorised(
    coords: np.ndarray,
    cutoff: float,
    gamma: float,
) -> np.ndarray:
    """Build the 3N x 3N ANM Hessian using vectorised numpy operations.

    The off-diagonal 3 x 3 super-elements are computed via broadcasting
    and the diagonal blocks are obtained as the negative row-block sums,
    avoiding a separate O(N^2) Python loop.

    Parameters
    ----------
    coords : np.ndarray
        Shape ``(N, 3)`` -- C-alpha coordinates.
    cutoff : float
        Distance cutoff in the same length unit as *coords*.
    gamma : float
        Uniform spring constant.

    Returns
    -------
    np.ndarray
        Shape ``(3N, 3N)`` symmetric Hessian matrix.
    """
    n: int = len(coords)

    # Pairwise difference vectors: diff[i,j] = coords[j] - coords[i]
    diff: np.ndarray = coords[np.newaxis, :, :] - coords[:, np.newaxis, :]  # (n, n, 3)

    # Squared distances
    dist_sq: np.ndarray = np.sum(diff ** 2, axis=2)  # (n, n)

    # Contact mask (within cutoff, excluding self-pairs)
    mask: np.ndarray = (dist_sq < cutoff ** 2) & (dist_sq > 0)

    # Spring constants: uniform gamma within cutoff (standard ANM, Bahar et al. 1997)
    k_values: np.ndarray = np.where(mask, -gamma, 0.0)  # (n, n)

    # 3x3 super-elements for all pairs:
    # H_ij[a,b] = k_values[i,j] * diff[i,j,a] * diff[i,j,b]
    super_elements: np.ndarray = (
        k_values[:, :, np.newaxis, np.newaxis]
        * diff[:, :, :, np.newaxis]
        * diff[:, :, np.newaxis, :]
    )  # (n, n, 3, 3)

    # Assemble off-diagonal part of the Hessian by reshaping
    # (n, n, 3, 3) -> (n, 3, n, 3) -> (3n, 3n)
    hessian: np.ndarray = super_elements.transpose(0, 2, 1, 3).reshape(
        3 * n, 3 * n
    )

    # Diagonal blocks = -sum of all off-diagonal blocks in the same row
    # super_elements[i, i, :, :] is zero (self-pair excluded by mask),
    # so summing over axis 1 gives the sum of off-diagonal blocks.
    diag_blocks: np.ndarray = -super_elements.sum(axis=1)  # (n, 3, 3)

    # Place diagonal blocks using vectorised fancy indexing
    idx: np.ndarray = np.arange(n)
    for a in range(3):
        for b in range(3):
            hessian[3 * idx + a, 3 * idx + b] = diag_blocks[:, a, b]

    return hessian
