"""Time-lagged Independent Component Analysis (tICA).

Identifies slow collective motions in the trajectory by solving the
generalised eigenvalue problem on the time-lagged covariance matrix.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import MDAnalysis as mda
from scipy.linalg import eigh

from ..utils.trajectory_utils import (
    select_ca_atoms,
    collect_ca_coords_flat,
    residue_contributions_from_eigenvector,
)

logger = logging.getLogger("md_ai_analyzer")


def compute_tica(
    universe: mda.Universe,
    lag_time: int = 10,
    n_components: int = 5,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Perform tICA on C-alpha coordinates to find slow motions.

    tICA finds the linear combinations of coordinates that decorrelate
    most slowly, capturing the slowest dynamical processes in the
    trajectory.

    Parameters
    ----------
    universe : mda.Universe
        MDAnalysis Universe with a loaded trajectory.
    lag_time : int
        Lag time in frames for the time-lagged covariance.
    n_components : int
        Number of tICA components to retain.
    **kwargs : Any
        Ignored; accepted for orchestrator compatibility.

    Returns
    -------
    dict[str, Any]
        Keys:

        * ``projections`` -- tICA projections per frame.
        * ``timescales`` -- implied timescales of each tIC.
        * ``eigenvalues`` -- tICA eigenvalues.
        * ``n_components`` -- number of components returned.
        * ``lag_time`` -- lag time used.
        * ``resids`` -- residue IDs.
        * ``tic1_residue_contributions`` -- per-residue contribution to tIC1.
        * ``tic2_residue_contributions`` -- per-residue contribution to tIC2.

    Notes
    -----
    Numerical stability is ensured by Tikhonov regularisation of the
    instantaneous covariance matrix C(0) before solving the generalised
    eigenvalue problem.  If the primary ``scipy.linalg.eigh`` call fails,
    a whitening-based fallback is used.
    """
    try:
        ca: mda.AtomGroup = select_ca_atoms(universe)
        n_atoms: int = len(ca)
        resids: list[int] = ca.resids.tolist()

        # Collect centred coordinates via shared utility
        coords: np.ndarray = collect_ca_coords_flat(universe, atoms=ca)
        mean_coords: np.ndarray = coords.mean(axis=0)
        X: np.ndarray = coords - mean_coords

        n_frames: int = X.shape[0]
        if n_frames <= lag_time + 1:
            msg = f"Trajectory too short for lag_time={lag_time}"
            logger.error(msg)
            return {"error": msg}

        # Instantaneous covariance C(0)
        C0: np.ndarray = (X.T @ X) / (n_frames - 1)

        # Time-lagged covariance C(tau) -- symmetrised
        X_t: np.ndarray = X[: n_frames - lag_time]
        X_tau: np.ndarray = X[lag_time:]
        Ctau: np.ndarray = (X_t.T @ X_tau) / (n_frames - lag_time - 1)
        Ctau_sym: np.ndarray = 0.5 * (Ctau + Ctau.T)

        # Tikhonov regularisation for numerical stability
        reg_strength: float = 1e-6
        reg: np.ndarray = reg_strength * np.eye(C0.shape[0])
        C0_reg: np.ndarray = C0 + reg

        # Solve generalised eigenvalue problem: Ctau v = lambda C0 v
        try:
            eigenvalues, eigenvectors = eigh(Ctau_sym, C0_reg)
        except np.linalg.LinAlgError:
            logger.warning(
                "Primary eigh failed; using whitening-based fallback"
            )
            eigenvalues, eigenvectors = _whitened_eig_fallback(
                Ctau_sym, C0_reg
            )

        # Sort by eigenvalue (descending -- slowest modes first)
        idx: np.ndarray = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Retain requested number of components
        n_comp: int = min(n_components, len(eigenvalues))
        eigenvalues = eigenvalues[:n_comp]
        eigenvectors = eigenvectors[:, :n_comp]

        # Project data onto tICA components
        projections: np.ndarray = X @ eigenvectors

        # Implied timescales: tau_i = -lag_time / ln(lambda_i)
        timescales: list[float] = []
        for ev in eigenvalues:
            if 0 < ev < 1:
                timescales.append(float(-lag_time / np.log(np.clip(ev, 1e-15, 1.0 - 1e-15))))
            else:
                timescales.append(float("inf"))

        # Per-residue contributions (vectorised via shared utility)
        tic1_contrib: list[float] = (
            residue_contributions_from_eigenvector(
                eigenvectors[:, 0], n_atoms
            ).tolist()
            if n_comp > 0
            else []
        )
        tic2_contrib: list[float] = (
            residue_contributions_from_eigenvector(
                eigenvectors[:, 1], n_atoms
            ).tolist()
            if n_comp > 1
            else []
        )

        logger.info(
            "tICA computed: %d components, lag=%d, slowest timescale=%.1f",
            n_comp,
            lag_time,
            timescales[0] if timescales else 0.0,
        )

        return {
            "projections": projections.tolist(),
            "timescales": timescales,
            "eigenvalues": eigenvalues.tolist(),
            "n_components": n_comp,
            "lag_time": lag_time,
            "resids": resids,
            "tic1_residue_contributions": tic1_contrib,
            "tic2_residue_contributions": tic2_contrib,
        }

    except Exception as e:
        logger.exception("tICA computation failed")
        return {"error": str(e), "projections": [], "timescales": []}


# ------------------------------------------------------------------ #
#  Internal helpers                                                    #
# ------------------------------------------------------------------ #

def _whitened_eig_fallback(
    Ctau_sym: np.ndarray,
    C0_reg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the generalised eigenvalue problem via explicit whitening.

    Computes C0^{-1/2} Ctau C0^{-1/2} and solves the resulting standard
    eigenvalue problem.  This is numerically more robust when C0 is
    ill-conditioned.

    Parameters
    ----------
    Ctau_sym : np.ndarray
        Symmetrised time-lagged covariance matrix.
    C0_reg : np.ndarray
        Regularised instantaneous covariance matrix.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(eigenvalues, eigenvectors)`` sorted ascending.
    """
    # Eigendecomposition of C0_reg for stable inverse square-root
    evals_c0, evecs_c0 = np.linalg.eigh(C0_reg)
    # Clamp small eigenvalues for numerical safety
    evals_c0 = np.maximum(evals_c0, 1e-12)
    inv_sqrt_evals = 1.0 / np.sqrt(evals_c0)

    # C0^{-1/2} = V diag(1/sqrt(lambda)) V^T
    C0_inv_sqrt: np.ndarray = (evecs_c0 * inv_sqrt_evals[np.newaxis, :]) @ evecs_c0.T

    M: np.ndarray = C0_inv_sqrt @ Ctau_sym @ C0_inv_sqrt
    eigenvalues, eigenvectors_t = np.linalg.eigh(M)

    # Transform eigenvectors back to original basis
    eigenvectors: np.ndarray = C0_inv_sqrt @ eigenvectors_t
    return eigenvalues, eigenvectors
