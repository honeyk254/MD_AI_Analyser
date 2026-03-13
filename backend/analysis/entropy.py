"""Configurational Entropy Estimation.

Uses Schlitter's method to estimate an upper bound on the configurational
entropy from the mass-weighted covariance matrix of C-alpha atom positions.

Reference
---------
Schlitter, J. (1993). Estimation of absolute and relative entropies of
macromolecules using the covariance matrix. *Chem. Phys. Lett.*, 215,
617--621.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
from scipy.linalg import eigvalsh

import MDAnalysis as mda

from ..utils.trajectory_utils import select_ca_atoms, collect_ca_positions

logger = logging.getLogger("md_ai_analyzer")

# --------------- Physical constants (SI) --------------- #
KB: float = 1.380649e-23       # Boltzmann constant  (J/K)
HBAR: float = 1.054571817e-34  # Reduced Planck constant  (J*s)
NA: float = 6.02214076e23      # Avogadro's number
AMU_TO_KG: float = 1.66054e-27 # atomic mass unit -> kg
ANG_TO_M: float = 1e-10        # angstrom -> metres


def compute_entropy(
    universe: mda.Universe,
    temperature: float = 300.0,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Estimate configurational entropy using Schlitter's formula.

    .. math::

        S_{\\text{Schlitter}} \\le \\frac{k_B}{2}
        \\sum_i \\ln\\!\\left(1 + \\frac{k_B T \\, e^2}{\\hbar^2}
        \\lambda_i\\right)

    where :math:`\\lambda_i` are eigenvalues of the mass-weighted
    covariance matrix and :math:`k_B T` is the thermal energy.

    Parameters
    ----------
    universe : mda.Universe
        MDAnalysis Universe with a loaded trajectory.
    temperature : float
        Temperature in Kelvin (default 300 K).
    **kwargs : Any
        Ignored; accepted for orchestrator compatibility.

    Returns
    -------
    dict[str, Any]
        Keys:

        * ``global_entropy_J_mol_K`` -- physically rigorous upper bound
          computed from the full covariance matrix.
        * ``total_entropy_J_mol_K`` -- alias for ``global_entropy_J_mol_K``.
        * ``total_entropy_kJ_mol_K`` -- global entropy in kJ/(mol*K).
        * ``per_residue_entropy`` -- per-residue entropy contribution
          under the "Independent Residue Approximation."
        * ``resids`` -- residue IDs.
        * ``entropy_convergence`` -- global entropy computed from increasing
          trajectory fractions (20 %, 40 %, ..., 100 %).
        * ``temperature_K`` -- temperature used.
        * ``n_frames_used`` -- total number of frames.

    Notes
    -----
    **Independent Residue Approximation**: The ``per_residue_entropy`` values
    are computed by considering each residue's degrees of freedom in isolation.
    Summing these values will significantly overestimate the total entropy
    because it ignores all cross-residue correlations. Use the
    ``global_entropy_J_mol_K`` for a physically meaningful (though still
    upper-bound) estimate of the system's configurational entropy.
    """
    try:
        ca: mda.AtomGroup = select_ca_atoms(universe)
        n_res: int = len(ca)
        resids: list[int] = ca.resids.tolist()

        # Masses in AMU (fall back to carbon mass if topology lacks masses)
        try:
            masses: np.ndarray = ca.masses  # type: ignore[assignment]
        except (AttributeError, mda.NoDataError):
            logger.warning("Masses unavailable; using default 12.0 AMU for CA")
            masses = np.full(n_res, 12.0)

        # Collect positions via shared utility: (n_frames, n_res, 3) in Ang
        positions: np.ndarray = collect_ca_positions(universe, atoms=ca)
        n_frames: int = positions.shape[0]

        if n_frames < 10:
            msg = "Too few frames for entropy estimation"
            logger.error(msg)
            return {"error": msg}

        # Convert to SI units
        pos_m: np.ndarray = positions * ANG_TO_M          # metres
        masses_kg: np.ndarray = masses * AMU_TO_KG         # kg

        # Full-trajectory entropy (Global)
        global_entropy: float = _schlitter_entropy(pos_m, masses_kg, temperature)

        # Per-residue entropy (3 DOF each) - Independent Residue Approximation
        logger.info("Computing per-residue entropy using Independent Residue Approximation (ignoring correlations)")
        per_res_entropy: np.ndarray = _per_residue_entropy(
            pos_m, masses_kg, temperature
        )

        # Convergence over trajectory fractions
        fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
        convergence: list[dict[str, Any]] = []
        for frac in fractions:
            n = max(10, int(n_frames * frac))
            s = _schlitter_entropy(pos_m[:n], masses_kg, temperature)
            convergence.append(
                {
                    "fraction": frac,
                    "n_frames": n,
                    "entropy_J_mol_K": round(float(s), 2),
                }
            )

        logger.info(
            "Global Entropy (Schlitter) at T=%.1f K: %.2f J/(mol*K) from %d frames. "
            "This is the physically rigorous upper bound.",
            temperature,
            global_entropy,
            n_frames,
        )

        return {
            "global_entropy_J_mol_K": round(float(global_entropy), 2),
            "total_entropy_J_mol_K": round(float(global_entropy), 2),
            "total_entropy_kJ_mol_K": round(float(global_entropy / 1000.0), 4),
            "per_residue_entropy": [round(float(x), 3) for x in per_res_entropy],
            "resids": resids,
            "entropy_convergence": convergence,
            "temperature_K": temperature,
            "n_frames_used": n_frames,
            "approximation_warning": (
                "Per-residue entropy values use the Independent Residue Approximation, "
                "which ignores cross-correlations and overestimates total entropy. "
                "Refer to global_entropy_J_mol_K for the rigorous system-level upper bound."
            )
        }

    except Exception as e:
        logger.exception("Entropy computation failed")
        return {"error": str(e)}


# ------------------------------------------------------------------ #
#  Internal helpers                                                    #
# ------------------------------------------------------------------ #


def _schlitter_entropy(
    positions_m: np.ndarray,
    masses_kg: np.ndarray,
    temperature: float,
) -> float:
    """Compute the Schlitter entropy upper bound.

    Parameters
    ----------
    positions_m : np.ndarray
        Shape ``(n_frames, n_atoms, 3)`` in metres.
    masses_kg : np.ndarray
        Shape ``(n_atoms,)`` in kg.
    temperature : float
        Temperature in K.

    Returns
    -------
    float
        Entropy in J/(mol*K).
    """
    n_frames, n_atoms, _ = positions_m.shape

    # Flatten to (n_frames, 3N)
    flat: np.ndarray = positions_m.reshape(n_frames, n_atoms * 3)
    mean: np.ndarray = flat.mean(axis=0)
    delta: np.ndarray = flat - mean

    # Mass-weight each coordinate by sqrt(m_i)
    mass_weights: np.ndarray = np.repeat(np.sqrt(masses_kg), 3)
    delta_mw: np.ndarray = delta * mass_weights[np.newaxis, :]

    # Covariance of mass-weighted fluctuations (units: kg * m**2)
    cov: np.ndarray = np.cov(delta_mw.T)

    # Eigenvalues -- clamp to non-negative for numerical safety
    eigenvalues: np.ndarray = eigvalsh(cov)
    eigenvalues = np.maximum(eigenvalues, 0.0)

    # Schlitter factor: kB*T*e**2 / hbar**2  (units: 1/(kg*m**2))
    kT: float = KB * temperature
    factor: float = kT * np.e ** 2 / (HBAR ** 2)

    # S = (kB / 2) * N_A * sum_i ln(1 + factor * lambda_i)
    log_terms: np.ndarray = np.log1p(factor * eigenvalues)
    entropy: float = 0.5 * KB * NA * float(np.sum(log_terms))

    return entropy


def _per_residue_entropy(
    positions_m: np.ndarray,
    masses_kg: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Compute per-residue entropy using each residue's 3 DOF.

    Parameters
    ----------
    positions_m : np.ndarray
        Shape ``(n_frames, n_atoms, 3)`` in metres.
    masses_kg : np.ndarray
        Shape ``(n_atoms,)`` in kg.
    temperature : float
        Temperature in K.

    Returns
    -------
    np.ndarray
        Shape ``(n_atoms,)`` -- per-residue entropy in J/(mol*K).
    """
    n_frames, n_atoms, _ = positions_m.shape

    kT: float = KB * temperature
    factor: float = kT * np.e ** 2 / (HBAR ** 2)

    per_res: np.ndarray = np.zeros(n_atoms, dtype=np.float64)

    # Vectorise the mean subtraction and mass-weighting across all residues
    mean_pos: np.ndarray = positions_m.mean(axis=0)           # (n_atoms, 3)
    delta: np.ndarray = positions_m - mean_pos                 # (n_frames, n_atoms, 3)
    sqrt_masses: np.ndarray = np.sqrt(masses_kg)               # (n_atoms,)
    delta_mw: np.ndarray = delta * sqrt_masses[np.newaxis, :, np.newaxis]  # mass-weight

    # For each residue the 3x3 covariance is small; compute all at once
    # delta_mw[:, i, :] has shape (n_frames, 3)
    # Batched covariance: C_i = delta_mw[:, i, :].T @ delta_mw[:, i, :] / (N-1)
    # Using einsum: (n_atoms, 3, 3)
    cov_batch: np.ndarray = np.einsum(
        "fia,fib->iab", delta_mw, delta_mw
    ) / (n_frames - 1)

    # Eigenvalues of each 3x3 matrix (vectorised via np.linalg.eigvalsh)
    eig_batch: np.ndarray = np.linalg.eigvalsh(cov_batch)     # (n_atoms, 3)
    eig_batch = np.maximum(eig_batch, 0.0)

    log_terms: np.ndarray = np.log1p(factor * eig_batch)      # (n_atoms, 3)
    per_res = 0.5 * KB * NA * log_terms.sum(axis=1)

    return per_res
