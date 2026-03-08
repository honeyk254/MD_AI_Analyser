"""Per-Residue Energy Decomposition (Simplified).

Estimates per-residue non-bonded interaction energies using distance-based
Lennard-Jones and Coulomb approximations on C-alpha atoms.  This is an
**approximate** calculation intended for qualitative comparison; it does
not replace a full force-field energy decomposition.

Residue-type-specific LJ parameters are used:

* Hydrophobic (ALA, VAL, LEU, ILE, PHE, TRP, MET, PRO): eps=0.8, sig=4.0
* Polar (SER, THR, ASN, GLN, TYR, CYS, HIS):             eps=0.4, sig=3.6
* Charged (ARG, LYS, ASP, GLU):                           eps=0.3, sig=3.4
* Default (GLY, others):                                   eps=0.5, sig=3.8

Combining rules: Lorentz--Berthelot.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Set

import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array

logger = logging.getLogger("md_ai_analyzer")

# Residue classification sets
_HYDROPHOBIC: Set[str] = {"ALA", "VAL", "LEU", "ILE", "PHE", "TRP", "MET", "PRO"}
_POLAR: Set[str] = {"SER", "THR", "ASN", "GLN", "TYR", "CYS", "HIS"}
_CHARGED: Set[str] = {"ARG", "LYS", "ASP", "GLU"}

# Coulomb constant in kcal/(mol*A*e^2)  ->  multiply by 4.184 for kJ
_COULOMB_CONST: float = 332.0637
_KCAL_TO_KJ: float = 4.184


def compute_energy_decomposition(
    universe: mda.Universe,
    cutoff: float = 12.0,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Estimate per-residue interaction energies from distance-based LJ + Coulomb.

    Parameters
    ----------
    universe : mda.Universe
        MDAnalysis Universe with a loaded trajectory.
    cutoff : float, optional
        Non-bonded interaction cutoff in Angstrom (default 12.0).
    **kwargs : Any
        Additional keyword arguments (unused, accepted for API consistency).

    Returns
    -------
    dict[str, Any]
        ``resids``
            Residue IDs.
        ``total_energy``
            Per-residue total interaction energy (kJ/mol).
        ``vdw_energy``
            Per-residue van der Waals energy (kJ/mol).
        ``elec_energy``
            Per-residue electrostatic energy (kJ/mol).
        ``top_pairs``
            Top 30 interacting residue pairs sorted by energy.
        ``energy_matrix``
            Pairwise energy matrix (truncated to 200x200 for large systems).
        ``mean_total_energy``
            Mean per-residue total energy (kJ/mol).
        ``has_charges``
            Whether partial charges were available from the topology.
        ``resnames``
            Residue names corresponding to ``resids``.
        ``parameter_source``
            Description of the LJ parameter source used.
    """
    try:
        ca = universe.select_atoms("protein and name CA")
        if len(ca) == 0:
            logger.warning("No CA atoms found for energy decomposition")
            return {"error": "No CA atoms found"}

        n_res: int = len(ca)
        resids: List[int] = ca.resids.tolist()
        logger.info(
            "Energy decomposition: %d residues, cutoff=%.1f A", n_res, cutoff
        )

        # ── Assign residue-type-specific LJ parameters ──────────
        resnames: List[str] = []
        epsilon_arr = np.empty(n_res, dtype=np.float64)
        sigma_arr = np.empty(n_res, dtype=np.float64)

        for i, res in enumerate(ca.residues):
            rn: str = res.resname
            resnames.append(rn)
            if rn in _HYDROPHOBIC:
                epsilon_arr[i], sigma_arr[i] = 0.8, 4.0
            elif rn in _POLAR:
                epsilon_arr[i], sigma_arr[i] = 0.4, 3.6
            elif rn in _CHARGED:
                epsilon_arr[i], sigma_arr[i] = 0.3, 3.4
            else:
                epsilon_arr[i], sigma_arr[i] = 0.5, 3.8

        # Lorentz-Berthelot combining rules (pre-computed once)
        eps_ij: np.ndarray = np.sqrt(np.outer(epsilon_arr, epsilon_arr))
        sig_ij: np.ndarray = np.add.outer(sigma_arr, sigma_arr) / 2.0
        sig6_ij: np.ndarray = sig_ij ** 6
        sig12_ij: np.ndarray = sig6_ij ** 2

        # Dielectric constant (implicit water screening)
        dielectric: float = 80.0

        # Attempt to read partial charges from topology
        try:
            charges: np.ndarray = ca.charges
            has_charges: bool = bool(np.any(charges != 0))
        except Exception:
            charges = np.zeros(n_res, dtype=np.float64)
            has_charges = False

        if has_charges:
            qq: np.ndarray = np.outer(charges, charges)
            logger.info("Using partial charges from topology")
        else:
            qq = np.zeros((n_res, n_res), dtype=np.float64)
            logger.info("No partial charges available; electrostatic term will be zero")

        # ── Accumulate energies over frames ──────────────────────
        vdw_per_residue = np.zeros(n_res, dtype=np.float64)
        elec_per_residue = np.zeros(n_res, dtype=np.float64)
        pair_energies = np.zeros((n_res, n_res), dtype=np.float64)
        n_frames: int = 0

        for ts in universe.trajectory:
            n_frames += 1
            dists = distance_array(ca.positions, ca.positions, box=ts.dimensions)

            # Avoid self-interaction and numerical issues at very short range
            np.fill_diagonal(dists, 1e10)
            np.clip(dists, 2.0, None, out=dists)

            # Cutoff mask
            mask = dists < cutoff

            # LJ energy: 4 * eps * [(sig/r)^12 - (sig/r)^6]
            r6 = np.where(mask, dists ** 6, 1e60)
            r12 = r6 ** 2
            lj = np.where(
                mask,
                4.0 * eps_ij * (sig12_ij / r12 - sig6_ij / r6),
                0.0,
            )

            # Coulomb energy: k * q_i * q_j / (eps_r * r)
            if has_charges:
                coulomb = np.where(
                    mask,
                    _COULOMB_CONST * _KCAL_TO_KJ * qq / (dielectric * dists),
                    0.0,
                )
            else:
                coulomb = np.zeros_like(lj)

            # Accumulate
            vdw_per_residue += lj.sum(axis=1)
            elec_per_residue += coulomb.sum(axis=1)
            pair_energies += lj + coulomb

        # ── Average over frames ──────────────────────────────────
        if n_frames > 0:
            vdw_per_residue /= n_frames
            elec_per_residue /= n_frames
            pair_energies /= n_frames

        total_per_residue: np.ndarray = vdw_per_residue + elec_per_residue

        # ── Top interacting pairs ────────────────────────────────
        # Extract upper triangle, filter by significance
        ii, jj = np.triu_indices(n_res, k=1)
        pair_vals = pair_energies[ii, jj]
        significant = np.abs(pair_vals) > 0.1
        sig_ii = ii[significant]
        sig_jj = jj[significant]
        sig_vals = pair_vals[significant]

        # Sort by energy (most negative first)
        order = np.argsort(sig_vals)
        top_pairs: List[Dict[str, Any]] = []
        for idx in order[:30]:
            i, j = int(sig_ii[idx]), int(sig_jj[idx])
            top_pairs.append(
                {
                    "resid_i": int(resids[i]),
                    "resid_j": int(resids[j]),
                    "resname_i": resnames[i],
                    "resname_j": resnames[j],
                    "energy_kj": round(float(sig_vals[idx]), 2),
                    "vdw_kj": round(
                        float(vdw_per_residue[i] + vdw_per_residue[j]) / n_res,
                        3,
                    ),
                    "elec_kj": round(
                        float(elec_per_residue[i] + elec_per_residue[j]) / n_res,
                        3,
                    ),
                }
            )

        # Truncate energy matrix for large systems
        matrix_size = min(n_res, 200)
        energy_matrix_trunc = pair_energies[:matrix_size, :matrix_size].tolist()

        logger.info(
            "Energy decomposition complete: mean total=%.3f kJ/mol, "
            "has_charges=%s",
            float(np.mean(total_per_residue)),
            has_charges,
        )

        return {
            "resids": resids,
            "total_energy": [round(float(x), 3) for x in total_per_residue],
            "vdw_energy": [round(float(x), 3) for x in vdw_per_residue],
            "elec_energy": [round(float(x), 3) for x in elec_per_residue],
            "top_pairs": top_pairs,
            "energy_matrix": energy_matrix_trunc if n_res <= 200 else [],
            "mean_total_energy": round(float(np.mean(total_per_residue)), 3),
            "has_charges": bool(has_charges),
            "resnames": resnames,
            "parameter_source": "residue-type-specific Lorentz-Berthelot",
        }

    except Exception as e:
        logger.error("Energy decomposition failed: %s", e, exc_info=True)
        return {"error": str(e)}
