"""
Configurational Entropy Estimation.
Uses Schlitter's method to estimate configurational entropy
from the mass-weighted covariance matrix of Cα atom positions.
"""
import numpy as np
from scipy.linalg import eigvalsh


# Physical constants
KB = 1.380649e-23     # Boltzmann constant (J/K)
HBAR = 1.054571817e-34  # Reduced Planck constant (J·s)
NA = 6.02214076e23     # Avogadro's number
AMU_TO_KG = 1.66054e-27  # atomic mass unit → kg
ANG_TO_M = 1e-10        # Ångström → meters
TEMP = 300.0             # Default temperature (K)


def compute_entropy(universe, temperature=300.0, **kwargs):
    """
    Estimate configurational entropy using Schlitter's method.

    S_Schlitter = 0.5 * kB * sum_i ln(1 + kBTe²/(ħ²) * λ_i)
    where λ_i are eigenvalues of the mass-weighted covariance matrix.

    Returns dict with:
        - total_entropy_J_mol_K: total entropy in J/(mol·K)
        - total_entropy_kJ_mol_K: total entropy in kJ/(mol·K)
        - per_residue_entropy: per-residue entropy contribution
        - resids: residue IDs
        - entropy_convergence: entropy estimated from increasing trajectory fractions
    """
    try:
        ca = universe.select_atoms("protein and name CA")
        if len(ca) == 0:
            return {"error": "No CA atoms found"}

        n_res = len(ca)
        resids = ca.resids.tolist()

        # Get masses
        try:
            masses = ca.masses  # in amu
        except Exception:
            masses = np.full(n_res, 12.0)  # default carbon mass

        # Collect positions
        positions = []
        for ts in universe.trajectory:
            positions.append(ca.positions.copy())
        positions = np.array(positions)  # (n_frames, n_res, 3)
        n_frames = positions.shape[0]

        if n_frames < 10:
            return {"error": "Too few frames for entropy estimation"}

        # Convert to meters and kg
        pos_m = positions * ANG_TO_M  # (n_frames, n_res, 3) in meters
        masses_kg = masses * AMU_TO_KG  # in kg

        # Compute total entropy from full trajectory
        total_entropy = _schlitter_entropy(pos_m, masses_kg, temperature)

        # Per-residue entropy (3 DOF per residue)
        per_res_entropy = _per_residue_entropy(pos_m, masses_kg, temperature)

        # Convergence: compute entropy from 20%, 40%, 60%, 80%, 100% of trajectory
        convergence = []
        fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
        for frac in fractions:
            n = max(10, int(n_frames * frac))
            s = _schlitter_entropy(pos_m[:n], masses_kg, temperature)
            convergence.append({
                "fraction": frac,
                "n_frames": n,
                "entropy_J_mol_K": round(float(s), 2),
            })

        return {
            "total_entropy_J_mol_K": round(float(total_entropy), 2),
            "total_entropy_kJ_mol_K": round(float(total_entropy / 1000.0), 4),
            "per_residue_entropy": [round(float(x), 3) for x in per_res_entropy],
            "resids": resids,
            "entropy_convergence": convergence,
            "temperature_K": temperature,
            "n_frames_used": n_frames,
        }

    except Exception as e:
        return {"error": str(e)}


def _schlitter_entropy(positions_m, masses_kg, temp):
    """
    Compute Schlitter entropy.
    positions_m: (n_frames, n_atoms, 3) in meters
    masses_kg: (n_atoms,) in kg
    """
    n_frames, n_atoms, _ = positions_m.shape

    # Flatten to (n_frames, 3N)
    flat = positions_m.reshape(n_frames, n_atoms * 3)
    mean = flat.mean(axis=0)
    delta = flat - mean

    # Mass-weight: multiply each coordinate by sqrt(mass)
    mass_weights = np.repeat(np.sqrt(masses_kg), 3)
    delta_mw = delta * mass_weights[np.newaxis, :]

    # Covariance matrix
    cov = np.cov(delta_mw.T)

    # Schlitter formula: S = 0.5 * kB * sum ln(1 + kBTe²/ħ² * λ_i)
    eigenvalues = eigvalsh(cov)
    eigenvalues = np.maximum(eigenvalues, 0)  # numerical safety

    factor = KB * temp * np.e ** 2 / (HBAR ** 2)
    log_terms = np.log(1.0 + factor * eigenvalues)
    entropy = 0.5 * KB * NA * np.sum(log_terms)  # J/(mol·K)

    return entropy


def _per_residue_entropy(positions_m, masses_kg, temp):
    """Compute per-residue entropy using 3 DOF per residue."""
    n_frames, n_atoms, _ = positions_m.shape
    per_res = np.zeros(n_atoms)

    factor = KB * temp * np.e ** 2 / (HBAR ** 2)

    for i in range(n_atoms):
        pos_i = positions_m[:, i, :]  # (n_frames, 3)
        mass_i = masses_kg[i]
        delta = pos_i - pos_i.mean(axis=0)
        delta_mw = delta * np.sqrt(mass_i)
        cov_i = np.cov(delta_mw.T)  # 3x3

        eigenvalues = eigvalsh(cov_i)
        eigenvalues = np.maximum(eigenvalues, 0)

        log_terms = np.log(1.0 + factor * eigenvalues)
        per_res[i] = 0.5 * KB * NA * np.sum(log_terms)

    return per_res
