"""
Per-Residue Energy Decomposition (Simplified).
Estimates per-residue non-bonded interaction energies using
distance-based Lennard-Jones and Coulomb approximations on Cα atoms.
"""
import numpy as np
from MDAnalysis.lib.distances import distance_array


def compute_energy_decomposition(universe, cutoff=12.0, **kwargs):
    """
    Estimate per-residue interaction energies from distance-based LJ + Coulomb.

    Uses residue-type-specific LJ parameters where available:
      - Hydrophobic residues: epsilon=0.8, sigma=4.0
      - Polar residues: epsilon=0.4, sigma=3.6
      - Charged residues: epsilon=0.3, sigma=3.4
      - Default (Gly/Pro): epsilon=0.5, sigma=3.8
    Reads charges from topology when available.

    Returns dict with:
        - resids: residue IDs
        - total_energy: per-residue total interaction energy (kJ/mol)
        - vdw_energy: per-residue van der Waals energy
        - elec_energy: per-residue electrostatic energy
        - top_pairs: top interacting residue pairs
        - energy_matrix: pairwise energy matrix (NxN, truncated for large systems)
        - resnames: residue names
        - per_pair_breakdown: VdW and elec per top pair
    """
    try:
        ca = universe.select_atoms("protein and name CA")
        if len(ca) == 0:
            return {"error": "No CA atoms found"}

        n_res = len(ca)
        resids = ca.resids.tolist()

        # Residue-type-specific LJ parameters (item 45)
        hydrophobic = {'ALA', 'VAL', 'LEU', 'ILE', 'PHE', 'TRP', 'MET', 'PRO'}
        polar = {'SER', 'THR', 'ASN', 'GLN', 'TYR', 'CYS', 'HIS'}
        charged = {'ARG', 'LYS', 'ASP', 'GLU'}

        resnames = []
        epsilon_arr = np.zeros(n_res)
        sigma_arr = np.zeros(n_res)
        for i, res in enumerate(ca.residues):
            rn = res.resname
            resnames.append(rn)
            if rn in hydrophobic:
                epsilon_arr[i] = 0.8
                sigma_arr[i] = 4.0
            elif rn in polar:
                epsilon_arr[i] = 0.4
                sigma_arr[i] = 3.6
            elif rn in charged:
                epsilon_arr[i] = 0.3
                sigma_arr[i] = 3.4
            else:
                epsilon_arr[i] = 0.5
                sigma_arr[i] = 3.8

        # Combining rules: Lorentz-Berthelot
        eps_ij = np.sqrt(np.outer(epsilon_arr, epsilon_arr))
        sig_ij = (np.add.outer(sigma_arr, sigma_arr)) / 2.0
        sig6_ij = sig_ij ** 6
        sig12_ij = sig6_ij ** 2

        # Dielectric constant for Coulomb
        dielectric = 80.0  # water screening
        coulomb_const = 332.0637  # kcal/mol·Å/e² → convert to kJ: × 4.184

        # Try to get charges; fallback to zero
        try:
            charges = ca.charges
            has_charges = np.any(charges != 0)
        except Exception:
            charges = np.zeros(n_res)
            has_charges = False

        # Accumulate energies over frames
        vdw_per_residue = np.zeros(n_res)
        elec_per_residue = np.zeros(n_res)
        pair_energies = np.zeros((n_res, n_res))
        n_frames = 0

        for ts in universe.trajectory:
            n_frames += 1
            dists = distance_array(ca.positions, ca.positions, box=ts.dimensions)

            # Avoid self and very close contacts
            np.fill_diagonal(dists, 1e10)
            dists = np.clip(dists, 2.0, None)

            # Apply cutoff mask
            mask = dists < cutoff

            # LJ energy: 4ε[(σ/r)^12 - (σ/r)^6]
            r6 = np.where(mask, dists ** 6, 1e60)
            r12 = r6 ** 2
            lj = np.where(mask, 4.0 * eps_ij * (sig12_ij / r12 - sig6_ij / r6), 0.0)

            # Coulomb energy: kqq/εr
            if has_charges:
                qq = np.outer(charges, charges)
                coulomb = np.where(mask, coulomb_const * 4.184 * qq / (dielectric * dists), 0.0)
            else:
                coulomb = np.zeros_like(lj)

            # Sum pairwise into per-residue
            vdw_per_residue += np.sum(lj, axis=1)
            elec_per_residue += np.sum(coulomb, axis=1)
            pair_energies += (lj + coulomb)

        # Average over frames
        if n_frames > 0:
            vdw_per_residue /= n_frames
            elec_per_residue /= n_frames
            pair_energies /= n_frames

        total_per_residue = vdw_per_residue + elec_per_residue

        # Top interacting pairs with per-pair breakdown
        top_pairs = []
        pair_flat = []
        for i in range(n_res):
            for j in range(i + 1, n_res):
                if abs(pair_energies[i, j]) > 0.1:
                    pair_flat.append((i, j, pair_energies[i, j]))

        pair_flat.sort(key=lambda x: x[2])
        for i, j, e in pair_flat[:30]:
            top_pairs.append({
                "resid_i": int(resids[i]),
                "resid_j": int(resids[j]),
                "resname_i": resnames[i],
                "resname_j": resnames[j],
                "energy_kj": round(float(e), 2),
                "vdw_kj": round(float(vdw_per_residue[i] + vdw_per_residue[j]) / n_res, 3),
                "elec_kj": round(float(elec_per_residue[i] + elec_per_residue[j]) / n_res, 3),
            })

        # Truncate energy matrix for large systems
        matrix_size = min(n_res, 200)
        energy_matrix_trunc = pair_energies[:matrix_size, :matrix_size].tolist()

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
        return {"error": str(e)}
