"""
Perturbation Response Scanning (PRS).
Predicts the structural response to perturbation at each residue
using the covariance-based linear response theory.
"""
import numpy as np
from scipy.linalg import pinv


def compute_prs(universe, **kwargs):
    """
    Perturbation Response Scanning based on covariance matrix inversion.

    For each residue i, apply a unit perturbation and compute the
    mean-square response of all other residues j using:
      response_j = sum_k (C_jk * F_k) where F is the perturbation vector.

    Returns dict with:
        - resids: residue IDs
        - effector_scores: per-residue effector score (how much it affects others)
        - sensor_scores: per-residue sensor score (how sensitive to perturbation)
        - response_matrix: PRS response matrix (effector × sensor)
        - top_effectors: top 20 effector residues
        - top_sensors: top 20 sensor residues
        - top_pairs: top effector→sensor pairs
    """
    try:
        ca = universe.select_atoms("protein and name CA")
        if len(ca) == 0:
            return {"error": "No CA atoms found"}

        n_res = len(ca)
        resids = ca.resids.tolist()

        if n_res > 500:
            # For very large proteins, use a stride
            stride = max(1, n_res // 300)
            ca_subset = ca[::stride]
            n_res_eff = len(ca_subset)
            resids_eff = ca_subset.resids.tolist()
        else:
            ca_subset = ca
            n_res_eff = n_res
            resids_eff = resids

        # Collect Cα positions over trajectory
        positions = []
        for ts in universe.trajectory:
            positions.append(ca_subset.positions.copy())
        positions = np.array(positions)  # (n_frames, n_res, 3)
        n_frames = positions.shape[0]

        if n_frames < 10:
            return {"error": "Too few frames for PRS analysis"}

        # Flatten to (n_frames, n_res*3)
        flat = positions.reshape(n_frames, n_res_eff * 3)
        mean_pos = flat.mean(axis=0)
        delta = flat - mean_pos

        # Covariance matrix (3N x 3N)
        cov = np.cov(delta.T)

        # Pseudo-inverse of covariance
        cov_inv = pinv(cov, rcond=1e-6)

        # PRS: for each residue i (effector), apply unit perturbation along x,y,z
        # Response of residue j = sqrt(sum over 3 components of (sum_k cov[3j+α, 3i+β] * F_β)^2)
        response_matrix = np.zeros((n_res_eff, n_res_eff))

        for i in range(n_res_eff):
            for direction in range(3):
                # Unit force on residue i in direction
                force = np.zeros(n_res_eff * 3)
                force[3 * i + direction] = 1.0

                # Linear response: Δr = C · F
                displacement = cov @ force  # using covariance, not inverse

                # Compute per-residue response magnitude
                for j in range(n_res_eff):
                    dx = displacement[3 * j: 3 * j + 3]
                    response_matrix[i, j] += np.sum(dx ** 2)

        # Normalize: average over 3 directions, then sqrt
        response_matrix /= 3.0
        response_matrix = np.sqrt(response_matrix)

        # Effector score: how much does perturbing i affect the rest?
        effector_scores = np.mean(response_matrix, axis=1)
        # Sensor score: how sensitive is j to perturbation anywhere?
        sensor_scores = np.mean(response_matrix, axis=0)

        # Normalize scores to [0, 1]
        if effector_scores.max() > 0:
            effector_norm = effector_scores / effector_scores.max()
        else:
            effector_norm = effector_scores
        if sensor_scores.max() > 0:
            sensor_norm = sensor_scores / sensor_scores.max()
        else:
            sensor_norm = sensor_scores

        # Top effectors
        eff_order = np.argsort(-effector_norm)
        top_effectors = [{"resid": int(resids_eff[i]), "score": round(float(effector_norm[i]), 4)}
                         for i in eff_order[:20]]

        # Top sensors
        sens_order = np.argsort(-sensor_norm)
        top_sensors = [{"resid": int(resids_eff[i]), "score": round(float(sensor_norm[i]), 4)}
                       for i in sens_order[:20]]

        # Top effector→sensor pairs
        top_pairs = []
        flat_resp = []
        for i in range(n_res_eff):
            for j in range(n_res_eff):
                if i != j and abs(i - j) > 5:
                    flat_resp.append((i, j, response_matrix[i, j]))
        flat_resp.sort(key=lambda x: -x[2])
        for i, j, r in flat_resp[:30]:
            top_pairs.append({
                "effector_resid": int(resids_eff[i]),
                "sensor_resid": int(resids_eff[j]),
                "response": round(float(r), 4),
            })

        # Truncate matrix for JSON
        matrix_size = min(n_res_eff, 150)
        response_trunc = response_matrix[:matrix_size, :matrix_size]

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
        return {"error": str(e)}
