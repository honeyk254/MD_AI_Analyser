"""
RMSD (Root Mean Square Deviation) analysis.
Computes backbone RMSD over the trajectory relative to the first frame.
"""
import numpy as np
from MDAnalysis.analysis.rms import RMSD as MDA_RMSD


def compute_rmsd(universe, **kwargs):
    """
    Compute RMSD of backbone atoms over the trajectory.
    
    Returns dict with:
        - time: list of timestamps (ps)
        - rmsd: list of RMSD values (Å)
        - mean_rmsd: float
        - std_rmsd: float
        - max_rmsd: float
        - equilibration_frame: estimated frame where RMSD stabilizes
    """
    protein = universe.select_atoms("protein and backbone")
    if len(protein) == 0:
        protein = universe.select_atoms("all")

    ref = universe.copy()
    ref.trajectory[0]

    R = MDA_RMSD(universe, ref, select="backbone", ref_frame=0)
    R.run()

    times = R.results.rmsd[:, 1].tolist()
    rmsd_values = R.results.rmsd[:, 2].tolist()

    rmsd_arr = np.array(rmsd_values)

    # Estimate equilibration: find first frame where running avg stabilizes
    window = max(len(rmsd_arr) // 10, 5)
    if len(rmsd_arr) > window:
        running_avg = np.convolve(rmsd_arr, np.ones(window)/window, mode='valid')
        diffs = np.abs(np.diff(running_avg))
        threshold = np.mean(diffs) * 0.5
        equil_candidates = np.where(diffs < threshold)[0]
        equil_frame = int(equil_candidates[0]) if len(equil_candidates) > 0 else 0
    else:
        equil_frame = 0

    return {
        "time": times,
        "rmsd": rmsd_values,
        "mean_rmsd": float(np.mean(rmsd_arr)),
        "std_rmsd": float(np.std(rmsd_arr)),
        "max_rmsd": float(np.max(rmsd_arr)),
        "min_rmsd": float(np.min(rmsd_arr)),
        "equilibration_frame": equil_frame,
    }
