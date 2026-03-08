"""
Radius of Gyration analysis.
Measures protein compactness over time.
"""
import numpy as np


def compute_rg(universe, **kwargs):
    """
    Compute radius of gyration over the trajectory.

    Returns dict with:
        - time: list of timestamps
        - rg: list of Rg values (Å)
        - mean_rg: float
        - std_rg: float
        - trend: 'compacting', 'expanding', or 'stable'
    """
    protein = universe.select_atoms("protein")
    if len(protein) == 0:
        protein = universe.select_atoms("all")

    times = []
    rg_values = []

    for ts in universe.trajectory:
        times.append(float(ts.time))
        rg_values.append(float(protein.radius_of_gyration()))

    rg_arr = np.array(rg_values)
    mean_rg = float(np.mean(rg_arr))
    std_rg = float(np.std(rg_arr))

    # Detect trend
    if len(rg_arr) > 10:
        first_quarter = np.mean(rg_arr[:len(rg_arr)//4])
        last_quarter = np.mean(rg_arr[-len(rg_arr)//4:])
        diff = last_quarter - first_quarter
        if diff < -0.5:
            trend = "compacting"
        elif diff > 0.5:
            trend = "expanding"
        else:
            trend = "stable"
    else:
        trend = "stable"

    return {
        "time": times,
        "rg": rg_values,
        "mean_rg": mean_rg,
        "std_rg": std_rg,
        "min_rg": float(np.min(rg_arr)),
        "max_rg": float(np.max(rg_arr)),
        "trend": trend,
    }
