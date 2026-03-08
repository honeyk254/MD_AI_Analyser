"""
Trajectory Comparison (item 46).
Compare two trajectories (e.g., wild-type vs mutant, different conditions).
"""
import numpy as np


def compare_trajectories(universe, universe_ref=None, **kwargs):
    """
    Compare two trajectories on structural and dynamic metrics.

    If universe_ref is None, compares first-half vs second-half of the
    single trajectory (useful for equilibration assessment).

    Returns dict with per-metric comparisons.
    """
    try:
        ca = universe.select_atoms("protein and name CA")
        if len(ca) == 0:
            ca = universe.select_atoms("all")

        n_frames = len(universe.trajectory)
        if n_frames < 10:
            return {"error": "Too few frames for comparison"}

        # Collect metrics for trajectory A and B
        if universe_ref is not None:
            ca_ref = universe_ref.select_atoms("protein and name CA")
            if len(ca_ref) == 0:
                ca_ref = universe_ref.select_atoms("all")
            metrics_a = _collect_metrics(universe, ca)
            metrics_b = _collect_metrics(universe_ref, ca_ref)
            comparison_type = "two_trajectories"
        else:
            # Split single trajectory in half
            half = n_frames // 2
            metrics_a = _collect_metrics(universe, ca, start=0, end=half)
            metrics_b = _collect_metrics(universe, ca, start=half, end=n_frames)
            comparison_type = "half_split"

        # Compare
        results = {
            "comparison_type": comparison_type,
            "n_frames_a": metrics_a["n_frames"],
            "n_frames_b": metrics_b["n_frames"],
            "comparisons": {},
        }

        # RMSD comparison
        if metrics_a["rg_values"] and metrics_b["rg_values"]:
            rg_a = np.array(metrics_a["rg_values"])
            rg_b = np.array(metrics_b["rg_values"])
            results["comparisons"]["rg"] = {
                "mean_a": round(float(np.mean(rg_a)), 3),
                "mean_b": round(float(np.mean(rg_b)), 3),
                "std_a": round(float(np.std(rg_a)), 3),
                "std_b": round(float(np.std(rg_b)), 3),
                "difference": round(float(np.mean(rg_b) - np.mean(rg_a)), 3),
                "ks_test": _ks_test(rg_a, rg_b),
            }

        # Per-residue RMSF comparison
        if len(metrics_a["rmsf"]) == len(metrics_b["rmsf"]) and len(metrics_a["rmsf"]) > 0:
            rmsf_a = np.array(metrics_a["rmsf"])
            rmsf_b = np.array(metrics_b["rmsf"])
            diff = rmsf_b - rmsf_a
            significant = np.where(np.abs(diff) > np.std(diff))[0]
            results["comparisons"]["rmsf"] = {
                "mean_rmsf_a": round(float(np.mean(rmsf_a)), 3),
                "mean_rmsf_b": round(float(np.mean(rmsf_b)), 3),
                "max_diff_resid": int(metrics_a["resids"][np.argmax(np.abs(diff))]) if len(diff) > 0 else None,
                "max_diff_value": round(float(np.max(np.abs(diff))), 3) if len(diff) > 0 else 0,
                "n_significant_changes": int(len(significant)),
                "significant_resids": [int(metrics_a["resids"][i]) for i in significant[:20]],
            }

        # Contact map difference
        if (metrics_a["contact_map"] is not None and metrics_b["contact_map"] is not None
                and metrics_a["contact_map"].shape == metrics_b["contact_map"].shape):
            diff_map = metrics_b["contact_map"] - metrics_a["contact_map"]
            changed = np.where(np.abs(diff_map) > 0.2)
            results["comparisons"]["contacts"] = {
                "n_changed_contacts": int(len(changed[0])),
                "mean_contact_diff": round(float(np.mean(np.abs(diff_map))), 4),
            }

        return results

    except Exception as e:
        return {"error": str(e)}


def _collect_metrics(universe, ca, start=None, end=None):
    """Collect basic metrics from a trajectory range."""
    rg_values = []
    coords_list = []
    resids = ca.resids.tolist()

    frames = list(universe.trajectory)
    if start is not None:
        frames = frames[start:end]

    for ts in frames:
        rg_values.append(float(ca.radius_of_gyration()))
        coords_list.append(ca.positions.copy())

    n_frames = len(frames)

    # RMSF
    if coords_list:
        coords_arr = np.array(coords_list)
        mean_pos = coords_arr.mean(axis=0)
        rmsf = np.sqrt(np.mean((coords_arr - mean_pos) ** 2, axis=0).mean(axis=1))
    else:
        rmsf = []

    # Average contact map (Cα-Cα)
    contact_map = None
    if coords_list and len(ca) < 500:
        contact_maps = []
        for c in coords_list[::max(1, len(coords_list) // 50)]:
            from scipy.spatial.distance import cdist
            dm = cdist(c, c)
            contact_maps.append((dm < 8.0).astype(float))
        if contact_maps:
            contact_map = np.mean(contact_maps, axis=0)

    return {
        "n_frames": n_frames,
        "rg_values": rg_values,
        "rmsf": rmsf.tolist() if isinstance(rmsf, np.ndarray) else rmsf,
        "resids": resids,
        "contact_map": contact_map,
    }


def _ks_test(a, b):
    """Kolmogorov-Smirnov test between two distributions."""
    try:
        from scipy.stats import ks_2samp
        stat, pval = ks_2samp(a, b)
        return {"statistic": round(float(stat), 4), "p_value": round(float(pval), 6)}
    except ImportError:
        return {"statistic": None, "p_value": None}
