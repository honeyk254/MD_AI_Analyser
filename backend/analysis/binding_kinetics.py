"""
Binding Kinetics Analysis.
Computes residence time, kon/koff estimates, and contact survival
for ligand-protein interactions.
"""
import numpy as np
from MDAnalysis.lib.distances import distance_array


def compute_binding_kinetics(universe, ligand_sel="resname LIG", contact_cutoff=4.5, **kwargs):
    """
    Analyze ligand binding kinetics from MD trajectory.

    Computes:
    - Per-residue contact survival function
    - Ligand residence time (continuous and intermittent)
    - Estimated kon/koff from contact/dissociation events
    - Distance to binding site over time

    Returns dict with:
        - residence_time_continuous: mean continuous contact time (ps)
        - residence_time_intermittent: mean intermittent contact time (ps)
        - contact_survival: survival probability over time
        - binding_events: list of bind/unbind events
        - per_residue_contact_time: per-residue average contact duration
        - com_distance: center-of-mass distance over time
        - koff_estimate: estimated off-rate (1/ps)
        - kon_estimate: estimated on-rate (1/ps)
    """
    try:
        ligand = universe.select_atoms(ligand_sel)
        if len(ligand) == 0:
            return {"error": f"No atoms matched ligand selection: {ligand_sel}"}

        protein = universe.select_atoms("protein")
        if len(protein) == 0:
            return {"error": "No protein atoms found"}

        ca = universe.select_atoms("protein and name CA")
        resids = ca.resids.tolist()
        n_res = len(ca)

        n_frames = len(universe.trajectory)
        dt = getattr(universe.trajectory, 'dt', 1.0)  # ps per frame

        # Track per-residue contacts and COM distance
        per_res_contact = np.zeros((n_frames, n_res), dtype=bool)
        com_distances = []
        times = []
        any_contact = np.zeros(n_frames, dtype=bool)

        for frame_idx, ts in enumerate(universe.trajectory):
            times.append(float(ts.time))

            # COM distance
            lig_com = ligand.center_of_mass()
            prot_com = protein.center_of_mass()
            com_dist = np.linalg.norm(lig_com - prot_com)
            com_distances.append(float(com_dist))

            # Per-residue min distance to ligand
            for ri, res_atom in enumerate(ca):
                dists = distance_array(
                    np.array([res_atom.position]),
                    ligand.positions
                )
                min_dist = dists.min()
                if min_dist < contact_cutoff:
                    per_res_contact[frame_idx, ri] = True

            any_contact[frame_idx] = per_res_contact[frame_idx].any()

        # ── Contact Survival Function ──────────────────────────
        # S(t) = <h(0)h(t)> / <h(0)h(0)> where h=1 if in contact
        max_lag = min(n_frames // 2, 500)
        survival = _contact_survival(any_contact, max_lag)

        # ── Residence Times ────────────────────────────────────
        continuous_times, intermittent_times = _compute_residence_times(any_contact, dt)

        # ── Binding Events ─────────────────────────────────────
        events = []
        in_contact = any_contact[0]
        event_start = 0 if in_contact else None

        for i in range(1, n_frames):
            if any_contact[i] and not in_contact:
                # Binding event
                event_start = i
                events.append({
                    "type": "bind",
                    "frame": i,
                    "time_ps": round(times[i], 2),
                })
            elif not any_contact[i] and in_contact:
                # Unbinding event
                duration = (i - (event_start or 0)) * dt
                events.append({
                    "type": "unbind",
                    "frame": i,
                    "time_ps": round(times[i], 2),
                    "contact_duration_ps": round(duration, 2),
                })
            in_contact = any_contact[i]

        # ── Rate Estimates ─────────────────────────────────────
        n_unbind = sum(1 for e in events if e["type"] == "unbind")
        n_bind = sum(1 for e in events if e["type"] == "bind")
        total_contact_time = np.sum(any_contact) * dt
        total_unbound_time = np.sum(~any_contact) * dt

        koff = n_unbind / max(total_contact_time, dt) if total_contact_time > 0 else 0.0
        kon = n_bind / max(total_unbound_time, dt) if total_unbound_time > 0 else 0.0

        # ── Per-Residue Contact Time ───────────────────────────
        per_res_time = []
        for ri in range(n_res):
            contact_frames = np.sum(per_res_contact[:, ri])
            mean_contact_ps = contact_frames * dt
            occupancy = contact_frames / max(n_frames, 1)
            if occupancy > 0.01:
                per_res_time.append({
                    "resid": int(resids[ri]),
                    "total_contact_ps": round(float(mean_contact_ps), 2),
                    "occupancy": round(float(occupancy), 4),
                })

        per_res_time.sort(key=lambda x: -x["occupancy"])

        return {
            "residence_time_continuous_ps": round(float(np.mean(continuous_times)), 2) if continuous_times else 0.0,
            "residence_time_intermittent_ps": round(float(np.mean(intermittent_times)), 2) if intermittent_times else 0.0,
            "contact_survival": survival,
            "binding_events": events[:200],
            "per_residue_contact_time": per_res_time[:50],
            "com_distance": [round(d, 2) for d in com_distances],
            "time": times,
            "koff_estimate_per_ps": round(float(koff), 8),
            "kon_estimate_per_ps": round(float(kon), 8),
            "n_bind_events": n_bind,
            "n_unbind_events": n_unbind,
            "total_contact_fraction": round(float(np.mean(any_contact)), 4),
        }

    except Exception as e:
        return {"error": str(e)}


def _contact_survival(contact_array, max_lag):
    """Compute contact survival function S(t)."""
    n = len(contact_array)
    survival = []
    c0 = np.sum(contact_array)
    if c0 == 0:
        return []

    for lag in range(0, max_lag, max(1, max_lag // 50)):
        if lag >= n:
            break
        overlap = np.sum(contact_array[:n - lag] & contact_array[lag:])
        s = overlap / c0
        survival.append({
            "lag_frames": lag,
            "survival": round(float(s), 4),
        })
    return survival


def _compute_residence_times(contact_array, dt):
    """Compute continuous and intermittent residence times."""
    n = len(contact_array)
    continuous = []
    current_run = 0

    for i in range(n):
        if contact_array[i]:
            current_run += 1
        else:
            if current_run > 0:
                continuous.append(current_run * dt)
            current_run = 0
    if current_run > 0:
        continuous.append(current_run * dt)

    # Intermittent: allow gaps of up to 5 frames
    intermittent = []
    gap_tolerance = 5
    in_contact = False
    contact_start = 0
    gap_count = 0

    for i in range(n):
        if contact_array[i]:
            if not in_contact:
                contact_start = i
                in_contact = True
            gap_count = 0
        else:
            if in_contact:
                gap_count += 1
                if gap_count > gap_tolerance:
                    duration = (i - gap_tolerance - contact_start) * dt
                    if duration > 0:
                        intermittent.append(duration)
                    in_contact = False
                    gap_count = 0

    if in_contact:
        intermittent.append((n - contact_start) * dt)

    return continuous, intermittent
