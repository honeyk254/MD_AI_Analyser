"""Binding Kinetics Analysis.

Computes residence time, kon/koff estimates, and contact-survival
functions for ligand--protein interactions from an MD trajectory.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array

logger = logging.getLogger("md_ai_analyzer")


def compute_binding_kinetics(
    universe: mda.Universe,
    ligand_sel: str = "resname LIG",
    contact_cutoff: float = 4.5,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Analyse ligand binding kinetics from an MD trajectory.

    Metrics computed:

    * Per-residue contact survival function.
    * Ligand residence time (continuous and intermittent).
    * Estimated kon/koff from contact/dissociation events.
    * Centre-of-mass distance over time.

    Parameters
    ----------
    universe : mda.Universe
        MDAnalysis Universe with a loaded trajectory.
    ligand_sel : str, optional
        Atom-selection string for the ligand (default ``"resname LIG"``).
    contact_cutoff : float, optional
        Distance cutoff in Angstrom for a residue--ligand contact
        (default 4.5).
    **kwargs : Any
        Additional keyword arguments (unused, accepted for API consistency).

    Returns
    -------
    dict[str, Any]
        ``residence_time_continuous_ps``
            Mean continuous contact time (ps).
        ``residence_time_intermittent_ps``
            Mean intermittent contact time allowing short gaps (ps).
        ``contact_survival``
            Contact survival probability S(t) over lag time.
        ``binding_events``
            List of bind/unbind events (up to 200).
        ``per_residue_contact_time``
            Per-residue average contact duration and occupancy (up to 50).
        ``com_distance``
            Centre-of-mass distance over time (Angstrom).
        ``time``
            List of timestamps (ps).
        ``koff_estimate_per_ps``
            Estimated off-rate (1/ps).
        ``kon_estimate_per_ps``
            Estimated on-rate (1/ps).
        ``n_bind_events``
            Number of binding events detected.
        ``n_unbind_events``
            Number of unbinding events detected.
        ``total_contact_fraction``
            Fraction of frames with any ligand--protein contact.
    """
    try:
        ligand = universe.select_atoms(ligand_sel)
        if len(ligand) == 0:
            msg = f"No atoms matched ligand selection: {ligand_sel}"
            logger.warning(msg)
            return {"error": msg}

        protein = universe.select_atoms("protein")
        if len(protein) == 0:
            logger.warning("No protein atoms found for binding kinetics")
            return {"error": "No protein atoms found"}

        ca = universe.select_atoms("protein and name CA")
        resids: List[int] = ca.resids.tolist()
        n_res: int = len(ca)

        n_frames: int = len(universe.trajectory)
        dt: float = float(getattr(universe.trajectory, "dt", 1.0))  # ps/frame

        logger.info(
            "Binding kinetics: ligand '%s' (%d atoms), %d CA residues, "
            "%d frames, dt=%.2f ps",
            ligand_sel,
            len(ligand),
            n_res,
            n_frames,
            dt,
        )

        # Track per-residue contacts and COM distance
        per_res_contact = np.zeros((n_frames, n_res), dtype=bool)
        com_distances = np.empty(n_frames, dtype=np.float64)
        times = np.empty(n_frames, dtype=np.float64)
        any_contact = np.zeros(n_frames, dtype=bool)

        for frame_idx, ts in enumerate(universe.trajectory):
            times[frame_idx] = ts.time

            # Centre-of-mass distance
            lig_com = ligand.center_of_mass()
            prot_com = protein.center_of_mass()
            com_distances[frame_idx] = float(np.linalg.norm(lig_com - prot_com))

            # Vectorised per-residue minimum distance to ligand
            dists = distance_array(ca.positions, ligand.positions)
            min_dists = dists.min(axis=1)  # shape (n_res,)
            per_res_contact[frame_idx] = min_dists < contact_cutoff

            any_contact[frame_idx] = per_res_contact[frame_idx].any()

        # ── Contact survival function ────────────────────────────
        max_lag: int = min(n_frames // 2, 500)
        survival: List[Dict[str, Any]] = _contact_survival(any_contact, max_lag)

        # ── Residence times ──────────────────────────────────────
        continuous_times, intermittent_times = _compute_residence_times(
            any_contact, dt
        )

        # ── Binding events ───────────────────────────────────────
        events: List[Dict[str, Any]] = []
        in_contact: bool = bool(any_contact[0])
        event_start: int = 0 if in_contact else 0

        for i in range(1, n_frames):
            if any_contact[i] and not in_contact:
                event_start = i
                events.append(
                    {
                        "type": "bind",
                        "frame": i,
                        "time_ps": round(float(times[i]), 2),
                    }
                )
            elif not any_contact[i] and in_contact:
                duration = (i - event_start) * dt
                events.append(
                    {
                        "type": "unbind",
                        "frame": i,
                        "time_ps": round(float(times[i]), 2),
                        "contact_duration_ps": round(duration, 2),
                    }
                )
            in_contact = bool(any_contact[i])

        # ── Rate estimates ───────────────────────────────────────
        n_unbind: int = sum(1 for e in events if e["type"] == "unbind")
        n_bind: int = sum(1 for e in events if e["type"] == "bind")
        total_contact_time: float = float(np.sum(any_contact)) * dt
        total_unbound_time: float = float(np.sum(~any_contact)) * dt

        koff: float = (
            n_unbind / max(total_contact_time, dt) if total_contact_time > 0 else 0.0
        )
        kon: float = (
            n_bind / max(total_unbound_time, dt) if total_unbound_time > 0 else 0.0
        )

        # ── Per-residue contact time ─────────────────────────────
        contact_frames_per_res: np.ndarray = per_res_contact.sum(axis=0)
        occupancy_per_res: np.ndarray = contact_frames_per_res / max(n_frames, 1)

        per_res_time: List[Dict[str, Any]] = []
        for ri in range(n_res):
            occ = float(occupancy_per_res[ri])
            if occ > 0.01:
                per_res_time.append(
                    {
                        "resid": int(resids[ri]),
                        "total_contact_ps": round(
                            float(contact_frames_per_res[ri]) * dt, 2
                        ),
                        "occupancy": round(occ, 4),
                    }
                )

        per_res_time.sort(key=lambda x: -x["occupancy"])

        logger.info(
            "Binding kinetics complete: %d bind events, %d unbind events, "
            "contact fraction=%.4f, koff=%.2e, kon=%.2e",
            n_bind,
            n_unbind,
            float(np.mean(any_contact)),
            koff,
            kon,
        )

        return {
            "residence_time_continuous_ps": (
                round(float(np.mean(continuous_times)), 2)
                if continuous_times
                else 0.0
            ),
            "residence_time_intermittent_ps": (
                round(float(np.mean(intermittent_times)), 2)
                if intermittent_times
                else 0.0
            ),
            "contact_survival": survival,
            "binding_events": events[:200],
            "per_residue_contact_time": per_res_time[:50],
            "com_distance": [round(float(d), 2) for d in com_distances],
            "time": times.tolist(),
            "koff_estimate_per_ps": round(float(koff), 8),
            "kon_estimate_per_ps": round(float(kon), 8),
            "n_bind_events": n_bind,
            "n_unbind_events": n_unbind,
            "total_contact_fraction": round(float(np.mean(any_contact)), 4),
        }

    except Exception as e:
        logger.error("Binding kinetics analysis failed: %s", e, exc_info=True)
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────


def _contact_survival(
    contact_array: np.ndarray,
    max_lag: int,
) -> List[Dict[str, Any]]:
    """Compute the contact survival function S(t).

    Parameters
    ----------
    contact_array : np.ndarray
        Boolean array of length *n_frames* indicating contact per frame.
    max_lag : int
        Maximum lag (in frames) to compute.

    Returns
    -------
    list[dict[str, Any]]
        Each entry contains ``lag_frames`` and ``survival`` probability.
    """
    n: int = len(contact_array)
    c0: int = int(np.sum(contact_array))
    if c0 == 0:
        return []

    step = max(1, max_lag // 50)
    survival: List[Dict[str, Any]] = []
    for lag in range(0, max_lag, step):
        if lag >= n:
            break
        overlap = int(np.sum(contact_array[: n - lag] & contact_array[lag:]))
        s = overlap / c0
        survival.append(
            {
                "lag_frames": lag,
                "survival": round(float(s), 4),
            }
        )
    return survival


def _compute_residence_times(
    contact_array: np.ndarray,
    dt: float,
) -> Tuple[List[float], List[float]]:
    """Compute continuous and intermittent residence times.

    Parameters
    ----------
    contact_array : np.ndarray
        Boolean array of length *n_frames*.
    dt : float
        Time step per frame (ps).

    Returns
    -------
    tuple[list[float], list[float]]
        ``(continuous_times, intermittent_times)`` in ps.
    """
    n: int = len(contact_array)

    # --- Continuous residence times ---
    continuous: List[float] = []
    current_run: int = 0
    for i in range(n):
        if contact_array[i]:
            current_run += 1
        else:
            if current_run > 0:
                continuous.append(current_run * dt)
            current_run = 0
    if current_run > 0:
        continuous.append(current_run * dt)

    # --- Intermittent residence times (allow gaps of up to 5 frames) ---
    intermittent: List[float] = []
    gap_tolerance: int = 5
    in_contact: bool = False
    contact_start: int = 0
    gap_count: int = 0

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
