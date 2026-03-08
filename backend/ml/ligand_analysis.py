from __future__ import annotations

"""Ligand interaction analysis.

Computes per-residue protein--ligand contact frequencies over a trajectory
and identifies key binding residues.  The per-atom inner loop has been
replaced with vectorised numpy indexing on the distance matrix.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from MDAnalysis.lib.distances import distance_array

logger = logging.getLogger("md_ai_analyzer")

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_ligand_interactions(
    universe: Any,
    ligand_sel: Optional[str] = None,
    cutoff: float = 4.5,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Analyse protein--ligand interactions over the trajectory.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        Loaded trajectory universe.
    ligand_sel : str, optional
        MDAnalysis selection string for the ligand.  If *None*, the
        function attempts common auto-detection strategies.
    cutoff : float
        Distance cutoff (angstroms) for a contact.
    **kwargs
        Additional keyword arguments (unused; accepted for orchestrator
        compatibility).

    Returns
    -------
    dict
        Keys:
        - ``contact_residues`` : residues contacting ligand with frequency
        - ``key_binding_residues`` : residues with > 50 % contact frequency
        - ``moderate_binding_residues`` : residues with 20--50 % frequency
        - ``contact_timeline`` : per-frame contact count
        - ``mean_contacts_per_frame`` : average contact count
        - ``binding_stability`` : qualitative stability label
        - ``ligand_selection`` : final ligand selection string used
        - ``n_ligand_atoms`` : number of ligand atoms
    """
    try:
        # ── Auto-detect ligand if needed ─────────────────────────
        if not ligand_sel:
            candidates = [
                "resname LIG",
                "resname UNK",
                "resname DRG",
                "not protein and not resname HOH and not resname WAT "
                "and not resname NA and not resname CL",
            ]
            for sel_str in candidates:
                try:
                    lig = universe.select_atoms(sel_str)
                    if 0 < len(lig) < 500:
                        ligand_sel = sel_str
                        logger.info("Auto-detected ligand with selection '%s' (%d atoms).", sel_str, len(lig))
                        break
                except Exception:
                    continue

        if not ligand_sel:
            logger.error("No ligand found and no ligand_selection provided.")
            return {"error": "No ligand found. Provide ligand_selection parameter."}

        ligand = universe.select_atoms(ligand_sel)
        if len(ligand) == 0:
            logger.error("Ligand selection '%s' matched no atoms.", ligand_sel)
            return {"error": f"Ligand selection '{ligand_sel}' matched no atoms"}

        protein = universe.select_atoms("protein")
        if len(protein) == 0:
            logger.error("No protein atoms found.")
            return {"error": "No protein atoms found"}

        # Pre-extract per-atom residue info (avoid per-atom attribute
        # lookups inside the frame loop).
        prot_resids: np.ndarray = protein.resids  # (n_protein,)
        prot_resnames: np.ndarray = protein.resnames  # (n_protein,)

        residue_contacts: Dict[tuple, int] = {}
        timeline: List[int] = []
        n_frames: int = 0

        for ts in universe.trajectory:
            dists = distance_array(
                protein.positions, ligand.positions, box=ts.dimensions
            )
            min_dists = dists.min(axis=1)  # (n_protein,)

            # Vectorised contact detection
            contact_mask = min_dists <= cutoff
            contact_resids = prot_resids[contact_mask]
            contact_resnames = prot_resnames[contact_mask]

            # Unique (resid, resname) pairs for this frame
            seen_this_frame: set = set()
            for rid, rname in zip(contact_resids, contact_resnames):
                key = (int(rid), str(rname))
                if key not in seen_this_frame:
                    seen_this_frame.add(key)

            # Accumulate counts
            for key in seen_this_frame:
                residue_contacts[key] = residue_contacts.get(key, 0) + 1

            timeline.append(int(np.sum(contact_mask)))
            n_frames += 1

        # ── Process results ──────────────────────────────────────
        contact_residues: List[Dict[str, Any]] = []
        for (resid, resname), count in sorted(
            residue_contacts.items(), key=lambda x: -x[1]
        ):
            freq = count / max(n_frames, 1)
            contact_residues.append(
                {
                    "resid": resid,
                    "resname": resname,
                    "count": count,
                    "frequency": round(freq, 3),
                }
            )

        key_binding = [r for r in contact_residues if r["frequency"] > 0.5]
        moderate_binding = [
            r for r in contact_residues if 0.2 < r["frequency"] <= 0.5
        ]

        # Binding stability (coefficient of variation)
        if timeline:
            mean_contacts = float(np.mean(timeline))
            cv = float(np.std(timeline) / mean_contacts) if mean_contacts > 0 else 0.0
            stability = "stable" if cv < 0.3 else "moderate" if cv < 0.6 else "unstable"
        else:
            mean_contacts = 0.0
            stability = "unknown"

        logger.info(
            "Ligand analysis complete: %d contacting residues, stability=%s.",
            len(contact_residues),
            stability,
        )

        return {
            "contact_residues": contact_residues[:50],
            "key_binding_residues": key_binding,
            "moderate_binding_residues": moderate_binding,
            "contact_timeline": timeline,
            "mean_contacts_per_frame": mean_contacts if timeline else 0,
            "binding_stability": stability,
            "ligand_selection": ligand_sel,
            "n_ligand_atoms": len(ligand),
        }

    except Exception as e:
        logger.exception("Ligand interaction analysis failed: %s", e)
        return {"error": str(e)}
