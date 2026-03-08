"""
Ligand interaction analysis.
Computes residue-ligand contact frequencies and identifies key binding residues.
"""
import numpy as np
from MDAnalysis.lib.distances import distance_array


def analyze_ligand_interactions(universe, ligand_sel=None, cutoff=4.5, **kwargs):
    """
    Analyze protein-ligand interactions over the trajectory.

    Returns dict with:
        - contact_residues: residues contacting ligand with frequency
        - key_binding_residues: residues with >50% contact frequency
        - contact_timeline: per-frame contact counts
        - binding_stability: overall binding stability metric
    """
    try:
        if not ligand_sel:
            # Try to auto-detect ligand
            for sel_str in ["resname LIG", "resname UNK", "resname DRG",
                           "not protein and not resname HOH and not resname WAT and not resname NA and not resname CL"]:
                try:
                    lig = universe.select_atoms(sel_str)
                    if len(lig) > 0 and len(lig) < 500:
                        ligand_sel = sel_str
                        break
                except Exception:
                    continue

        if not ligand_sel:
            return {"error": "No ligand found. Provide ligand_selection parameter."}

        ligand = universe.select_atoms(ligand_sel)
        if len(ligand) == 0:
            return {"error": f"Ligand selection '{ligand_sel}' matched no atoms"}

        protein = universe.select_atoms("protein")
        if len(protein) == 0:
            return {"error": "No protein atoms found"}

        # Get per-residue closest atoms
        residue_contacts = {}
        timeline = []
        n_frames = 0

        for ts in universe.trajectory:
            dists = distance_array(protein.positions, ligand.positions, box=ts.dimensions)
            min_dists = dists.min(axis=1)

            frame_contacts = 0
            for i, atom in enumerate(protein):
                if min_dists[i] <= cutoff:
                    resid = int(atom.resid)
                    resname = atom.resname
                    key = (resid, resname)
                    residue_contacts[key] = residue_contacts.get(key, 0) + 1
                    frame_contacts += 1

            timeline.append(frame_contacts)
            n_frames += 1

        # Process results
        contact_residues = []
        for (resid, resname), count in sorted(residue_contacts.items(), key=lambda x: -x[1]):
            freq = count / n_frames
            contact_residues.append({
                "resid": resid,
                "resname": resname,
                "count": count,
                "frequency": round(freq, 3),
            })

        key_binding = [r for r in contact_residues if r["frequency"] > 0.5]
        moderate_binding = [r for r in contact_residues if 0.2 < r["frequency"] <= 0.5]

        # Binding stability: coefficient of variation of contact count
        if timeline:
            cv = float(np.std(timeline) / np.mean(timeline)) if np.mean(timeline) > 0 else 0
            stability = "stable" if cv < 0.3 else "moderate" if cv < 0.6 else "unstable"
        else:
            stability = "unknown"

        return {
            "contact_residues": contact_residues[:50],
            "key_binding_residues": key_binding,
            "moderate_binding_residues": moderate_binding,
            "contact_timeline": timeline,
            "mean_contacts_per_frame": float(np.mean(timeline)) if timeline else 0,
            "binding_stability": stability,
            "ligand_selection": ligand_sel,
            "n_ligand_atoms": len(ligand),
        }

    except Exception as e:
        return {"error": str(e)}
