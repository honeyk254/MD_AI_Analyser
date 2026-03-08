"""
Secondary structure evolution using MDTraj DSSP.
Tracks helix, sheet, and coil content over time.
"""
import numpy as np


def compute_secondary_structure(universe, **kwargs):
    """
    Compute secondary structure per residue over time using MDTraj.

    Returns dict with:
        - resids: list of residue IDs
        - ss_timeline: 2D list [frame][residue] of SS codes (H/E/C)
        - helix_fraction: per-frame helix content
        - sheet_fraction: per-frame sheet content
        - coil_fraction: per-frame coil content
        - per_residue_dominant_ss: dominant SS per residue
    """
    try:
        import mdtraj as md

        # Convert MDAnalysis Universe to mdtraj trajectory
        protein = universe.select_atoms("protein")
        if len(protein) == 0:
            return {"error": "No protein atoms found"}

        positions = []
        for ts in universe.trajectory:
            positions.append(protein.positions.copy() / 10.0)  # Å to nm

        topology = md.Topology()
        chain = topology.add_chain()
        prev_resid = None
        residue_map = {}
        for atom in protein:
            if atom.resid != prev_resid:
                res = topology.add_residue(atom.resname, chain)
                prev_resid = atom.resid
                residue_map[atom.resid] = res
            topology.add_atom(atom.name, md.element.Element.getBySymbol(atom.element) if hasattr(atom, 'element') else md.element.carbon, residue_map[atom.resid])

        traj = md.Trajectory(np.array(positions), topology)
        dssp_result = md.compute_dssp(traj, simplified=True)

        resids = sorted(set(protein.resids.tolist()))

        helix_frac = []
        sheet_frac = []
        coil_frac = []

        for frame_ss in dssp_result:
            n = len(frame_ss)
            helix_frac.append(float(np.sum(frame_ss == 'H')) / n if n > 0 else 0)
            sheet_frac.append(float(np.sum(frame_ss == 'E')) / n if n > 0 else 0)
            coil_frac.append(float(np.sum(frame_ss == 'C')) / n if n > 0 else 0)

        # Dominant SS per residue
        n_residues = dssp_result.shape[1] if len(dssp_result.shape) > 1 else 0
        dominant_ss = []
        for r in range(n_residues):
            col = dssp_result[:, r]
            counts = {'H': np.sum(col == 'H'), 'E': np.sum(col == 'E'), 'C': np.sum(col == 'C')}
            dominant_ss.append(max(counts, key=counts.get))

        return {
            "resids": resids[:n_residues],
            "helix_fraction": helix_frac,
            "sheet_fraction": sheet_frac,
            "coil_fraction": coil_frac,
            "per_residue_dominant_ss": dominant_ss,
            "mean_helix": float(np.mean(helix_frac)) if helix_frac else 0,
            "mean_sheet": float(np.mean(sheet_frac)) if sheet_frac else 0,
            "mean_coil": float(np.mean(coil_frac)) if coil_frac else 0,
        }

    except Exception as e:
        return {"error": str(e), "helix_fraction": [], "sheet_fraction": [], "coil_fraction": []}
