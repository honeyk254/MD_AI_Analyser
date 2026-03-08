"""
Solvent Accessible Surface Area (SASA) analysis.
Tracks protein surface exposure over time.
"""
import numpy as np


def compute_sasa(universe, **kwargs):
    """
    Compute SASA over the trajectory using MDTraj.

    Returns dict with:
        - time: timestamps
        - total_sasa: total SASA per frame (nm²)
        - per_residue_sasa: average SASA per residue
        - resids: residue IDs
        - buried_residues: residues with very low SASA (core)
        - exposed_residues: residues with high SASA (surface)
    """
    try:
        import mdtraj as md

        protein = universe.select_atoms("protein")
        if len(protein) == 0:
            return {"error": "No protein atoms found"}

        positions = []
        times = []
        for ts in universe.trajectory:
            positions.append(protein.positions.copy() / 10.0)  # Å to nm
            times.append(float(ts.time))

        # Build an mdtraj topology
        topology = md.Topology()
        chain = topology.add_chain()
        prev_resid = None
        res_obj = None
        for atom in protein:
            if atom.resid != prev_resid:
                res_obj = topology.add_residue(atom.resname, chain)
                prev_resid = atom.resid
            try:
                element = md.element.Element.getBySymbol(atom.element)
            except Exception:
                element = md.element.carbon
            topology.add_atom(atom.name, element, res_obj)

        traj = md.Trajectory(np.array(positions), topology)
        sasa = md.shrake_rupley(traj, mode='residue')  # (n_frames, n_residues)

        total_sasa = sasa.sum(axis=1).tolist()
        avg_per_residue = sasa.mean(axis=0).tolist()

        resids = sorted(set(protein.resids.tolist()))
        n_res = min(len(resids), len(avg_per_residue))

        mean_sasa = np.mean(avg_per_residue[:n_res])
        std_sasa = np.std(avg_per_residue[:n_res])

        buried = [int(resids[i]) for i in range(n_res) if avg_per_residue[i] < mean_sasa - std_sasa]
        exposed = [int(resids[i]) for i in range(n_res) if avg_per_residue[i] > mean_sasa + std_sasa]

        return {
            "time": times,
            "total_sasa": total_sasa,
            "per_residue_sasa": avg_per_residue[:n_res],
            "resids": resids[:n_res],
            "mean_total_sasa": float(np.mean(total_sasa)),
            "buried_residues": buried,
            "exposed_residues": exposed,
        }

    except Exception as e:
        return {"error": str(e)}
