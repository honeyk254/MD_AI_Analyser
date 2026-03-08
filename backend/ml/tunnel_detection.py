"""
Simplified Tunnel / Cavity Detection.
Identifies persistent cavities and solvent-accessible voids around the protein
using a grid-based probe approach with optional Delaunay tessellation.
"""
import numpy as np
from MDAnalysis.lib.distances import distance_array


def detect_tunnels(universe, probe_radius=1.4, grid_spacing=2.0, **kwargs):
    """
    Detect cavities using a grid-based solvent accessibility probe.

    Places a 3D grid around the protein and identifies grid points that are:
    - Within the protein's bounding box (+ padding)
    - Far enough from protein atoms to be solvent-accessible
    - Close enough to be within a cavity (not bulk solvent)

    Returns dict with:
        - cavity_volume_per_frame: estimated cavity volume per frame (ų)
        - time: timestamps
        - mean_cavity_volume: average cavity volume
        - bottleneck_residues: residues that line the cavity most frequently
        - cavity_persistence: fraction of frames each grid region is a cavity
        - alpha_spheres: Delaunay-based void centers (if scipy available)
    """
    try:
        protein = universe.select_atoms("protein")
        if len(protein) == 0:
            return {"error": "No protein atoms found"}

        ca = universe.select_atoms("protein and name CA")
        resids = ca.resids.tolist()
        heavy = universe.select_atoms("protein and not (name H*)")

        n_frames = len(universe.trajectory)
        volumes = []
        times = []
        residue_cavity_counts = np.zeros(len(ca))

        # VdW radii approximation
        vdw_radius = 1.7  # average heavy atom vdW radius

        for ts in universe.trajectory:
            times.append(float(ts.time))

            # Bounding box with padding
            pos = heavy.positions
            bbox_min = pos.min(axis=0) - 5.0
            bbox_max = pos.max(axis=0) + 5.0

            # Create grid
            x_range = np.arange(bbox_min[0], bbox_max[0], grid_spacing)
            y_range = np.arange(bbox_min[1], bbox_max[1], grid_spacing)
            z_range = np.arange(bbox_min[2], bbox_max[2], grid_spacing)

            # Limit grid size for performance
            max_points_per_dim = 40
            if len(x_range) > max_points_per_dim:
                x_range = np.linspace(bbox_min[0], bbox_max[0], max_points_per_dim)
            if len(y_range) > max_points_per_dim:
                y_range = np.linspace(bbox_min[1], bbox_max[1], max_points_per_dim)
            if len(z_range) > max_points_per_dim:
                z_range = np.linspace(bbox_min[2], bbox_max[2], max_points_per_dim)

            grid = np.array(np.meshgrid(x_range, y_range, z_range)).reshape(3, -1).T
            n_grid = len(grid)

            if n_grid == 0:
                volumes.append(0.0)
                continue

            # Compute distances from grid points to protein heavy atoms
            # Use Cα for efficiency (approximate)
            ca_pos = ca.positions
            dists = distance_array(grid, ca_pos)  # (n_grid, n_ca)

            min_dist = dists.min(axis=1)

            # Cavity points: minimum distance is between (vdw + probe) and a max cutoff
            inner_cutoff = vdw_radius + probe_radius
            outer_cutoff = inner_cutoff + 6.0  # not bulk solvent

            cavity_mask = (min_dist > inner_cutoff) & (min_dist < outer_cutoff)
            n_cavity = np.sum(cavity_mask)
            volume = float(n_cavity) * (grid_spacing ** 3)
            volumes.append(volume)

            # Identify which residues are closest to cavity points
            if n_cavity > 0:
                cavity_dists = dists[cavity_mask]  # (n_cavity, n_ca)
                closest_res = np.argmin(cavity_dists, axis=1)
                for idx in closest_res:
                    residue_cavity_counts[idx] += 1

        # Normalize bottleneck scores
        if n_frames > 0:
            residue_cavity_freq = residue_cavity_counts / n_frames
        else:
            residue_cavity_freq = residue_cavity_counts

        # Top bottleneck residues
        top_idx = np.argsort(-residue_cavity_freq)[:20]
        bottleneck_residues = []
        for idx in top_idx:
            if residue_cavity_freq[idx] > 0.1:
                bottleneck_residues.append({
                    "resid": int(resids[idx]),
                    "cavity_frequency": round(float(residue_cavity_freq[idx]), 3),
                })

        return {
            "cavity_volume_per_frame": [round(v, 1) for v in volumes],
            "time": times,
            "mean_cavity_volume": round(float(np.mean(volumes)), 1) if volumes else 0,
            "std_cavity_volume": round(float(np.std(volumes)), 1) if volumes else 0,
            "bottleneck_residues": bottleneck_residues,
            "bottleneck_scores": [round(float(x), 3) for x in residue_cavity_freq],
            "resids": resids,
            "alpha_spheres": _delaunay_voids(ca, universe, probe_radius, vdw_radius),
        }

    except Exception as e:
        return {"error": str(e)}


def _delaunay_voids(ca, universe, probe_radius, vdw_radius):
    """
    Delaunay tessellation-based void detection (item 50).
    Uses Delaunay triangulation of Cα atoms to find tetrahedral voids
    large enough to be cavities.
    """
    try:
        from scipy.spatial import Delaunay

        # Use average structure
        positions = []
        for ts in universe.trajectory:
            positions.append(ca.positions.copy())
        mean_pos = np.mean(positions, axis=0)

        tri = Delaunay(mean_pos)
        simplices = tri.simplices

        voids = []
        for simplex in simplices:
            vertices = mean_pos[simplex]
            center = vertices.mean(axis=0)
            # Circumradius approximation: max distance from center to vertices
            radii = np.linalg.norm(vertices - center, axis=1)
            circum_r = radii.max()

            # Void if circumradius > vdw + probe (i.e., large enough for solvent)
            if circum_r > (vdw_radius + probe_radius) and circum_r < 15.0:
                resids_in_void = [int(ca.resids[i]) for i in simplex]
                voids.append({
                    "center": [round(float(c), 2) for c in center],
                    "radius": round(float(circum_r), 2),
                    "resids": resids_in_void,
                })

        # Sort by radius (largest first) and limit
        voids.sort(key=lambda x: -x["radius"])
        return voids[:50]

    except Exception:
        return []
