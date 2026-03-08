from __future__ import annotations

"""Simplified Tunnel / Cavity Detection.

Identifies persistent cavities and solvent-accessible voids around the
protein using a grid-based probe approach with optional Delaunay
tessellation.
"""

import logging
from typing import Any, Dict, List

import numpy as np
from MDAnalysis.lib.distances import distance_array

logger = logging.getLogger("md_ai_analyzer")

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_tunnels(
    universe: Any,
    probe_radius: float = 1.4,
    grid_spacing: float = 2.0,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Detect cavities using a grid-based solvent-accessibility probe.

    Places a 3-D grid around the protein and identifies grid points that
    are within the protein bounding box (plus padding), far enough from
    protein atoms to be solvent-accessible, yet close enough not to be
    bulk solvent.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        Loaded trajectory universe.
    probe_radius : float
        Solvent probe radius in angstroms.
    grid_spacing : float
        Grid spacing in angstroms for the cavity search.
    **kwargs
        Additional keyword arguments (unused; accepted for orchestrator
        compatibility).

    Returns
    -------
    dict
        Keys:
        - ``cavity_volume_per_frame`` : estimated cavity volume per frame (A^3)
        - ``time`` : timestamps
        - ``mean_cavity_volume`` : average cavity volume
        - ``std_cavity_volume`` : standard deviation of cavity volume
        - ``bottleneck_residues`` : residues that line the cavity most frequently
        - ``bottleneck_scores`` : per-residue normalised cavity frequency
        - ``resids`` : residue IDs
        - ``alpha_spheres`` : Delaunay-based void centres
    """
    try:
        protein = universe.select_atoms("protein")
        if len(protein) == 0:
            logger.error("No protein atoms found for tunnel detection.")
            return {"error": "No protein atoms found"}

        ca = universe.select_atoms("protein and name CA")
        resids: List[int] = ca.resids.tolist()
        heavy = universe.select_atoms("protein and not (name H*)")

        n_frames = len(universe.trajectory)
        volumes: List[float] = []
        times: List[float] = []
        residue_cavity_counts = np.zeros(len(ca), dtype=np.float64)

        # VdW radii approximation
        vdw_radius: float = 1.7  # average heavy-atom vdW radius

        # Pre-compute mean CA positions for Delaunay (avoid second
        # trajectory iteration inside _delaunay_voids).
        ca_positions_all = np.empty(
            (n_frames, len(ca), 3), dtype=np.float64
        )

        max_points_per_dim: int = 40
        inner_cutoff = vdw_radius + probe_radius
        outer_cutoff = inner_cutoff + 6.0

        for frame_idx, ts in enumerate(universe.trajectory):
            times.append(float(ts.time))
            ca_positions_all[frame_idx] = ca.positions

            # Bounding box with padding
            pos = heavy.positions
            bbox_min = pos.min(axis=0) - 5.0
            bbox_max = pos.max(axis=0) + 5.0

            # Create grid (clamped per dimension)
            ranges: List[np.ndarray] = []
            for dim in range(3):
                r = np.arange(bbox_min[dim], bbox_max[dim], grid_spacing)
                if len(r) > max_points_per_dim:
                    r = np.linspace(bbox_min[dim], bbox_max[dim], max_points_per_dim)
                ranges.append(r)

            grid = np.array(
                np.meshgrid(ranges[0], ranges[1], ranges[2])
            ).reshape(3, -1).T
            n_grid = len(grid)

            if n_grid == 0:
                volumes.append(0.0)
                continue

            # Distances from grid points to CA atoms (approximate)
            ca_pos = ca.positions
            dists = distance_array(grid, ca_pos)  # (n_grid, n_ca)
            min_dist = dists.min(axis=1)

            # Cavity points: between inner and outer cutoff
            cavity_mask = (min_dist > inner_cutoff) & (min_dist < outer_cutoff)
            n_cavity = int(np.sum(cavity_mask))
            volume = float(n_cavity) * (grid_spacing ** 3)
            volumes.append(volume)

            # Which residues are closest to cavity points (vectorised)
            if n_cavity > 0:
                cavity_dists = dists[cavity_mask]  # (n_cavity, n_ca)
                closest_res = np.argmin(cavity_dists, axis=1)
                # Vectorised counting via bincount
                counts = np.bincount(closest_res, minlength=len(ca))
                residue_cavity_counts += counts.astype(np.float64)

        # Normalise bottleneck scores
        if n_frames > 0:
            residue_cavity_freq = residue_cavity_counts / n_frames
        else:
            residue_cavity_freq = residue_cavity_counts

        # Top bottleneck residues
        top_idx = np.argsort(-residue_cavity_freq)[:20]
        bottleneck_residues: List[Dict[str, Any]] = []
        for idx in top_idx:
            if residue_cavity_freq[idx] > 0.1:
                bottleneck_residues.append(
                    {
                        "resid": int(resids[idx]),
                        "cavity_frequency": round(float(residue_cavity_freq[idx]), 3),
                    }
                )

        # Pre-computed mean positions for Delaunay (no second iteration)
        mean_ca_pos = ca_positions_all.mean(axis=0)

        logger.info(
            "Tunnel detection complete: mean volume %.1f A^3, %d bottleneck residues.",
            float(np.mean(volumes)) if volumes else 0.0,
            len(bottleneck_residues),
        )

        return {
            "cavity_volume_per_frame": [round(v, 1) for v in volumes],
            "time": times,
            "mean_cavity_volume": round(float(np.mean(volumes)), 1) if volumes else 0,
            "std_cavity_volume": round(float(np.std(volumes)), 1) if volumes else 0,
            "bottleneck_residues": bottleneck_residues,
            "bottleneck_scores": [round(float(x), 3) for x in residue_cavity_freq],
            "resids": resids,
            "alpha_spheres": _delaunay_voids(
                ca, mean_ca_pos, probe_radius, vdw_radius
            ),
        }

    except Exception as e:
        logger.exception("Tunnel detection failed: %s", e)
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _delaunay_voids(
    ca: Any,
    mean_pos: np.ndarray,
    probe_radius: float,
    vdw_radius: float,
) -> List[Dict[str, Any]]:
    """Delaunay tessellation-based void detection.

    Uses Delaunay triangulation of the *pre-computed* mean C-alpha positions
    to find tetrahedral voids large enough to represent cavities, avoiding a
    redundant second pass over the trajectory.

    Parameters
    ----------
    ca : MDAnalysis.AtomGroup
        C-alpha atom group (used to retrieve residue IDs).
    mean_pos : np.ndarray
        Pre-computed mean C-alpha positions of shape ``(n_atoms, 3)``.
    probe_radius : float
        Solvent probe radius in angstroms.
    vdw_radius : float
        Average heavy-atom van der Waals radius.

    Returns
    -------
    list[dict]
        Up to 50 largest voids, each containing ``center``, ``radius``,
        and ``resids`` keys.
    """
    try:
        from scipy.spatial import Delaunay

        tri = Delaunay(mean_pos)
        simplices = tri.simplices  # (n_simplices, 4)

        # Vectorised: compute centres and circumradii for all simplices
        vertices = mean_pos[simplices]  # (n_simplices, 4, 3)
        centres = vertices.mean(axis=1)  # (n_simplices, 3)

        # Circumradius approximation: max distance from centre to any vertex
        diff = vertices - centres[:, None, :]  # (n_simplices, 4, 3)
        radii = np.sqrt(np.sum(diff ** 2, axis=2))  # (n_simplices, 4)
        circum_r = radii.max(axis=1)  # (n_simplices,)

        # Filter: void if circumradius within (vdw + probe, 15.0)
        min_r = vdw_radius + probe_radius
        void_mask = (circum_r > min_r) & (circum_r < 15.0)

        void_indices = np.where(void_mask)[0]
        void_radii = circum_r[void_indices]

        # Sort by radius (largest first), take top 50
        sort_order = np.argsort(-void_radii)[:50]
        selected = void_indices[sort_order]

        resids_arr = ca.resids
        voids: List[Dict[str, Any]] = []
        for idx in selected:
            center = centres[idx]
            simplex = simplices[idx]
            resids_in_void = [int(resids_arr[i]) for i in simplex]
            voids.append(
                {
                    "center": [round(float(c), 2) for c in center],
                    "radius": round(float(circum_r[idx]), 2),
                    "resids": resids_in_void,
                }
            )

        logger.info("Delaunay void detection found %d voids.", len(voids))
        return voids

    except ImportError:
        logger.warning("scipy not available; skipping Delaunay void detection.")
        return []
    except Exception as exc:
        logger.warning("Delaunay void detection failed: %s", exc)
        return []
