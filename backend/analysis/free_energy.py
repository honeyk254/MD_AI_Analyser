"""Free Energy Landscape computation.

Constructs a 2-D free energy surface from PCA projections using
Boltzmann inversion: ``F = -kT * ln(P)``, where *P* is the probability
density estimated from a 2-D histogram of the first two principal
components.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import MDAnalysis as mda

from ..utils.trajectory_utils import select_ca_atoms, collect_ca_coords_flat
from ..utils.ml_feature_utils import pca_reduce

logger = logging.getLogger("md_ai_analyzer")

# Boltzmann constant in kJ/(mol*K)
_KB_KJ: float = 8.314e-3


def compute_free_energy_landscape(
    universe: mda.Universe,
    n_bins: int = 50,
    temperature: float = 300.0,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Compute a 2-D free energy landscape from PCA projections.

    The free energy surface is obtained by:

    1. Collecting flattened C-alpha coordinates for every frame.
    2. Reducing to two principal components via PCA.
    3. Building a 2-D probability-density histogram.
    4. Applying Boltzmann inversion: ``F = -kT * ln(P)``.
    5. Shifting so that the global minimum is at zero.

    Parameters
    ----------
    universe : mda.Universe
        MDAnalysis Universe with a loaded trajectory.
    n_bins : int, optional
        Number of histogram bins along each axis (default 50).
    temperature : float, optional
        Temperature in Kelvin used for kT (default 300).
    **kwargs : Any
        Additional keyword arguments (unused, accepted for API consistency).

    Returns
    -------
    dict[str, Any]
        ``fel``
            2-D free energy values (kJ/mol), list of lists.
        ``pc1_edges``
            Bin edges for PC1.
        ``pc2_edges``
            Bin edges for PC2.
        ``minima``
            Up to 20 local free-energy minima sorted by energy.
        ``temperature``
            Temperature used (K).
        ``n_bins``
            Number of bins used.
        ``kT``
            Thermal energy kT (kJ/mol).
        ``pc1_range``
            [min, max] of PC1 projection.
        ``pc2_range``
            [min, max] of PC2 projection.
    """
    try:
        ca = select_ca_atoms(universe)
        logger.info(
            "Computing free energy landscape: %d CA atoms, %d frames, "
            "T=%.1f K, bins=%d",
            len(ca),
            len(universe.trajectory),
            temperature,
            n_bins,
        )

        # --- Collect flattened coordinates via shared utility ---
        coords: np.ndarray = collect_ca_coords_flat(universe, atoms=ca)

        # Need at least 2 components
        n_comp_avail = min(2, coords.shape[0] - 1, coords.shape[1])
        if n_comp_avail < 2:
            logger.warning("Not enough data for FEL (n_comp_avail=%d)", n_comp_avail)
            return {"error": "Not enough data for FEL"}

        # --- PCA via shared utility ---
        projections, _pca_model = pca_reduce(coords, n_components=2)

        pc1: np.ndarray = projections[:, 0]
        pc2: np.ndarray = projections[:, 1]

        # --- 2-D histogram (probability density) ---
        H, xedges, yedges = np.histogram2d(pc1, pc2, bins=n_bins, density=True)

        # --- Boltzmann inversion: F = -kT * ln(P) ---
        kT: float = _KB_KJ * temperature

        # Numerical stability: replace zero-density bins with a small floor
        positive_mask = H > 0
        if positive_mask.any():
            density_floor = H[positive_mask].min() * 0.01
        else:
            density_floor = 1e-30
        H_safe = np.where(positive_mask, H, density_floor)

        F: np.ndarray = -kT * np.log(H_safe)
        F -= F.min()  # shift global minimum to zero

        # --- Locate local minima (8-connected neighbourhood) ---
        minima: List[Dict[str, Any]] = []
        f_mean_half = float(np.mean(F)) * 0.5
        nx, ny = F.shape
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                val = F[i, j]
                neighbours = [
                    F[i - 1, j],
                    F[i + 1, j],
                    F[i, j - 1],
                    F[i, j + 1],
                    F[i - 1, j - 1],
                    F[i - 1, j + 1],
                    F[i + 1, j - 1],
                    F[i + 1, j + 1],
                ]
                if all(val <= n for n in neighbours) and val < f_mean_half:
                    x_center = (xedges[i] + xedges[i + 1]) / 2.0
                    y_center = (yedges[j] + yedges[j + 1]) / 2.0
                    minima.append(
                        {
                            "pc1": round(float(x_center), 2),
                            "pc2": round(float(y_center), 2),
                            "free_energy": round(float(val), 2),
                        }
                    )

        minima.sort(key=lambda x: x["free_energy"])

        logger.info(
            "FEL complete: kT=%.4f kJ/mol, %d minima detected",
            kT,
            len(minima),
        )

        return {
            "fel": F.tolist(),
            "pc1_edges": xedges.tolist(),
            "pc2_edges": yedges.tolist(),
            "minima": minima[:20],
            "temperature": temperature,
            "n_bins": n_bins,
            "kT": round(kT, 4),
            "pc1_range": [float(pc1.min()), float(pc1.max())],
            "pc2_range": [float(pc2.min()), float(pc2.max())],
        }

    except Exception as e:
        logger.error(
            "Free energy landscape computation failed: %s", e, exc_info=True
        )
        return {"error": str(e)}
