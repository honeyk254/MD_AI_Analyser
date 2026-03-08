"""
Shared trajectory utilities for coordinate extraction and atom selection.

Eliminates duplicated trajectory-iteration and coordinate-collection patterns
across analysis and ML modules.

Functions
---------
select_ca_atoms
    Robust Cα atom selection with fallback.
collect_ca_positions
    Collect Cα positions across all frames as a single ndarray.
    Performs least-squares alignment to the first frame by default to
    remove rigid-body rotation/translation (Kabsch, 1976).
collect_ca_coords_flat
    Collect flattened Cα coordinate vectors (n_frames, n_atoms * 3).
    Performs alignment by default.
collect_frames_metadata
    Collect timestamps and frame count.
compute_mean_positions
    Compute the average structure from trajectory positions.
compute_fluctuations
    Compute per-residue displacement fluctuations from the mean.
compute_dccm_from_positions
    Compute and normalise the Dynamic Cross-Correlation Matrix.
"""
import logging
from typing import Optional, Tuple

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import align as mda_align

logger = logging.getLogger("md_ai_analyzer")


def _align_positions_kabsch(
    positions: np.ndarray,
    ref_index: int = 0,
) -> np.ndarray:
    """Align trajectory positions to a reference frame using Kabsch superposition.

    Removes rigid-body translation and rotation so that downstream analyses
    (RMSF, PCA, DCCM, tICA, entropy, etc.) capture only internal motions.

    Parameters
    ----------
    positions : np.ndarray
        Shape ``(n_frames, n_atoms, 3)``.
    ref_index : int
        Frame index to use as the alignment reference (default 0).

    Returns
    -------
    np.ndarray
        Aligned positions with the same shape.
    """
    ref = positions[ref_index].copy()
    ref_center = ref.mean(axis=0)
    ref_centered = ref - ref_center

    aligned = np.empty_like(positions)
    for i in range(len(positions)):
        mobile = positions[i]
        mobile_center = mobile.mean(axis=0)
        mobile_centered = mobile - mobile_center

        # Kabsch algorithm: find optimal rotation via SVD
        H = mobile_centered.T @ ref_centered
        U, _S, Vt = np.linalg.svd(H)

        # Correct for reflection
        d = np.linalg.det(Vt.T @ U.T)
        sign_matrix = np.eye(3)
        sign_matrix[2, 2] = np.sign(d)

        R = Vt.T @ sign_matrix @ U.T
        aligned[i] = (mobile_centered @ R.T) + ref_center

    return aligned


def select_ca_atoms(
    universe: mda.Universe,
    selection: str = "protein and name CA",
    fallback: str = "all",
) -> mda.AtomGroup:
    """Select Cα atoms with a configurable fallback.

    Parameters
    ----------
    universe : mda.Universe
        MDAnalysis Universe object.
    selection : str
        Primary atom selection string.
    fallback : str
        Fallback selection if primary yields no atoms.

    Returns
    -------
    mda.AtomGroup
        Selected atoms (never empty if *fallback* matches anything).

    Raises
    ------
    ValueError
        If both selections yield zero atoms.
    """
    atoms = universe.select_atoms(selection)
    if len(atoms) == 0:
        logger.warning(
            "Selection '%s' matched 0 atoms; falling back to '%s'",
            selection, fallback,
        )
        atoms = universe.select_atoms(fallback)
    if len(atoms) == 0:
        raise ValueError(
            f"Neither '{selection}' nor '{fallback}' matched any atoms"
        )
    return atoms


def collect_ca_positions(
    universe: mda.Universe,
    atoms: Optional[mda.AtomGroup] = None,
    align: bool = True,
) -> np.ndarray:
    """Collect per-frame (n_atoms, 3) positions into a single array.

    By default, performs least-squares alignment to the first frame
    (Kabsch superposition) to remove rigid-body rotation and translation.
    This is essential for analyses like RMSF, PCA, DCCM, tICA, and
    entropy that assume internal-motion-only coordinates.

    Parameters
    ----------
    universe : mda.Universe
    atoms : mda.AtomGroup, optional
        If *None*, ``select_ca_atoms(universe)`` is used.
    align : bool
        If True (default), align all frames to the first frame via
        Kabsch superposition.

    Returns
    -------
    np.ndarray
        Shape ``(n_frames, n_atoms, 3)``.
    """
    if atoms is None:
        atoms = select_ca_atoms(universe)
    positions = np.empty(
        (len(universe.trajectory), len(atoms), 3), dtype=np.float64
    )
    for i, _ts in enumerate(universe.trajectory):
        positions[i] = atoms.positions
    if align:
        positions = _align_positions_kabsch(positions, ref_index=0)
    return positions


def collect_ca_coords_flat(
    universe: mda.Universe,
    atoms: Optional[mda.AtomGroup] = None,
    align: bool = True,
) -> np.ndarray:
    """Collect flattened Cα coordinate vectors.

    By default, performs least-squares alignment to the first frame
    before flattening.

    Parameters
    ----------
    universe : mda.Universe
    atoms : mda.AtomGroup, optional
    align : bool
        If True (default), align all frames to the first frame via
        Kabsch superposition.

    Returns
    -------
    np.ndarray
        Shape ``(n_frames, n_atoms * 3)``.
    """
    if atoms is None:
        atoms = select_ca_atoms(universe)
    n_atoms = len(atoms)
    n_frames = len(universe.trajectory)
    positions_3d = np.empty((n_frames, n_atoms, 3), dtype=np.float64)
    for i, _ts in enumerate(universe.trajectory):
        positions_3d[i] = atoms.positions
    if align:
        positions_3d = _align_positions_kabsch(positions_3d, ref_index=0)
    coords = positions_3d.reshape(n_frames, n_atoms * 3)
    return coords


def collect_frames_metadata(
    universe: mda.Universe,
) -> Tuple[list, int]:
    """Return (timestamps_list, n_frames) for the trajectory.

    Parameters
    ----------
    universe : mda.Universe

    Returns
    -------
    tuple[list[float], int]
    """
    times = [float(ts.time) for ts in universe.trajectory]
    return times, len(times)


def compute_mean_positions(positions: np.ndarray) -> np.ndarray:
    """Compute the mean structure from a ``(n_frames, n_atoms, 3)`` array.

    Parameters
    ----------
    positions : np.ndarray

    Returns
    -------
    np.ndarray
        Shape ``(n_atoms, 3)``.
    """
    return positions.mean(axis=0)


def compute_fluctuations(positions: np.ndarray) -> np.ndarray:
    """Per-atom RMS fluctuation magnitude from mean.

    Parameters
    ----------
    positions : np.ndarray
        Shape ``(n_frames, n_atoms, 3)``.

    Returns
    -------
    np.ndarray
        Shape ``(n_atoms,)`` — RMS displacement per atom.
    """
    mean_pos = positions.mean(axis=0)
    delta = positions - mean_pos
    return np.sqrt(np.mean(np.sum(delta ** 2, axis=2), axis=0))


def compute_dccm_from_positions(positions: np.ndarray) -> np.ndarray:
    """Compute the normalised Dynamic Cross-Correlation Matrix.

    Parameters
    ----------
    positions : np.ndarray
        Shape ``(n_frames, n_atoms, 3)``.

    Returns
    -------
    np.ndarray
        Shape ``(n_atoms, n_atoms)`` with values in ``[-1, 1]``.
    """
    mean_pos = positions.mean(axis=0)
    delta = positions - mean_pos
    n_frames = delta.shape[0]

    # Vectorised cross-correlation
    dccm = np.einsum("fid,fjd->ij", delta, delta) / n_frames

    # Normalise
    diag = np.sqrt(np.diag(dccm))
    diag = np.where(diag == 0, 1e-10, diag)
    dccm_normalised = dccm / np.outer(diag, diag)
    return dccm_normalised


def residue_contributions_from_eigenvector(
    eigenvector: np.ndarray, n_atoms: int
) -> np.ndarray:
    """Per-residue contribution magnitudes from a 3N eigenvector.

    Parameters
    ----------
    eigenvector : np.ndarray
        Shape ``(n_atoms * 3,)``.
    n_atoms : int

    Returns
    -------
    np.ndarray
        Shape ``(n_atoms,)``.
    """
    reshaped = eigenvector.reshape(n_atoms, 3)
    return np.sqrt(np.sum(reshaped ** 2, axis=1))
