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
collect_ca_coords_flat
    Collect flattened Cα coordinate vectors (n_frames, n_atoms * 3).
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

logger = logging.getLogger("md_ai_analyzer")


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
) -> np.ndarray:
    """Collect per-frame (n_atoms, 3) positions into a single array.

    Parameters
    ----------
    universe : mda.Universe
    atoms : mda.AtomGroup, optional
        If *None*, ``select_ca_atoms(universe)`` is used.

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
    return positions


def collect_ca_coords_flat(
    universe: mda.Universe,
    atoms: Optional[mda.AtomGroup] = None,
) -> np.ndarray:
    """Collect flattened Cα coordinate vectors.

    Parameters
    ----------
    universe : mda.Universe
    atoms : mda.AtomGroup, optional

    Returns
    -------
    np.ndarray
        Shape ``(n_frames, n_atoms * 3)``.
    """
    if atoms is None:
        atoms = select_ca_atoms(universe)
    n_atoms = len(atoms)
    n_frames = len(universe.trajectory)
    coords = np.empty((n_frames, n_atoms * 3), dtype=np.float64)
    for i, _ts in enumerate(universe.trajectory):
        coords[i] = atoms.positions.ravel()
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
