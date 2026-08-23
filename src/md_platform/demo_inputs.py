"""Bundled demo trajectories for zero-setup deployment."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

ResidueSpec = Tuple[str, List[Tuple[str, Tuple[float, float, float], str]]]


RESIDUES: List[ResidueSpec] = [
    (
        "ALA",
        [
            ("N", (-0.8, 0.0, 0.0), "N"),
            ("CA", (0.6, 0.2, 0.0), "C"),
            ("C", (1.8, -0.2, 0.1), "C"),
            ("O", (2.8, -0.5, 0.0), "O"),
            ("CB", (0.8, 1.3, 0.6), "C"),
        ],
    ),
    (
        "LYS",
        [
            ("N", (3.2, 0.1, 0.0), "N"),
            ("CA", (4.6, 0.3, 0.0), "C"),
            ("C", (5.8, -0.1, 0.1), "C"),
            ("O", (6.8, -0.4, 0.0), "O"),
            ("CB", (4.8, 1.4, 0.6), "C"),
            ("CG", (5.8, 2.2, 0.9), "C"),
            ("CD", (6.8, 2.9, 1.1), "C"),
            ("CE", (7.8, 3.6, 1.3), "C"),
            ("NZ", (8.8, 4.2, 1.5), "N"),
        ],
    ),
    (
        "ASP",
        [
            ("N", (7.2, 0.0, 0.0), "N"),
            ("CA", (8.6, 0.2, 0.0), "C"),
            ("C", (9.8, -0.2, 0.1), "C"),
            ("O", (10.8, -0.5, 0.0), "O"),
            ("CB", (8.9, 1.3, 0.6), "C"),
            ("CG", (9.9, 2.0, 0.8), "C"),
            ("OD1", (10.8, 2.6, 1.0), "O"),
            ("OD2", (9.7, 2.5, -0.2), "O"),
        ],
    ),
    (
        "GLY",
        [
            ("N", (10.8, 0.1, 0.0), "N"),
            ("CA", (12.2, 0.2, 0.0), "C"),
            ("C", (13.4, -0.2, 0.1), "C"),
            ("O", (14.4, -0.5, 0.0), "O"),
        ],
    ),
    (
        "LEU",
        [
            ("N", (13.8, 0.1, 0.0), "N"),
            ("CA", (15.2, 0.2, 0.0), "C"),
            ("C", (16.4, -0.2, 0.1), "C"),
            ("O", (17.4, -0.5, 0.0), "O"),
            ("CB", (15.5, 1.2, 0.6), "C"),
            ("CG", (16.5, 2.0, 0.9), "C"),
            ("CD1", (17.4, 2.6, 1.1), "C"),
            ("CD2", (16.7, 2.7, -0.2), "C"),
        ],
    ),
    (
        "GLU",
        [
            ("N", (17.8, 0.1, 0.0), "N"),
            ("CA", (19.2, 0.2, 0.0), "C"),
            ("C", (20.4, -0.2, 0.1), "C"),
            ("O", (21.4, -0.5, 0.0), "O"),
            ("CB", (19.5, 1.2, 0.6), "C"),
            ("CG", (20.5, 2.0, 0.9), "C"),
            ("CD", (21.5, 2.7, 1.1), "C"),
            ("OE1", (22.4, 3.2, 1.3), "O"),
            ("OE2", (21.4, 3.1, -0.1), "O"),
        ],
    ),
]


def ensure_demo_inputs(base_dir: Path) -> Dict[str, Dict[str, str]]:
    """Create bundled demo trajectories if they are missing."""
    demo_dir = base_dir / "examples"
    demo_dir.mkdir(parents=True, exist_ok=True)

    examples: Dict[str, Dict[str, Any]] = {
        "stable": {
            "label": "Stable peptide",
            "amplitude": 0.12,
        },
        "flexible": {
            "label": "Flexible peptide",
            "amplitude": 0.45,
        },
        "kinetics": {
            "label": "Synthetic two-state kinetics (demonstrates the TICA/MSM layer)",
            "amplitude": 0.05,
        },
    }

    created: Dict[str, Dict[str, str]] = {}
    for name, spec in examples.items():
        topology_path = demo_dir / f"{name}_topology.pdb"
        trajectory_path = demo_dir / f"{name}_trajectory.pdb"
        if name == "kinetics":
            # Two metastable conformers sampled from a seeded Markov chain:
            # real interconversion statistics for the kinetic layer, on data
            # that is clearly synthetic. 4000 frames: VAMPnets need enough
            # samples that the slow mode beats per-frame noise memorization.
            topology_path.write_text(_build_kinetics_pdb(n_models=1), encoding="utf-8")
            trajectory_path.write_text(_build_kinetics_pdb(n_models=4000), encoding="utf-8")
        else:
            topology_path.write_text(_build_pdb(amplitude=0.0, n_models=1), encoding="utf-8")
            trajectory_path.write_text(
                _build_pdb(amplitude=float(spec["amplitude"]), n_models=12), encoding="utf-8"
            )
        created[name] = {
            "name": name,
            "label": str(spec["label"]),
            "topology_file": str(topology_path),
            "trajectory_file": str(trajectory_path),
        }

    return created


KINETICS_SEED = 42
KINETICS_STAY_PROBABILITY = 0.92
KINETICS_SCALE_A = 0.8  # compact conformer: x contracted about the centroid
KINETICS_SCALE_B = 1.2  # extended conformer: x expanded about the centroid
KINETICS_NOISE_ANGSTROM = 0.08  # white per-frame noise: no spurious slow modes


def _build_kinetics_pdb(n_models: int) -> str:
    """Two-state trajectory: a breathing conformer change sampled from a Markov chain.

    The conformers differ by an internal coordinate (x-scaling about the
    centroid), so alignment cannot strip the slow mode, and the only
    structured, autocorrelated signal is the state flips.
    """
    rng = random.Random(KINETICS_SEED)
    centroid_x = sum(x for _resname, atoms in RESIDUES for _name, (x, _y, _z), _el in atoms)
    centroid_x /= sum(len(atoms) for _resname, atoms in RESIDUES)
    lines: List[str] = []
    state = 0
    for model_idx in range(1, n_models + 1):
        serial = 1  # per-model numbering: 43 atoms x 4000 models would overflow 5 digits
        lines.append(f"MODEL     {model_idx}")
        if rng.random() > KINETICS_STAY_PROBABILITY:
            state = 1 - state
        scale = KINETICS_SCALE_B if state else KINETICS_SCALE_A
        for resid, (resname, atoms) in enumerate(RESIDUES, start=1):
            for atom_name, (x, y, z), element in atoms:
                lines.append(
                    _atom_line(
                        serial=serial,
                        atom_name=atom_name,
                        resname=resname,
                        chain="A",
                        resid=resid,
                        x=centroid_x + (x - centroid_x) * scale + rng.gauss(0.0, KINETICS_NOISE_ANGSTROM),
                        y=y + rng.gauss(0.0, KINETICS_NOISE_ANGSTROM),
                        z=z + rng.gauss(0.0, KINETICS_NOISE_ANGSTROM),
                        element=element,
                    )
                )
                serial += 1
        lines.append("ENDMDL")
    lines.append("END")
    return "\n".join(lines) + "\n"


def _build_pdb(amplitude: float, n_models: int) -> str:
    lines: List[str] = []
    serial = 1
    for model_idx in range(1, n_models + 1):
        lines.append(f"MODEL     {model_idx}")
        phase = model_idx / 3.0
        for resid, (resname, atoms) in enumerate(RESIDUES, start=1):
            residue_shift = math.sin(phase + resid * 0.5) * amplitude
            for atom_name, (x, y, z), element in atoms:
                wobble = math.cos(phase + serial * 0.07) * amplitude
                lines.append(
                    _atom_line(
                        serial=serial,
                        atom_name=atom_name,
                        resname=resname,
                        chain="A",
                        resid=resid,
                        x=x + residue_shift,
                        y=y + wobble,
                        z=z + wobble * 0.5,
                        element=element,
                    )
                )
                serial += 1
        lines.append("ENDMDL")
    lines.append("END")
    return "\n".join(lines) + "\n"


def _atom_line(
    serial: int,
    atom_name: str,
    resname: str,
    chain: str,
    resid: int,
    x: float,
    y: float,
    z: float,
    element: str,
) -> str:
    # Strict PDB column layout (serial 7-11, name 13-16, resName 18-20,
    # chainID 22, resSeq 23-26, coords 31-54, element 77-78). A previous
    # version was off by one column, which broke protein/resname selection.
    return (
        f"ATOM  {serial:5d} {atom_name:>4}{'':1}{resname:>3} {chain}{resid:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2}"
    )
