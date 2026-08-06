"""Preloaded demo dataset.

The demo path exists so the platform can be evaluated without anyone supplying
files. It uses the 38-model solution-NMR ensemble of the Trp-cage miniprotein
TC5B (PDB 1L2Y, Neidigh, Fesinmeyer & Andersen, Nat. Struct. Biol. 9:425, 2002),
which is small, has explicit hydrogens (so the geometric hydrogen-bond criteria
apply) and contains a real Asp9-Arg16 salt bridge.

It is an experimental conformational ensemble, not a molecular-dynamics
trajectory: successive frames are independent NMR models, so there is no
physical time axis and any "time" in the demo report is frame index only. The
demo response and the report say so rather than implying a simulation was run.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .config import settings

DEMO_CAVEAT = (
    "Demo dataset: the 38-model solution-NMR ensemble of Trp-cage TC5B (PDB 1L2Y) "
    "treated as a frame series. It is not a molecular-dynamics simulation, so frame "
    "intervals are nominal and kinetic interpretation does not apply."
)


@dataclass(frozen=True)
class DemoDataset:
    """One preloaded example a caller can analyse with no upload."""

    key: str
    name: str
    description: str
    filename: str
    n_frames: int

    @property
    def path(self) -> Path:
        return Path(settings.demo_dir) / self.filename

    def exists(self) -> bool:
        return self.path.is_file()


DEMO_DATASETS: Dict[str, DemoDataset] = {
    "trp_cage": DemoDataset(
        key="trp_cage",
        name="Trp-cage TC5B (PDB 1L2Y)",
        description=(
            "20-residue designed miniprotein, 38-model NMR ensemble, explicit "
            "hydrogens, contains an Asp9-Arg16 salt bridge. " + DEMO_CAVEAT
        ),
        filename="1l2y.pdb",
        n_frames=38,
    ),
}

DEFAULT_DEMO_KEY = "trp_cage"


def available_datasets() -> List[DemoDataset]:
    """Demo datasets actually present on disk."""
    return [dataset for dataset in DEMO_DATASETS.values() if dataset.exists()]


def get_dataset(key: str) -> DemoDataset:
    dataset = DEMO_DATASETS.get(key)
    if dataset is None:
        raise KeyError(f"Unknown demo dataset '{key}'. Available: {sorted(DEMO_DATASETS)}")
    if not dataset.exists():
        raise FileNotFoundError(
            f"Demo dataset '{key}' is not present at {dataset.path}. "
            "It ships with the repository under data/demo/."
        )
    return dataset
