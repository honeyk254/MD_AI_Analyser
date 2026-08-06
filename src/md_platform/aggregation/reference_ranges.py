"""Reference ranges for classical MD metrics.

Every range here is either a literature-derived expectation or an explicitly
labelled heuristic, and each one carries its source and its caveats. They are
sanity bands for a globular, folded, solvated protein at ~300 K — being outside
a band is a prompt to look closer, never a verdict about the simulation.

Ranges live in code (not in a prompt) so that comparisons are deterministic,
reviewable and diffable, and so the LLM layer can only *read* the verdict.
"""

from dataclasses import dataclass
from typing import Dict, Literal, Optional

from ..schemas.summary import ReferenceComparison

Scaling = Literal["absolute", "per_residue"]


@dataclass(frozen=True)
class ReferenceRange:
    """An expected band for one metric."""

    low: float
    high: float
    unit: str
    source: str
    note: str
    scaling: Scaling = "absolute"

    def bounds(self, n_residues: int) -> "tuple[float, float]":
        if self.scaling == "per_residue":
            return self.low * n_residues, self.high * n_residues
        return self.low, self.high


HEURISTIC = "Documented heuristic (see aggregation/reference_ranges.py), not a literature range"

REFERENCE_RANGES: Dict[str, ReferenceRange] = {
    "backbone_rmsd": ReferenceRange(
        low=0.5,
        high=3.0,
        unit="Angstrom",
        source=(
            "Lindorff-Larsen et al., PLoS ONE 7:e32131 (2012), systematic force-field "
            "validation: equilibrated globular proteins in explicit solvent typically "
            "stay within a few Angstrom of the experimental starting structure"
        ),
        note=(
            "Backbone RMSD from the first analysed frame. Values above the band are "
            "common and legitimate for flexible or multi-domain proteins, and for "
            "simulations that start far from equilibrium."
        ),
    ),
    "mean_rmsf": ReferenceRange(
        low=0.4,
        high=2.5,
        unit="Angstrom",
        source=(
            "Comparable in magnitude to crystallographic B-factor-derived "
            "fluctuations for folded proteins; band itself is a heuristic"
        ),
        note=(
            "Mean C-alpha RMSF averaged over all residues. Termini and long loops "
            "routinely exceed this band without indicating a problem."
        ),
    ),
    "hbond_count": ReferenceRange(
        low=0.4,
        high=1.2,
        unit="count",
        scaling="per_residue",
        source=HEURISTIC,
        note=(
            "Intra-protein hydrogen bonds per residue, scaled by residue count. "
            "Strongly dependent on the geometric criteria used (here: donor-acceptor "
            "distance and D-H-A angle cutoffs recorded in the run card) and on "
            "secondary-structure content."
        ),
    ),
    "salt_bridge_count": ReferenceRange(
        low=0.01,
        high=0.12,
        unit="count",
        scaling="per_residue",
        source=HEURISTIC,
        note=(
            "Salt bridges per residue, scaled by residue count. Highly sequence- and "
            "pH-dependent; halophilic and highly charged proteins sit far above it."
        ),
    ),
    "helix_fraction": ReferenceRange(
        low=0.0,
        high=1.0,
        unit="fraction",
        source=HEURISTIC,
        note=(
            "No universal expectation: helix content is a property of the fold, not "
            "of simulation quality. Only the stability of the fraction over time is "
            "interpretable here."
        ),
    ),
    "sheet_fraction": ReferenceRange(
        low=0.0,
        high=1.0,
        unit="fraction",
        source=HEURISTIC,
        note=(
            "No universal expectation, as for helix content; judge drift over time "
            "rather than the absolute value."
        ),
    ),
}

# Metrics whose absolute value scales with system size in a way that no simple
# band captures. They are reported with an explicit "no reference" verdict
# rather than compared against an invented range.
NO_REFERENCE_NOTES: Dict[str, str] = {
    "radius_of_gyration": (
        "Radius of gyration scales with chain length, so there is no size-independent "
        "expected range. The informative signal is drift over the trajectory."
    ),
    "total_sasa": (
        "Total SASA scales with molecular mass (cf. Miller et al., J. Mol. Biol. "
        "196:641 (1987)); this pipeline does not compute the mass-scaled expectation, "
        "so no range is asserted."
    ),
    "coil_fraction": (
        "Coil content is the complement of assigned helix and sheet content and has "
        "no universal expected range."
    ),
}


def compare_to_reference(
    metric: str, value: float, unit: str, n_residues: int
) -> ReferenceComparison:
    """Compare one metric value to its reference band, deterministically."""
    ref = REFERENCE_RANGES.get(metric)
    if ref is None:
        return ReferenceComparison(
            metric=metric,
            value=value,
            unit=unit,
            verdict="no_reference",
            note=NO_REFERENCE_NOTES.get(
                metric, "No reference range is defined for this metric."
            ),
        )

    low, high = ref.bounds(n_residues)
    if value < low:
        verdict = "below_range"
    elif value > high:
        verdict = "above_range"
    else:
        verdict = "within_range"

    return ReferenceComparison(
        metric=metric,
        value=value,
        unit=unit,
        verdict=verdict,
        reference_low=low,
        reference_high=high,
        source=ref.source,
        note=ref.note,
    )
