"""Phase 1 numerical regression suite (plan: 100% pass rate required to merge).

Reference system: adenylate kinase DIMS closed->open transition trajectory
(MDAnalysisTests PSF/DCD), the same dataset used by the MDAnalysis User Guide
quickstart (Beckstein et al. 2009). Literature anchors are quoted from that
page; drift bands are +/-10% around values recorded on 2026-08-22.
Pipeline runtime baseline (rmsd+rg+sasa on this trajectory): ~2.6 s.
"""

import time

import MDAnalysis as mda
import pytest
from MDAnalysisTests.datafiles import DCD, PSF

from md_platform.domain.classical.com import compute_com
from md_platform.domain.classical.contacts import compute_contact_map
from md_platform.domain.classical.dihedrals import compute_dihedrals
from md_platform.domain.classical.hbonds import compute_hbonds
from md_platform.domain.classical.radius_of_gyration import compute_rg
from md_platform.domain.classical.rmsd import compute_rmsd
from md_platform.domain.classical.rmsf import compute_rmsf
from md_platform.domain.classical.salt_bridges import compute_salt_bridges
from md_platform.domain.classical.sasa import compute_sasa
from md_platform.domain.classical.secondary_structure import compute_secondary_structure
from md_platform.domain.parsing import parse_metadata
from md_platform.domain.validation import validate_trajectory

TOL = 0.10  # plan: +/-10% or the published range if narrower

MODULES = [
    ("rmsd", compute_rmsd),
    ("rmsf", compute_rmsf),
    ("radius_of_gyration", compute_rg),
    ("sasa", compute_sasa),
    ("hbonds", compute_hbonds),
    ("contacts", compute_contact_map),
    ("secondary_structure", compute_secondary_structure),
    ("salt_bridges", compute_salt_bridges),
    ("dihedrals", compute_dihedrals),
    ("com", compute_com),
]


@pytest.fixture(scope="module")
def reference_run():
    universe = mda.Universe(PSF, DCD)
    started = time.perf_counter()
    metadata = parse_metadata(universe, PSF, DCD)
    modules = {name: func(universe) for name, func in MODULES}
    qc = validate_trajectory(universe, metadata)
    return {
        "metadata": metadata,
        "modules": modules,
        "qc": qc,
        "runtime_seconds": time.perf_counter() - started,
        "universe": universe,
    }


def test_reference_metadata_matches_published_system(reference_run):
    meta = reference_run["metadata"]
    assert meta.n_atoms == 3341
    assert meta.n_residues == 214
    assert meta.n_frames_analyzed == 98
    assert meta.timestep_ps == pytest.approx(1.0, abs=0.01)
    assert reference_run["qc"].sufficient_frames


def test_rmsd_within_literature_range(reference_run):
    rmsd = reference_run["modules"]["rmsd"].scalar_metrics["backbone_rmsd"]
    # User Guide quickstart: backbone RMSD of last vs first frame = 6.85 A.
    assert rmsd.max == pytest.approx(6.85, rel=TOL)
    # Drift bands around recorded baseline.
    assert rmsd.mean == pytest.approx(4.394, rel=TOL)
    assert rmsd.std == pytest.approx(2.034, rel=TOL)


def test_rg_within_literature_range(reference_run):
    rg = reference_run["modules"]["radius_of_gyration"].scalar_metrics["radius_of_gyration"]
    # User Guide quickstart: Rg at frame 0 = 16.669 A, rising as the protein opens.
    assert rg.time_series[0] == pytest.approx(16.669, rel=0.01)
    assert rg.mean == pytest.approx(18.265, rel=TOL)
    assert rg.time_series[-1] > rg.time_series[0]


def test_sasa_within_recorded_band(reference_run):
    sasa = reference_run["modules"]["sasa"].scalar_metrics["total_sasa"]
    # No published anchor for this dataset; band is the recorded baseline
    # for a 214-residue globular protein (sanity: 90-130 nm^2).
    assert sasa.mean == pytest.approx(108.044, rel=TOL)
    assert 90.0 < sasa.mean < 130.0


def test_full_classical_pipeline_runs_clean(reference_run):
    """All 8 modules complete without error on the reference trajectory."""
    for name, _ in MODULES:
        result = reference_run["modules"][name]
        assert result.error is None, f"{name} failed"
        assert result.runtime_seconds >= 0


def test_remaining_modules_sane(reference_run):
    modules = reference_run["modules"]
    rmsf = modules["rmsf"].residue_metrics["rmsf"]
    assert len(rmsf.values) == 214
    assert 0.3 < rmsf.values[0] < 5.0  # terminal/structured CA, per-residue scale

    hbonds = modules["hbonds"].scalar_metrics["hbond_count"]
    assert hbonds.mean > 30  # 214-residue protein in vacuum-ish DIMS keeps many hbonds

    contacts = modules["contacts"].data
    assert len(contacts["contact_map"]) == 214

    ss = modules["secondary_structure"].scalar_metrics
    total = ss["helix_fraction"].mean + ss["sheet_fraction"].mean + ss["coil_fraction"].mean
    assert total == pytest.approx(1.0, abs=0.05)

    salt = modules["salt_bridges"].scalar_metrics["salt_bridge_count"]
    # Carboxylate-oxygen charge centres (Barlow & Thornton 1983), recorded
    # 2026-08-25. The carbon-centred selection this replaced found only
    # ~30.2/frame on this trajectory — a silent ~24% underestimate.
    assert salt.mean > 0
    assert salt.mean == pytest.approx(37.408, rel=TOL)

    dihedrals = modules["dihedrals"].residue_metrics
    assert len(dihedrals["phi_circular_std"].values) == dihedrals["phi_circular_std"].resids[-1]
    flex = modules["dihedrals"].scalar_metrics["mean_backbone_circular_std"]
    assert 0.0 < flex.mean < 180.0  # circular std of a torsion is bounded by (0, 180]

    com = modules["com"].scalar_metrics["com_drift"]
    assert com.min == pytest.approx(0.0)  # drift is measured from frame 0
    assert com.max >= com.mean >= 0


def test_pipeline_runtime_recorded(reference_run):
    # Plan: tracked, not gated. Baseline ~13 s (98 frames, 8 modules, laptop CPU).
    assert 0 < reference_run["runtime_seconds"] < 300
