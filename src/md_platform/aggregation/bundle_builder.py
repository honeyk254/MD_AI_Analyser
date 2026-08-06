"""Aggregation layer.

Collects module results and trajectory metadata into the final, versioned
AnalysisBundle.
"""

import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import MDAnalysis
import mdtraj
import numpy

from ..schemas.analysis_bundle import (
    AnalysisBundle,
    ModuleResult,
    QCFlag,
    QCFlags,
    TrajectoryMetadata,
)
from ..schemas.run_card import FileProvenance, RunCard, ToolVersions


def tool_versions() -> ToolVersions:
    """Versions of the libraries whose numerics the results depend on."""
    return ToolVersions(
        python=sys.version.split()[0],
        mdanalysis=MDAnalysis.__version__,
        mdtraj=mdtraj.__version__,
        numpy=numpy.__version__,
    )


def build_bundle(
    run_id: str,
    trajectory_metadata: TrajectoryMetadata,
    qc_flags: QCFlags,
    module_results: Dict[str, ModuleResult],
    inputs: Dict[str, FileProvenance],
    parameters: Dict[str, Any],
    container_digest: Optional[str] = None,
) -> AnalysisBundle:
    """Build the final AnalysisBundle from all upstream components."""
    rmsd = module_results.get("rmsd")
    if rmsd is not None and not rmsd.error:
        equil_frame = rmsd.data.get("equilibration_frame")
        qc_flags.is_equilibrated = equil_frame is not None
        qc_flags.flags.append(
            QCFlag(
                check_name="is_equilibrated",
                passed=qc_flags.is_equilibrated,
                details=(
                    f"RMSD trace settles at frame {equil_frame} "
                    f"({rmsd.data.get('equilibration_method', 'heuristic')})."
                    if qc_flags.is_equilibrated
                    else "No stable RMSD equilibration point found."
                ),
            )
        )

    for name, result in module_results.items():
        if result.error:
            qc_flags.flags.append(
                QCFlag(
                    check_name=f"module_{name}",
                    passed=False,
                    details=f"Module did not produce results: {result.error}",
                )
            )

    run_card = RunCard(
        inputs=inputs,
        tools=tool_versions(),
        container_digest=container_digest,
        parameters=parameters,
    )

    return AnalysisBundle(
        run_id=run_id,
        created_at=datetime.now(timezone.utc),
        trajectory_metadata=trajectory_metadata,
        qc_flags=qc_flags,
        modules=module_results,
        run_card=run_card,
    )
