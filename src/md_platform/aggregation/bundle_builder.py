"""Aggregation layer.

Collects individual module results and trajectory metadata to build the
final, versioned AnalysisBundle.
"""

from typing import Dict, Any
from datetime import datetime, timezone
import sys
import MDAnalysis

from ..schemas.analysis_bundle import AnalysisBundle, ModuleResult, TrajectoryMetadata, QCFlags, QCFlag
from ..schemas.run_card import RunCard, ToolVersions, FileProvenance


def build_bundle(
    run_id: str,
    trajectory_metadata: TrajectoryMetadata,
    qc_flags: QCFlags,
    module_results: Dict[str, ModuleResult],
    inputs: Dict[str, Any],
    parameters: Dict[str, Any],
) -> AnalysisBundle:
    """Build the final AnalysisBundle from all upstream components."""
    
    # 1. Update QC flags based on module outputs
    # For example, check if RMSD found an equilibration point
    if "rmsd" in module_results and not module_results["rmsd"].error:
        equil_frame = module_results["rmsd"].data.get("equilibration_frame", 0)
        qc_flags.is_equilibrated = equil_frame > 0
        if not qc_flags.is_equilibrated:
            qc_flags.flags.append(
                QCFlag(check_name="is_equilibrated", passed=False, details="No stable RMSD equilibration point found.")
            )
        else:
            qc_flags.flags.append(
                QCFlag(check_name="is_equilibrated", passed=True, details=f"Equilibrated at frame {equil_frame}.")
            )

    # 2. Build the RunCard
    # In a full implementation, `inputs` would contain file hashes
    import mdtraj
    import numpy
    
    tool_versions = ToolVersions(
        python=sys.version.split()[0],
        mdanalysis=MDAnalysis.__version__,
        mdtraj=mdtraj.__version__,
        numpy=numpy.__version__,
    )
    
    import os
    norm_inputs = {}
    for k, v in inputs.items():
        if isinstance(v, dict) and "file" in v:
            path = v["file"]
            try:
                size = os.path.getsize(path)
            except Exception:
                size = 0
            norm_inputs[k] = FileProvenance(filename=path, sha256=v.get("hash", "skipped"), size_bytes=size)
        else:
            norm_inputs[k] = v

    run_card = RunCard(
        inputs=norm_inputs,
        tools=tool_versions,
        container_digest=None,
        parameters=parameters,
    )
    
    # 3. Assemble
    bundle = AnalysisBundle(
        run_id=run_id,
        created_at=datetime.now(timezone.utc),
        trajectory_metadata=trajectory_metadata,
        qc_flags=qc_flags,
        modules=module_results,
        run_card=run_card,
    )
    
    return bundle
