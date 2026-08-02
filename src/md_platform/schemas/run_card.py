"""RunCard schema for tracking provenance and reproducibility."""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class FileProvenance(BaseModel):
    """Hashes and metadata for input files."""

    filename: str
    sha256: str
    size_bytes: int


class ToolVersions(BaseModel):
    """Versions of critical libraries used in the analysis."""

    python: str
    mdanalysis: str
    mdtraj: Optional[str] = None
    numpy: str


class RunCard(BaseModel):
    """Provenance record for a single analysis run.

    Provides the reproducibility backbone by recording exactly what software,
    inputs, and parameters produced this AnalysisBundle.
    """

    inputs: Dict[str, FileProvenance] = Field(
        description="Hashes of the input trajectory and topology files."
    )
    tools: ToolVersions = Field(description="Library versions.")
    container_digest: Optional[str] = Field(
        None, description="SHA256 digest of the Docker image used for the run, if any."
    )
    parameters: Dict[str, Any] = Field(
        description="Global parameters requested for this run (e.g. stride, temperature)."
    )
    seed: Optional[int] = Field(
        None, description="Random seed used for stochastic algorithms (if any)."
    )
