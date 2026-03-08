"""
Pydantic models for API request/response schemas.
"""
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
from enum import Enum


class AnalysisStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class UploadResponse(BaseModel):
    job_id: str
    message: str
    files: Dict[str, str]


class AnalysisRequest(BaseModel):
    job_id: str
    stride: int = 1
    run_gnn: bool = True
    run_transformer: bool = True
    run_msm: bool = True
    ligand_selection: Optional[str] = None
    reference_pdb: Optional[str] = None
    # Subtrajectory analysis (item 49)
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    # Configurable analysis parameters (item 44)
    hbond_cutoff: float = 3.5
    contact_cutoff: float = 8.0
    salt_bridge_cutoff: float = 4.0
    fel_bins: int = 50
    temperature: float = 300.0
    msm_lag_time: int = 5
    grid_spacing: float = 2.0
    correlation_threshold: float = 0.5
    vae_latent_dim: int = 2


class ProgressUpdate(BaseModel):
    job_id: str
    status: AnalysisStatus
    current_module: str
    progress_percent: float
    message: str


class InsightItem(BaseModel):
    type: str
    residues: List[int]
    description: str
    confidence: float
    evidence: List[str]
    category: str  # structural, dynamic, allosteric, binding, transition


class AnalysisResult(BaseModel):
    job_id: str
    status: AnalysisStatus
    trajectory_info: Dict[str, Any] = {}
    rmsd: Dict[str, Any] = {}
    rmsf: Dict[str, Any] = {}
    rg: Dict[str, Any] = {}
    secondary_structure: Dict[str, Any] = {}
    hbonds: Dict[str, Any] = {}
    salt_bridges: Dict[str, Any] = {}
    contacts: Dict[str, Any] = {}
    pca: Dict[str, Any] = {}
    dccm: Dict[str, Any] = {}
    clustering: Dict[str, Any] = {}
    free_energy: Dict[str, Any] = {}
    sasa: Dict[str, Any] = {}
    tica: Dict[str, Any] = {}
    ml_states: Dict[str, Any] = {}
    msm: Dict[str, Any] = {}
    allosteric: Dict[str, Any] = {}
    domains: Dict[str, Any] = {}
    ligand: Dict[str, Any] = {}
    dimensionality: Dict[str, Any] = {}
    water_bridges: Dict[str, Any] = {}
    energy_decomposition: Dict[str, Any] = {}
    prs: Dict[str, Any] = {}
    nma: Dict[str, Any] = {}
    entropy: Dict[str, Any] = {}
    interaction_fingerprints: Dict[str, Any] = {}
    tunnels: Dict[str, Any] = {}
    vae: Dict[str, Any] = {}
    dynamic_network: Dict[str, Any] = {}
    gnn_results: Dict[str, Any] = {}
    transformer_results: Dict[str, Any] = {}
    convergence: Dict[str, Any] = {}
    binding_kinetics: Dict[str, Any] = {}
    biological_insights: List[Dict[str, Any]] = []
    plots: Dict[str, str] = {}  # plot name -> JSON string of plotly figure
