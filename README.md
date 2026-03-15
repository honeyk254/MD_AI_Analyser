# MD AI Analyzer

**AI-Assisted Molecular Dynamics Trajectory Analysis Platform**

An advanced local platform that analyzes GROMACS MD simulation outputs using classical MD metrics, machine learning, graph neural networks, transformers, variational autoencoders, and biological inference -- then proposes biology-oriented hypotheses with explicit methodological caveats.

---

## Features

### Classical MD Analysis (21 modules)
- RMSD, RMSF, Radius of Gyration
- Secondary Structure Evolution (DSSP)
- Hydrogen Bonds, Salt Bridges (carboxylate-carbon selection to avoid double-counting), Water Bridges
- Contact Maps & Distance Matrices
- Principal Component Analysis (PCA)
- Dynamic Cross-Correlation Matrix (DCCM)
- Conformational Clustering (KMeans / HDBSCAN / GMM)
- Free Energy Landscape (Boltzmann inversion)
- Solvent Accessible Surface Area (SASA)
- Time-lagged Independent Component Analysis (tICA)
- Normal Mode Analysis (ANM-based, distance-normalised Hessian, uniform spring constant, vectorised)
- Perturbation Response Scanning (effector/sensor identification)
- Interaction Score Decomposition (per-residue Cα-based proximity scores using LJ/Coulomb forms; coarse-grained heuristic ranking only, not physical energies)
- Configurational Entropy (Schlitter's method upper bound with explicit kBT)
- Convergence Assessment (block-average SEM ratio, autocorrelation, cosine content)
- Binding Contact Persistence (residence time, contact survival, conservative kon/koff reporting only when event counts are sufficient)
- Trajectory Comparison (two-trajectory or half-split equilibration check)

### Machine Learning (10 modules)
- Conformational State Discovery (HDBSCAN / GMM / KMeans)
- Markov State Models (tICA/PCA-reduced states, reversible transition matrix, MFPT, implied timescales, Chapman-Kolmogorov test with direct T(k·τ) estimation, conservative `is_markovian` assessment)
- Allosteric Pathway Detection (graph centrality, community detection)
- Dynamic Domain Detection (spectral clustering on DCCM)
- Ligand Interaction Analysis
- Dimensionality Reduction (PCA / UMAP / t-SNE)
- Interaction Fingerprints (hydrophobic, salt bridge, aromatic contacts)
- Cavity/Void Detection (grid-based probe + Delaunay tessellation; detects static per-frame volumes, not connected tunnels — use CAVER/MOLE for tunnel tracing)
- Dynamic Network Analysis (time-windowed DCCM, community evolution)
- VAE Latent Space Analysis (variational autoencoder with KL annealing for conformational landscapes)

### Deep Learning
- **Graph Neural Networks** (GAT + GCN hybrid via PyTorch Geometric)
  - Self-supervised RMSF prediction from contact graph structure
  - Residue importance ranked by graph-topological distinctiveness (not validated functional importance)
  - Attention-based interaction detection
  - Community detection from learned embeddings
- **Transformer** (self-supervised masked reconstruction)
  - Temporal change-point detection (nonlinear change-point detector in hidden-state space)
  - Temporal importance scoring
  - Per-residue dynamic attribution via Integrated Gradients
- **Variational Autoencoder** (configurable latent dimension, KL annealing warmup)
  - Conformational landscape mapping
  - Reconstruction quality assessment
  - Latent density estimation

### AI Biological Inference Engine
Automatically generates 38 types of biological interpretations via a detector dispatch with detector-specific heuristic confidence scores. All insights include explicit caveats about limitations and cross-validation requirements, including:

**Structural:**
- Hinge residue detection
- Flexible loop identification
- Stable core classification
- Local stiffness mapping

**Dynamic:**
- Conformational transition analysis
- Breathing motions & cracking events
- Domain motion interpretation
- NMA collective motion characterization
- Entropy estimation
- Dynamic network evolution

**Allosteric & Communication:**
- Allosteric communication pathways
- Force propagation pathways
- PRS effector/sensor identification
- Communication hub detection
- H-bond network rewiring

**Binding & Functional:**
- Binding pocket dynamics
- Cryptic binding site detection
- Druggability scoring
- Ligand binding kinetics interpretation
- PPI interface hotspots
- Tunnel/cavity characterization

**Stability & Prediction:**
- Overall stability assessment
- Convergence assessment
- Mutation sensitivity heuristic ranking
- Stability risk heuristic
- Aggregation-prone region detection
- PTM accessibility screen

**Deep Learning Insights:**
- GNN key residue interpretation
- Transformer transition interpretation
- VAE conformational landscape interpretation

### Interactive Web Interface
- File upload with drag & drop
- Real-time SSE progress streaming
- Interactive Plotly charts (31 plot types)
- 3D molecular viewer (3Dmol.js) with 5 representations and 4 coloring modes
- 13 residue highlight modes:
  - *Structural:* Flexible, Hinge, Stable Core, Stiff Residues
  - *Functional:* Communication Hubs, GNN Top Residues, Mutation-Sensitive, PTM Sites
  - *Binding:* Cryptic Binding Sites, PPI Hotspots, Druggable Pockets
  - *Other:* Aggregation-Prone, Electrostatic Funnels
- Advanced parameter configuration (frame range, distance cutoffs, simulation parameters, ML settings)
- HTML/CSV/JSON/PDF report downloads

---

## Quick Start

### 1. Install Dependencies

```bash
cd md_ai_analyzer
pip install -r requirements.txt
```

> **GPU Support**: If you have an NVIDIA GPU, install PyTorch with CUDA:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> pip install torch-geometric
> ```

### 2. Run the Server

```bash
python run.py
```

### 3. Open in Browser

Navigate to: **http://localhost:8000**

### 4. Upload & Analyze

1. Upload your GROMACS files (trajectory + structure)
2. Configure analysis options (GNN, Transformer, MSM, advanced parameters)
3. Click "Upload & Analyze"
4. Watch real-time progress
5. Explore interactive results and biological insights
6. Download reports (HTML/CSV/JSON/PDF)

---

## Supported Input Files

| Type | Extensions | Required |
|------|-----------|----------|
| Trajectory | `.xtc`, `.trr` | Recommended |
| Topology | `.tpr` | Optional |
| Structure | `.pdb`, `.gro` | **Required** |
| Reference | `.pdb`, `.gro` | Optional |

---

## Configurable Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Start/End Frame | -- | Analyze a subtrajectory |
| Stride | 1 | Analyze every Nth frame |
| H-bond Cutoff | 3.5 A | Hydrogen bond distance threshold |
| Contact Cutoff | 8.0 A | Contact map distance threshold |
| Salt Bridge Cutoff | 4.0 A | Salt bridge distance threshold |
| FEL Bins | 50 | Free energy landscape grid resolution |
| Temperature | 300 K | For entropy and free energy calculations |
| MSM Lag Time | 5 | Markov state model lag time (frames) |
| Grid Spacing | 2.0 A | Tunnel/cavity detection grid resolution |
| DCCM Threshold | 0.5 | Correlation threshold for network edges |
| VAE Latent Dim | 2 | Variational autoencoder latent dimensions |
| Ligand Selection | -- | MDAnalysis selection string for ligand |
| Discard Equilibration | Off | Auto-detect and discard pre-equilibrium frames via RMSD |

---

## Architecture

```
md_ai_analyzer/
├── run.py                          # Entry point (uvicorn launcher)
├── requirements.txt                # Dependencies
├── backend/
│   ├── main.py                     # FastAPI app, routes, middleware
│   ├── config.py                   # Configuration, paths, device detection
│   ├── models.py                   # Pydantic schemas (request/response/result)
│   ├── orchestrator.py             # Analysis pipeline coordinator
│   │
│   ├── utils/                      # Shared utility modules
│   │   ├── trajectory_utils.py     # CA selection, coordinate collection, DCCM
│   │   ├── ml_feature_utils.py     # PCA, clustering, scaling, seed management
│   │   └── plotting_utils.py       # Dark theme, color constants, safe_plot
│   │
│   ├── analysis/                   # Classical MD analysis (21 modules)
│   │   ├── rmsd.py                 # Root mean square deviation
│   │   ├── rmsf.py                 # Root mean square fluctuation
│   │   ├── radius_of_gyration.py   # Radius of gyration
│   │   ├── secondary_structure.py  # DSSP secondary structure
│   │   ├── hbonds.py               # Hydrogen bond analysis
│   │   ├── salt_bridges.py         # Salt bridge detection
│   │   ├── water_bridges.py        # Water-mediated bridges
│   │   ├── contacts.py             # Contact map computation
│   │   ├── pca.py                  # Principal component analysis
│   │   ├── dccm.py                 # Dynamic cross-correlation
│   │   ├── clustering.py           # Conformational clustering
│   │   ├── free_energy.py          # Free energy landscape
│   │   ├── sasa.py                 # Solvent accessible surface area
│   │   ├── tica.py                 # Time-lagged ICA
│   │   ├── nma.py                  # Normal mode analysis (vectorised ANM)
│   │   ├── prs.py                  # Perturbation response scanning
│   │   ├── energy_decomposition.py # Per-residue interaction score decomposition (heuristic)
│   │   ├── entropy.py              # Configurational entropy (Schlitter)
│   │   ├── convergence.py          # Simulation convergence assessment
│   │   ├── binding_kinetics.py     # Ligand binding kinetics
│   │   └── trajectory_comparison.py# Two-trajectory / half-split comparison
│   │
│   ├── ml/                         # Machine learning (10 modules)
│   │   ├── state_discovery.py      # HDBSCAN/GMM/KMeans clustering
│   │   ├── msm.py                  # Markov State Models
│   │   ├── allosteric.py           # Allosteric network analysis
│   │   ├── domain_detection.py     # Spectral domain detection
│   │   ├── ligand_analysis.py      # Ligand contact analysis
│   │   ├── dimensionality.py       # UMAP/t-SNE reduction
│   │   ├── interaction_fingerprints.py # Interaction fingerprints
│   │   ├── tunnel_detection.py     # Cavity/tunnel detection
│   │   ├── dynamic_network.py      # Time-windowed network analysis
│   │   └── vae_latent.py           # Variational autoencoder
│   │
│   ├── gnn_models/                 # Graph Neural Networks
│   │   └── residue_gnn.py          # GAT+GCN hybrid
│   │
│   ├── transformer_models/         # Transformer architectures
│   │   └── trajectory_transformer.py
│   │
│   ├── bio_inference/              # Biological interpretation
│   │   └── engine.py               # 38-detector dispatch engine
│   │
│   └── visualization/              # Plotting & reports
│       ├── plots.py                # 31 Plotly chart generators
│       └── report_generator.py     # HTML, CSV & PDF reports
│
├── frontend/
│   ├── index.html                  # Main SPA
│   ├── style.css                   # Dark theme design system
│   └── app.js                      # Application logic
│
├── uploads/                        # Uploaded files (gitignored)
└── results/                        # Analysis outputs (gitignored)
```

### Shared Utilities (`backend/utils/`)

The `utils` package eliminates code duplication across 40+ analysis and ML modules:

**`trajectory_utils.py`** -- Central trajectory data access:
- `select_ca_atoms()` -- C-alpha atom selection with validation
- `collect_ca_positions()` -- Materialise full trajectory positions `(n_frames, n_res, 3)` with optional Kabsch superposition
- `collect_ca_coords_flat()` -- Flattened coordinates `(n_frames, n_res*3)` for ML input (with optional alignment)
- `compute_dccm_from_positions()` -- Vectorised DCCM via `einsum`, computed once and shared
- `residue_contributions_from_eigenvector()` -- Per-residue contributions from PCA/tICA eigenvectors

**`ml_feature_utils.py`** -- Shared ML preprocessing:
- `pca_reduce()` -- PCA with automatic component clamping
- `find_optimal_k()` -- Silhouette-based optimal cluster count
- `standardise_features()` -- Zero-mean unit-variance scaling
- `set_global_seed()` -- Reproducibility across numpy + torch

**`plotting_utils.py`** -- Unified visualisation theme:
- `apply_dark_theme()` -- Consistent Plotly dark layout
- `safe_plot()` -- Decorator that catches errors per-plot without aborting the pipeline
- Named colour constants (`ACCENT_CYAN`, `ACCENT_RED`, ..., `COMMUNITY_COLORS`)

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serve frontend |
| `GET` | `/api/health` | System info, GPU status, dependency check |
| `POST` | `/api/upload` | Upload trajectory/topology/structure files |
| `POST` | `/api/analyze` | Start analysis pipeline |
| `GET` | `/api/progress/{job_id}` | SSE real-time progress stream |
| `GET` | `/api/results/{job_id}` | Fetch analysis results (JSON) |
| `GET` | `/api/report/{job_id}` | Download HTML report |
| `GET` | `/api/csv/{job_id}` | Download CSV metrics |
| `GET` | `/api/pdf/{job_id}` | Download PDF report |
| `GET` | `/api/structure/{job_id}` | Fetch structure file for 3D viewer |

**Security:** Rate limiting (60 req/min per IP), CORS, security headers (X-Content-Type-Options, X-Frame-Options, Referrer-Policy, Permissions-Policy), request ID tracing, input validation, filename sanitization.

---

## Key Design Decisions

### Trajectory Alignment
All fluctuation-based analyses (RMSF, DCCM, PCA, etc.) operate on Kabsch-aligned Calpha coordinates, removing rigid-body rotation and translation before computing per-residue displacements.

### Concurrency Safety
The orchestrator uses a local `params` dict per `run_analysis()` call rather than shared instance state, preventing race conditions when multiple jobs run concurrently.

### Trajectory I/O
Requested frame windows are materialised in memory before analysis so all downstream modules operate on the same selected subtrajectory. Shared utility functions (`collect_ca_positions`, `collect_ca_coords_flat`) then reuse those coordinates, avoiding redundant `O(n_frames)` iterations per analysis.

### Vectorised Computation
Performance-critical code uses numpy vectorisation:
- **NMA Hessian**: Built via broadcasting with distance-normalised spring constant (`k/|r|²`) per the standard ANM formulation (Atilgan et al. 2001)
- **DCCM**: Computed via `np.einsum("fid,fjd->ij", delta, delta)`
- **Interaction fingerprints**: `np.triu_indices` + boolean masking
- **Salt bridges / H-bonds**: `np.argwhere` and `np.bincount`

### Salt Bridge Counting
Negatively charged groups are selected using the carboxylate *carbon* (ASP:CG, GLU:CD) rather than both oxygens. This avoids double-counting when both oxygens of the same carboxylate are within the cutoff.

### MDTraj Bond Inference
Both SASA (Shrake-Rupley) and DSSP (secondary structure) call `topology.create_standard_bonds()` on the reconstructed MDTraj topology. This ensures correct atom radii are assigned for SASA and backbone hydrogen-bond geometry is correctly identified for DSSP.

### VAE KL Annealing
The VAE training schedule linearly ramps the KL weight β from 0 → 1 over the first 40% of epochs (Bowman et al. 2016 warmup), preventing posterior collapse and producing more expressive latent representations.

### Chapman-Kolmogorov Validation
The MSM CK test computes both predicted T(τ)^k and directly estimated T(k·τ) from trajectory data. Full-matrix and diagonal deviations are reported, and the `is_markovian` flag is only set when CK consistency and lag-time implied-timescale stability are both satisfactory.

### ML Reproducibility
All ML and deep learning modules call `set_global_seed(42)` before model initialisation, ensuring deterministic results across runs (given the same input).

### Graceful Degradation
- PyTorch is optional: if not installed, GNN, Transformer, and VAE modules are skipped
- Each analysis module catches its own exceptions; a single module failure does not abort the pipeline
- The `@safe_plot` decorator ensures individual plot failures do not prevent other plots from rendering

---

## Extensibility

The modular architecture supports extension for:
- **AlphaFold models** -- upload predicted structures as reference
- **Docking trajectories** -- use ligand selection parameter
- **Coarse-grained simulations** -- works with any MDAnalysis-compatible format
- **Enhanced sampling** -- tICA and MSM handle replica exchange / metadynamics data
- **Custom analysis** -- add new modules to `backend/analysis/` or `backend/ml/`
- **Custom plots** -- add a generator to `plots.py` and register it in the `generators` list

---

## System Requirements

- Python 3.10+
- 8 GB RAM minimum (16+ GB recommended for large trajectories)
- NVIDIA GPU recommended (CUDA support for PyTorch)
- Modern web browser (Chrome, Firefox, Edge)
