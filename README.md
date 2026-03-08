# 🧬 MD AI Analyzer

**AI-Powered Molecular Dynamics Trajectory Analysis Platform**

An advanced local platform that analyzes GROMACS MD simulation outputs using classical MD metrics, machine learning, graph neural networks, transformers, and biological inference — then explains the biology behind the motions.

---

## Features

### Classical MD Analysis
- RMSD, RMSF, Radius of Gyration
- Secondary Structure Evolution (DSSP)
- Hydrogen Bonds, Salt Bridges
- Contact Maps & Distance Matrices
- Principal Component Analysis (PCA)
- Dynamic Cross-Correlation Matrix (DCCM)
- Conformational Clustering
- Free Energy Landscape
- Solvent Accessible Surface Area (SASA)
- Time-lagged Independent Component Analysis (tICA)

### Machine Learning
- Conformational State Discovery (HDBSCAN / GMM / KMeans)
- Markov State Models (transition matrix, MFPT, timescales)
- Allosteric Pathway Detection (graph centrality, community detection)
- Dynamic Domain Detection (spectral clustering)
- Ligand Interaction Analysis
- Dimensionality Reduction (PCA / UMAP / t-SNE)

### Deep Learning
- **Graph Neural Networks** (GAT + GCN hybrid via PyTorch Geometric)
  - Self-supervised residue importance learning
  - Attention-based interaction detection
  - Community detection from learned embeddings
- **Transformer** (self-supervised masked reconstruction)
  - Structural transition detection
  - Temporal importance scoring
  - Per-residue dynamic attribution via gradient analysis

### AI Biological Inference Engine
Automatically generates interpretations including:
- Hinge residue detection
- Flexible loop identification
- Stable core classification
- Allosteric communication pathways
- Binding pocket dynamics
- Conformational transition analysis
- Domain motion interpretation
- Overall stability assessment
- GNN and Transformer insight interpretation

### Interactive Web Interface
- File upload with drag & drop
- Real-time SSE progress streaming
- Interactive Plotly charts (15 plot types)
- 3D molecular viewer (3Dmol.js)
- Residue highlighting (flexible, hinge, hubs, GNN top)
- HTML/CSV/JSON report downloads

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
2. Configure analysis options (GNN, Transformer, MSM)
3. Click "Upload & Analyze"
4. Watch real-time progress
5. Explore interactive results and biological insights
6. Download reports (HTML/CSV/JSON)

---

## Supported Input Files

| Type | Extensions | Required |
|------|-----------|----------|
| Trajectory | `.xtc`, `.trr` | Recommended |
| Topology | `.tpr` | Optional |
| Structure | `.pdb`, `.gro` | **Required** |
| Reference | `.pdb`, `.gro` | Optional |

---

## Project Structure

```
md_ai_analyzer/
├── run.py                          # Entry point
├── requirements.txt                # Dependencies
├── backend/
│   ├── main.py                     # FastAPI application
│   ├── config.py                   # Configuration & GPU detection
│   ├── models.py                   # Pydantic schemas
│   ├── orchestrator.py             # Analysis pipeline manager
│   ├── analysis/                   # Classical MD analysis modules
│   │   ├── rmsd.py, rmsf.py, radius_of_gyration.py
│   │   ├── secondary_structure.py, hbonds.py, salt_bridges.py
│   │   ├── contacts.py, pca.py, dccm.py
│   │   ├── clustering.py, free_energy.py
│   │   ├── sasa.py, tica.py
│   ├── ml/                         # Machine learning modules
│   │   ├── state_discovery.py      # HDBSCAN/GMM clustering
│   │   ├── msm.py                  # Markov State Models
│   │   ├── allosteric.py           # Network analysis
│   │   ├── domain_detection.py     # Spectral clustering
│   │   ├── ligand_analysis.py      # Ligand contacts
│   │   ├── dimensionality.py       # UMAP/t-SNE
│   ├── gnn_models/                 # Graph Neural Networks
│   │   └── residue_gnn.py          # GAT+GCN hybrid
│   ├── transformer_models/         # Transformer architectures
│   │   └── trajectory_transformer.py
│   ├── bio_inference/              # Biological interpretation
│   │   └── engine.py
│   └── visualization/              # Plotting & reports
│       ├── plots.py                # 15 Plotly chart generators
│       └── report_generator.py     # HTML & CSV reports
├── frontend/
│   ├── index.html                  # Main SPA
│   ├── style.css                   # Dark theme design system
│   └── app.js                      # Application logic
├── uploads/                        # Uploaded files
├── results/                        # Analysis outputs
└── reports/                        # Generated reports
```

---

## Extensibility

The modular architecture supports extension for:
- **AlphaFold models** — upload predicted structures as reference
- **Docking trajectories** — use ligand selection parameter
- **Coarse-grained simulations** — works with any MDAnalysis-compatible format
- **Enhanced sampling** — tICA and MSM handle replica exchange / metadynamics data
- **Custom analysis** — add new modules to `backend/analysis/` or `backend/ml/`

---

## System Requirements

- Python 3.10+
- 8 GB RAM minimum (16+ GB recommended for large trajectories)
- NVIDIA GPU recommended (CUDA support for PyTorch)
- Modern web browser (Chrome, Firefox, Edge)
