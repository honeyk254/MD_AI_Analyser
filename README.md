# MD AI Platform

An advanced, highly-modular platform for molecular dynamics trajectory analysis. This repository is currently implementing the **Phase 1 Architecture**, which provides a robust, decoupled, strictly-typed foundation for classical MD metrics analysis, paving the way for advanced machine learning models in future phases.

## 🚀 Features (Phase 1)

* **Strict Data Contracts:** All modules communicate using rigorously defined Pydantic v2 schemas (`AnalysisBundle`, `ModuleResult`, etc.).
* **Decoupled Architecture:** Domain logic (MD analysis) is completely separated from the API layer (FastAPI) and the Reporting layer (HTML/Plotly).
* **Automated Quality Control:** Trajectories are validated for consistency and sufficient length before any analysis runs.
* **Classical MD Modules:**
  * Root Mean Square Deviation (RMSD)
  * Root Mean Square Fluctuation (RMSF)
  * Radius of Gyration (Rg)
  * Solvent Accessible Surface Area (SASA)
  * Hydrogen Bonds
  * Contact Maps (Distance Matrices)
  * Secondary Structure Evolution
  * Salt Bridges
* **Containerized Deployment:** Ready to run anywhere using Docker and Docker Compose.
* **Standalone HTML Reporting:** Generates beautiful, dark-themed, interactive Plotly charts bundled into a single portable HTML file.

## 🧠 AI Taxonomy

| Tier | What it is | Examples here | Rule |
|---|---|---|---|
| Deterministic algorithms | No learned parameters, exact math | RMSD, RMSF, Rg, SASA, H-bond geometry, DSSP, contacts, salt bridges | Ground truth. Never replace with ML. |
| Classical unsupervised statistics | Learned from data, not deep learning | PCA, k-means/hierarchical clustering, 2D free-energy histograms | Standard, interpretable, cheap. Don't call it "AI." |
| Deep learning (narrow, justified) | Neural nets, used only where classical methods are provably insufficient | TICA+MSM/VAMPnets (kinetics), autoencoder CVs | Each one must cite literature and state its failure mode. |
| LLM layer | Application-layer synthesis, outside the taxonomy | Report generation | Never produces a number. Only narrates numbers that already exist. |

## 📦 Architecture Overview

The system is organized into a modular monolith under `src/md_platform`:

```
src/md_platform/
├── schemas/         # Pydantic v2 data models (AnalysisBundle, RunCard)
├── domain/          # Core scientific logic (MDAnalysis, Parsing, Validation)
│   └── classical/   # RMSD, RMSF, Rg, SASA, etc.
├── aggregation/     # Assembles ModuleResults into the final AnalysisBundle
├── orchestration/   # Ties the pipeline together asynchronously
├── reporting/       # Generates JSON Plotly charts and standalone HTML reports
├── api/             # FastAPI endpoints (submit, status, results)
└── config.py        # Global environment settings
```

## 🛠 Installation & Usage

The easiest way to run the platform is using Docker.

### 1. Run via Docker Compose (Recommended)

Ensure you have Docker installed, then run:

```bash
docker compose up -d --build
```

The API will be available at `http://localhost:8000`.

Bundled demo trajectories are available at `GET /api/v1/demo/examples`, and you can launch one with `POST /api/v1/demo/{example_name}/submit`.

For a host deployment, use the included `docker-compose.yml` stack: the API stays private on the Docker network and Caddy exposes port `80`.

### 2. Local Installation

If you prefer to run natively, ensure you have Python 3.10+ installed.

```bash
# Clone the repository
git clone https://github.com/honeyk254/MD_AI_Analyser.git
cd MD_AI_Analyser

# Install dependencies
pip install -e .

# Run the API server
uvicorn md_platform.api.app:app --reload
```

## 🔌 API Endpoints

* `POST /api/v1/analysis/submit`: Submit a new analysis run with a trajectory and topology.
* `GET /api/v1/analysis/{run_id}/status`: Poll the current status of the pipeline.
* `GET /api/v1/analysis/{run_id}/results`: Retrieve the final JSON `AnalysisBundle` with all computed metrics.
* `GET /api/v1/demo/examples`: List bundled demo trajectories.
* `POST /api/v1/demo/{example_name}/submit`: Run a bundled demo trajectory without uploading files.

## 🧪 Development & Testing

We use `pytest` for all unit and integration testing.

```bash
pip install -e .[dev]
pytest tests/
```

## 🚢 Deployment

The shipped Docker image is self-contained. `docker compose up -d --build` starts the API and Caddy proxy, seeds bundled demo trajectories under `data/inputs`, and exposes the demo on port `80`.

## 📋 Roadmap

This project is currently executing against the `md-ai-platform-master-plan.md`. 
* **Phase 1:** Deterministic classical engine (Completed)
* **Phase 2:** Grounded LLM reporting (Target Demo State)
* **Phase 3:** Lightweight deployment
* **Phase 4:** Statistical/ML layer (TICA + MSM)

## Demo

A local demo run completed and produced a standalone HTML report.

- Run ID: f47b4f30-02fe-4174-8f82-6682f7ec3711
- Report path: data\outputs\f47b4f30-02fe-4174-8f82-6682f7ec3711\analysis_report.html

Reproduce the demo:

1. Start the API (Docker Compose recommended) or run `uvicorn md_platform.api.app:app --reload`.
2. List bundled demos: GET /api/v1/demo/examples
3. Submit the "stable" demo: POST /api/v1/demo/stable/submit
4. Poll run status: GET /api/v1/analysis/{run_id}/status
5. Retrieve results or open the HTML path above: GET /api/v1/analysis/{run_id}/results

Notes: toy demo inputs may skip some protein-specific modules (SASA, secondary structure, etc.). Use a real PDB+XTC pair for full classical analyses.
