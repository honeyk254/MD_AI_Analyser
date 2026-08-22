# MD AI Platform

[![CI](https://github.com/honeyk254/MD_AI_Analyser/actions/workflows/ci.yml/badge.svg)](https://github.com/honeyk254/MD_AI_Analyser/actions/workflows/ci.yml)

An advanced, modular platform for molecular dynamics (MD) trajectory analysis. The platform combines **ground-truth deterministic physical calculations** with **grounded, anti-hallucination LLM narrative reporting** and **production-ready containerized deployment**.

---

## 🚀 Features

### 1. Deterministic Classical Engine (Ground Truth)
* **Strict Data Contracts:** All modules communicate using rigorously defined Pydantic v2 schemas (`AnalysisBundle`, `ModuleResult`, `RunCard`, etc.).
* **Decoupled Architecture:** Scientific computation (MDAnalysis / MDTraj) is completely separated from the API layer (FastAPI) and reporting layer (HTML/Plotly).
* **Automated Quality Control:** Trajectories are validated for consistency, file integrity, frame sufficiency, and equilibration before analysis runs.
* **Classical MD Modules:**
  * **RMSD** (Root Mean Square Deviation)
  * **RMSF** (Root Mean Square Fluctuation)
  * **Rg** (Radius of Gyration)
  * **SASA** (Solvent Accessible Surface Area)
  * **Hydrogen Bonds** (Geometric Donor-Acceptor criteria)
  * **Contact Maps** (Residue-residue distance matrices)
  * **Secondary Structure Evolution** (DSSP / simplified assignments)
  * **Salt Bridges** (Ionic interaction distance monitoring)

### 2. Grounded LLM Reporting & Synthesis
* **Tool-Calling Architecture:** Orchestrates Anthropic Claude via structured tools (`get_metric_summary`, `get_qc_flags`, `compare_to_reference_ranges`) to query real computed values rather than guessing.
* **Anti-Hallucination Grounding Checker:** Regex-based validation scans narrative reports against the `AnalysisBundle` to detect and flag any unsupported numbers with `[QC WARNING]`.
* **Deterministic Offline Fallback:** Automatically generates a structured scientific report even when no API key or LLM provider is available.
* **Standalone HTML Dashboards:** Bundles dark-themed, interactive Plotly visualizations and grounded scientific narratives into a single portable HTML report file.
* **Human Review & Sign-Off:** Supports review and approval workflows (`POST /api/v1/analysis/{run_id}/review`).

### 3. Production Deployment & Demo System
* **Reverse-Proxy Architecture:** Production Docker Compose stack with Caddy proxying port `80` to private FastAPI instances.
* **Built-in Security Guards:** Request payload size limits (64 KB default, `MAX_REQUEST_BODY_BYTES`) and IP-based rate limiting respecting `X-Forwarded-For`.
* **Bundled Zero-Setup Demos:** Includes pre-packaged synthetic peptide trajectories (`stable` and `flexible`) for instant testing.

---

## 🧠 AI Taxonomy

| Tier | Category | Implemented Modules | Governance Rule |
|---|---|---|---|
| **Tier 1** | Deterministic algorithms | RMSD, RMSF, Rg, SASA, H-bond geometry, DSSP, contacts, salt bridges | **Ground truth.** Exact physics and geometry. Never replace with generative ML. |
| **Tier 2** | Classical unsupervised statistics | PCA, k-means/hierarchical clustering, 2D free-energy landscapes | Standard, interpretable statistical mechanics. |
| **Tier 3** | Deep learning (narrow, justified) | TICA + MSM / VAMPnets (kinetics), autoencoder CVs | Only used where classical methods are provably insufficient. Must state failure modes. |
| **Tier 4** | Grounded LLM layer | Scientific narrative synthesis, QC assessment | **Never invents numbers.** Strictly queries and narrates existing metrics via tool calling and grounding validation. |

---

## 📦 Architecture Overview

```
src/md_platform/
├── schemas/         # Pydantic v2 data models (AnalysisBundle, RunCard, API models)
├── domain/          # Core scientific logic (MDAnalysis, MDTraj, parsing, QC validation)
│   └── classical/   # RMSD, RMSF, Rg, SASA, contacts, hbonds, salt bridges, DSSP
├── aggregation/     # Aggregates ModuleResults into the final AnalysisBundle & LLM summaries
├── orchestration/   # Async analysis orchestrator and pipeline execution
├── llm/             # Tool-calling LLM orchestrator & anti-hallucination grounding checker
├── reporting/       # Plotly chart generators and standalone single-file HTML reports
├── api/             # FastAPI application, middleware (rate limiting/size guards), and routes
├── demo_inputs.py   # Bundled demo datasets (stable & flexible trajectories)
└── config.py        # Global environment settings and path configuration
```

---

## 🛠 Installation & Usage

### 1. Run via Docker Compose (Recommended)

```bash
# Build and run the Caddy reverse-proxy + API stack
docker compose up -d --build
```
The API is available at `http://localhost:80` (or `http://localhost:8000` for direct API access).

### 2. Local Installation

Ensure Python 3.10+ is installed:

```bash
# Clone repository
git clone https://github.com/honeyk254/MD_AI_Analyser.git
cd MD_AI_Analyser

# Install platform and dependencies
pip install -e .

# (Optional) Provide Anthropic API key for LLM report synthesis
# export ANTHROPIC_API_KEY="your-api-key"

# Run the API server
uvicorn md_platform.api.app:app --reload --port 8000
```

---

## 🔌 API Endpoints

### Analysis
* `POST /api/v1/analysis/submit`: Submit a new analysis run with uploaded topology and trajectory files.
* `GET /api/v1/analysis/{run_id}/status`: Poll the current status (`pending`, `running`, `completed`, `failed`).
* `GET /api/v1/analysis/{run_id}/results`: Retrieve the full JSON `AnalysisBundle` with computed metrics and narrative report.
* `POST /api/v1/analysis/{run_id}/review`: Approve or sign off on an analysis run after review.

### Zero-Setup Demo Trajectories
* `GET /api/v1/demo/examples`: List available bundled demo datasets (`stable`, `flexible`).
* `POST /api/v1/demo/{example_name}/submit`: Submit a bundled demo trajectory without uploading any files.

### System & Health
* `GET /health`: Health check endpoint (`{"status": "ok", "version": "2.0.0"}`).

---

## ⚡ Quick Demo

You can test the entire pipeline in under 10 seconds:

### Via Python CLI:
```bash
python run_demo.py
```
This executes the bundled demo trajectory and generates the standalone HTML report at `data/outputs/<run_id>/analysis_report.html`.

### Via REST API:
```bash
# 1. Submit the stable peptide demo
curl -X POST http://localhost:8000/api/v1/demo/stable/submit

# 2. Poll status (replace with returned run_id)
curl http://localhost:8000/api/v1/analysis/{run_id}/status

# 3. Retrieve final AnalysisBundle
curl http://localhost:8000/api/v1/analysis/{run_id}/results
```

---

## 🧪 Testing & Validation

The CI pipeline (GitHub Actions) runs lint (ruff), type-check (mypy), the full test suite, and a domain-layer coverage gate (≥80%).

```bash
pip install -e .[dev]
pytest tests/                                   # full suite
pytest tests/test_regression.py -v              # numerical regression vs. reference trajectory
pytest tests/ --cov=src/md_platform/domain --cov-fail-under=80
ruff check src tests run_demo.py
mypy src
```

**Numerical regression suite** (`tests/test_regression.py`): runs all 8 classical modules against the adenylate-kinase DIMS reference trajectory (the MDAnalysis User Guide dataset, Beckstein et al. 2009) and asserts RMSD/Rg/SASA against published values (e.g. frame-0 Rg = 16.669 Å, endpoint backbone RMSD = 6.85 Å) plus ±10% drift bands. Pipeline runtime baseline on this trajectory: ~13 s (8 modules, 98 frames).

---

## 📋 Roadmap

Live demo report: **https://honeyk254.github.io/MD_AI_Analyser/** (reference-system report with all 8 analyses, generated end-to-end by the pipeline through the grounding check and human review gate).

This project executes against the [Master Plan](md-ai-platform-master-plan.md):
* ✅ **Phase 0:** Data contracts, provenance run-card, CI (lint + type-check + tests + coverage gate), reference trajectory with published ranges
* ✅ **Phase 1:** Classical MD engine — 8 modules, numerical regression suite vs. literature ranges, domain coverage ≥80%
* ✅ **Phase 2:** Grounded LLM reporting, anti-hallucination verifier (100% catch rate on injected-error fixtures), human review gate, report latency/cost/review-turnaround metrics logged
* ✅ **Phase 3:** Containerized deployment, Caddy reverse proxy, middleware guards, demo endpoints
* 🔄 **Phase 4:** Statistical & Machine Learning Layer (TICA, Markov State Models / MSM, and Free-Energy Landscapes)

