# MD AI Platform

[![CI](https://github.com/honeyk254/MD_AI_Analyser/actions/workflows/ci.yml/badge.svg)](https://github.com/honeyk254/MD_AI_Analyser/actions/workflows/ci.yml)

An advanced, modular platform for molecular dynamics (MD) trajectory analysis. The platform combines **ground-truth deterministic physical calculations** with **grounded, anti-hallucination LLM narrative reporting**, an **opt-in statistical/ML kinetic layer (TICA + MSM + VAMPnet ablation)**, and **production-ready containerized deployment**. All six phases of the [master plan](md-ai-platform-master-plan.md) are complete.

---

## 🚀 Features

### 1. Deterministic Classical Engine (Ground Truth)
* **Strict Data Contracts:** All modules communicate using rigorously defined Pydantic v2 schemas (`AnalysisBundle`, `ModuleResult`, `RunCard`, etc.).
* **Decoupled Architecture:** Scientific computation (MDAnalysis / MDTraj) is completely separated from the API layer (FastAPI) and reporting layer (HTML/Plotly).
* **Automated Quality Control:** Trajectories are validated for consistency, file integrity, frame sufficiency, and equilibration before analysis runs.
* **Classical MD Modules (10):**
  * **RMSD** (Root Mean Square Deviation)
  * **RMSF** (Root Mean Square Fluctuation)
  * **Rg** (Radius of Gyration)
  * **SASA** (Solvent Accessible Surface Area)
  * **Hydrogen Bonds** (Geometric Donor-Acceptor criteria)
  * **Contact Maps** (Residue-residue distance matrices)
  * **Secondary Structure Evolution** (DSSP / simplified assignments)
  * **Salt Bridges** (Ionic interaction distance monitoring)
  * **Dihedrals** (Ramachandran phi/psi flexibility via circular standard deviation)
  * **COM Drift** (Center-of-mass translational stability)

### 2. Statistical / ML Kinetic Layer (opt-in, gated)
* **TICA + MSM:** hand-rolled NumPy implementation, enabled per-run via `enable_ml`.
* **Minimum-data gating:** refuses to run (with reasons) below minimum frames / transition counts — no silent unreliable kinetics.
* **Chapman-Kolmogorov validation** decides `is_markovian` before any kinetic number is reported.
* **Mandatory PCA-vs-TICA baseline:** state-agreement NMI and timescale error reported side-by-side with every MSM result.
* **VAMPnet ablation (Phase 6):** a small VAMP-2-scored network ablated against the TICA/MSM baseline (timescale agreement + state NMI as specific numbers). Requires the optional `torch` extra; degrades gracefully without it.
* **Analysis Cards:** every ML bundle carries a required, machine-readable model card (see [ANALYSIS_CARDS.md](ANALYSIS_CARDS.md)).

### 3. Grounded LLM Reporting & Synthesis
* **Tool-Calling Architecture:** Orchestrates Anthropic Claude via structured tools (`get_metric_summary`, `get_qc_flags`, `compare_to_reference_ranges`) to query real computed values rather than guessing.
* **Anti-Hallucination Grounding Checker:** Regex-based validation scans narrative reports against the `AnalysisBundle` to detect and flag any unsupported numbers with `[QC WARNING]`.
* **Deterministic Offline Fallback:** Automatically generates a structured scientific report even when no API key or LLM provider is available.
* **Standalone HTML Dashboards:** Bundles dark-themed, interactive Plotly visualizations and grounded scientific narratives into a single portable HTML report file.
* **Human Review & Sign-Off:** Supports review and approval workflows (`POST /api/v1/analysis/{run_id}/review`).

### 4. Observability: LLM Tracing + Cost/Latency Dashboard (Phase 6)
* **OpenTelemetry spans** around every LLM report generation (`llm.generate_report`) with run id, mode, latency, cost, and ungrounded-claim attributes — in-memory exporter, no collector required (OTel SDK optional at runtime).
* **Cost/latency metrics:** every report (LLM or deterministic fallback) is recorded with tokens, cost (target < $0.50/report), and grounding-check outcome.
* **Metrics endpoints:** `GET /api/v1/metrics/llm` (JSON aggregates + recent calls + spans) and `GET /api/v1/metrics/dashboard` (HTML cost/latency dashboard).

### 5. Production Deployment & Demo System
* **Reverse-Proxy Architecture:** Production Docker Compose stack with Caddy proxying port `80` to private FastAPI instances.
* **Built-in Security Guards:** Request payload size limits (64 KB default, `MAX_REQUEST_BODY_BYTES`) and IP-based rate limiting respecting `X-Forwarded-For`.
* **Bundled Zero-Setup Demos:** Includes pre-packaged synthetic trajectories (`stable`, `flexible`, and `kinetics` — a two-state system with real interconversion statistics for the ML layer) for instant testing.

---

## 🧠 AI Taxonomy

| Tier | Category | Implemented Modules | Governance Rule |
|---|---|---|---|
| **Tier 1** | Deterministic algorithms | RMSD, RMSF, Rg, SASA, H-bond geometry, DSSP, contacts, salt bridges, dihedrals, COM drift | **Ground truth.** Exact physics and geometry. Never replace with generative ML. |
| **Tier 2** | Classical unsupervised statistics | PCA, k-means/hierarchical clustering, 2D free-energy landscapes | Standard, interpretable statistical mechanics. |
| **Tier 3** | Deep learning (narrow, justified) | TICA + MSM, VAMPnet ablation (kinetics) | Only used where classical methods are provably insufficient. Must state failure modes. |
| **Tier 4** | Grounded LLM layer | Scientific narrative synthesis, QC assessment | **Never invents numbers.** Strictly queries and narrates existing metrics via tool calling and grounding validation. |

---

## 📦 Architecture Overview

```
src/md_platform/
├── schemas/         # Pydantic v2 data models (AnalysisBundle, RunCard, API models)
├── domain/          # Core scientific logic (MDAnalysis, MDTraj, parsing, QC validation)
│   └── classical/   # rmsd, rmsf, rg, sasa, contacts, hbonds, salt bridges, DSSP, dihedrals, com
├── aggregation/     # Aggregates ModuleResults into the final AnalysisBundle & LLM summaries
├── orchestrator.py  # Async analysis orchestrator and pipeline execution
├── ml/              # Opt-in kinetic layer: TICA/MSM, PCA baseline, VAMPnet ablation, Analysis Cards
├── llm/             # Tool-calling LLM orchestrator & anti-hallucination grounding checker
├── reporting/       # Plotly chart generators and standalone single-file HTML reports
├── api/             # FastAPI application, middleware (rate limiting/size guards), and routes
├── observability.py # OpenTelemetry spans + LLM cost/latency metrics store
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

# Optional extras:
pip install -e ".[vampnets]"      # torch, for the VAMPnet ablation (CI installs CPU torch)
pip install -e ".[observability]" # OpenTelemetry SDK, for real tracing spans

# (Optional) Provide Anthropic API key for LLM report synthesis
# export ANTHROPIC_API_KEY="your-api-key"

# Run the API server
uvicorn md_platform.api.app:app --reload --port 8000
```

### 3. Full AWS via Terraform (Phase 6 artifact)

`deploy/aws/main.tf` defines the complete ECS Fargate deployment (ECR, cluster, ALB, service, CloudWatch logs). Per the master plan's cost rule it is meant to be applied briefly for demo capture and destroyed afterwards:

```bash
cd deploy/aws
terraform init
docker build -t md-ai-platform ../..   # then tag + push to the ECR URL terraform creates
terraform apply                        # capture the demo
terraform destroy                      # when done — no idle costs
```

SQS/Step Functions are deliberately omitted: the service is synchronous, and the single-host Compose stack above covers the always-on demo.

---

## 🔌 API Endpoints

### Analysis
* `POST /api/v1/analysis/submit`: Submit a new analysis run with uploaded topology and trajectory files (set `enable_ml: true` for the kinetic layer).
* `GET /api/v1/analysis/{run_id}/status`: Poll the current status (`pending`, `running`, `completed`, `failed`).
* `GET /api/v1/analysis/{run_id}/results`: Retrieve the full JSON `AnalysisBundle` with computed metrics and narrative report.
* `GET /api/v1/analysis/{run_id}/ml-results`: Retrieve the Phase 4 ML bundle (TICA/MSM/baseline/VAMPnet ablation + Analysis Card).
* `POST /api/v1/analysis/{run_id}/review`: Approve or sign off on an analysis run after review.

### Zero-Setup Demo Trajectories
* `GET /api/v1/demo/examples`: List available bundled demo datasets (`stable`, `flexible`, `kinetics`).
* `POST /api/v1/demo/{example_name}/submit`: Submit a bundled demo trajectory without uploading any files.

### Observability
* `GET /api/v1/metrics/llm`: LLM cost/latency aggregates, recent calls, and recent trace spans.
* `GET /api/v1/metrics/dashboard`: HTML cost/latency dashboard (mean cost vs the <$0.50/report target).

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
pytest tests/test_regression.py -v              # numerical regression vs. reference trajectory (10 modules)
pytest tests/test_contracts.py -v               # AnalysisBundle producer/consumer contract tests
pytest tests/test_llm_eval.py -v                # LLM eval harness: scored rubric, 100% required
pytest tests/ --cov=src/md_platform/domain --cov-fail-under=80
ruff check src tests run_demo.py
mypy src
```

**Numerical regression suite** (`tests/test_regression.py`): runs all 10 classical modules against the adenylate-kinase DIMS reference trajectory (the MDAnalysis User Guide dataset, Beckstein et al. 2009) and asserts RMSD/Rg/SASA against published values (e.g. frame-0 Rg = 16.669 Å, endpoint backbone RMSD = 6.85 Å) plus ±10% drift bands.

---

## 📋 Roadmap

Live demo report: **https://honeyk254.github.io/MD_AI_Analyser/** — a single-trajectory showcase generated end-to-end by the pipeline (through the grounding check and human review gate):
* **[Reference system](https://honeyk254.github.io/MD_AI_Analyser/adk-report.html)** — adenylate kinase (Beckstein et al. 2009), all 10 classical modules; the ML layer refuses to run because DIMS is a directed, non-equilibrium trajectory with too few interconversions (the gate doing its job).

This project executes against the [Master Plan](md-ai-platform-master-plan.md):
* ✅ **Phase 0:** Data contracts, provenance run-card, CI (lint + type-check + tests + coverage gate), reference trajectory with published ranges
* ✅ **Phase 1:** Classical MD engine — 8 modules, numerical regression suite vs. literature ranges, domain coverage ≥80%
* ✅ **Phase 2:** Grounded LLM reporting, anti-hallucination verifier (100% catch rate on injected-error fixtures), human review gate, report latency/cost/review-turnaround metrics logged
* ✅ **Phase 3:** Containerized deployment, Caddy reverse proxy, middleware guards, demo endpoints
* ✅ **Phase 4:** Statistical & Machine Learning Layer — opt-in TICA/MSM with minimum-frames/transition-count gating, Chapman-Kolmogorov validation before `is_markovian` is reported, and a mandatory PCA-vs-TICA baseline comparison
* ✅ **Phase 5:** Testing rigor + Analysis Cards — `AnalysisBundle` contract tests (producer/consumer + JSON round-trip), formalized LLM eval harness (4 known-correct + 5 injected-error fixtures, scored rubric at 100%), and Analysis Cards ([ANALYSIS_CARDS.md](ANALYSIS_CARDS.md), machine-readable card required on every ML bundle)
* ✅ **Phase 6 (stretch):** VAMPnet ablation vs. TICA/MSM (VAMP-2-scored network, timescale/state agreement reported as specific numbers; CI runs it on CPU torch), OpenTelemetry tracing + LLM cost/latency dashboard (`/api/v1/metrics/*`), full-AWS Terraform artifact (`deploy/aws/`), and 2 more classical modules (dihedrals, COM drift) bringing the engine to the plan's 10-module ceiling. Autoencoder CVs and SQS/Step Functions queues were skipped by design (see the master plan's "explicitly skip" rules).

