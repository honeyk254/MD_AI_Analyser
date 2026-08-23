# MD AI Platform — Master Execution Plan

**Purpose of this document:** everything needed to build, evaluate, and pitch this project, in one place. If nothing else survives, this file should be enough to pick the project back up from zero and execute it correctly.

**Owner:** Honey Kameshwar, B.Tech Biotechnology (Minor: ML & Data Analytics), NSUT Delhi.
**Positioning:** structural biology analysis infrastructure that could plug into a drug-discovery pipeline. **Not** "AI for Healthcare" in the clinical sense — a single MD trajectory says nothing about diagnosis or patient outcomes, and overclaiming this is a credibility risk with any domain-literate reviewer. The correct comparison set is Recursion / Isomorphic Labs / Iambic / Genesis Therapeutics / Insitro-style structural biology tooling, not clinical AI.

---

## Part 1 — Foundations

### 1.1 What this project is

A platform that ingests GROMACS (and eventually AMBER/CHARMM) MD simulation output, runs a battery of classical structural-dynamics analyses, adds a small set of literature-justified statistical/ML methods, and uses an LLM as a **grounded narration layer** over pre-computed numbers to produce a scientific report — never as a source of numbers itself.

### 1.2 Non-negotiable design principle

> **Each layer only consumes the structured output of the layer below it, never raw trajectory data past the classical-analysis layer.**

This single rule is what makes the system auditable, cheap to run, and defensible under technical questioning. Every architectural decision below either serves this rule or doesn't belong in the project.

### 1.3 Critical assumptions — resolved

| Original assumption | Resolution |
|---|---|
| "Infer force field, water model, sim parameters from any file" | Only true when the format embeds it (AMBER `.prmtop`, GROMACS `.tpr`, CHARMM PSF+params) — parse deterministically. For raw PDB+XTC/DCD with no companion log/mdp/cntrl, report **"insufficient metadata — not recoverable."** Never build a classifier that guesses this from bond-length statistics. |
| "Feed all structured outputs into the LLM" | The LLM only ever sees **pre-aggregated summary statistics** (means, stds, trends, changepoints, threshold comparisons) — never raw per-frame arrays. Its job is narrative synthesis, not arithmetic. |
| "Run only scientifically defensible ML/DL" | PCA, k-means, hierarchical clustering, t-SNE/UMAP are **classical unsupervised statistics, not deep learning.** Label them honestly; don't market them as "AI." |
| "More ML = more impressive" | False for this audience. Restraint — 20 classical analyses done well, 2–3 ML methods that are genuinely state-of-the-art for a specific sub-problem, each with a stated literature basis and a documented failure mode — reads as senior. A DL module bolted on "because AI" reads as junior. |
| Self-supervised GNN/VAE/Transformer trained and evaluated on one trajectory (original codebase) | Not a valid prediction claim — the "ground truth" was already computed two steps earlier in the same pipeline. Either restructure to train across multiple trajectories/proteins with a genuine held-out test, or keep strictly labeled "exploratory pattern discovery," never "prediction." |

---

## Part 2 — Architecture

### 2.1 System layers

```
1. INGESTION & IDENTIFICATION
   Upload → format detection → topology/trajectory parse → deterministic metadata extraction
              │
2. CLASSICAL ANALYSIS ENGINE (deterministic, no learned parameters)
   RMSD / RMSF / Rg / SASA / H-bonds / DSSP / PCA / clustering / contacts /
   salt bridges / distances / angles / free-energy surfaces
   → structured, versioned feature tables
              │
3. STATISTICAL / ML LAYER (narrow, justified, opt-in per system)
   TICA + MSM (kinetics) · autoencoder CVs (enhanced sampling) · QC anomaly checks
              │
4. AGGREGATION & GROUNDING LAYER
   Deterministic summarization: stats, trends, changepoints, rule-based QC flags
   → single structured AnalysisBundle
              │
5. LLM / AGENT REPORTING LAYER
   Tool-calling orchestrator → narrative report → grounding/consistency check
   → human review gate → final report
```

### 2.2 AI taxonomy (put this front-and-center in the README)

| Tier | What it is | Examples here | Rule |
|---|---|---|---|
| Deterministic algorithms | No learned parameters, exact math | RMSD, RMSF, Rg, SASA, H-bond geometry, DSSP, contacts, salt bridges | Ground truth. Never replace with ML. |
| Classical unsupervised statistics | Learned from data, not deep learning | PCA, k-means/hierarchical clustering, 2D free-energy histograms | Standard, interpretable, cheap. Don't call it "AI." |
| Deep learning (narrow, justified) | Neural nets, used only where classical methods are provably insufficient | TICA+MSM/VAMPnets (kinetics), autoencoder CVs | Each one must cite literature and state its failure mode. |
| LLM layer | Application-layer synthesis, outside the taxonomy | Report generation | Never produces a number. Only narrates numbers that already exist. |

### 2.3 Backend architecture

Modular monolith with clean internal boundaries — not microservices. A solo-maintained portfolio project that's over-decomposed is harder to explain end-to-end than a well-organized monolith, and that hurts more than it helps in review.

- **API layer:** FastAPI, Pydantic v2 schemas for every request/response — this doubles as data-contract documentation.
- **Orchestration layer:** task queue (Celery+Redis, or Prefect 2 for nicer built-in observability/retries) — added when jobs are actually long-running, not from day one.
- **Compute layer:** stateless worker processes, one job type per analysis module.
- **Domain layer:** pure-Python scientific functions, no framework dependencies — this is what makes the classical code unit-testable against reference datasets.

```
Client → FastAPI (validation, auth, job submission)
            │
      Job Queue (Celery/Redis or Prefect)
   ┌────────┼────────────┐
   ▼        ▼             ▼
Parsing  Classical     ML/DL
Workers  Analysis      Workers
         Workers
            │
   Feature Store (Parquet/Postgres)
            │
   Aggregation Job → AnalysisBundle (JSON)
            │
   LLM Reporting Service (stateless, tool-calling)
            │
   Report Store + Human Review Queue
```

### 2.4 Data pipeline

```
Raw upload (topology + trajectory + optional log/mdp/cntrl files)
   │
Format detection (extension + magic bytes + parser probing via MDAnalysis,
   fallback chain across parsers before failing)
   │
Deterministic metadata extraction
   - Chain/residue typing → protein/DNA/RNA/lipid/ligand/ion/solvent
   - Force field / water model: parsed if embedded, else "unknown — not recoverable"
   - Trajectory stats: n_frames, timestep, total time, box info
   │
Validation gate (schema check, min frame count, unit sanity checks)
   → REJECT early, loudly, with a specific reason
   │
Classical analysis batch (parallelized per-module workers)
   │
Feature store write (Parquet, partitioned by run_id + analysis_type)
   │
ML/DL batch (opt-in only; refuse to run and say why if the trajectory
   doesn't meet literature-justified minimum requirements — e.g. MSM
   needs enough independent transitions to be statistically meaningful)
   │
Aggregation job → AnalysisBundle (versioned JSON schema)
```

Every `AnalysisBundle` records exact tool versions, parameters, and input file hashes — the reproducibility backbone (see 2.9).

### 2.5 Agent / LLM architecture

Single tool-calling orchestrator with a verification pass. Not a multi-agent swarm — that adds failure surface without adding capability and is harder to defend line-by-line in review.

```
AnalysisBundle (structured JSON, no raw arrays)
        │
Orchestrator LLM call (function/tool-calling, structured output)
   Tools:
     - get_metric_summary(metric_name)
     - get_qc_flags()
     - compare_to_reference_ranges(metric_name)   ← literature heuristics, not LLM opinion
        │
Draft report (QC / structural behavior / biological interpretation / limitations / follow-ups)
        │
Grounding/consistency checker (deterministic, non-LLM):
   extracts every numeric claim from the draft, verifies against the
   AnalysisBundle, flags or auto-corrects mismatches
        │
Human review gate (required before "final" status — a feature, not a gap)
        │
Final report (PDF/HTML) + audit trail (prompt version, model version,
   bundle hash, reviewer sign-off)
```

**The grounding checker is the single highest-leverage component in the entire system.** Cheap to build (number extraction + lookup against the bundle), and it's the concrete answer to "how do you prevent hallucination in a scientific report" — the question this domain's reviewers ask first.

### 2.6 ML/DL opportunities — ranked by defensibility

| Method | Purpose | Literature basis | Notes |
|---|---|---|---|
| TICA + MSM | Slow collective motions, kinetics, metastable states | Pérez-Hernández 2013 (TICA); Bowman, Noé, Pande (MSM); Prinz et al. 2011 | Use `deeptime`, not PyEMMA (unmaintained). Require CK validation before reporting `is_markovian`. |
| VAMPnets / deep MSM | Nonlinear feature learning for kinetics | Mardt et al. 2018 | Stretch — only after TICA+MSM exists as a baseline to ablate against. |
| Autoencoder collective variables | Enhanced-sampling CV discovery | Chen & Ferguson 2018 and related | Only if there's an actual enhanced-sampling use case; don't add for its own sake. |
| NMA / ANM | Collective motion, low-frequency modes | Atilgan et al. 2001 | Classical, not ML — already correctly implemented in the existing codebase. |
| PRS | Effector/sensor identification | Atilgan & Atilgan | Legitimate, keep as classical network analysis, not "AI." |
| GNN for RMSF | Cross-protein flexibility prediction | Precedent exists for GNN-based B-factor/flexibility prediction generalizing across proteins | Only valid with genuine held-out evaluation across multiple proteins. Self-supervised on one trajectory = not a prediction claim, cut or relabel "exploratory." |
| VAE conformational landscape | Nonlinear dimensionality reduction | Bowman et al. 2016 (KL annealing) | Only keep if reported against a PCA/tICA variance-explained baseline. No baseline, no VAE. |

### 2.7 What stays classical — no ML branding, ever

RMSD, RMSF, Rg, SASA, H-bonds, salt bridges, secondary structure (DSSP), contact maps, PCA, DCCM, clustering, free-energy landscape, convergence, entropy (Schlitter). All deterministic or standard unsupervised statistics. Resist any temptation to "AI-ify" these in marketing copy — a domain-literate reviewer notices immediately, and it undercuts credibility on the parts that *are* genuinely deep learning.

### 2.8 Cloud vs. local execution strategy

- **Local-first for compute.** Trajectory files are large, GPU-bound, and often unpublished research data — a legitimate reason, not just a resource constraint.
- **Public demo:** lightweight single-host deployment (Fly.io/Render) pre-loaded with 1–2 open reference trajectories, so a recruiter can interact without supplying their own files.
- **Full AWS (ECS Fargate, SQS, Step Functions, Terraform):** optional, later-phase artifact. Stand it up briefly for demo screenshots, `terraform destroy` afterward — stating this cost-conscious workflow explicitly is a stronger signal than a 24/7 deployment nobody uses at 3am.

### 2.9 Storage architecture

| Data | Store | Why |
|---|---|---|
| Raw trajectories/topologies | Object storage (S3 / MinIO locally) | Large binary, immutable once uploaded |
| Per-frame/per-residue features | Parquet on object storage, or DuckDB locally | Columnar, fast for repeated aggregation queries |
| Run/job metadata, user data | Postgres | Relational integrity, queryable |
| Plots/figures/reports | Object storage, versioned by run_id | Immutable, directly servable |
| AnalysisBundle | Postgres JSONB or object storage | Small, queryable, the canonical "what did the LLM see" audit record |
| (Optional) methodology reference corpus | pgvector | Only if the LLM should cite methodology papers — nice-to-have, not required |

### 2.10 Deployment architecture

**Docker image split** (by dependency weight and blast radius, not by "microservice" dogma):

| Image | Contains | Why separate |
|---|---|---|
| `api` | FastAPI app, Pydantic schemas | Small, fast to build/deploy, changes most often |
| `worker-classical` | MDAnalysis/MDTraj/biotite + classical modules | CPU-only, no torch, scales cheaply |
| `worker-ml` | `deeptime`, PyTorch (CPU by default) | Isolated so it never bloats the classical deploy path |
| `llm-service` | Thin Anthropic/Bedrock client wrapper, orchestrator, grounding checker | No scientific-compute deps — should be genuinely tiny |
| `migrations` | Alembic only, run as one-off task | Avoids migration races between replicas |

`docker compose up` brings up the entire stack — this is deliberately both the demo environment and the CI integration-test environment.

**CI/CD (GitHub Actions):**
```
On every PR: lint → type-check → unit tests → integration tests →
             numerical regression tests → build all images (no push) →
             terraform plan (if infra changed, posted as PR comment)

On merge to main: build+tag with commit SHA → push to registry →
             auto-deploy to dev → smoke test →
             manual approval gate → promote SAME image tag to prod
```
The "same image promoted, not rebuilt between environments" detail is a specific, checkable claim worth stating explicitly.

### 2.11 Testing strategy

| Layer | Covers | Tooling | Runs |
|---|---|---|---|
| Unit tests | RMSD/RMSF/Rg math, metadata-parsing rules, grounding checker's number extraction | pytest | Every PR |
| **Numerical regression** | Full classical pipeline vs. 1–2 reference trajectories, computed values within literature-reported ranges | pytest + fixture trajectories | Every PR — highest-value suite, non-negotiable |
| Contract tests | `AnalysisBundle` schema, producer/consumer validation | pytest + Pydantic | Every PR |
| Integration tests | API → queue → worker → DB round trip | pytest + docker-compose | Every PR |
| **LLM eval harness** | Fixed `AnalysisBundle` fixtures with known-correct and deliberately-injected-error claims; assert grounding checker catches errors | pytest + scored rubric | Every PR touching prompts, nightly against live model |
| Infra tests | `terraform validate` + plan diff review | GitHub Actions | Every PR touching infra |
| Smoke tests | End-to-end job submission against deployed dev env | pytest | Post-deploy, pre-prod promotion |

Deliberately skip: Locust load clusters, chaos engineering — not where the risk lives in this project.

### 2.12 MLOps/LLMOps

- **Data versioning:** DVC or lakeFS for trajectory/feature datasets.
- **Experiment tracking:** MLflow/W&B — only for actually-trained models (MSM/VAMPnets), not for deterministic RMSD.
- **Model registry:** trained kinetics/CV models, with hyperparameters and training trajectory subset recorded.
- **Prompt versioning:** treat prompts as code — version, diff, re-run the eval harness on every change.
- **Observability:** structured logging + OpenTelemetry tracing across the pipeline; a cost/latency panel for LLM calls (tokens in/out, $/report).

### 2.13 Security and reproducibility

- **Provenance:** hash every input file; record tool versions, container digest, seeds, full params in a per-run "run card."
- **Environment pinning:** Docker + locked dependencies (uv/poetry/conda-lock).
- **Secrets:** secrets manager or documented `.env` path — never in code.
- **Access control:** per-project scoping even in a single-tenant demo.
- **Data sensitivity:** MD trajectories of public/synthetic systems aren't PHI — don't pretend otherwise — but design storage/access *as if* sensitive data could arrive later (encryption at rest, audit logs on raw-file reads). State this explicitly; it shows forward thinking without overclaiming the present.

### 2.14 Tech stack

| Layer | Choice | Note |
|---|---|---|
| Trajectory/topology parsing | MDAnalysis (primary), MDTraj (fallback) | Broadest format support |
| Secondary structure | DSSP via MDTraj/biotite | Classical, don't reinvent |
| Kinetics/dimensionality reduction | `deeptime` | Maintained successor to PyEMMA |
| Backend API | FastAPI + Pydantic v2 | Schema-first |
| Job orchestration | Celery+Redis (or Prefect 2) locally; SQS+Step Functions on AWS | One execution model, not two parallel systems |
| Metadata store | Postgres (RDS in AWS) | |
| Feature store | Parquet on object storage, DuckDB locally | |
| Object storage | S3 (prod), MinIO (local) | Same API |
| LLM | Claude via Anthropic API or Bedrock, tool-calling, structured output | Avoid LangChain unless the abstraction cost is justified |
| Frontend | Plotly for plots; Streamlit for fast iteration, or React+Plotly for polish | Don't over-invest in frontend at the science pipeline's expense |
| Testing | pytest — unit/integration/regression/contract/LLM-eval | Credibility anchor |
| CI/CD | GitHub Actions | Lint, type-check, test, build, plan/apply, staged deploy |
| IaC | Terraform | Diffable infra changes as PRs |
| Containerization | Docker (multi-stage) + Compose; optional k8s manifests as documented extension | |
| Cloud target (optional, later) | AWS: ECS Fargate, S3, RDS, SQS, Step Functions, ECR, Secrets Manager, CloudWatch | |

### 2.15 What makes this stand out to recruiters

- The **Analysis Card** pattern — a model-card-style one-pager per ML component (purpose, literature basis, data requirements, failure modes, baseline comparison).
- Numerical validation against a known reference system.
- The grounding/consistency checker, demonstrably catching injected errors in CI.
- Honest metadata handling ("not recoverable" instead of guessing).
- Baseline ablations for every ML method.
- A CI pipeline whose interesting tests are scientific (regression + LLM-eval), not just structural.
- An explicit, stated cloud cost strategy.
- A phased roadmap in the README — signals product judgment, not just execution.

### 2.16 What to avoid — scientifically weak or resume fluff

- Fine-tuning an LLM from scratch.
- "AI-generated" protein visualizations via generative image models.
- Blockchain for "data integrity."
- A free-form chatbot answering directly against raw trajectory data — reintroduces the hallucination risk the grounding layer exists to prevent.
- Any claim of disease detection/diagnosis/clinical relevance from MD output.
- Deep learning applied to Section 2.7's classical calculations "because it's more AI."
- A multi-agent LLM swarm for what is fundamentally single-pass report generation.
- A full production Kubernetes cluster / service mesh / multi-region deployment for a solo project.
- A NAT gateway, always-on GPU instance, or multi-AZ RDS left running 24/7 for a project nobody uses at 3am.
- Choosing EKS/Aurora/multi-region "because AWS best practice" without a specific requirement that forces it over the cheaper option.

---

## Part 3 — Phased execution plan

**Total to a strong, demoable state (through Phase 2): 6–8 weeks part-time.** Everything after is upside, not requirement. Every phase is scoped so stopping right after it still leaves a coherent, honest artifact.

### Phase 0 — Foundation & contracts

- **Effort:** ~1 week
- **Deliverables:**
  - `AnalysisBundle` Pydantic v2 schema defined (even before real numbers populate it)
  - 1–2 reference trajectories downloaded, with published RMSD/Rg ranges recorded
  - Repo skeleton: `docker-compose.yml` with `api` + `postgres` only (no queue yet)
  - GitHub Actions running lint + type-check on every PR
  - Run-card/provenance schema defined
- **Definition of done:** contracts exist and are documented, even with zero analysis modules implemented yet.
- **Metrics:** N/A — this phase produces no measurable output, only contracts.
- **Explicitly skip:** Celery/Redis, any ML dependency, any cloud config.

### Phase 1 — Deterministic classical engine

- **Effort:** ~3–4 weeks
- **Deliverables:**
  - 6–10 classical modules ported (RMSD, RMSF, Rg, SASA, H-bonds, contacts, secondary structure, salt bridges — prioritized for report narrative value)
  - Every module writes to the Phase 0 `AnalysisBundle` contract
  - Numerical regression tests against reference trajectories
  - Basic HTML/PDF report (plots + tables, no LLM narration yet)
  - `docker-compose up` runs the full pipeline synchronously (no queue needed at this trajectory size)
- **Definition of done:** CI badge is real — computed RMSD/Rg/SASA fall within literature-reported ranges (±10% tolerance, or the range actually reported in the source paper if narrower) on every commit.
- **Metrics:**
  - Regression suite: 100% pass rate required to merge
  - Test coverage on domain layer: ≥80%
  - Full pipeline runtime on reference trajectory: recorded as a baseline number (not gated yet, but tracked)
- **Explicitly skip:** parallel workers, Parquet feature store, any ML/DL module.

### Phase 2 — Grounded LLM reporting (target demo state)

- **Effort:** ~2–3 weeks
- **Deliverables:**
  - Deterministic aggregation job (stats, trends, threshold comparisons — pure Python, no LLM)
  - Orchestrator LLM call with exactly 3 tools: `get_metric_summary`, `get_qc_flags`, `compare_to_reference_ranges`
  - Grounding/consistency checker built and tested **before** prompt polishing
  - Human review gate (status flag + reviewer sign-off field — minimal implementation)
  - Report template updated with LLM narrative sections alongside Phase 1 plots
- **Definition of done:** given any report, you can point to a specific, running, tested component that verifies every numeric claim against the `AnalysisBundle` — not a design intention.
- **Metrics:**
  - Grounding checker catch rate on injected-error fixtures: **100%** (n ≥ 5 fixtures minimum)
  - Report generation latency: target < 30s end-to-end (aggregation + LLM call + grounding check)
  - Cost per report: track $/report from day one; target < $0.50 at default model tier
  - Human review turnaround: not gated, just logged, to have a real number to cite later
- **This is the version to demo if time-constrained.** Everything past this is depth, not missing core.

### Phase 3 — Lightweight deployment

- **Effort:** ~1 week
- **Deliverables:**
  - `docker-compose` stack deployed to a single low-cost host (Fly.io/Render)
  - 1–2 pre-loaded example trajectories so the demo works with zero setup
  - Basic rate limiting and input size caps
- **Definition of done:** a URL exists that a recruiter can open and get a real, grounded report without supplying their own files.
- **Metrics:**
  - Uptime target for demo period: informal, just "works when a recruiter clicks it"
  - Cold-start to first report: target < 2 minutes
- **Explicitly skip:** Terraform, ECS, SQS, multi-AZ anything.

### Phase 4 — Statistical/ML layer: TICA + MSM

- **Effort:** ~2–3 weeks
- **Deliverables:**
  - TICA + MSM via `deeptime`, opt-in, gated behind a minimum-frame/minimum-transition-count check
  - Chapman-Kolmogorov validation before reporting `is_markovian`
  - Explicit baseline comparison (e.g., PCA-based clustering vs. TICA-based metastable states) reported side-by-side
  - MSM summary fed into the existing Phase 2 `AnalysisBundle` → LLM pipeline (no new report infra needed)
- **Definition of done:** the MSM module refuses to run and states why on trajectories that don't meet minimum statistical requirements, rather than producing an unreliable result silently.
- **Metrics:**
  - Minimum observed transitions per state pair before kinetics are reported: ≥10 (literature-typical floor, adjust with justification)
  - CK-test deviation threshold for `is_markovian = true`: define and document a specific numeric cutoff (e.g., <15% deviation between predicted and directly-estimated transition matrices) — the exact number matters less than that one exists and is applied consistently
  - Baseline comparison present in 100% of MSM reports (no MSM output without its classical baseline alongside it)
- **Explicitly skip:** VAMPnets, autoencoder CVs (Phase 6 ablations).

### Phase 5 — Testing rigor + Analysis Cards

- **Effort:** ~1–2 weeks
- **Deliverables:**
  - Contract tests on `AnalysisBundle` schema (producer/consumer validation)
  - LLM eval harness formalized: 3–5 known-correct fixtures, 3–5 injected-error fixtures, scored rubric (not just pass/fail on "did it run")
  - Analysis Cards written for MSM/TICA (and any other ML component present)
- **Definition of done:** CI has a numerical-regression suite and an LLM-eval suite, both green, both checkable in commit history.
- **Metrics:**
  - LLM eval harness: 100% catch rate on injected errors, maintained across prompt changes (re-run on every prompt-touching PR)
  - Analysis Card coverage: 100% of ML/statistical components have one
  - CI pipeline total runtime: target < 10 minutes for PR checks (keeps the regression/eval suites actually used, not skipped for speed)

### Phase 6 — Stretch (optional, only if time remains)

- **Effort:** open-ended
- **Priority order if pursued:**
  1. VAMPnets, explicitly ablated against the Phase 4 TICA+MSM baseline (report timescale/state agreement between the two)
  2. Full AWS via Terraform (ECS Fargate, SQS, Step Functions) — stand up briefly for demo capture, `terraform destroy` after
  3. OpenTelemetry tracing + a cost/latency dashboard for LLM calls
  4. Remaining classical modules ported from the existing 21-module codebase (mechanical once the Phase 1 pattern exists)
- **Metrics:** VAMPnets vs. MSM agreement on implied timescales, reported as a specific number, not a qualitative claim.

---

## Part 4 — Master metrics dashboard

A single reference table of every numeric target across the whole project, so nothing has to be re-derived from memory.

| Metric | Target | Phase introduced |
|---|---|---|
| Regression suite pass rate | 100% (blocks merge) | 1 |
| RMSD/Rg/SASA vs. literature range tolerance | ±10% (or source-paper range if narrower) | 1 |
| Domain-layer test coverage | ≥80% | 1 |
| Grounding checker catch rate on injected errors | 100% (n≥5 fixtures) | 2 |
| Report generation latency | <30s end-to-end | 2 |
| Cost per report | <$0.50 | 2 |
| Cold start to first demo report | <2 min | 3 |
| Minimum transitions per state pair (MSM) | ≥10 | 4 |
| CK-test deviation cutoff for `is_markovian` | Documented fixed threshold (e.g. <15%) | 4 |
| MSM baseline comparison present | 100% of MSM reports | 4 |
| Analysis Card coverage | 100% of ML components | 5 |
| CI pipeline runtime (PR checks) | <10 min | 5 |
| VAMPnets vs. MSM timescale agreement | Reported number, not qualitative | 6 (stretch) |

---

## Part 5 — Risk register / cut-line rules

Use this when scope pressure hits and something has to give.

1. **Never cut the grounding checker or the numerical regression suite to save time.** These two are the entire credibility case for the project. Cut module breadth, frontend polish, or cloud infra scope first.
2. **Never expand Phase 1 module count at the cost of starting Phase 2.** 8 well-tested modules with a working grounding checker beats 21 modules with an ungrounded LLM layer.
3. **Never run a paid cloud environment continuously "just in case."** `terraform apply` for a demo, `terraform destroy` after. Document this explicitly as a cost-management decision.
4. **Never add a DL module without a stated literature basis and a documented failure mode.** If you can't write its Analysis Card in one page, it's not ready to ship.
5. **If time runs out mid-phase, stop at the nearest completed phase boundary, not mid-feature.** Every phase above is designed to be independently demoable — a half-finished Phase 4 is worse than a finished Phase 2.

---

## Part 6 — Quick-reference checklist

- [x] Phase 0: `AnalysisBundle` schema, reference trajectories, repo skeleton, CI lint/type-check
- [x] Phase 1: 6–10 classical modules, regression tests green, basic report
- [x] Phase 2: aggregation job, 3-tool orchestrator, grounding checker, human review gate
- [x] Phase 3: single-host deploy, pre-loaded demo trajectories
- [x] Phase 4: TICA+MSM with baseline comparison, minimum-data gating, CK validation
- [x] Phase 5: contract tests, formal LLM eval harness, Analysis Cards
- [ ] Phase 6 (optional): VAMPnets ablation, full AWS, tracing/cost dashboard, remaining modules
