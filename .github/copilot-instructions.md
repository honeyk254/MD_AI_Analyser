# Copilot instructions for MD AI Platform

## Project shape

- This is a Python 3.10+ FastAPI project for molecular dynamics trajectory analysis.
- The core package lives under `src/md_platform/` and is organized as a modular monolith:
  - `api/` for FastAPI app, routes, and middleware (rate limiting, body-size guards)
  - `domain/` for scientific calculations, parsing, and validation (`domain/classical/` holds the 10 modules)
  - `aggregation/` for assembling the final bundle
  - `reporting/` for Plotly/HTML output
  - `schemas/` for Pydantic v2 data contracts
  - `llm/` for narrative generation grounded in the computed bundle
  - `ml/` for the opt-in kinetic layer (TICA/MSM, PCA baseline, VAMPnet ablation)
- `observability.py` holds OpenTelemetry spans and LLM cost/latency metrics (`/api/v1/metrics/*`).
- `demo_inputs.py` materializes the bundled zero-setup demo trajectories (stable, flexible, kinetics).
- The main pipeline is orchestrated by `md_platform.orchestrator.AnalysisOrchestrator`, which:
  1. loads an `MDAnalysis.Universe`
  2. parses metadata and validates the trajectory
  3. runs classical analysis modules
  4. builds an `AnalysisBundle`
  5. generates plots and HTML
  6. generates an LLM draft report (deterministic offline fallback when no API key is set)

## Key conventions

- Treat `AnalysisBundle` as the single contract between analysis, reporting, and LLM layers. ML outputs live in a separate `MLAnalysisBundle` that requires an `analysis_card`.
- Classical module outputs should fit `ModuleResult`; prefer strict schema-backed data over ad hoc dicts.
- LLM reporting must narrate existing results only; it must not invent or compute new scientific values.
- The ML layer is opt-in (`enable_ml`) and must refuse to run with explicit reasons when gating minimums (frames, transition counts) are not met.
- Keep deterministic scientific logic in `domain/`; keep API and presentation code out of scientific modules.
- The app exposes a `/health` endpoint and FastAPI routes under the API package.
- Tests use `pytest` with `pythonpath = ["src"]`, so imports should resolve from `md_platform`.

## Build, test, and lint

- Install dev dependencies: `pip install -e .[dev]`
- Run tests: `pytest tests/` (coverage gate: `pytest tests/ --cov=src/md_platform/domain --cov-fail-under=80`)
- Run a single test file: `pytest tests/test_api.py`
- Run a single test: `pytest tests/test_api.py -k test_health_check`
- Lint: `ruff check src tests run_demo.py`
- Type-check: `mypy src`
- CI (`.github/workflows/ci.yml`) installs CPU torch plus `.[dev]`, then runs ruff, mypy, and the full test suite with the coverage gate.

## When editing code

- Preserve Pydantic v2 models and the existing bundle shape unless the change explicitly requires a schema update.
- Keep changes aligned with the current pipeline order in `AnalysisOrchestrator`.
- If you touch reporting, keep Plotly output compatible with the standalone HTML report format.
- If you touch API code, preserve the existing health endpoint and versioned response shape used by tests.
- If you touch the classical modules or the ML layer, numerical regression tests in `tests/test_regression.py` and `tests/test_phase4.py` are the correctness anchor — update pinned baselines deliberately, never by deleting assertions.
