# Copilot instructions for MD AI Platform

## Project shape

- This is a Python 3.9+ FastAPI project for molecular dynamics trajectory analysis.
- The core package lives under `src/md_platform/` and is organized as a modular monolith:
  - `api/` for FastAPI app and routes
  - `domain/` for scientific calculations, parsing, and validation
  - `aggregation/` for assembling the final bundle
  - `reporting/` for Plotly/HTML output
  - `schemas/` for Pydantic v2 data contracts
  - `llm/` for narrative generation grounded in the computed bundle
- The main pipeline is orchestrated by `md_platform.orchestrator.AnalysisOrchestrator`, which:
  1. loads an `MDAnalysis.Universe`
  2. parses metadata and validates the trajectory
  3. runs classical analysis modules
  4. builds an `AnalysisBundle`
  5. generates plots and HTML
  6. generates an LLM draft report

## Key conventions

- Treat `AnalysisBundle` as the single contract between analysis, reporting, and LLM layers.
- Classical module outputs should fit `ModuleResult`; prefer strict schema-backed data over ad hoc dicts.
- LLM reporting must narrate existing results only; it must not invent or compute new scientific values.
- Keep deterministic scientific logic in `domain/`; keep API and presentation code out of scientific modules.
- The app exposes a `/health` endpoint and FastAPI routes under the API package.
- Tests use `pytest` with `pythonpath = ["src"]`, so imports should resolve from `md_platform`.

## Build, test, and lint

- Install dev dependencies: `pip install -e .[dev]`
- Run tests: `pytest tests/`
- Run a single test file: `pytest tests/test_api.py`
- Run a single test: `pytest tests/test_api.py -k test_health_check`
- Lint/format tools available in the project:
  - `black`
  - `isort`
  - `flake8`
- CI currently installs `.[dev]` and runs `pytest tests/`.

## When editing code

- Preserve Pydantic v2 models and the existing bundle shape unless the change explicitly requires a schema update.
- Keep changes aligned with the current pipeline order in `AnalysisOrchestrator`.
- If you touch reporting, keep Plotly output compatible with the standalone HTML report format.
- If you touch API code, preserve the existing health endpoint and versioned response shape used by tests.
