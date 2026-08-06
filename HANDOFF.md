# Handoff: what is done, what is left

Branch: `devin/1785989700-phase1-repair`. Scope: repair the Phase 1 foundation,
then implement Phase 2 (grounded reporting) and Phase 3 (single-host demo) of the
master plan.

## Done and verified by hand

Phase 1 repair:

- `schemas/api.py` rewritten: `RunStatus`, `AnalysisParameters`, `SubmitRequest`,
  `AnalysisRequest`, `AnalysisResponse`, `StatusResponse`, upload/demo schemas.
  `schemas/__init__.py` re-exports the real names (it exported deleted ones).
- `domain/frames.py`: one `FrameWindow` abstraction; every classical module now
  honours stride and start/end frames, and the orchestrator threads the request's
  cutoffs through. Previously all of these were silently ignored.
- Classical modules corrected: RMSD reference frame = first analysed frame; RMSF
  on aligned C-alphas; SASA/DSSP via a topology-preserving MDTraj bridge
  (`domain/mdtraj_bridge.py`); salt bridges on carboxylate **oxygens** (was
  carbons); H-bond donors/acceptors selected explicitly by atom name, because
  MDAnalysis' guesser needs partial charges a PDB does not carry.
- `store.py`: filesystem-backed run store (status, bundle, report, per-run input
  dirs, atomic writes). Runs no longer live in a dict that a restart erases.
- Provenance: real `FileProvenance` with streaming sha256; `QCFlag` objects
  instead of raw dicts; failed modules survive as `ModuleResult.error` plus a QC
  flag rather than vanishing from the bundle.
- Deps: `pydantic-settings`, `python-multipart`, `httpx` (dev), PEP 621 package
  discovery, `requires-python >= 3.10`.

Phase 2:

- `aggregation/summarizer.py`: bundle -> `GroundedSummary` (mean/std/min/max, CV,
  half-to-half drift, trend, binary-segmentation changepoints, QC view, module
  errors, deterministic observations). No per-frame arrays reach the narrator.
- `aggregation/reference_ranges.py`: documented sanity bands with sources and
  explicit "no reference" cases.
- `reporting/tools.py`: exactly the three sanctioned tools; dispatch refuses any
  other name.
- `reporting/narrator.py`: `TemplateNarrator` (deterministic, offline) and
  `LLMNarrator` (bounded 8-turn tool loop) behind one interface; both audited.
- `reporting/llm.py` + `prompts.py`: swappable client, versioned prompt, token /
  latency / cost accounting.
- `reporting/grounding.py`: extracts every number from the narrative, matches it
  against facts re-derived from the summary, tolerance from written precision,
  unit-aware; verdicts verified / mismatch / unsupported. Summary-supplied
  literature citations are masked (they contain non-measurement numbers); an
  invented citation is still checked digit by digit.
- `reporting/report_service.py`: summarize -> narrate -> check -> review gate. A
  report that fails grounding stays `draft` and **cannot** be approved.
- `reporting/html_report.py`: narrative, grounding table, review banner and full
  provenance alongside the existing Plotly sections; all model/user text escaped.

Phase 3:

- `demo.py` + `data/demo/1l2y.pdb`: preloaded Trp-cage TC5B 38-model NMR ensemble
  (has hydrogens, has a real Asp9-Arg16 salt bridge), labelled honestly as an
  experimental ensemble rather than an MD simulation.
- `api/demo_routes.py`: `GET /` landing, `GET /demo` (computes on first visit,
  then serves the cached run), `POST /api/v1/demo/run`, dataset listing.
- `api/routes.py`: upload -> submit -> status -> results -> report -> review, all
  wired to the store. Clients send a `run_id`, never a server path.
- `api/uploads.py`: streamed uploads, extension allowlist, size cap enforced
  mid-stream with cleanup.
- `api/rate_limit.py`: fixed-window limiter with a tighter budget for
  work-starting endpoints; `X-Forwarded-For` aware.
- `config.py`: settings-driven limits (upload bytes, frames, atoms, rate limits,
  model, CORS). Orchestrator strides over-long windows down instead of refusing.
- CORS fixed: wildcard origins are fine only because credentials are now off.

Hand-verified once (see TESTING.md §3): all eight modules `ok` on the demo
ensemble in ~1 s, report generated, grounding passed with 78 verified claims,
`pending_review`, HTML written.

## Left to do

1. **Tests** (nothing new committed yet; `tests/` still holds the two Phase 1
   tests, and `tests/test_api.py` asserts the old `version: 2.0.0` health body so
   it fails as written):
   - `conftest.py` with a synthetic `AnalysisBundle` fixture (metric keys:
     `backbone_rmsd`, `mean_rmsf`, `radius_of_gyration`, `total_sasa`,
     `hbond_count`, `salt_bridge_count`, plus secondary-structure fractions) so
     tests never run MDAnalysis;
   - `test_grounding.py`: unmodified template narrative passes; **>= 5 injected
     error fixtures** (perturbed mean, invented count, wrong unit, swapped metric
     value, fabricated changepoint frame) with an assertion that 100% are caught;
   - `test_summarizer.py` (trend/drift/changepoints), `test_tools.py` (only three
     tools), `test_review_gate.py` (draft -> pending -> approved; ungrounded
     report refused), `test_imports.py` (import every module),
     `test_api.py` rewrite (health, 404s, submit-without-upload, demo dataset
     list) using `TestClient` and a `tmp_path` output dir.
2. **Docker / deploy**: Dockerfile still installs a dummy source tree before
   copying the package (Phase 1 bug) — rewrite it, add `docker-compose.yml` with
   a `data/outputs` volume and a healthcheck, plus `fly.toml` or `render.yaml`
   and a deployment section in the README covering env vars
   (`ANTHROPIC_API_KEY`, `MAX_UPLOAD_BYTES`, `RATE_LIMIT_*`, `CORS_ORIGINS`).
3. **README**: it still claims Phase 1 is complete and documents the old request
   shape. Needs the upload -> submit -> report -> review flow, the `/demo` URL,
   the grounding/review guarantees, and the demo-data caveat.
4. **Lint**: run `black`, `isort`, `flake8` over `src/` and fix fallout (line
   length is set to 90 in `pyproject.toml`).
5. **Verification before claiming done**: `pytest`, the TESTING.md §3 pipeline
   check, `docker compose up` + `/demo`, and the negative checks in §6.
6. **Environment blueprint**: record `pip install -U "setuptools>=64" wheel` and
   `pip install -e ".[dev]" --no-build-isolation` (plain `-e .` fails on this
   box's pip), via `update_environment_config`.
7. **PR**: fetch the template, open one PR for the whole branch (or split
   Phase 1 repair / Phase 2 / Phase 3 if a smaller review is preferred).

## Known risks worth a second look

- `parse_metadata` reports `timestep_ps = 1.0` for the PDB demo data because
  MDAnalysis defaults it; the demo caveat covers it, but a QC flag saying "frame
  interval is nominal" would be more honest.
- The grounding checker matches a claim against the nearest fact of a named
  metric; a narrative that cites two metrics in one sentence can be verified
  against the wrong one. Tightening this needs the injected-error fixtures first.
- `RunStore` is not concurrency-safe across processes (single-worker deployment
  assumed); document or serialise before scaling out.
