# Manual verification guide

Everything below is runnable on one machine with no API key: without
`ANTHROPIC_API_KEY` the reporting layer uses the deterministic template narrator,
which goes through the identical three tools and the identical grounding check.

## 0. Install

```bash
pip install -U "setuptools>=64" wheel
pip install -e ".[dev]" --no-build-isolation   # build isolation fails on old pip
```

## 1. Import smoke test

```bash
python -c "import md_platform.api.app as a; print(a.app.title)"
```

Expected: `MD AI Platform`. Any ImportError here means the schema contracts and
their consumers have drifted apart again — that was the Phase 1 breakage.

## 2. Unit tests

```bash
pytest -q
```

Must-pass expectations:

- grounding fixtures: every injected numeric error is detected (mismatch or
  unsupported), and the unmodified template narrative passes with 0 failures;
- only `get_metric_summary`, `get_qc_flags`, `compare_to_reference_ranges` are
  dispatchable; any other tool name raises;
- review gate: a report whose grounding failed cannot be approved.

## 3. Classical pipeline on the demo dataset

```bash
python -W ignore -c "
from md_platform.api.dependencies import get_orchestrator, get_report_service
from md_platform.schemas.api import AnalysisRequest
o, r = get_orchestrator(), get_report_service()
b = o.run_analysis_sync(AnalysisRequest(run_id='manual-check',
        topology_file='data/demo/1l2y.pdb', trajectory_file='data/demo/1l2y.pdb'))
print({k: (v.error or 'ok') for k, v in b.modules.items()})
rep = r.generate(b)
print('grounded:', rep.grounding.passed, rep.grounding.n_verified, 'claims verified')
for c in rep.grounding.failures(): print('FAIL', c.status, c.claim.value, c.detail)
print(rep.review.status, rep.html_path)
"
```

Expected (last verified 2026-08-06):

- all eight modules `ok` (rmsd, rmsf, radius_of_gyration, sasa, hbonds, contacts,
  secondary_structure, salt_bridges);
- runtime ~1 s for the 38-frame demo ensemble;
- `grounded: True`, `review status: pending_review`;
- an HTML report at `data/outputs/manual-check/analysis_report.html`.

Any `FAIL` line is a real finding: either the narrator wrote a number the bundle
does not contain, or the checker is matching too strictly. Do not "fix" it by
loosening the checker without understanding which of the two happened.

## 4. API and the zero-setup demo

```bash
uvicorn md_platform.api.app:app --port 8000
```

| Check | Request | Expected |
| --- | --- | --- |
| Health | `GET /health` | `status: ok`, narrator `template`, demo dataset `trp_cage` |
| Landing | `GET /` | page with an "Open the demo report" link |
| Zero-setup demo | `GET /demo` | full HTML report; first call computes, later calls serve the cached run |
| Demo JSON | `POST /api/v1/demo/run` | `grounding_passed: true`, `review_status: pending_review` |
| Unknown run | `GET /api/v1/analysis/nope/status` | 404 |
| Upload | `POST /api/v1/analysis/upload` with `topology=@data/demo/1l2y.pdb`, `trajectory=@data/demo/1l2y.pdb` | 201 with a `run_id` |
| Submit | `POST /api/v1/analysis/submit` `{"run_id": "<id>"}` | 200, then poll `/status` to `completed` |
| Results | `GET /api/v1/analysis/<id>/results` | the AnalysisBundle |
| Report | `POST /api/v1/analysis/<id>/report` | grounded report, `pending_review` |
| Approve | `POST /api/v1/analysis/<id>/review` `{"reviewer": "you", "approve": true}` | `approved`, banner updated in the HTML |
| Rate limit | 6+ rapid POSTs to `/api/v1/demo/run` | 429 with `Retry-After` |
| Size cap | upload a >100 MB file | 413, and the partial file is deleted |
| Path safety | any request naming a server path | not possible: clients only send `run_id` |

Golden path for a reviewer with no files: start the server, open `/`, click
through to `/demo`, read the report, then approve it via the review endpoint.

## 5. Docker

```bash
docker compose up --build
curl localhost:8000/health
open http://localhost:8000/demo
```

Check that `data/demo/1l2y.pdb` is present inside the image (the demo path 503s
without it) and that `data/outputs` is a writable volume.

## 6. Negative / honesty checks

These are the ones worth doing by hand, because they are what the project claims:

1. Edit a number in a narrative section before the grounding call (or use the
   fixtures in `tests/test_grounding.py`) and confirm the report is refused
   approval and the HTML shows a red FAILED banner.
2. Confirm the report never states a force field for the demo data: a PDB carries
   none, so it must read `unknown — not recoverable`.
3. Confirm the demo report carries the NMR-ensemble caveat and does not describe
   the frame series as a simulation with physical kinetics.
