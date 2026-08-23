# Analysis Cards

Model-card style documentation for every statistical/ML component in the
platform. The machine-readable version of this card is a **required field**
(`analysis_card`) on every `MLAnalysisBundle` (see
`src/md_platform/ml/schemas.py`), so a component cannot emit results without
one — coverage is enforced by `tests/test_contracts.py`.

## TICA + MSM kinetic module (`phase4-v1`)

**Purpose.** Estimate slow collective motions and metastable kinetics from
aligned trajectory features.

**Literature basis.** Pérez-Hernández et al. (TICA); Bowman, Noè, and Pande
(MSM); Prinz et al. (MSM validation).

**Data requirements.**

- At least the configured minimum number of analyzed frames (default gate).
- At least the configured minimum observed transitions per state pair
  (default 10 per the master plan; enforced before any kinetic number is
  reported).
- Non-zero lag in frames.

**Failure modes (and what the platform does about them).**

| Failure | Behavior |
|---|---|
| Too few frames for stable transition statistics | Gate refuses to run; `status="blocked"` with reasons |
| Too few transitions for a reliable MSM | Gate refuses to run; `status="blocked"` with reasons |
| CK deviation above the declared cutoff | Model runs but `is_markovian=false` and kinetics are not reported |

**Baseline protocol.** Compare PCA-clustered states against TICA-clustered
states and report state/timescale agreement (NMI, timescale relative error)
side-by-side — no MSM report is produced without its classical baseline.

### Component notes

- **TICA** — time-lagged independent component analysis on CA-atom features;
  output is a `KineticEmbedding` with explained variance and projections.
- **MSM** — lagged transition counts → transition matrix, stationary
  distribution, implied timescales; Chapman-Kolmogorov deviation vs the
  configured cutoff decides `is_markovian`.
- **PCA baseline** — same clustering on PCA components; exists solely as the
  comparison baseline, never reported alone.
- **VAMPnet ablation** (Phase 6) — a small lobet-style network trained with
  the differentiable VAMP-2 score (Wu & Noé 2017); torch is an optional
  dependency (`pip install md-ai-platform[vampnets]`). Without torch the
  ablation reports `available=false` and the rest of the ML layer is
  unaffected. Reported numbers: leading implied timescale vs the TICA/MSM
  baseline, timescale relative error, and state-agreement NMI.

Implementation: `src/md_platform/ml/analysis.py` (opt-in via
`enable_ml`, hand-rolled NumPy — no external MSM dependency).
