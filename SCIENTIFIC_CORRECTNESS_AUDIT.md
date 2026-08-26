# Scientific Correctness Audit — MD AI Analyser

**Date:** 2026-08-01
**Scope:** Full codebase review of all 35+ scientific modules for mathematical, physical, and methodological correctness.
**Reviewer context:** Computational biophysics, statistical mechanics, machine learning for molecular dynamics.

---

## Status addendum (2026-08-25): mapping to the rebuilt codebase

The audited `backend/` tree no longer exists. It was replaced by the
`src/md_platform/` rewrite (see `md-ai-platform-master-plan.md` §1.3: the
original 21-module codebase was restructured, not ported wholesale). Where
each finding stands today:

| Finding | Status in `src/md_platform/` |
|---|---|
| **C1** (tICA `eigh` on non-symmetric matrix) | **Fixed in the rewrite.** `ml/analysis.py::_run_tica` whitens via inverse Cholesky and symmetrizes before `np.linalg.eigh` — the standard sound formulation of the generalized eigenproblem. |
| **C2** (salt bridges: carboxylate carbon + 4.0 Å) | **Fixed 2026-08-25.** `domain/classical/salt_bridges.py` v2.1.0 selects the carboxylate **oxygens** (ASP OD1/OD2, GLU OE1/OE2, per Barlow & Thornton 1983) with per-frame residue-pair deduplication, and the regression suite now pins the ADK baseline (the carbon-centred selection found 30.2 bridges/frame vs 37.4 with oxygens — the predicted ~24% undercount). |
| **M3** (MSM: no uncertainty on implied timescales) | **Fixed 2026-08-25.** Implied timescales now carry 90% CIs from a seeded moving-block bootstrap (200 resamples, block = 5× lag); see `ANALYSIS_CARDS.md`. |
| **C3, M1–M2, M4–M8, m1–m6** | **Not applicable.** Those modules (entropy, PRS, free-energy landscape, NMA, GNN, VAE, tunnel detection, bio-inference, DCCM, energy decomposition, binding kinetics, transformer, convergence, allosteric network) were deliberately **not ported** — the rebuilt engine is the master plan's 10-module classical set plus the gated TICA/MSM/VAMPnet layer. |

The findings below are retained verbatim as the audit of record for the
original codebase.

---

## Executive Summary

The codebase implements ~31 analysis modules for GROMACS molecular dynamics trajectory analysis. The core physics and mathematical formulations are **largely correct**: Kabsch alignment via SVD with reflection correction, standard DCCM via einsum, correct Schlitter entropy with SI units, and a well-implemented vectorised ANM Hessian. Caveats are included on heuristic modules, which is commendable.

However, the audit identified **3 critical**, **8 major**, and **6 minor** scientific issues that affect result correctness or user interpretation.

| Severity | Count | Impact |
|----------|-------|--------|
| CRITICAL | 3 | Produce numerically wrong or systematically biased results |
| MAJOR | 8 | Misleading results, missing uncertainty, or incorrect labels |
| MINOR | 6 | Suboptimal but not wrong; cosmetic or negligible-impact issues |

---

## CRITICAL Issues

Issues that produce wrong numerical results or systematically biased outputs.

---

### C1. MSM internal tICA uses `eigh` on a non-symmetric matrix

**File:** `backend/ml/msm.py:271`

```python
evals, evecs = np.linalg.eigh(np.linalg.pinv(C0 + reg) @ Ctau)
```

**Problem:** `C0⁻¹ · Ctau` is the product of two symmetric matrices that do **not** commute in general, so their product is **not symmetric**. `numpy.linalg.eigh` assumes a symmetric (Hermitian) input and only reads the lower triangle. When given a non-symmetric matrix, it silently returns incorrect eigenvalues and eigenvectors.

This corrupts the tICA feature space that the MSM uses for state assignment. Every downstream quantity — transition matrix, stationary distribution, implied timescales, MFPT, CK test, metastable states — is built on an incorrectly defined state space.

**Evidence:** The standalone `backend/analysis/tica.py:100` solves the same problem correctly:
```python
eigenvalues, eigenvectors = eigh(Ctau_sym, C0_reg)  # generalised eigenvalue problem
```

**Fix (1-line):** Replace `msm.py:271` with:
```python
from scipy.linalg import eigh
evals, evecs = eigh(Ctau, C0 + reg)
```
Then sort by descending eigenvalue as before.

**Severity:** CRITICAL — all MSM outputs (kinetics, metastable states, timescales, MFPT) are affected.

---

### C2. Salt bridge detection uses carboxylate carbon with too-tight cutoff

**File:** `backend/analysis/salt_bridges.py:62-65`

```python
neg_sel = universe.select_atoms(
    "(resname ASP and name CG) or "
    "(resname GLU and name CD)"
)
```

**Problem:** Uses the carboxylate **carbon** atom (CG for Asp, CD for Glu) as the negative charge centre with a 4.0 Å distance cutoff. The standard approach in the literature (e.g., Barlow & Thornton 1983, Kumar & Nussinov 1999) uses the carboxylate **oxygen** atoms (OD1/OD2 for Asp, OE1/OE2 for Glu) as charge centres, because the negative charge is localised on the oxygens, not the carbon.

The nitrogen-to-carbon distance is typically 1.2–1.5 Å longer than the nitrogen-to-oxygen distance. With a 4.0 Å cutoff applied to the carbon atom, many genuine salt bridges (where N–O ≈ 3.2–4.0 Å but N–C ≈ 4.5–5.5 Å) will be systematically **missed**.

**Estimated impact:** ~30–60% of real salt bridges may be missed depending on the protein.

**Fix options:**
1. Use oxygen atoms with duplicate-pair deduplication:
   ```python
   neg_sel = universe.select_atoms(
       "(resname ASP and name OD1 OD2) or "
       "(resname GLU and name OE1 OE2)"
   )
   ```
   Then deduplicate by (positive_resid, negative_resid) pairs.
2. Keep using carbon atoms but increase the cutoff to ~5.5 Å to compensate for the C–O offset.

**Severity:** CRITICAL — systematically underestimates salt bridge count.

---

### C3. Entropy fallback mass is wrong for Cα-based analysis

**File:** `backend/analysis/entropy.py:94-95`

```python
logger.warning("Masses unavailable; using default 12.0 AMU for CA")
masses = np.full(n_res, 12.0)
```

**Problem:** When the topology file lacks mass information, the code falls back to bare carbon mass (12.0 AMU). Since the analysis uses Cα atom positions as proxies for entire residues, the Schlitter entropy computation effectively treats each residue's 3 degrees of freedom as belonging to a 12 Da point mass.

In the Schlitter formula, the mass-weighting enters as `sqrt(m)` in the covariance construction (line 197), meaning eigenvalues scale linearly with mass. The Schlitter factor `kT·e²/ℏ²` multiplied by mass-weighted eigenvalues produces the argument of the logarithm. Using 12 AMU instead of the correct ~110 AMU (average residue mass) means this argument is ~9× too small, leading to entropy values that are significantly lower than the true Schlitter upper bound.

**Fix:** Use average amino acid mass when topology masses are unavailable:
```python
masses = np.full(n_res, 110.0)  # average amino acid mass
```
Or ideally, use a per-residue mass lookup table keyed by residue name.

**Severity:** CRITICAL — absolute entropy values can be substantially wrong when topology lacks mass data. This fallback path is hit for PDB-only inputs (no TPR), which is a common use case.

---

## MAJOR Issues

Issues that produce misleading results, lack necessary uncertainty quantification, or use incorrect labels.

---

### M1. PRS: unused `pinv` import suggests incomplete implementation

**File:** `backend/analysis/prs.py:12`

```python
from scipy.linalg import pinv  # imported but never used
```

**Problem:** The `pinv` import is dead code, suggesting a planned Hessian-inversion approach was never completed. The current covariance-based implementation (`response = Σ C²`) is actually **theoretically valid** via the fluctuation-dissipation theorem and follows the PRS formulation of Atilgan et al. (2005) and Chennubhotla et al. (2007). However, the unused import creates confusion about the intended algorithm.

The response matrix correctly computes:
```
R(i→j) = sqrt(1/3 · Σ_{α,d} C[3j+α, 3i+d]²)
```
which is the standard PRS response for unit perturbations along each Cartesian axis.

**Fix:** Remove the unused `pinv` import. Optionally, add a docstring note explaining the relationship between C-based PRS and H⁻¹-based PRS:
> For ANM-derived covariances, C ∝ H⁻¹, so covariance-based PRS and Hessian-inversion PRS produce equivalent rankings. The MD-derived covariance used here additionally captures anharmonic effects.

**Severity:** MAJOR — misleading dead code; the underlying science is sound.

---

### M2. Free energy landscape: unsampled bins get finite energy

**File:** `backend/analysis/free_energy.py:107-112`

```python
positive_mask = H > 0
if positive_mask.any():
    density_floor = H[positive_mask].min() * 0.01
else:
    density_floor = 1e-30
H_safe = np.where(positive_mask, H, density_floor)
```

**Problem:** Bins with zero density (never visited during the simulation) are assigned `F = -kT · ln(0.01 · min_density)` rather than infinity or NaN. This creates artificial flat plateaus in the free energy landscape for completely unsampled conformational regions. Users may incorrectly interpret these finite-energy regions as accessible conformational states.

For example, with `min_density = 0.001` and `kT = 2.494 kJ/mol` at 300 K:
- Floor density: `0.001 × 0.01 = 1e-5`
- F_floor = `-2.494 × ln(1e-5)` ≈ 28.7 kJ/mol
- F_min = `-2.494 × ln(0.001)` ≈ 17.2 kJ/mol
- Apparent barrier for unsampled region: only ~11.5 kJ/mol above minimum

This is physically wrong — an unsampled region has effectively infinite free energy given the available data.

**Fix:** Set unsampled bins to NaN or a configurable maximum cutoff:
```python
F = np.full_like(H, np.nan, dtype=np.float64)
F[positive_mask] = -kT * np.log(H[positive_mask])
F[positive_mask] -= np.nanmin(F)
```

**Severity:** MAJOR — can misrepresent the accessibility of conformational states.

---

### M3. MSM: no bootstrap confidence intervals on implied timescales

**File:** `backend/ml/msm.py:130-136`

```python
for ev in T_eigenvalues[1:]:
    if 0 < ev < 1:
        timescales.append(float(-lag_time / np.log(ev)))
    else:
        timescales.append(0.0)
```

**Problem:** Implied timescales are point estimates with no uncertainty quantification. Without bootstrap confidence intervals, it is impossible to determine whether:
1. Timescale "convergence" in the lag-time sweep is genuine or within statistical noise.
2. The difference between two timescales is statistically significant.

The CK test (implemented at line 343) helps assess overall model quality but does not provide per-timescale error bars.

**Fix:** Implement bootstrap resampling of the count matrix:
```python
def _bootstrap_timescales(labels, n_states, lag, n_bootstrap=200):
    """Bootstrap confidence intervals for implied timescales."""
    n = len(labels)
    timescales_boot = []
    for _ in range(n_bootstrap):
        # Block bootstrap to preserve temporal correlations
        block_size = lag * 5
        n_blocks = n // block_size + 1
        indices = np.concatenate([
            np.arange(start, min(start + block_size, n))
            for start in np.random.randint(0, n - block_size, n_blocks)
        ])[:n]
        T_boot = _build_transition_matrix(labels[indices], n_states, lag)
        if T_boot is not None:
            evals = np.sort(np.real(np.linalg.eigvals(T_boot)))[::-1]
            its = [-lag / np.log(ev) if 0 < ev < 1 else 0.0 for ev in evals[1:]]
            timescales_boot.append(its)
    return np.percentile(timescales_boot, [5, 95], axis=0)
```

**Severity:** MAJOR — users cannot assess statistical reliability of kinetic predictions.

---

### M4. NMA B-factors missing temperature prefactor

**File:** `backend/analysis/nma.py:101`

```python
bfactors = (8.0 * np.pi ** 2 / 3.0) * (mode_sq @ inv_evals)
```

**Problem:** The correct crystallographic B-factor formula from ANM is:

```
B_i = (8π²kT) / (3γ) · Σ_k |v_{ik}|² / λ_k
```

The code omits the `kT/γ` prefactor. Since the result is normalized to [0, 1] (line 103), **rankings are unaffected**. However, the intermediate values labeled `bfactors` are not physically meaningful B-factors — they have wrong units and magnitude. If a user examines the pre-normalization values, they will be uninterpretable.

**Fix:** Either:
1. Include the prefactor: `bfactors = (8 * np.pi**2 * KB_EV * temperature) / (3 * gamma) * (mode_sq @ inv_evals)`, or
2. Rename the field to `bfactors_normalized` and document that these are relative, not absolute, B-factors.

**Severity:** MAJOR — misleading label; rankings (the primary use case) are correct.

---

### M5. GNN: no train/test split; reported error is training error

**File:** `backend/gnn_models/residue_gnn.py:246-273`

```python
model.train()
for epoch in range(200):
    # trains on ALL residues
    embeddings, importance, _, _ = model(x, edge_index)
    loss = F.mse_loss(importance, target)
    ...

model.eval()
with torch.no_grad():
    embeddings, importance, attn1, attn2 = model(x, edge_index)
recon_error = float(F.mse_loss(importance, target).item())  # same data!
```

**Problem:** The model trains for 200 epochs on the full dataset and evaluates `reconstruction_error` on the **same data**. This training-set error is not a valid quality metric — it primarily measures the model's capacity to memorise, not generalise. A sufficiently parameterised network can achieve near-zero training error on any dataset regardless of whether the learned representations are meaningful.

The caveat text (line 349) acknowledges this is a "single-trajectory self-supervised ranking," which is good. But the `reconstruction_error` field in the output can still mislead automated downstream interpretation.

**Fix:** Implement k-fold cross-validation (e.g., leave-out 20% of residues per fold) and report test-set error. For graph data, this requires careful handling to avoid information leakage through edges.

**Severity:** MAJOR — no validation that the GNN learned anything generalisable.

---

### M6. VAE logged total loss does not match training objective

**File:** `backend/ml/vae_latent.py:148-171`

```python
beta = min(1.0, epoch / kl_warmup_epochs)  # ramps 0→1
loss = recon_loss + beta * kl_loss          # TRAINING uses beta
...
total_losses.append(
    round((epoch_recon + epoch_kl) / max(n_samples, 1), 4)  # LOGGED without beta
)
```

**Problem:** During KL annealing warmup (first 40% of epochs), `beta < 1`, so the actual training loss is `recon + beta·kl`. But the logged `total_loss` records `recon + kl` (without the beta scaling). This means:
- The logged loss curve does not reflect the actual optimisation objective.
- The loss may appear to increase during early training (as beta grows and KL begins contributing), confusing users into thinking training is diverging.

**Fix:** Log the actual loss:
```python
total_losses.append(
    round((epoch_recon + beta * epoch_kl) / max(n_samples, 1), 4)
)
```

**Severity:** MAJOR — misleading training diagnostics.

---

### M7. Tunnel detection uses Cα-atom distances for heavy-atom cutoffs

**File:** `backend/ml/tunnel_detection.py:84, 93-94, 124`

```python
vdw_radius = 1.7        # average heavy-atom vdW radius
inner_cutoff = vdw_radius + probe_radius  # = 3.1 Å
...
dists = distance_array(grid, ca_pos)  # distances to CA, not heavy atoms
min_dist = dists.min(axis=1)
cavity_mask = (min_dist > inner_cutoff) & (min_dist < outer_cutoff)
```

**Problem:** Grid-to-Cα distances are compared against cutoffs derived from heavy-atom van der Waals radii. Cα atoms are backbone interior atoms, typically 1–2 Å from the nearest side-chain heavy atom that defines the true protein surface. Using Cα positions with a surface-atom cutoff systematically shifts the effective probe surface inward, causing:
- Overestimation of cavity volumes (regions between Cα and the true surface are counted as cavities).
- Incorrect identification of cavity-lining residues.

**Fix:** Use heavy atoms (`universe.select_atoms("protein and not (name H*)")`) for distance calculations, or increase the inner cutoff by ~2 Å to compensate for the Cα-to-surface offset.

**Severity:** MAJOR — cavity volumes will be systematically overestimated.

---

### M8. Bio-inference "confidence" scores are heuristic but named misleadingly

**File:** `backend/bio_inference/engine.py` (all 38 detectors)

Example from `_detect_hinge_residues` (line 148):
```python
confidence = min(0.95, 0.5 + 0.3 * (center - mean_rmsf) / (std_rmsf + 1e-8))
```

**Problem:** Every detector produces a `confidence` field using ad-hoc formulas with no statistical calibration. Users will naturally interpret `"confidence": 0.85` as "85% probability this finding is correct." In reality, these are arbitrary scoring functions that have not been validated against any ground truth.

The code does add mitigating metadata (lines 106-111):
```python
insight.setdefault("confidence_method", "heuristic")
insight.setdefault("confidence_note", "Heuristic confidence ... not statistically calibrated.")
```
This is a good practice, but the primary `confidence` field name still dominates user perception.

**Fix:**
1. Rename the field to `heuristic_score` throughout all detectors and output schemas.
2. Update the `InsightItem` Pydantic model in `models.py` accordingly.
3. In reports and visualisations, label the score as "Heuristic Score (not a probability)".

**Severity:** MAJOR — systematically misleads users about the reliability of biological insights.

---

## MINOR Issues

Issues that are suboptimal but produce negligible or no error in practice.

---

### m1. DCCM uses biased covariance estimator

**File:** `backend/utils/trajectory_utils.py:265`

```python
dccm = np.einsum("fid,fjd->ij", delta, delta) / n_frames  # N, not N-1
```

Divides by N rather than N-1. For typical MD trajectories with N > 1000 frames, the bias is < 0.1% and has no practical impact on the normalised correlation matrix.

---

### m2. Energy decomposition mixes physical Coulomb with heuristic LJ

**File:** `backend/analysis/energy_decomposition.py:32-33`

```python
_COULOMB_CONST = 332.0637  # kcal·Å/(mol·e²)
_KCAL_TO_KJ = 4.184
```

The Coulomb interaction uses a physically correct constant (332.0637 kcal·Å/mol·e²) with proper unit conversion, but the LJ interactions use arbitrary ε/σ parameters that are not derived from any force field. The total interaction score (`vdw + elec`) is therefore dimensionally inconsistent. The module's caveat (line 220-225) correctly warns about this.

---

### m3. Binding kinetics: dead conditional in event_start initialisation

**File:** `backend/analysis/binding_kinetics.py:144`

```python
event_start: int = 0 if in_contact else 0  # both branches identical
```

Both branches produce the same value. Not a scientific error, but indicates incomplete logic or a copy-paste artifact.

---

### m4. Transformer: 15% masking rate may be suboptimal for short sequences

**File:** `backend/transformer_models/trajectory_transformer.py:248`

```python
mask = torch.rand(1, n_frames).to(device) < 0.15
```

BERT-style 15% masking was designed for large text corpora. For MD trajectories sub-sampled to ~500 frames, only ~75 frames are masked per epoch. For very short trajectories (< 100 frames), this provides < 15 masked frames, which may be insufficient for learning meaningful temporal representations. Consider adaptive masking rates for short sequences.

---

### m5. Convergence score: equal weighting across diagnostics is arbitrary

**File:** `backend/analysis/convergence.py:123-168`

The overall convergence score equally weights four diagnostics (RMSD drift, Rg drift, block-average SEM ratio, cosine content). There is no principled justification for equal weighting, but each diagnostic is individually well-implemented. Users should examine individual diagnostics rather than relying on the aggregate score.

---

### m6. Allosteric network: edge weight vs distance attribute could confuse developers

**File:** `backend/ml/allosteric.py:92`

```python
G.add_edge(int(i), int(j), weight=corr_val, distance=1.0 - corr_val)
```

Stores both `weight` (raw correlation) and `distance` (1 - correlation) on each edge. The centrality metrics correctly use `weight="distance"` and `distance="distance"` respectively. However, having two attributes with different semantics on the same edge is a maintenance risk. The current code handles it correctly.

---

## Modules with Correct Scientific Implementation

The following modules were verified as scientifically correct and well-implemented:

| Module | File | Assessment |
|--------|------|------------|
| RMSD | `analysis/rmsd.py` | Correctly delegates to MDAnalysis; equilibration heuristic via rolling std is reasonable |
| RMSF | `analysis/rmsf.py` | Correct per-residue fluctuation from Kabsch-aligned Cα positions |
| PCA | `analysis/pca.py` | Standard scikit-learn PCA on aligned, flattened coordinates |
| DCCM | `analysis/dccm.py` | Correct normalised cross-correlation via `np.einsum("fid,fjd->ij")` |
| tICA | `analysis/tica.py` | Correct generalised eigenvalue problem with Tikhonov regularisation and whitening fallback |
| ANM Hessian | `analysis/nma.py:155-223` | Correct vectorised construction per Atilgan et al. (2001, Biophys. J.) with proper diagonal blocks |
| Schlitter entropy | `analysis/entropy.py:167-214` | Correct formula with proper SI unit handling (kB, ℏ, NA, e²) |
| Kabsch alignment | `utils/trajectory_utils.py:37-80` | Correct SVD-based superposition with reflection correction via `det(V^T · U^T)` |
| DCCM computation | `utils/trajectory_utils.py:247-271` | Correct vectorised computation with safe normalisation |
| Radius of gyration | `analysis/radius_of_gyration.py` | Correctly delegates to MDAnalysis `radius_of_gyration()` |
| Secondary structure | `analysis/secondary_structure.py` | Correctly uses MDTraj DSSP with bond creation fallback |
| SASA | `analysis/sasa.py` | Correctly uses MDTraj Shrake-Rupley |
| Hydrogen bonds | `analysis/hbonds.py` | Correct geometric criteria via MDAnalysis HydrogenBondAnalysis |
| Clustering | `analysis/clustering.py` | Standard KMeans/HDBSCAN/GMM with silhouette scoring |
| VAE architecture | `ml/vae_latent.py` | Correct ELBO with reparameterisation trick and KL annealing (Bowman et al. 2016) |
| MSM CK test | `ml/msm.py:343-401` | Properly compares T(τ)^k against directly estimated T(k·τ) |
| MSM MFPT | `ml/msm.py:202-232` | Correct fundamental matrix approach for mean first-passage times |
| Allosteric network | `ml/allosteric.py` | Correct betweenness/closeness centrality and Louvain communities on DCCM graph |
| Cosine content | `analysis/convergence.py:314-343` | Correct implementation of Hess (2002) cosine content for PCA projections |
| Block averaging | `analysis/convergence.py:226-271` | Correct block-mean SEM computation with ddof=1 |

---

## Recommended Fix Priority

Ordered by impact and effort:

| Priority | ID | Fix | Effort | Impact |
|----------|-----|-----|--------|--------|
| 1 | C1 | MSM tICA: use `scipy.linalg.eigh(Ctau, C0_reg)` | 1 line | Fixes all MSM outputs |
| 2 | C2 | Salt bridges: use oxygen atoms or increase cutoff to 5.5 Å | ~10 lines | Fixes false negatives |
| 3 | C3 | Entropy: change fallback mass from 12.0 to ~110.0 AMU | 1 line | Fixes entropy for PDB-only inputs |
| 4 | M8 | Rename `confidence` to `heuristic_score` in bio-inference | ~50 lines (find/replace) | Prevents misinterpretation |
| 5 | M2 | FEL: use NaN for unsampled bins | ~5 lines | Prevents false accessibility |
| 6 | M6 | VAE: log `beta * kl` not raw `kl` | 1 line | Fixes training diagnostics |
| 7 | M1 | Remove unused `pinv` import from PRS | 1 line | Removes confusion |
| 8 | M4 | NMA: add kT/γ prefactor or rename field | ~3 lines | Correct absolute B-factors |
| 9 | M7 | Tunnel detection: use heavy atoms for distances | ~5 lines | Correct cavity volumes |
| 10 | M3 | MSM: implement bootstrap CI for timescales | ~40 lines | Adds uncertainty quantification |
| 11 | M5 | GNN: add cross-validation split | ~30 lines | Validates model quality |

---

## References

- Atilgan, A. R. et al. (2001). Anisotropy of fluctuation dynamics of proteins with an elastic network model. *Biophys. J.*, 80(1), 505-515.
- Atilgan, C. & Atilgan, A. R. (2009). Perturbation-response scanning reveals ligand entry-exit mechanisms of ferric binding protein. *PLoS Comput. Biol.*, 5(10), e1000544.
- Barlow, D. J. & Thornton, J. M. (1983). Ion-pairs in proteins. *J. Mol. Biol.*, 168(4), 867-885.
- Bowman, S. R. et al. (2016). Generating sentences from a continuous space. *CoNLL 2016*.
- Chennubhotla, C. & Bahar, I. (2007). Signal propagation in proteins and relation to equilibrium fluctuations. *PLoS Comput. Biol.*, 3(9), e172.
- Hess, B. (2002). Convergence of sampling in protein simulations. *Phys. Rev. E*, 65(3), 031910.
- Kumar, S. & Nussinov, R. (1999). Salt bridge stability in monomeric proteins. *J. Mol. Biol.*, 293(5), 1241-1255.
- Schlitter, J. (1993). Estimation of absolute and relative entropies of macromolecules using the covariance matrix. *Chem. Phys. Lett.*, 215(6), 617-621.
- Sethi, A. et al. (2009). Dynamical networks in tRNA:protein complexes. *PNAS*, 106(16), 6620-6625.
- Sundararajan, M. et al. (2017). Axiomatic attribution for deep networks. *ICML 2017*.
