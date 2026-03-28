# Implementation Roadmap

## MD AI Analyzer - Scientific Strengthening Project

**Based on:** SCIENTIFIC_ASSESSMENT.md evaluation
**Priority Framework:** Critical → High → Medium → Low
**Timeline:** 12 weeks for Critical + High priority items

---

## Phase 1: CRITICAL FIXES (Weeks 1-3)

### 1.1 Fix PRS Theoretical Issue 🔴 **BLOCKING**

**Problem:** PRS currently uses covariance matrix (spontaneous fluctuations) instead of inverted Hessian (elastic response to force).

**Location:** `backend/analysis/prs.py`

**Implementation:**

```python
# NEW FILE: backend/analysis/prs_hessian.py

def compute_prs_hessian(universe, cutoff=15.0, gamma=1.0, **kwargs):
    """PRS using inverted ANM Hessian (theoretically correct)."""

    # 1. Build ANM Hessian using existing code from nma.py
    from .nma import _build_anm_hessian_vectorised

    ca = select_ca_atoms(universe)
    positions = collect_ca_positions(universe, atoms=ca)
    mean_pos = positions.mean(axis=0)

    # Build Hessian
    hessian = _build_anm_hessian_vectorised(mean_pos, cutoff, gamma)

    # 2. Pseudo-inverse (removes 6 rigid-body modes automatically)
    from scipy.linalg import pinv
    H_inv = pinv(hessian, rcond=1e-6)

    # 3. Apply unit perturbations
    n_res = len(ca)
    response_matrix = np.zeros((n_res, n_res))

    for i_res in range(n_res):
        for direction in range(3):
            # Unit force on residue i_res in given direction
            force = np.zeros(3 * n_res)
            force[3*i_res + direction] = 1.0

            # Linear response: displacement = H^(-1) @ force
            displacement = H_inv @ force

            # Residue-block response magnitudes
            disp_reshaped = displacement.reshape(n_res, 3)
            response_per_res = np.linalg.norm(disp_reshaped, axis=1)

            response_matrix[i_res] += response_per_res

    # Average over 3 directions
    response_matrix /= 3.0

    # Rest of processing (effector/sensor scores) same as current prs.py
    ...
```

**Testing:**

1. Compare covariance-PRS vs. Hessian-PRS rankings on benchmark systems
2. Expected: High correlation (ρ > 0.7) but not identical
3. Validate against literature PRS studies (e.g., General et al. 2014)

**Documentation Update:**

- Add `prs_method` parameter: `"covariance"` (fast) or `"hessian"` (rigorous)
- Default to `"hessian"`
- Explain difference in README and paper

**Timeline:** 5 days
**Difficulty:** Medium
**Impact:** HIGH – fixes theoretical unsoundness

---

### 1.2 Rename "Confidence" to "Heuristic Score" 🔴 **CRITICAL**

**Problem:** Biological inference uses term "confidence" for uncalibrated heuristic scores.

**Locations:**

- `backend/bio_inference/engine.py` (all detector methods)
- `backend/models.py` (Pydantic schema)
- `frontend/app.js` (display logic)
- `frontend/index.html` (UI labels)
- `backend/visualization/report_generator.py` (HTML/PDF reports)

**Implementation:**

```python
# backend/bio_inference/engine.py

# BEFORE:
insights.append({
    "confidence": round(confidence, 2),
    "confidence_method": "heuristic",
    "confidence_note": "Heuristic confidence derived..."
})

# AFTER:
insights.append({
    "heuristic_score": round(score, 2),
    "score_interpretation": "Relative ranking strength (not probability)",
    "score_range": [0.0, 1.0],
    "caveat": (
        "This score represents heuristic ranking strength based on "
        "trajectory features. It is NOT a calibrated probability or "
        "statistical confidence. Use for hypothesis generation only. "
        "Validate with experimental data or conservation analysis."
    ),
})
```

**Global Changes:**

```bash
# Search and replace across codebase
grep -r "confidence" backend/bio_inference/ backend/models.py frontend/
# Replace with "heuristic_score"
# Update UI labels: "Confidence" → "Heuristic Score"
# Add tooltip: "Relative ranking strength (0-1), not statistical probability"
```

**Frontend Display:**

```javascript
// frontend/app.js
// BEFORE:
<span class="confidence-badge">${(insight.confidence * 100).toFixed(0)}%</span>

// AFTER:
<span class="score-badge">Score: ${insight.heuristic_score.toFixed(2)}</span>
<span class="tooltip">ⓘ Heuristic ranking, not statistical probability</span>
```

**Timeline:** 2 days
**Difficulty:** Easy (search-replace + UI updates)
**Impact:** CRITICAL – scientific integrity

---

### 1.3 Add MSM Uncertainty Quantification 🔴 **ESSENTIAL**

**Problem:** MSM reports point estimates without confidence intervals.

**Location:** `backend/ml/msm.py`

**Implementation:**

```python
# backend/ml/msm.py

def _bootstrap_msm_uncertainty(labels, n_states, lag_time, reversible, n_bootstrap=100):
    """Bootstrap confidence intervals for MSM transition probabilities."""

    n_frames = len(labels)
    T_bootstrap = []
    timescales_bootstrap = []

    for b in range(n_bootstrap):
        # Resample frames with replacement
        boot_indices = np.random.choice(n_frames, size=n_frames, replace=True)
        boot_labels = labels[boot_indices]

        # Build MSM on bootstrap sample
        T_boot = _build_transition_matrix(boot_labels, n_states, lag_time, reversible)

        if T_boot is not None:
            T_bootstrap.append(T_boot)

            # Compute implied timescales
            evals = np.sort(np.real(np.linalg.eigvals(T_boot)))[::-1]
            its = []
            for ev in evals[1:]:
                if 0 < ev < 1:
                    its.append(-lag_time / np.log(ev))
                else:
                    its.append(0.0)
            timescales_bootstrap.append(its)

    # Compute 95% confidence intervals
    T_bootstrap = np.array(T_bootstrap)  # (n_bootstrap, n_states, n_states)
    T_ci_lower = np.percentile(T_bootstrap, 2.5, axis=0)
    T_ci_upper = np.percentile(T_bootstrap, 97.5, axis=0)

    timescales_bootstrap = np.array(timescales_bootstrap)
    timescales_ci_lower = np.percentile(timescales_bootstrap, 2.5, axis=0).tolist()
    timescales_ci_upper = np.percentile(timescales_bootstrap, 97.5, axis=0).tolist()

    return {
        "transition_matrix_ci_lower": T_ci_lower.tolist(),
        "transition_matrix_ci_upper": T_ci_upper.tolist(),
        "implied_timescales_ci_lower": timescales_ci_lower,
        "implied_timescales_ci_upper": timescales_ci_upper,
        "n_bootstrap": n_bootstrap,
    }


def build_msm(...):
    # ... existing code ...

    # Add bootstrap uncertainty quantification
    uncertainty = _bootstrap_msm_uncertainty(labels, n_states, lag_time, reversible)

    return {
        # ... existing keys ...
        "uncertainty": uncertainty,
        "implied_timescales_ci": [
            {
                "index": i,
                "mean": timescales[i],
                "ci_lower": uncertainty["implied_timescales_ci_lower"][i],
                "ci_upper": uncertainty["implied_timescales_ci_upper"][i],
            }
            for i in range(min(5, len(timescales)))
        ],
    }
```

**Visualization Update:**

```python
# backend/visualization/plots.py

def plot_msm_implied_timescales_with_ci(msm_data):
    """Plot implied timescales with bootstrap confidence intervals."""

    fig = go.Figure()

    if "implied_timescales_ci" in msm_data:
        for item in msm_data["implied_timescales_ci"]:
            fig.add_trace(go.Scatter(
                x=[item["index"], item["index"]],
                y=[item["ci_lower"], item["ci_upper"]],
                mode='lines',
                line=dict(color='rgba(0,176,246,0.3)', width=8),
                showlegend=False,
                name='95% CI'
            ))
            fig.add_trace(go.Scatter(
                x=[item["index"]],
                y=[item["mean"]],
                mode='markers',
                marker=dict(color=ACCENT_CYAN, size=10),
                showlegend=False,
            ))

    # ... rest of plotting code ...
```

**Timeline:** 5 days
**Difficulty:** Medium
**Impact:** HIGH – enables quantitative uncertainty reporting

---

## Phase 2: HIGH PRIORITY (Weeks 4-7)

### 2.1 Implement Multiple Testing Correction

**Location:** `backend/utils/statistics_utils.py` (new file)

```python
# backend/utils/statistics_utils.py

from statsmodels.stats.multitest import fdrcorrection

def apply_fdr_correction(p_values, alpha=0.05, method='indep'):
    """Apply False Discovery Rate correction to p-values.

    Parameters
    ----------
    p_values : array-like
        Raw p-values.
    alpha : float
        Family-wise error rate (default 0.05).
    method : str
        'indep' (independent) or 'negcorr' (negative correlation).

    Returns
    -------
    dict
        {
            'significant': bool array,
            'corrected_alpha': float,
            'n_tests': int,
            'n_significant': int,
            'method': str
        }
    """
    reject, pvals_corrected = fdrcorrection(p_values, alpha=alpha, method=method)

    return {
        "significant": reject,
        "p_values_corrected": pvals_corrected,
        "corrected_alpha": np.max(pvals_corrected[reject]) if np.any(reject) else 0.0,
        "n_tests": len(p_values),
        "n_significant": int(np.sum(reject)),
        "method": f"FDR-BH ({method})",
    }
```

**Integration Example:**

```python
# backend/bio_inference/engine.py

def _detect_hinge_residues(self, result):
    """Hinge detection with multiple testing correction."""

    # ... existing heuristic detection ...

    # Add p-value calculation (permutation test)
    p_values = []
    for insight in insights:
        # Null hypothesis: random residue with same RMSF distribution
        null_scores = []
        for _ in range(1000):
            null_rmsf = np.random.permutation(rmsf)
            null_score = _compute_hinge_score(null_rmsf, insight["residues"][0])
            null_scores.append(null_score)

        obs_score = insight["heuristic_score"]
        p_val = np.mean(null_scores >= obs_score)
        p_values.append(p_val)

    # FDR correction
    from ..utils.statistics_utils import apply_fdr_correction
    correction = apply_fdr_correction(p_values, alpha=0.05)

    # Add corrected significance to insights
    for insight, is_sig in zip(insights, correction["significant"]):
        insight["significant_fdr_corrected"] = bool(is_sig)
        insight["fdr_alpha"] = correction["corrected_alpha"]

    return insights
```

**Timeline:** 6 days
**Difficulty:** Medium-Hard
**Impact:** HIGH – reduces false positive rate

---

### 2.2 Add GNN Cross-Validation

**Location:** `backend/gnn_models/residue_gnn.py`

```python
# backend/gnn_models/residue_gnn.py

def run_gnn_analysis_with_cv(universe, k_folds=5, **kwargs):
    """GNN analysis with k-fold cross-validation."""

    # ... collect data ...

    n_frames = coords.shape[0]
    fold_size = n_frames // k_folds

    fold_results = []

    for fold in range(k_folds):
        # Train/test split
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size
        test_mask = np.zeros(n_frames, dtype=bool)
        test_mask[test_start:test_end] = True
        train_mask = ~test_mask

        # Build graph on training frames
        train_coords = coords[train_mask]
        train_rmsf = compute_rmsf(train_coords)

        # Train model
        model = ResidueGNN(...)
        train_loss = _train_gnn(model, train_data, epochs=100)

        # Test model
        test_coords = coords[test_mask]
        test_rmsf_true = compute_rmsf(test_coords)
        test_rmsf_pred = model.predict(test_graph)
        test_mse = np.mean((test_rmsf_true - test_rmsf_pred) ** 2)

        fold_results.append({
            "fold": fold,
            "train_loss": train_loss,
            "test_mse": test_mse,
            "test_r2": r2_score(test_rmsf_true, test_rmsf_pred),
        })

    # Aggregate cross-validation results
    mean_test_mse = np.mean([f["test_mse"] for f in fold_results])
    mean_test_r2 = np.mean([f["test_r2"] for f in fold_results])

    return {
        # ... existing keys ...
        "cross_validation": {
            "k_folds": k_folds,
            "fold_results": fold_results,
            "mean_test_mse": round(mean_test_mse, 4),
            "mean_test_r2": round(mean_test_r2, 4),
            "generalization_warning": (
                "Low test R² suggests overfitting. Model may not generalize "
                "beyond this trajectory." if mean_test_r2 < 0.5 else
                "Reasonable generalization within trajectory."
            ),
        },
    }
```

**Timeline:** 5 days
**Difficulty:** Medium
**Impact:** HIGH – demonstrates model generalization

---

### 2.3 Validate Biological Detectors Against Benchmarks

**Location:** `backend/bio_inference/validation.py` (new file)

```python
# backend/bio_inference/validation.py

def validate_hinge_detector_against_dyndom(engine, test_trajectories, dyndom_hinges):
    """Validate hinge detector against DynDom database."""

    all_predictions = []
    all_labels = []

    for traj_id, (universe, known_hinges) in enumerate(zip(test_trajectories, dyndom_hinges)):
        # Run analysis
        result = run_full_analysis(universe)
        insights = engine.interpret(result)

        # Extract hinge predictions
        hinge_insights = [i for i in insights if i["type"] == "hinge_residue"]
        predicted_hinges = set()
        for h in hinge_insights:
            predicted_hinges.update(h["residues"])

        # Binary classification: is each residue a hinge?
        n_res = len(universe.select_atoms("protein and name CA"))
        for res_id in range(1, n_res + 1):
            all_predictions.append(1 if res_id in predicted_hinges else 0)
            all_labels.append(1 if res_id in known_hinges else 0)

    # Compute metrics
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='binary'
    )
    auc = roc_auc_score(all_labels, all_predictions)

    return {
        "detector": "hinge_residue",
        "n_test_systems": len(test_trajectories),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1, 3),
        "roc_auc": round(auc, 3),
        "benchmark_dataset": "DynDom",
    }
```

**Benchmark Datasets to Curate:**

1. **Hinges:** DynDom / MolMovDB (domain motion proteins)
2. **Stable Core:** ProTherm (mutational stability data - core residues resist mutation)
3. **Allosteric Pathways:** AlloSigMA2 database
4. **Druggable Pockets:** DUD-E / sc-PDB (validated binding sites)
5. **Cryptic Sites:** CryptoSite database

**Timeline:** 2 weeks (1 week data curation, 1 week implementation)
**Difficulty:** Hard (requires dataset curation)
**Impact:** **CRITICAL** – establishes scientific validity

---

### 2.4 Add Uncertainty Quantification Framework

**Location:** `backend/utils/uncertainty.py` (new file)

```python
# backend/utils/uncertainty.py

def bootstrap_confidence_interval(data, statistic_func, n_bootstrap=1000, alpha=0.05):
    """Generic bootstrap confidence interval calculator.

    Parameters
    ----------
    data : array-like or dict
        Input data (trajectory coordinates, time series, etc.).
    statistic_func : callable
        Function that computes the statistic of interest from data.
    n_bootstrap : int
        Number of bootstrap samples.
    alpha : float
        Confidence level (0.05 → 95% CI).

    Returns
    -------
    dict
        {
            'point_estimate': float,
            'ci_lower': float,
            'ci_upper': float,
            'standard_error': float,
            'n_bootstrap': int,
        }
    """

    # Point estimate
    point_est = statistic_func(data)

    # Bootstrap distribution
    boot_estimates = []
    n_samples = len(data) if hasattr(data, '__len__') else data.shape[0]

    for _ in range(n_bootstrap):
        # Resample with replacement
        boot_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_data = data[boot_indices] if isinstance(data, np.ndarray) else \
                    [data[i] for i in boot_indices]

        boot_est = statistic_func(boot_data)
        boot_estimates.append(boot_est)

    boot_estimates = np.array(boot_estimates)

    # Percentile CI
    ci_lower = np.percentile(boot_estimates, 100 * alpha / 2)
    ci_upper = np.percentile(boot_estimates, 100 * (1 - alpha / 2))
    se = np.std(boot_estimates, ddof=1)

    return {
        "point_estimate": float(point_est),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "standard_error": float(se),
        "n_bootstrap": n_bootstrap,
        "confidence_level": 1 - alpha,
    }


# Integration example:
def compute_rmsf_with_uncertainty(universe, n_bootstrap=100):
    """RMSF with bootstrap confidence intervals."""

    from .trajectory_utils import collect_ca_positions

    ca = select_ca_atoms(universe)
    positions = collect_ca_positions(universe, atoms=ca)

    def rmsf_func(pos_data):
        mean_pos = pos_data.mean(axis=0)
        fluctuations = np.sqrt(np.mean((pos_data - mean_pos) ** 2, axis=0))
        return fluctuations.mean(axis=1)  # per-residue RMSF

    # Bootstrap
    rmsf_point = rmsf_func(positions)

    rmsf_with_ci = []
    for residue_idx in range(len(ca)):
        # Bootstrap CI for this residue
        def residue_rmsf(pos):
            return rmsf_func(pos)[residue_idx]

        ci = bootstrap_confidence_interval(positions, residue_rmsf, n_bootstrap=n_bootstrap)
        rmsf_with_ci.append(ci)

    return {
        "resids": ca.resids.tolist(),
        "rmsf": [x["point_estimate"] for x in rmsf_with_ci],
        "rmsf_ci_lower": [x["ci_lower"] for x in rmsf_with_ci],
        "rmsf_ci_upper": [x["ci_upper"] for x in rmsf_with_ci],
        "rmsf_standard_error": [x["standard_error"] for x in rmsf_with_ci],
    }
```

**Timeline:** 7 days
**Difficulty:** Medium
**Impact:** HIGH – quantifies prediction uncertainty

---

## Phase 3: MEDIUM PRIORITY (Weeks 8-10)

### 3.1 Add Trajectory Comparison Framework

**Location:** `backend/analysis/multi_trajectory.py` (new file)

```python
def compare_multiple_replicas(universes, metric="rmsd"):
    """Compare multiple MD replicas for convergence assessment."""

    n_replicas = len(universes)

    # Compute metric for each replica
    replica_data = []
    for i, u in enumerate(universes):
        if metric == "rmsd":
            from .rmsd import compute_rmsd
            data = compute_rmsd(u)["rmsd"]
        elif metric == "rmsf":
            from .rmsf import compute_rmsf
            data = compute_rmsf(u)["rmsf"]
        else:
            raise ValueError(f"Unknown metric: {metric}")

        replica_data.append(data)

    # Cross-replica statistics
    replica_data = np.array(replica_data)  # (n_replicas, n_points)

    mean_across_replicas = replica_data.mean(axis=0)
    std_across_replicas = replica_data.std(axis=0, ddof=1)
    cv_across_replicas = std_across_replicas / (mean_across_replicas + 1e-8)

    # Convergence: low coefficient of variation
    well_converged = np.mean(cv_across_replicas) < 0.15

    return {
        "metric": metric,
        "n_replicas": n_replicas,
        "mean_across_replicas": mean_across_replicas.tolist(),
        "std_across_replicas": std_across_replicas.tolist(),
        "coefficient_of_variation": cv_across_replicas.tolist(),
        "mean_cv": round(float(np.mean(cv_across_replicas)), 4),
        "well_converged": bool(well_converged),
        "convergence_criterion": "mean CV < 0.15",
    }
```

**Timeline:** 4 days
**Difficulty:** Medium
**Impact:** Medium – improves convergence assessment

---

### 3.2 Implement Experimental Validation Hooks

**Location:** `backend/validation/experimental.py` (new file)

```python
def compare_to_experimental_bfactors(predicted_rmsf, pdb_structure, pdb_id):
    """Compare predicted RMSF to experimental B-factors."""

    from MDAnalysis import Universe
    import numpy as np
    from scipy.stats import pearsonr, spearmanr

    # Load PDB and extract B-factors
    u_exp = Universe(pdb_structure)
    ca_exp = u_exp.select_atoms("protein and name CA")
    bfactors_exp = ca_exp.bfactors

    # Convert RMSF to B-factor scale: B = (8π²/3) * RMSF²
    bfactors_pred = (8 * np.pi**2 / 3) * np.array(predicted_rmsf) ** 2

    # Align lengths (handle residue numbering differences)
    min_len = min(len(bfactors_exp), len(bfactors_pred))
    bfactors_exp = bfactors_exp[:min_len]
    bfactors_pred = bfactors_pred[:min_len]

    # Correlations
    pearson_r, pearson_p = pearsonr(bfactors_exp, bfactors_pred)
    spearman_r, spearman_p = spearmanr(bfactors_exp, bfactors_pred)

    # RMSE
    rmse = np.sqrt(np.mean((bfactors_exp - bfactors_pred) ** 2))

    return {
        "pdb_id": pdb_id,
        "n_residues_compared": min_len,
        "pearson_correlation": round(pearson_r, 3),
        "pearson_p_value": round(pearson_p, 6),
        "spearman_correlation": round(spearman_r, 3),
        "spearman_p_value": round(spearman_p, 6),
        "rmse": round(rmse, 2),
        "interpretation": (
            "Excellent agreement" if pearson_r > 0.7 else
            "Good agreement" if pearson_r > 0.5 else
            "Moderate agreement" if pearson_r > 0.3 else
            "Poor agreement"
        ),
    }
```

**Timeline:** 5 days
**Difficulty:** Medium
**Impact:** HIGH – enables experimental validation

---

## Phase 4: LOW PRIORITY (Weeks 11-12)

### 4.1 Bayesian MSM (Optional)

Implementation of Dirichlet posterior on transition probabilities using PyMC.

**Benefits:**

- Credible intervals on kinetic parameters
- Model selection via Bayes factors
- Handles sparse data better

**Complexity:** High
**Timeline:** 2 weeks

---

### 4.2 Transfer Learning for GNN (Optional)

Pre-train GNN on multiple protein families, fine-tune on target.

**Benefits:**

- Better generalization
- Reduced overfitting
- Cross-protein applicability

**Complexity:** Very High
**Timeline:** 1 month

---

## Testing Strategy

### Unit Tests (Add to `tests/` directory)

```python
# tests/test_prs_hessian.py

def test_prs_hessian_symmetry():
    """Test that Hessian-based PRS response matrix is symmetric."""
    from backend.analysis.prs_hessian import compute_prs_hessian

    universe = load_test_trajectory()
    result = compute_prs_hessian(universe)

    response = np.array(result["response_matrix"])
    assert np.allclose(response, response.T, atol=1e-6)


def test_prs_covariance_vs_hessian_ranking():
    """Test that covariance and Hessian PRS give similar rankings."""
    from backend.analysis.prs import compute_prs as compute_prs_cov
    from backend.analysis.prs_hessian import compute_prs_hessian

    universe = load_test_trajectory()

    result_cov = compute_prs_cov(universe)
    result_hess = compute_prs_hessian(universe)

    eff_cov = result_cov["effector_scores"]
    eff_hess = result_hess["effector_scores"]

    # Spearman rank correlation should be high
    from scipy.stats import spearmanr
    rho, _ = spearmanr(eff_cov, eff_hess)

    assert rho > 0.7, f"Expected high rank correlation, got {rho}"
```

### Integration Tests

```python
# tests/integration/test_full_pipeline.py

def test_full_pipeline_with_uncertainty():
    """Test that full analysis pipeline includes uncertainty quantification."""

    universe = load_test_trajectory()
    result = run_full_analysis(universe, enable_uncertainty=True)

    # Check that uncertainty fields are present
    assert "rmsf_ci_lower" in result.rmsf
    assert "uncertainty" in result.msm
    assert "cross_validation" in result.gnn_results
```

### Validation Tests

```python
# tests/validation/test_bio_inference_benchmarks.py

def test_hinge_detector_precision():
    """Test hinge detector against benchmark hinges."""
    from backend.bio_inference.validation import validate_hinge_detector_against_dyndom

    # Load benchmark data
    test_trajectories, known_hinges = load_dyndom_benchmark()

    engine = BiologicalInferenceEngine()
    metrics = validate_hinge_detector_against_dyndom(engine, test_trajectories, known_hinges)

    # Require at least 50% precision to pass
    assert metrics["precision"] > 0.5, f"Precision too low: {metrics['precision']}"
```

---

## Documentation Updates

### 1. README.md Additions

```markdown
## Scientific Validation

### Uncertainty Quantification
All derived quantities now include bootstrap confidence intervals. Enable with `enable_uncertainty=True`.

### Biological Inference Caveats
Detector outputs are **heuristic scores** for hypothesis generation, not validated predictions.
Cross-reference with:
- Conservation scores (ConSurf, Rate4Site)
- Experimental mutagenesis data
- AlphaFold pLDDT confidence scores

### Validation Against Experimental Data
Compare your results to:
- Crystallographic B-factors: `compare_to_experimental_bfactors()`
- Mutational ΔΔG: `validate_against_protherm()`
- NMR dynamics: `compare_to_nmr_order_parameters()`
```

### 2. New METHODS.md File

Detailed mathematical descriptions of:

- Schlitter entropy formula with derivation
- ANM Hessian construction
- PRS using Hessian inversion vs. covariance
- MSM Chapman-Kolmogorov test
- Bootstrap confidence interval calculation
- FDR multiple testing correction

### 3. VALIDATION.md File

Benchmark results on:

- DynDom (hinge detection)
- AlloSigMA2 (allosteric pathways)
- DUD-E (druggable pockets)
- ProTherm (stability predictions)

---

## Success Metrics

### Phase 1 (Critical) - Must Achieve:

- PRS uses Hessian inversion (or clearly labeled as covariance-based)
- No use of "confidence" for uncalibrated scores
- MSM reports uncertainty quantification
- All claims include appropriate caveats

### Phase 2 (High Priority) - Target:

- FDR correction available for all per-residue tests
- GNN cross-validation shows test R² > 0.4
- At least 3 biological detectors validated (precision > 0.5)
- Bootstrap CIs available for RMSD, RMSF, entropy

### Phase 3 (Medium Priority) - Desirable:

- Multi-replica comparison framework functional
- Experimental B-factor correlation tool working
- Null model permutation tests for pathways

### Publication Readiness Checklist:

- Methods section written (10+ pages)
- Validation section with benchmarks
- Code fully documented (docstrings + Sphinx)
- Tutorial notebooks (3-5 examples)
- Unit test coverage > 70%
- Comparison to existing tools (GROMACS, PyEMMA, ProDy)

---

## Resource Requirements

### Compute:

- **Phase 1:** Local machine (CPU sufficient)
- **Phase 2 (validation):** GPU for GNN training, otherwise CPU
- **Phase 3:** Cluster access for multi-replica simulations

### Time Estimate:

- **1 full-time developer:** 12 weeks for Phases 1-2
- **2 developers:** 6-7 weeks for Phases 1-2

### Dependencies to Add:

```txt
# requirements_dev.txt
statsmodels>=0.14.0  # FDR correction
scikit-learn>=1.3.0  # Cross-validation, metrics
pymc>=5.0.0  # Bayesian MSM (optional)
pytest>=7.4.0  # Unit testing
pytest-cov>=4.1.0  # Coverage
sphinx>=7.0.0  # Documentation
```

---

## Risk Mitigation

### Risk: PRS rankings change significantly with Hessian method

**Mitigation:**

- Validate on literature PRS studies first
- Offer both methods with documented trade-offs
- Emphasize that qualitative rankings matter more than absolute magnitudes

### Risk: Biological detector validation shows poor precision

**Mitigation:**

- Frame as "feature detection" not "prediction"
- Emphasize hypothesis-generation role
- Report both precision and recall (favor recall for screening)

### Risk: Bootstrap CI calculation too slow for large proteins

**Mitigation:**

- Implement optional uncertainty (off by default)
- Use n_bootstrap=100 instead of 1000 for large systems
- Parallelize bootstrap loops with joblib

---

## Conclusion

This roadmap prioritizes **scientific integrity** (Phase 1 Critical) over **feature additions**. The platform already has excellent physics-based implementations—the focus must be on:

1. **Fixing theoretical issues** (PRS)
2. **Proper uncertainty quantification** (MSM, all metrics)
3. **Appropriate framing** (heuristic scores not confidences)
4. **Experimental validation** (benchmarks, not just claims)

**After Phase 1-2 completion**, the platform will be ready for submission to a computational methods journal.

---

**Next Steps:** Begin with PRS Hessian implementation (1.1) while simultaneously updating terminology (1.2).