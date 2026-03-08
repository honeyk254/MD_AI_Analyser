"""
Convergence Assessment.
Block averaging and autocorrelation analysis to evaluate
whether the simulation has converged.
"""
import numpy as np


def compute_convergence(universe, **kwargs):
    """
    Assess trajectory convergence using block averaging, autocorrelation,
    and cosine content of principal components.

    Returns dict with:
        - rmsd_block_averages: block-averaged RMSD statistics
        - rg_block_averages: block-averaged Rg statistics
        - autocorrelation_rmsd: autocorrelation function of RMSD
        - cosine_content: cosine content of first PCA components
        - convergence_score: overall 0-1 convergence assessment
        - recommendations: text suggestions
    """
    try:
        ca = universe.select_atoms("protein and name CA")
        if len(ca) == 0:
            ca = universe.select_atoms("all")

        n_frames = len(universe.trajectory)
        if n_frames < 20:
            return {"error": "Too few frames for convergence assessment"}

        # ── Collect RMSD and Rg time series ────────────────────
        from MDAnalysis.analysis.rms import RMSD as MDA_RMSD
        ref = universe.copy()
        ref.trajectory[0]
        R = MDA_RMSD(universe, ref, select="backbone", ref_frame=0)
        R.run()
        rmsd_ts = R.results.rmsd[:, 2]
        times = R.results.rmsd[:, 1].tolist()

        # Rg
        rg_ts = []
        for ts in universe.trajectory:
            rg_ts.append(ca.radius_of_gyration())
        rg_ts = np.array(rg_ts)

        # ── Block Averaging ────────────────────────────────────
        rmsd_blocks = _block_average(rmsd_ts)
        rg_blocks = _block_average(rg_ts)

        # ── Autocorrelation of RMSD ────────────────────────────
        acf_rmsd = _autocorrelation(rmsd_ts, max_lag=min(n_frames // 2, 200))

        # ── Cosine Content of PCA ──────────────────────────────
        coords = []
        for ts_frame in universe.trajectory:
            coords.append(ca.positions.flatten().copy())
        coords = np.array(coords)

        cosine_content = []
        try:
            from sklearn.decomposition import PCA
            n_comp = min(5, coords.shape[0] - 1, coords.shape[1])
            if n_comp >= 1:
                pca = PCA(n_components=n_comp)
                projections = pca.fit_transform(coords)
                for i in range(n_comp):
                    cc = _cosine_content_pc(projections[:, i])
                    cosine_content.append({
                        "pc": i + 1,
                        "cosine_content": round(float(cc), 4),
                        "converged": cc < 0.5,
                    })
        except Exception:
            pass

        # ── Overall Convergence Score ──────────────────────────
        score = 0.0
        n_checks = 0

        # Check 1: RMSD stabilization (last 20% vs first 20%)
        split = max(1, n_frames // 5)
        first_mean = np.mean(rmsd_ts[:split])
        last_mean = np.mean(rmsd_ts[-split:])
        last_std = np.std(rmsd_ts[-split:])
        rmsd_drift = abs(last_mean - first_mean) / (last_mean + 1e-8)
        rmsd_score = max(0, 1.0 - rmsd_drift * 2)
        score += rmsd_score
        n_checks += 1

        # Check 2: Rg stabilization
        rg_first = np.mean(rg_ts[:split])
        rg_last = np.mean(rg_ts[-split:])
        rg_drift = abs(rg_last - rg_first) / (rg_last + 1e-8)
        rg_score = max(0, 1.0 - rg_drift * 2)
        score += rg_score
        n_checks += 1

        # Check 3: Block average SEM convergence
        if rmsd_blocks:
            last_sem = rmsd_blocks[-1]["sem"]
            first_sem = rmsd_blocks[0]["sem"] if rmsd_blocks[0]["sem"] > 0 else 1.0
            block_score = min(1.0, first_sem / (last_sem + 1e-8) * 0.3)
            score += block_score
            n_checks += 1

        # Check 4: Cosine content
        if cosine_content:
            cc_avg = np.mean([c["cosine_content"] for c in cosine_content[:3]])
            cc_score = max(0, 1.0 - cc_avg)
            score += cc_score
            n_checks += 1

        overall_score = round(score / max(n_checks, 1), 3)

        # Recommendations
        recommendations = []
        if rmsd_drift > 0.15:
            recommendations.append("RMSD has not stabilized — consider extending simulation")
        if rg_drift > 0.1:
            recommendations.append("Radius of gyration is still drifting — protein may not be equilibrated")
        if cosine_content and cosine_content[0]["cosine_content"] > 0.7:
            recommendations.append("PC1 has high cosine content — trajectory may be sampling a random walk, not converged dynamics")
        if overall_score > 0.7:
            recommendations.append("Trajectory appears reasonably well-converged")
        elif overall_score < 0.4:
            recommendations.append("Trajectory shows poor convergence — results should be interpreted with caution")

        return {
            "time": times,
            "rmsd_block_averages": rmsd_blocks,
            "rg_block_averages": rg_blocks,
            "autocorrelation_rmsd": acf_rmsd,
            "cosine_content": cosine_content,
            "convergence_score": overall_score,
            "rmsd_drift": round(float(rmsd_drift), 4),
            "rg_drift": round(float(rg_drift), 4),
            "recommendations": recommendations,
        }

    except Exception as e:
        return {"error": str(e)}


def _block_average(data, n_blocks_list=None):
    """Compute block averages with increasing block sizes."""
    n = len(data)
    if n_blocks_list is None:
        n_blocks_list = [2, 4, 5, 8, 10, 20]

    results = []
    for nb in n_blocks_list:
        if nb > n:
            continue
        block_size = n // nb
        if block_size < 2:
            continue
        block_means = []
        for i in range(nb):
            start = i * block_size
            end = start + block_size
            block_means.append(np.mean(data[start:end]))
        block_means = np.array(block_means)
        sem = np.std(block_means) / np.sqrt(nb)
        results.append({
            "n_blocks": nb,
            "block_size": block_size,
            "mean": round(float(np.mean(block_means)), 4),
            "sem": round(float(sem), 6),
        })
    return results


def _autocorrelation(data, max_lag=200):
    """Compute normalized autocorrelation function."""
    n = len(data)
    mean = np.mean(data)
    var = np.var(data)
    if var == 0:
        return []

    data_centered = data - mean
    max_lag = min(max_lag, n - 1)
    acf = []
    for lag in range(0, max_lag, max(1, max_lag // 50)):
        c = np.mean(data_centered[:n - lag] * data_centered[lag:])
        acf.append({
            "lag": lag,
            "acf": round(float(c / var), 4),
        })
    return acf


def _cosine_content_pc(projection):
    """
    Compute cosine content of a PC projection.
    cos_content = 2/T * (integral cos(pi*t/T) * q(t) dt)^2 / (integral q(t)^2 dt)
    Value close to 1 means random diffusion, close to 0 means converged sampling.
    """
    n = len(projection)
    if n < 2:
        return 1.0

    t = np.arange(n)
    cos_wave = np.cos(np.pi * t / (n - 1))

    numerator = (np.sum(projection * cos_wave)) ** 2
    denominator = np.sum(projection ** 2)

    if denominator == 0:
        return 1.0

    return (2.0 / n) * numerator / denominator
