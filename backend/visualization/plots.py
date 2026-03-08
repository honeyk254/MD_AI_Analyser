from __future__ import annotations

"""
Plotly chart generators for all analysis modules.

Each ``_plot_*`` function returns a Plotly ``Figure`` (or *None* on
invalid/empty data).  The :func:`generate_all_plots` orchestrator calls
every generator and serialises successful figures to JSON.
"""

import logging
from typing import Any, Optional

import numpy as np
import plotly.graph_objects as go
import plotly.express as px  # noqa: F401 — kept for potential downstream use
from plotly.subplots import make_subplots

from ..utils.plotting_utils import (
    apply_dark_theme,
    safe_plot,
    ACCENT_CYAN,
    ACCENT_RED,
    ACCENT_TEAL,
    ACCENT_YELLOW,
    ACCENT_PURPLE,
    ACCENT_PINK,
    ACCENT_ORANGE,
    ACCENT_GREEN,
    ACCENT_LIGHT_BLUE,
    ACCENT_DARK_PURPLE,
    ACCENT_DARK_TEAL,
    ACCENT_SAGE,
    COMMUNITY_COLORS,
    PAPER_BG,
    PLOT_BG,
    TEXT_COLOR,
)

logger = logging.getLogger("md_ai_analyzer")


# ──────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────

def generate_all_plots(result: Any) -> dict[str, str]:
    """Generate all visualisation plots from analysis results.

    Iterates over every registered ``(name, generator_func, data)`` triple.
    Each generator is decorated with :func:`safe_plot`, so individual
    failures are logged as warnings and do **not** abort the remaining
    plots.

    Parameters
    ----------
    result : AnalysisResult
        The completed analysis result object whose attributes hold
        per-module output dicts.

    Returns
    -------
    dict[str, str]
        Mapping of plot names to Plotly JSON strings.  Only plots that
        produced a non-*None* figure are included.
    """
    plots: dict[str, str] = {}

    generators: list[tuple[str, Any, Any]] = [
        ("rmsd_plot", _plot_rmsd, result.rmsd),
        ("rmsf_plot", _plot_rmsf, result.rmsf),
        ("rg_plot", _plot_rg, result.rg),
        ("ss_plot", _plot_secondary_structure, result.secondary_structure),
        ("hbond_plot", _plot_hbonds, result.hbonds),
        ("salt_bridges_plot", _plot_salt_bridges, result.salt_bridges),
        ("contact_map", _plot_contact_map, result.contacts),
        ("pca_plot", _plot_pca, result.pca),
        ("dccm_plot", _plot_dccm, result.dccm),
        ("fel_plot", _plot_free_energy, result.free_energy),
        ("clustering_plot", _plot_clustering, result.clustering),
        ("sasa_plot", _plot_sasa, result.sasa),
        ("tica_plot", _plot_tica, result.tica),
        ("dimensionality_plot", _plot_dimensionality, result.dimensionality),
        ("dimensionality_3d_plot", _plot_dimensionality_3d, result.dimensionality),
        ("gnn_plot", _plot_gnn, result.gnn_results),
        ("transformer_plot", _plot_transformer, result.transformer_results),
        ("msm_plot", _plot_msm, result.msm),
        # Part A plots
        ("water_bridges_plot", _plot_water_bridges, result.water_bridges),
        ("energy_plot", _plot_energy_decomposition, result.energy_decomposition),
        ("prs_plot", _plot_prs, result.prs),
        ("nma_plot", _plot_nma, result.nma),
        ("entropy_plot", _plot_entropy, result.entropy),
        ("ifp_plot", _plot_ifp, result.interaction_fingerprints),
        ("tunnel_plot", _plot_tunnels, result.tunnels),
        ("vae_plot", _plot_vae, result.vae),
        ("dynamic_network_plot", _plot_dynamic_network, result.dynamic_network),
        # Phase 4 new plots
        ("convergence_plot", _plot_convergence, result.convergence),
        ("binding_kinetics_plot", _plot_binding_kinetics, result.binding_kinetics),
        ("network_graph_plot", _plot_allosteric_network, result.allosteric),
        ("training_loss_plot", _plot_training_losses, {
            "gnn": result.gnn_results,
            "transformer": result.transformer_results,
            "vae": result.vae,
        }),
    ]

    for name, func, data in generators:
        if isinstance(data, dict) and not data.get("error"):
            fig = func(data)
            if fig is not None:
                plots[name] = fig.to_json()

    return plots


# ──────────────────────────────────────────────────────────────────
# Core analysis plots
# ──────────────────────────────────────────────────────────────────

@safe_plot
def _plot_rmsd(data: dict[str, Any]) -> Optional[go.Figure]:
    """RMSD time-series with equilibration and mean indicators.

    Parameters
    ----------
    data : dict
        Must contain ``time`` and ``rmsd`` lists.

    Returns
    -------
    go.Figure or None
    """
    if not data.get("time") or not data.get("rmsd"):
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["time"], y=data["rmsd"],
        mode="lines", name="RMSD",
        line=dict(color=ACCENT_CYAN, width=1.5),
        fill="tozeroy", fillcolor="rgba(0,212,255,0.1)",
    ))
    # Equilibration line
    if data.get("equilibration_frame", 0) > 0 and data["time"]:
        equil_time = data["time"][data["equilibration_frame"]]
        fig.add_vline(
            x=equil_time, line_dash="dash", line_color=ACCENT_RED,
            annotation_text="Equilibration",
        )
    # Mean line
    fig.add_hline(
        y=data["mean_rmsd"], line_dash="dot", line_color=ACCENT_YELLOW,
        annotation_text=f"Mean: {data['mean_rmsd']:.2f} \u00c5",
    )
    return apply_dark_theme(fig, "RMSD Over Time", "Time (ps)", "RMSD (\u00c5)")


@safe_plot
def _plot_rmsf(data: dict[str, Any]) -> Optional[go.Figure]:
    """Per-residue RMSF bar chart with colour coding.

    Parameters
    ----------
    data : dict
        Must contain ``rmsf`` and ``resids`` lists of equal length.

    Returns
    -------
    go.Figure or None
    """
    rmsf = data.get("rmsf", [])
    resids = data.get("resids", [])
    if not rmsf or not resids or len(rmsf) != len(resids):
        return None

    mean_r = data.get("mean_rmsf", float(np.mean(rmsf)))
    std_r = data.get("std_rmsf", float(np.std(rmsf)))
    colors: list[str] = []
    for v in rmsf:
        if v > mean_r + std_r:
            colors.append(ACCENT_RED)
        elif v < mean_r - 0.5 * std_r:
            colors.append(ACCENT_TEAL)
        else:
            colors.append(ACCENT_CYAN)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=resids, y=rmsf, marker_color=colors, name="RMSF"))
    fig.add_hline(
        y=mean_r, line_dash="dot", line_color=ACCENT_YELLOW,
        annotation_text=f"Mean: {mean_r:.2f} \u00c5",
    )
    return apply_dark_theme(fig, "Per-Residue RMSF", "Residue ID", "RMSF (\u00c5)")


@safe_plot
def _plot_rg(data: dict[str, Any]) -> Optional[go.Figure]:
    """Radius of gyration time-series.

    Parameters
    ----------
    data : dict
        Must contain ``time`` and ``rg`` lists.

    Returns
    -------
    go.Figure or None
    """
    if not data.get("time") or not data.get("rg"):
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["time"], y=data["rg"],
        mode="lines", name="Rg",
        line=dict(color=ACCENT_PURPLE, width=1.5),
        fill="tozeroy", fillcolor="rgba(162,155,254,0.1)",
    ))
    return apply_dark_theme(
        fig,
        f"Radius of Gyration (Trend: {data.get('trend', 'N/A')})",
        "Time (ps)", "Rg (\u00c5)",
    )


@safe_plot
def _plot_secondary_structure(data: dict[str, Any]) -> Optional[go.Figure]:
    """Secondary structure fraction evolution over frames.

    Parameters
    ----------
    data : dict
        Must contain ``helix_fraction``, ``sheet_fraction``, ``coil_fraction``.

    Returns
    -------
    go.Figure or None
    """
    n_frames = len(data.get("helix_fraction", []))
    if n_frames == 0:
        return None

    frames = list(range(n_frames))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frames, y=data["helix_fraction"], mode="lines",
        name="\u03b1-Helix", line=dict(color=ACCENT_RED, width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=frames, y=data["sheet_fraction"], mode="lines",
        name="\u03b2-Sheet", line=dict(color=ACCENT_TEAL, width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=frames, y=data["coil_fraction"], mode="lines",
        name="Coil", line=dict(color=ACCENT_YELLOW, width=1.5),
    ))
    return apply_dark_theme(fig, "Secondary Structure Evolution", "Frame", "Fraction")


@safe_plot
def _plot_hbonds(data: dict[str, Any]) -> Optional[go.Figure]:
    """Hydrogen bond count over time.

    Parameters
    ----------
    data : dict
        Must contain ``time`` and ``n_hbonds`` lists.

    Returns
    -------
    go.Figure or None
    """
    if not data.get("time") or not data.get("n_hbonds"):
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["time"], y=data["n_hbonds"],
        mode="lines", name="H-bonds",
        line=dict(color=ACCENT_DARK_PURPLE, width=1.5),
        fill="tozeroy", fillcolor="rgba(108,92,231,0.1)",
    ))
    return apply_dark_theme(
        fig,
        f"Hydrogen Bonds (Mean: {data.get('mean_hbonds', 0):.1f})",
        "Time (ps)", "Number of H-bonds",
    )


@safe_plot
def _plot_salt_bridges(data: dict[str, Any]) -> Optional[go.Figure]:
    """Salt bridge count over time and top pair occupancies.

    Parameters
    ----------
    data : dict
        Should contain ``pairs`` and/or ``total_per_frame`` / ``time``.

    Returns
    -------
    go.Figure or None
    """
    pairs = data.get("pairs", [])
    time = data.get("time", [])
    total_per_frame = data.get("total_per_frame", [])
    if not pairs and not total_per_frame:
        return None

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Salt Bridges Over Time", "Top Salt Bridge Pairs (Occupancy)"),
        row_heights=[0.5, 0.5],
    )
    if time and total_per_frame:
        fig.add_trace(go.Scatter(
            x=time, y=total_per_frame,
            mode="lines", name="Salt Bridges/Frame",
            line=dict(color=ACCENT_YELLOW, width=1.5),
            fill="tozeroy", fillcolor="rgba(255,217,61,0.1)",
        ), row=1, col=1)
    if pairs:
        labels = [f"{p['positive']}-{p['negative']}" for p in pairs[:20]]
        occupancies = [p["occupancy"] for p in pairs[:20]]
        fig.add_trace(go.Bar(
            x=labels, y=occupancies,
            marker_color="#fdcb6e", name="Occupancy",
        ), row=2, col=1)
    return apply_dark_theme(
        fig,
        f"Salt Bridges (Mean: {data.get('mean_salt_bridges', 0):.1f}/frame)",
        "", "", height=600,
    )


@safe_plot
def _plot_contact_map(data: dict[str, Any]) -> Optional[go.Figure]:
    """Average residue-residue contact frequency heatmap.

    Parameters
    ----------
    data : dict
        Must contain ``contact_map`` and ``resids``.

    Returns
    -------
    go.Figure or None
    """
    cmap = data.get("contact_map", [])
    resids = data.get("resids", [])
    if not cmap or not resids:
        return None

    fig = go.Figure(data=go.Heatmap(
        z=cmap, x=resids, y=resids,
        colorscale="Viridis",
        colorbar=dict(title="Contact Freq"),
    ))
    return apply_dark_theme(fig, "Average Contact Map", "Residue ID", "Residue ID")


@safe_plot
def _plot_pca(data: dict[str, Any]) -> Optional[go.Figure]:
    """PCA projection scatter and explained variance bar chart.

    Parameters
    ----------
    data : dict
        Must contain ``projections`` and ``cumulative_variance``.

    Returns
    -------
    go.Figure or None
    """
    proj = np.array(data.get("projections", []))
    if proj.ndim < 2 or proj.shape[1] < 2:
        return None

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("PCA Projection", "Explained Variance"),
    )
    fig.add_trace(go.Scatter(
        x=proj[:, 0], y=proj[:, 1],
        mode="markers", name="Conformations",
        marker=dict(
            color=list(range(len(proj))), colorscale="Plasma",
            size=4, colorbar=dict(title="Frame", x=0.45),
        ),
    ), row=1, col=1)

    cum_var = data["cumulative_variance"]
    fig.add_trace(go.Bar(
        x=list(range(1, len(cum_var) + 1)), y=cum_var,
        marker_color=ACCENT_CYAN, name="Cumulative Variance",
    ), row=1, col=2)

    return apply_dark_theme(fig, "Principal Component Analysis")


@safe_plot
def _plot_dccm(data: dict[str, Any]) -> Optional[go.Figure]:
    """Dynamic Cross-Correlation Matrix heatmap.

    Parameters
    ----------
    data : dict
        Must contain ``dccm`` and ``resids``.

    Returns
    -------
    go.Figure or None
    """
    dccm = data.get("dccm", [])
    resids = data.get("resids", [])
    if not dccm or not resids:
        return None

    fig = go.Figure(data=go.Heatmap(
        z=dccm, x=resids, y=resids,
        colorscale="RdBu_r", zmid=0,
        colorbar=dict(title="Correlation"),
    ))
    return apply_dark_theme(
        fig, "Dynamic Cross-Correlation Matrix", "Residue ID", "Residue ID",
    )


@safe_plot
def _plot_free_energy(data: dict[str, Any]) -> Optional[go.Figure]:
    """Free energy landscape contour plot with minima markers.

    Parameters
    ----------
    data : dict
        Must contain ``fel``, ``pc1_edges``, ``pc2_edges``.

    Returns
    -------
    go.Figure or None
    """
    if not data.get("fel") or not data.get("pc1_edges") or not data.get("pc2_edges"):
        return None

    fig = go.Figure(data=go.Contour(
        z=data["fel"],
        x=data["pc1_edges"][:-1],
        y=data["pc2_edges"][:-1],
        colorscale="Magma_r",
        colorbar=dict(title="\u0394G (kJ/mol)"),
        contours=dict(showlabels=True),
    ))
    minima = data.get("minima", [])
    if minima:
        fig.add_trace(go.Scatter(
            x=[m["pc1"] for m in minima],
            y=[m["pc2"] for m in minima],
            mode="markers+text",
            text=[f"Min {i + 1}" for i in range(len(minima))],
            marker=dict(color=ACCENT_RED, size=12, symbol="star"),
            textposition="top center",
            textfont=dict(color=ACCENT_RED),
            name="Minima",
        ))
    return apply_dark_theme(fig, "Free Energy Landscape", "PC1", "PC2")


@safe_plot
def _plot_clustering(data: dict[str, Any]) -> Optional[go.Figure]:
    """Conformational cluster assignment scatter.

    Parameters
    ----------
    data : dict
        Must contain ``labels`` and ``n_clusters``.

    Returns
    -------
    go.Figure or None
    """
    labels = data.get("labels")
    if labels is None:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(labels))), y=labels,
        mode="markers", name="Cluster",
        marker=dict(color=labels, colorscale="Set1", size=3),
    ))
    return apply_dark_theme(
        fig,
        f"Conformational Clusters (k={data['n_clusters']}, "
        f"silhouette={data.get('silhouette_score', 0):.2f})",
        "Frame", "Cluster ID",
    )


@safe_plot
def _plot_sasa(data: dict[str, Any]) -> Optional[go.Figure]:
    """Total solvent-accessible surface area over time.

    Parameters
    ----------
    data : dict
        Must contain ``time`` and ``total_sasa``.

    Returns
    -------
    go.Figure or None
    """
    if not data.get("time") or not data.get("total_sasa"):
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["time"], y=data["total_sasa"],
        mode="lines", name="Total SASA",
        line=dict(color=ACCENT_PINK, width=1.5),
        fill="tozeroy", fillcolor="rgba(253,121,168,0.1)",
    ))
    return apply_dark_theme(
        fig, "Solvent Accessible Surface Area", "Time (ps)", "SASA (nm\u00b2)",
    )


@safe_plot
def _plot_dimensionality(data: dict[str, Any]) -> Optional[go.Figure]:
    """Side-by-side 2-D PCA, UMAP, and t-SNE projections.

    Parameters
    ----------
    data : dict
        May contain ``pca_2d``, ``umap_2d``, ``tsne_2d``.

    Returns
    -------
    go.Figure or None
    """
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("PCA 2D", "UMAP 2D", "t-SNE 2D"),
    )

    for idx, (key, title) in enumerate([
        ("pca_2d", "PCA"), ("umap_2d", "UMAP"), ("tsne_2d", "t-SNE"),
    ]):
        proj = data.get(key, [])
        if proj and len(proj) > 0:
            arr = np.array(proj)
            fig.add_trace(go.Scatter(
                x=arr[:, 0], y=arr[:, 1],
                mode="markers", name=title,
                marker=dict(
                    color=list(range(len(arr))), colorscale="Plasma", size=3,
                ),
            ), row=1, col=idx + 1)

    return apply_dark_theme(fig, "Dimensionality Reduction")


@safe_plot
def _plot_gnn(data: dict[str, Any]) -> Optional[go.Figure]:
    """GNN-learned per-residue importance bar chart.

    Parameters
    ----------
    data : dict
        Should contain ``resids`` and ``residue_importance``.

    Returns
    -------
    go.Figure or None
    """
    resids = data.get("resids", [])
    importance = data.get("residue_importance", [])
    if not resids or not importance:
        return None

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=resids, y=importance,
        marker_color=ACCENT_ORANGE, name="GNN Importance",
    ))
    return apply_dark_theme(
        fig, "GNN Residue Importance Scores", "Residue ID", "Importance",
    )


@safe_plot
def _plot_transformer(data: dict[str, Any]) -> Optional[go.Figure]:
    """Transformer temporal importance with transition markers.

    Parameters
    ----------
    data : dict
        Should contain ``temporal_importance`` and optionally
        ``transition_frames``.

    Returns
    -------
    go.Figure or None
    """
    temporal = data.get("temporal_importance", [])
    if not temporal:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(temporal))), y=temporal,
        mode="lines", name="Frame Importance",
        line=dict(color=ACCENT_DARK_TEAL, width=1.5),
        fill="tozeroy", fillcolor="rgba(0,206,201,0.1)",
    ))
    transitions = data.get("transition_frames", [])
    if transitions:
        fig.add_trace(go.Scatter(
            x=[t["frame"] for t in transitions],
            y=[t["frame_importance"] for t in transitions],
            mode="markers", name="Transitions",
            marker=dict(color=ACCENT_RED, size=10, symbol="diamond"),
        ))
    return apply_dark_theme(
        fig, "Transformer: Temporal Importance & Transitions",
        "Frame", "Importance",
    )


@safe_plot
def _plot_msm(data: dict[str, Any]) -> Optional[go.Figure]:
    """Markov State Model transition probability heatmap.

    Parameters
    ----------
    data : dict
        Must contain ``transition_matrix``.

    Returns
    -------
    go.Figure or None
    """
    T = np.array(data.get("transition_matrix", []))
    if T.size == 0:
        return None

    fig = go.Figure(data=go.Heatmap(
        z=T, colorscale="Blues",
        colorbar=dict(title="Probability"),
    ))
    return apply_dark_theme(
        fig, "MSM Transition Probability Matrix", "To State", "From State",
    )


@safe_plot
def _plot_tica(data: dict[str, Any]) -> Optional[go.Figure]:
    """tICA projection scatter and implied timescales bar chart.

    Parameters
    ----------
    data : dict
        Must contain ``projections``; optionally ``timescales``.

    Returns
    -------
    go.Figure or None
    """
    projections = data.get("projections", [])
    timescales = data.get("timescales", [])
    if not projections:
        return None

    arr = np.array(projections)
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("tICA Projection (tIC1 vs tIC2)", "Implied Timescales"),
    )
    if arr.shape[1] >= 2:
        fig.add_trace(go.Scatter(
            x=arr[:, 0], y=arr[:, 1],
            mode="markers", name="Frames",
            marker=dict(
                color=list(range(len(arr))), colorscale="Viridis",
                size=4, colorbar=dict(title="Frame", x=0.45),
            ),
        ), row=1, col=1)
    if timescales:
        valid_ts = [
            (i + 1, t) for i, t in enumerate(timescales)
            if t != float("inf") and t > 0
        ]
        if valid_ts:
            fig.add_trace(go.Bar(
                x=[f"tIC{i}" for i, _ in valid_ts],
                y=[t for _, t in valid_ts],
                marker_color=ACCENT_GREEN, name="Timescale",
            ), row=1, col=2)

    return apply_dark_theme(
        fig, f"tICA Analysis (lag={data.get('lag_time', '?')})",
    )


# ──────────────────────────────────────────────────────────────────
# Part A -- New Plot Generators
# ──────────────────────────────────────────────────────────────────

@safe_plot
def _plot_water_bridges(data: dict[str, Any]) -> Optional[go.Figure]:
    """Top water-mediated bridge occupancies.

    Parameters
    ----------
    data : dict
        Must contain ``bridges`` list.

    Returns
    -------
    go.Figure or None
    """
    bridges = data.get("bridges", [])
    if not bridges:
        return None

    fig = go.Figure()
    labels = [f"{b['resid_1']}-{b['resid_2']}" for b in bridges[:30]]
    occupancies = [b["occupancy"] for b in bridges[:30]]
    fig.add_trace(go.Bar(
        x=labels, y=occupancies,
        marker_color=ACCENT_LIGHT_BLUE, name="Water Bridge Occupancy",
    ))
    return apply_dark_theme(fig, "Top Water-Mediated Bridges", "Residue Pair", "Occupancy")


@safe_plot
def _plot_energy_decomposition(data: dict[str, Any]) -> Optional[go.Figure]:
    """Per-residue energy decomposition (total, VdW, electrostatic).

    Parameters
    ----------
    data : dict
        Must contain ``resids`` and ``total_energy``.

    Returns
    -------
    go.Figure or None
    """
    resids = data.get("resids", [])
    total = data.get("total_energy", [])
    if not resids or not total:
        return None

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Per-Residue Total Interaction Energy", "VdW vs Electrostatic"),
    )
    fig.add_trace(go.Bar(
        x=resids, y=total, marker_color=ACCENT_ORANGE, name="Total",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=resids, y=data.get("vdw_energy", []),
        marker_color=ACCENT_DARK_TEAL, name="VdW",
    ), row=2, col=1)
    fig.add_trace(go.Bar(
        x=resids, y=data.get("elec_energy", []),
        marker_color="#fdcb6e", name="Electrostatic",
    ), row=2, col=1)
    return apply_dark_theme(
        fig, "Per-Residue Energy Decomposition", height=600,
    )


@safe_plot
def _plot_prs(data: dict[str, Any]) -> Optional[go.Figure]:
    """Perturbation Response Scanning heatmap and effector/sensor scores.

    Parameters
    ----------
    data : dict
        Must contain ``response_matrix``.

    Returns
    -------
    go.Figure or None
    """
    matrix = data.get("response_matrix", [])
    if not matrix:
        return None

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("PRS Response Matrix", "Effector vs Sensor Scores"),
        column_widths=[0.6, 0.4],
    )
    fig.add_trace(go.Heatmap(
        z=matrix, colorscale="Hot", colorbar=dict(title="Response", x=0.45),
    ), row=1, col=1)
    resids = data.get("resids", [])
    fig.add_trace(go.Scatter(
        x=resids, y=data.get("effector_scores", []),
        mode="lines", name="Effector", line=dict(color=ACCENT_RED, width=1.5),
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=resids, y=data.get("sensor_scores", []),
        mode="lines", name="Sensor", line=dict(color=ACCENT_LIGHT_BLUE, width=1.5),
    ), row=1, col=2)
    return apply_dark_theme(
        fig, "Perturbation Response Scanning", height=500,
    )


@safe_plot
def _plot_nma(data: dict[str, Any]) -> Optional[go.Figure]:
    """Normal Mode Analysis B-factors and mode collectivity.

    Parameters
    ----------
    data : dict
        Must contain ``resids`` and ``bfactors``.

    Returns
    -------
    go.Figure or None
    """
    resids = data.get("resids", [])
    bfactors = data.get("bfactors", [])
    if not resids or not bfactors:
        return None

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("ANM Predicted B-factors (Mobility)", "Mode Collectivity"),
    )
    fig.add_trace(go.Bar(
        x=resids, y=bfactors,
        marker_color=ACCENT_PURPLE, name="B-factor",
    ), row=1, col=1)
    collectivity = data.get("mode_collectivity", [])
    if collectivity:
        fig.add_trace(go.Bar(
            x=list(range(1, len(collectivity) + 1)), y=collectivity,
            marker_color=ACCENT_GREEN, name="Collectivity",
        ), row=2, col=1)
    return apply_dark_theme(fig, "Normal Mode Analysis", height=600)


@safe_plot
def _plot_entropy(data: dict[str, Any]) -> Optional[go.Figure]:
    """Per-residue configurational entropy and convergence curve.

    Parameters
    ----------
    data : dict
        Must contain ``resids`` and ``per_residue_entropy``.

    Returns
    -------
    go.Figure or None
    """
    resids = data.get("resids", [])
    per_res = data.get("per_residue_entropy", [])
    if not resids or not per_res:
        return None

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Per-Residue Configurational Entropy", "Entropy Convergence"),
    )
    fig.add_trace(go.Bar(
        x=resids, y=per_res,
        marker_color=ACCENT_PINK, name="Entropy",
    ), row=1, col=1)
    conv = data.get("entropy_convergence", [])
    if conv:
        fig.add_trace(go.Scatter(
            x=[c["fraction"] * 100 for c in conv],
            y=[c["entropy_J_mol_K"] for c in conv],
            mode="lines+markers", name="Convergence",
            line=dict(color="#ffeaa7", width=2),
            marker=dict(size=8),
        ), row=2, col=1)
    return apply_dark_theme(
        fig,
        f"Entropy Estimation (Total: {data.get('total_entropy_kJ_mol_K', 0):.2f} kJ/mol/K)",
        height=600,
    )


@safe_plot
def _plot_ifp(data: dict[str, Any]) -> Optional[go.Figure]:
    """Interaction fingerprint consensus per residue.

    Parameters
    ----------
    data : dict
        Must contain ``resids`` and ``consensus_fingerprint``.

    Returns
    -------
    go.Figure or None
    """
    resids = data.get("resids", [])
    consensus = data.get("consensus_fingerprint", [])
    if not resids or not consensus:
        return None

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=resids, y=consensus,
        marker_color="#e84393", name="Interaction Density",
    ))
    return apply_dark_theme(
        fig,
        f"Interaction Fingerprint Consensus "
        f"(Mean: {data.get('mean_interactions_per_frame', 0):.0f}/frame)",
        "Residue ID", "Interaction Score",
    )


@safe_plot
def _plot_tunnels(data: dict[str, Any]) -> Optional[go.Figure]:
    """Cavity volume over time and bottleneck residue frequencies.

    Parameters
    ----------
    data : dict
        Must contain ``cavity_volume_per_frame`` and ``time``.

    Returns
    -------
    go.Figure or None
    """
    volumes = data.get("cavity_volume_per_frame", [])
    times = data.get("time", [])
    if not volumes or not times:
        return None

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Cavity Volume Over Time", "Bottleneck Residues"),
    )
    fig.add_trace(go.Scatter(
        x=times, y=volumes, mode="lines", name="Volume",
        line=dict(color=ACCENT_SAGE, width=1.5),
        fill="tozeroy", fillcolor="rgba(0,184,148,0.1)",
    ), row=1, col=1)
    bottleneck = data.get("bottleneck_residues", [])
    if bottleneck:
        fig.add_trace(go.Bar(
            x=[b["resid"] for b in bottleneck],
            y=[b["cavity_frequency"] for b in bottleneck],
            marker_color=ACCENT_DARK_TEAL, name="Cavity Lining Freq",
        ), row=2, col=1)
    return apply_dark_theme(
        fig,
        f"Tunnel / Cavity Detection "
        f"(Mean: {data.get('mean_cavity_volume', 0):.0f} \u00c5\u00b3)",
        height=600,
    )


@safe_plot
def _plot_vae(data: dict[str, Any]) -> Optional[go.Figure]:
    """VAE latent-space scatter and training loss curves.

    Parameters
    ----------
    data : dict
        Must contain ``latent_coords``.

    Returns
    -------
    go.Figure or None
    """
    latent = data.get("latent_coords", [])
    if not latent or len(latent) == 0:
        return None

    arr = np.array(latent)
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("VAE Latent Space", "Training Loss"),
    )
    fig.add_trace(go.Scatter(
        x=arr[:, 0], y=arr[:, 1],
        mode="markers", name="Conformations",
        marker=dict(
            color=list(range(len(arr))), colorscale="Turbo",
            size=4, colorbar=dict(title="Frame", x=0.45),
        ),
    ), row=1, col=1)
    recon_loss = data.get("reconstruction_loss", [])
    kl_loss = data.get("kl_loss", [])
    if recon_loss:
        epochs = list(range(1, len(recon_loss) + 1))
        fig.add_trace(go.Scatter(
            x=epochs, y=recon_loss, mode="lines", name="Recon Loss",
            line=dict(color=ACCENT_RED, width=1.5),
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=epochs, y=kl_loss, mode="lines", name="KL Loss",
            line=dict(color=ACCENT_LIGHT_BLUE, width=1.5),
        ), row=1, col=2)
    return apply_dark_theme(fig, "Variational Autoencoder Latent Space")


@safe_plot
def _plot_dynamic_network(data: dict[str, Any]) -> Optional[go.Figure]:
    """Community evolution heatmap and per-residue stability.

    Parameters
    ----------
    data : dict
        Must contain ``community_evolution`` and ``resids``.

    Returns
    -------
    go.Figure or None
    """
    comm_evo = data.get("community_evolution", [])
    resids = data.get("resids", [])
    if not comm_evo or not resids:
        return None

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            "Community Evolution Over Time Windows",
            "Community Stability Per Residue",
        ),
    )
    fig.add_trace(go.Heatmap(
        z=comm_evo, x=resids,
        y=[f"Window {i + 1}" for i in range(len(comm_evo))],
        colorscale="Set3", colorbar=dict(title="Community", x=0.95),
    ), row=1, col=1)
    stability = data.get("community_stability", [])
    if stability:
        fig.add_trace(go.Bar(
            x=resids, y=stability,
            marker_color=ACCENT_DARK_PURPLE, name="Stability",
        ), row=2, col=1)
    return apply_dark_theme(fig, "Dynamic Network Analysis", height=600)


# ──────────────────────────────────────────────────────────────────
# Phase 4 -- New Plot Generators
# ──────────────────────────────────────────────────────────────────

@safe_plot
def _plot_dimensionality_3d(data: dict[str, Any]) -> Optional[go.Figure]:
    """3-D PCA / UMAP / t-SNE scatter plots.

    Parameters
    ----------
    data : dict
        May contain ``pca_3d``, ``umap_3d``, ``tsne_3d``.

    Returns
    -------
    go.Figure or None
    """
    pca_3d = data.get("pca_3d", [])
    umap_3d = data.get("umap_3d", [])
    tsne_3d = data.get("tsne_3d", [])

    has_any = any(len(d) > 0 for d in [pca_3d, umap_3d, tsne_3d])
    if not has_any:
        return None

    n_cols = sum(1 for d in [pca_3d, umap_3d, tsne_3d] if len(d) > 0)
    if n_cols == 0:
        return None

    titles: list[str] = []
    datasets: list[np.ndarray] = []
    if pca_3d:
        titles.append("PCA 3D")
        datasets.append(np.array(pca_3d))
    if umap_3d:
        titles.append("UMAP 3D")
        datasets.append(np.array(umap_3d))
    if tsne_3d:
        titles.append("t-SNE 3D")
        datasets.append(np.array(tsne_3d))

    fig = make_subplots(
        rows=1, cols=n_cols,
        subplot_titles=titles,
        specs=[[{"type": "scatter3d"}] * n_cols],
    )

    for idx, (arr, title) in enumerate(zip(datasets, titles)):
        fig.add_trace(go.Scatter3d(
            x=arr[:, 0], y=arr[:, 1], z=arr[:, 2],
            mode="markers", name=title,
            marker=dict(
                color=list(range(len(arr))), colorscale="Plasma",
                size=2, opacity=0.7,
            ),
        ), row=1, col=idx + 1)

    return apply_dark_theme(fig, "3D Dimensionality Reduction", height=600)


@safe_plot
def _plot_convergence(data: dict[str, Any]) -> Optional[go.Figure]:
    """Convergence assessment dashboard (block SEM, ACF, cosine content).

    Parameters
    ----------
    data : dict
        Convergence analysis output dict.

    Returns
    -------
    go.Figure or None
    """
    if not data:
        return None

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Block Average SEM",
            "Autocorrelation (RMSD)",
            "Cosine Content by PC",
            f"Convergence Score: {data.get('convergence_score', 'N/A')}",
        ),
    )

    # Block average SEM
    blocks = data.get("rmsd_block_averages", [])
    if blocks:
        fig.add_trace(go.Scatter(
            x=[b["n_blocks"] for b in blocks],
            y=[b["sem"] for b in blocks],
            mode="lines+markers", name="RMSD SEM",
            line=dict(color=ACCENT_CYAN, width=2),
            marker=dict(size=8),
        ), row=1, col=1)

    # Autocorrelation
    acf = data.get("autocorrelation_rmsd", [])
    if acf:
        fig.add_trace(go.Scatter(
            x=[a["lag"] for a in acf],
            y=[a["acf"] for a in acf],
            mode="lines", name="ACF",
            line=dict(color=ACCENT_PURPLE, width=2),
        ), row=1, col=2)
        fig.add_hline(y=0, row=1, col=2, line_dash="dot", line_color="#666")

    # Cosine content
    cc = data.get("cosine_content", [])
    if cc:
        fig.add_trace(go.Bar(
            x=[f"PC{c['pc']}" for c in cc],
            y=[c["cosine_content"] for c in cc],
            marker_color=[
                ACCENT_GREEN if c["converged"] else ACCENT_RED for c in cc
            ],
            name="Cosine Content",
        ), row=2, col=1)
        fig.add_hline(
            y=0.5, row=2, col=1, line_dash="dash", line_color=ACCENT_YELLOW,
            annotation_text="Threshold",
        )

    # Recommendations
    recs = data.get("recommendations", [])
    if recs:
        text = "<br>".join(f"* {r}" for r in recs)
        fig.add_annotation(
            text=text, xref="x4 domain", yref="y4 domain",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=11, color=TEXT_COLOR),
            align="left",
        )

    return apply_dark_theme(fig, "Convergence Assessment", height=600)


@safe_plot
def _plot_binding_kinetics(data: dict[str, Any]) -> Optional[go.Figure]:
    """Binding kinetics dashboard (COM distance, survival, contacts, events).

    Parameters
    ----------
    data : dict
        Binding kinetics analysis output dict.

    Returns
    -------
    go.Figure or None
    """
    if not data:
        return None

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "COM Distance Over Time",
            "Contact Survival Function",
            "Per-Residue Contact Occupancy",
            f"kon={data.get('kon_estimate_per_ps', 0):.2e}/ps  "
            f"koff={data.get('koff_estimate_per_ps', 0):.2e}/ps",
        ),
    )

    # COM distance
    times = data.get("time", [])
    com = data.get("com_distance", [])
    if times and com and len(times) == len(com):
        fig.add_trace(go.Scatter(
            x=times, y=com, mode="lines", name="COM dist",
            line=dict(color=ACCENT_CYAN, width=1.5),
        ), row=1, col=1)

    # Survival function
    survival = data.get("contact_survival", [])
    if survival:
        fig.add_trace(go.Scatter(
            x=[s["lag_frames"] for s in survival],
            y=[s["survival"] for s in survival],
            mode="lines", name="S(t)",
            line=dict(color=ACCENT_PINK, width=2),
        ), row=1, col=2)

    # Per-residue contact occupancy
    res_contacts = data.get("per_residue_contact_time", [])
    if res_contacts:
        fig.add_trace(go.Bar(
            x=[r["resid"] for r in res_contacts[:30]],
            y=[r["occupancy"] for r in res_contacts[:30]],
            marker_color=ACCENT_GREEN, name="Occupancy",
        ), row=2, col=1)

    # Binding events timeline
    events = data.get("binding_events", [])
    bind_times = [e["time_ps"] for e in events if e["type"] == "bind"]
    unbind_times = [e["time_ps"] for e in events if e["type"] == "unbind"]
    if bind_times:
        fig.add_trace(go.Scatter(
            x=bind_times, y=[1] * len(bind_times),
            mode="markers", name="Bind",
            marker=dict(color=ACCENT_GREEN, size=8, symbol="triangle-up"),
        ), row=2, col=2)
    if unbind_times:
        fig.add_trace(go.Scatter(
            x=unbind_times, y=[0] * len(unbind_times),
            mode="markers", name="Unbind",
            marker=dict(color=ACCENT_RED, size=8, symbol="triangle-down"),
        ), row=2, col=2)

    return apply_dark_theme(fig, "Binding Kinetics Analysis", height=600)


@safe_plot
def _plot_allosteric_network(data: dict[str, Any]) -> Optional[go.Figure]:
    """Interactive allosteric communication network graph.

    Parameters
    ----------
    data : dict
        Must contain ``network_edges`` and ``resids``; optionally
        ``communities`` and ``hub_residues``.

    Returns
    -------
    go.Figure or None
    """
    if not data:
        return None

    edges = data.get("network_edges", [])
    communities = data.get("communities", [])
    hubs = data.get("hub_residues", [])
    resids = data.get("resids", [])
    if not edges or not resids:
        return None

    import networkx as nx

    G = nx.Graph()
    for r in resids:
        G.add_node(r)
    for e in edges[:200]:
        G.add_edge(e["res_i"], e["res_j"], weight=abs(e["correlation"]))

    try:
        pos = nx.spring_layout(
            G, k=2.0 / np.sqrt(len(resids)), iterations=50, seed=42,
        )
    except Exception:
        return None

    # Colour by community
    comm_color: dict[int, str] = {}
    for comm in communities:
        c_idx = comm["id"] % len(COMMUNITY_COLORS)
        for r in comm["residues"]:
            comm_color[r] = COMMUNITY_COLORS[c_idx]

    hub_resids = {h["resid"] for h in hubs}

    # Edge traces
    edge_x: list[Optional[float]] = []
    edge_y: list[Optional[float]] = []
    for e in edges[:200]:
        if e["res_i"] in pos and e["res_j"] in pos:
            x0, y0 = pos[e["res_i"]]
            x1, y1 = pos[e["res_j"]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(color="rgba(100,100,150,0.3)", width=0.5),
        hoverinfo="none", name="Edges",
    ))

    # Node traces
    node_x = [pos[r][0] for r in resids if r in pos]
    node_y = [pos[r][1] for r in resids if r in pos]
    node_color = [comm_color.get(r, "#666680") for r in resids if r in pos]
    node_size = [12 if r in hub_resids else 5 for r in resids if r in pos]
    node_text = [
        f"Res {r}" + (" (HUB)" if r in hub_resids else "")
        for r in resids if r in pos
    ]

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers",
        marker=dict(
            color=node_color, size=node_size,
            line=dict(width=1, color="rgba(255,255,255,0.3)"),
        ),
        text=node_text, hoverinfo="text", name="Residues",
    ))

    modularity = data.get("modularity_score", 0)
    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return apply_dark_theme(
        fig,
        f"Allosteric Network (Modularity: {modularity:.3f}, {len(edges)} edges)",
        height=600,
    )


@safe_plot
def _plot_training_losses(data: dict[str, Any]) -> Optional[go.Figure]:
    """Combined training loss curves for GNN, Transformer, and VAE.

    Parameters
    ----------
    data : dict
        Mapping with keys ``gnn``, ``transformer``, ``vae``, each being an
        analysis output dict containing ``training_losses`` or ``total_loss``.

    Returns
    -------
    go.Figure or None
    """
    if not data:
        return None

    gnn = data.get("gnn", {})
    transformer = data.get("transformer", {})
    vae = data.get("vae", {})

    has_any = False
    fig = go.Figure()

    gnn_losses = gnn.get("training_losses", []) if isinstance(gnn, dict) else []
    trans_losses = transformer.get("training_losses", []) if isinstance(transformer, dict) else []
    vae_losses = vae.get("total_loss", []) if isinstance(vae, dict) else []

    if gnn_losses:
        fig.add_trace(go.Scatter(
            x=list(range(1, len(gnn_losses) + 1)), y=gnn_losses,
            mode="lines", name="GNN",
            line=dict(color=ACCENT_ORANGE, width=2),
        ))
        has_any = True

    if trans_losses:
        fig.add_trace(go.Scatter(
            x=list(range(1, len(trans_losses) + 1)), y=trans_losses,
            mode="lines", name="Transformer",
            line=dict(color=ACCENT_DARK_TEAL, width=2),
        ))
        has_any = True

    if vae_losses:
        fig.add_trace(go.Scatter(
            x=list(range(1, len(vae_losses) + 1)), y=vae_losses,
            mode="lines", name="VAE",
            line=dict(color=ACCENT_PURPLE, width=2),
        ))
        has_any = True

    if not has_any:
        return None

    return apply_dark_theme(fig, "Deep Learning Training Loss Curves", "Epoch", "Loss")
