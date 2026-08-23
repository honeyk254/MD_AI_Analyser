"""Plotly chart generators for all classical analysis modules.

Each function takes a ModuleResult and returns a Plotly Figure JSON, or None.
"""

import logging
from typing import Dict, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..ml.schemas import MLAnalysisBundle
from ..schemas.analysis_bundle import AnalysisBundle, ModuleResult

logger = logging.getLogger("md_ai_analyzer")

# --- Neo-brutalist styling tokens (mirror the CSS tokens in html_report.py) ---
BLACK = "#000000"
WHITE = "#FFFFFF"
PAPER_BEIGE = "#F5F5DC"
ACCENT_TEAL = "#00C2CB"
ACCENT_MAGENTA = "#FF00FF"
ACCENT_YELLOW = "#FFD700"
GRID_COLOR = "rgba(0,0,0,0.15)"
HEADING_FONT = "Arial Black, Arial, sans-serif"
BODY_FONT = "Helvetica Neue, Arial, sans-serif"
BRUTALIST_SCALE = [[0.0, WHITE], [0.5, ACCENT_TEAL], [1.0, BLACK]]

def apply_theme(
    fig: go.Figure,
    title: str,
    xaxis_title: str = "",
    yaxis_title: str = "",
    height: int = 500,
) -> go.Figure:
    """Apply the neo-brutalist chart theme: white card, black strokes, bold type."""
    fig.update_layout(
        title=dict(text=title, font=dict(color=BLACK, size=20, family=HEADING_FONT)),
        xaxis=dict(
            title=xaxis_title, color=BLACK, gridcolor=GRID_COLOR,
            zerolinecolor=BLACK, linecolor=BLACK, ticks="outside", tickcolor=BLACK,
        ),
        yaxis=dict(
            title=yaxis_title, color=BLACK, gridcolor=GRID_COLOR,
            zerolinecolor=BLACK, linecolor=BLACK, ticks="outside", tickcolor=BLACK,
        ),
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        font=dict(color=BLACK, family=BODY_FONT, size=14),
        height=height,
        margin=dict(t=60, b=50, l=60, r=20),
        legend=dict(font=dict(color=BLACK), bgcolor=WHITE, bordercolor=BLACK, borderwidth=2),
    )
    return fig


def generate_all_plots(bundle: AnalysisBundle, ml_bundle: Optional[MLAnalysisBundle] = None) -> Dict[str, str]:
    """Generate all visualization plots from the AnalysisBundle.

    Returns a dictionary mapping plot names to Plotly JSON strings.
    """
    plots: Dict[str, str] = {}

    generators = [
        ("rmsd_plot", _plot_rmsd, bundle.modules.get("rmsd")),
        ("rmsf_plot", _plot_rmsf, bundle.modules.get("rmsf")),
        ("rg_plot", _plot_rg, bundle.modules.get("radius_of_gyration")),
        ("sasa_plot", _plot_sasa, bundle.modules.get("sasa")),
        ("hbond_plot", _plot_hbonds, bundle.modules.get("hbonds")),
        ("contact_map", _plot_contact_map, bundle.modules.get("contacts")),
        ("ss_plot", _plot_secondary_structure, bundle.modules.get("secondary_structure")),
        ("salt_bridges_plot", _plot_salt_bridges, bundle.modules.get("salt_bridges")),
        ("dihedrals_plot", _plot_dihedrals, bundle.modules.get("dihedrals")),
        ("com_plot", _plot_com, bundle.modules.get("com")),
    ]

    for name, func, module_res in generators:
        if module_res and not module_res.error:
            try:
                fig = func(module_res)
                if fig is not None:
                    plots[name] = fig.to_json()
            except Exception as e:
                logger.warning(f"Failed to generate plot {name}: {e}")

    if ml_bundle and ml_bundle.status == "completed":
        for name, fig in _generate_ml_plots(ml_bundle).items():
            plots[name] = fig.to_json()

    return plots


def _generate_ml_plots(ml_bundle: MLAnalysisBundle) -> Dict[str, go.Figure]:
    plots: Dict[str, go.Figure] = {}
    if ml_bundle.pca and ml_bundle.tica and ml_bundle.baseline_comparison:
        comparison = _plot_ml_comparison(ml_bundle)
        if comparison is not None:
            plots["ml_comparison_plot"] = comparison
    if ml_bundle.tica:
        tica_plot = _plot_tica_embedding(ml_bundle)
        if tica_plot is not None:
            plots["tica_plot"] = tica_plot
    if ml_bundle.msm:
        msm_plot = _plot_msm_timescales(ml_bundle)
        if msm_plot is not None:
            plots["msm_timescales_plot"] = msm_plot
    return plots


def _plot_rmsd(res: ModuleResult) -> Optional[go.Figure]:
    if "backbone_rmsd" not in res.scalar_metrics or "time_ps" not in res.data:
        return None

    metric = res.scalar_metrics["backbone_rmsd"]
    times = res.data["time_ps"]
    equil_frame = res.data.get("equilibration_frame", 0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=metric.time_series,
        mode="lines", name="RMSD",
        line=dict(color=ACCENT_TEAL, width=3),
    ))
    if equil_frame > 0 and len(times) > equil_frame:
        fig.add_vline(
            x=times[equil_frame], line_dash="dash", line_color=ACCENT_MAGENTA, line_width=2,
            annotation_text="Equilibration",
        )
    fig.add_hline(
        y=metric.mean, line_dash="dot", line_color=ACCENT_YELLOW, line_width=2,
        annotation_text=f"Mean: {metric.mean:.2f} \u00c5",
    )
    return apply_theme(fig, "RMSD Over Time", "Time (ps)", "RMSD (\u00c5)")


def _plot_rmsf(res: ModuleResult) -> Optional[go.Figure]:
    if "rmsf" not in res.residue_metrics or "mean_rmsf" not in res.scalar_metrics:
        return None

    rmsf = res.residue_metrics["rmsf"].values
    resids = res.residue_metrics["rmsf"].resids
    mean_r = res.scalar_metrics["mean_rmsf"].mean
    std_r = res.scalar_metrics["mean_rmsf"].std

    colors = []
    for v in rmsf:
        if v > mean_r + std_r:
            colors.append(ACCENT_MAGENTA)
        elif v < mean_r - 0.5 * std_r:
            colors.append(ACCENT_TEAL)
        else:
            colors.append(BLACK)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=resids, y=rmsf, marker_color=colors, marker_line_color=BLACK, marker_line_width=1, name="RMSF",
    ))
    fig.add_hline(
        y=mean_r, line_dash="dot", line_color=ACCENT_YELLOW, line_width=2,
        annotation_text=f"Mean: {mean_r:.2f} \u00c5",
    )
    return apply_theme(fig, "Per-Residue RMSF", "Residue ID", "RMSF (\u00c5)")


def _plot_rg(res: ModuleResult) -> Optional[go.Figure]:
    if "radius_of_gyration" not in res.scalar_metrics or "time_ps" not in res.data:
        return None

    metric = res.scalar_metrics["radius_of_gyration"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=res.data["time_ps"], y=metric.time_series,
        mode="lines", name="Rg",
        line=dict(color=ACCENT_MAGENTA, width=3),
    ))
    trend = res.data.get("trend", "N/A")
    return apply_theme(fig, f"Radius of Gyration (Trend: {trend})", "Time (ps)", "Rg (\u00c5)")


def _plot_sasa(res: ModuleResult) -> Optional[go.Figure]:
    if "total_sasa" not in res.scalar_metrics or "time_ps" not in res.data:
        return None

    metric = res.scalar_metrics["total_sasa"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=res.data["time_ps"], y=metric.time_series,
        mode="lines", name="Total SASA",
        line=dict(color=ACCENT_TEAL, width=3),
    ))
    return apply_theme(fig, "Solvent Accessible Surface Area", "Time (ps)", "SASA (nm\u00b2)")


def _plot_hbonds(res: ModuleResult) -> Optional[go.Figure]:
    if "hbond_count" not in res.scalar_metrics or "time_ps" not in res.data:
        return None

    metric = res.scalar_metrics["hbond_count"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=res.data["time_ps"], y=metric.time_series,
        mode="lines", name="H-bonds",
        line=dict(color=BLACK, width=3),
    ))
    return apply_theme(fig, f"Hydrogen Bonds (Mean: {metric.mean:.1f})", "Time (ps)", "Number of H-bonds")


def _plot_contact_map(res: ModuleResult) -> Optional[go.Figure]:
    if "contact_map" not in res.data or "resids" not in res.data:
        return None

    cmap = res.data["contact_map"]
    resids = res.data["resids"]

    fig = go.Figure(data=go.Heatmap(
        z=cmap, x=resids, y=resids,
        colorscale=BRUTALIST_SCALE,
        colorbar=dict(title="Contact Freq", outlinecolor=BLACK, outlinewidth=2),
    ))
    return apply_theme(fig, "Average Contact Map", "Residue ID", "Residue ID")


def _plot_secondary_structure(res: ModuleResult) -> Optional[go.Figure]:
    if "helix_fraction" not in res.scalar_metrics:
        return None

    h = res.scalar_metrics["helix_fraction"].time_series
    s = res.scalar_metrics["sheet_fraction"].time_series
    c = res.scalar_metrics["coil_fraction"].time_series
    if not h or not s or not c:
        return None
    frames = list(range(len(h)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frames, y=h, mode="lines", name="\u03b1-Helix", line=dict(color=ACCENT_MAGENTA, width=3),
    ))
    fig.add_trace(go.Scatter(
        x=frames, y=s, mode="lines", name="\u03b2-Sheet", line=dict(color=ACCENT_TEAL, width=3),
    ))
    fig.add_trace(go.Scatter(
        x=frames, y=c, mode="lines", name="Coil", line=dict(color=BLACK, width=3),
    ))
    return apply_theme(fig, "Secondary Structure Evolution", "Frame", "Fraction")


def _plot_salt_bridges(res: ModuleResult) -> Optional[go.Figure]:
    if "salt_bridge_count" not in res.scalar_metrics or "time_ps" not in res.data:
        return None

    metric = res.scalar_metrics["salt_bridge_count"]
    times = res.data["time_ps"]
    pairs = res.data.get("pairs", [])

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Salt Bridges Over Time", "Top Salt Bridge Pairs (Occupancy)"),
        row_heights=[0.5, 0.5],
    )

    fig.add_trace(go.Scatter(
        x=times, y=metric.time_series,
        mode="lines", name="Salt Bridges/Frame",
        line=dict(color=ACCENT_TEAL, width=3),
    ), row=1, col=1)

    if pairs:
        labels = [f"{p['positive']}-{p['negative']}" for p in pairs[:20]]
        occupancies = [p["occupancy"] for p in pairs[:20]]
        fig.add_trace(go.Bar(
            x=labels, y=occupancies,
            marker_color=ACCENT_YELLOW, marker_line_color=BLACK, marker_line_width=1, name="Occupancy",
        ), row=2, col=1)

    return apply_theme(fig, f"Salt Bridges (Mean: {metric.mean:.1f}/frame)", "", "", height=600)


def _plot_ml_comparison(ml_bundle: MLAnalysisBundle) -> Optional[go.Figure]:
    if not ml_bundle.pca or not ml_bundle.tica or not ml_bundle.baseline_comparison:
        return None

    pca = np.asarray(ml_bundle.pca.projections, dtype=float)
    tica = np.asarray(ml_bundle.tica.projections, dtype=float)
    pca_labels = np.asarray(ml_bundle.baseline_comparison.pca_state_labels, dtype=int)
    tica_labels = np.asarray(ml_bundle.baseline_comparison.tica_state_labels, dtype=int)
    if pca.shape[1] < 2 or tica.shape[1] < 2:
        return None

    fig = make_subplots(rows=1, cols=2, subplot_titles=("PCA clustering", "TICA clustering"))
    fig.add_trace(
        go.Scatter(
            x=pca[:, 0],
            y=pca[:, 1],
            mode="markers",
            marker=dict(color=pca_labels, colorscale=BRUTALIST_SCALE, size=8, line=dict(color=BLACK, width=1)),
            name="PCA states",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=tica[:, 0],
            y=tica[:, 1],
            mode="markers",
            marker=dict(color=tica_labels, colorscale=BRUTALIST_SCALE, size=8, line=dict(color=BLACK, width=1)),
            name="TICA states",
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        title=dict(text="PCA vs TICA baseline comparison", font=dict(color=BLACK, size=20, family=HEADING_FONT)),
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        font=dict(color=BLACK, family=BODY_FONT, size=14),
        height=500,
        margin=dict(t=60, b=40, l=50, r=20),
        legend=dict(font=dict(color=BLACK), bgcolor=WHITE, bordercolor=BLACK, borderwidth=2),
    )
    fig.update_annotations(font=dict(color=BLACK, family=BODY_FONT))
    fig.update_xaxes(title_text="Component 1", color=BLACK, gridcolor=GRID_COLOR, linecolor=BLACK)
    fig.update_yaxes(title_text="Component 2", color=BLACK, gridcolor=GRID_COLOR, linecolor=BLACK)
    return fig


def _plot_tica_embedding(ml_bundle: MLAnalysisBundle) -> Optional[go.Figure]:
    if not ml_bundle.tica:
        return None
    tica = np.asarray(ml_bundle.tica.projections, dtype=float)
    if tica.shape[1] < 2:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=tica[:, 0],
            y=tica[:, 1],
            mode="markers",
            marker=dict(
                color=np.arange(len(tica)),
                size=8,
                colorscale=[[0.0, ACCENT_TEAL], [1.0, BLACK]],
                line=dict(color=BLACK, width=1),
                showscale=True,
                colorbar=dict(outlinecolor=BLACK, outlinewidth=2),
            ),
            name="TICA",
        )
    )
    return apply_theme(fig, "TICA embedding", "tIC1", "tIC2")


def _plot_msm_timescales(ml_bundle: MLAnalysisBundle) -> Optional[go.Figure]:
    if not ml_bundle.msm:
        return None
    timescales = ml_bundle.msm.implied_timescales_ps
    if not timescales:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[f"t{i+1}" for i in range(len(timescales))],
            y=timescales,
            marker_color=ACCENT_TEAL, marker_line_color=BLACK, marker_line_width=1,
            name="Implied timescales",
        )
    )
    fig.add_hline(
        y=0,
        line_color=BLACK,
    )
    return apply_theme(fig, "MSM implied timescales", "Mode", "Timescale (ps)")


def _plot_dihedrals(res: ModuleResult) -> Optional[go.Figure]:
    phi = res.residue_metrics.get("phi_circular_std")
    psi = res.residue_metrics.get("psi_circular_std")
    if not phi or not psi:
        return None

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=phi.resids, y=phi.values, name="phi",
        marker_color=ACCENT_TEAL, marker_line_color=BLACK, marker_line_width=1,
    ))
    fig.add_trace(go.Bar(
        x=psi.resids, y=psi.values, name="psi",
        marker_color=ACCENT_YELLOW, marker_line_color=BLACK, marker_line_width=1,
    ))
    fig.update_layout(barmode="group")
    return apply_theme(fig, "Backbone Dihedral Flexibility (Circular Std)", "Residue", "Degrees")


def _plot_com(res: ModuleResult) -> Optional[go.Figure]:
    metric = res.scalar_metrics.get("com_drift")
    if not metric or not metric.time_series or "time_ps" not in res.data:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=res.data["time_ps"], y=metric.time_series,
        mode="lines", name="COM drift",
        line=dict(color=ACCENT_MAGENTA, width=3),
    ))
    return apply_theme(fig, "Center-of-Mass Drift", "Time (ps)", "Drift (Angstrom)")
