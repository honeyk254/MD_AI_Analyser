"""
Report generator — creates HTML, CSV, and PDF reports from analysis results.
"""
import json
import csv
from pathlib import Path
from typing import Optional
import numpy as np


def generate_html_report(result, output_dir: Path) -> Path:
    """Generate a standalone HTML report with embedded Plotly charts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "analysis_report.html"

    # Build report sections
    info = result.trajectory_info or {}
    insights = result.biological_insights or []
    plots = result.plots or {}

    # Generate plot divs
    plot_html_sections = []
    plot_names_map = {
        "rmsd_plot": "RMSD Over Time",
        "rmsf_plot": "Per-Residue RMSF",
        "rg_plot": "Radius of Gyration",
        "ss_plot": "Secondary Structure Evolution",
        "hbond_plot": "Hydrogen Bonds",
        "contact_map": "Contact Map",
        "pca_plot": "Principal Component Analysis",
        "dccm_plot": "Dynamic Cross-Correlation Matrix",
        "fel_plot": "Free Energy Landscape",
        "clustering_plot": "Conformational Clustering",
        "sasa_plot": "Solvent Accessible Surface Area",
        "dimensionality_plot": "Dimensionality Reduction (2D)",
        "dimensionality_3d_plot": "Dimensionality Reduction (3D)",
        "gnn_plot": "GNN Residue Importance",
        "transformer_plot": "Transformer Temporal Analysis",
        "msm_plot": "Markov State Model",
        # Phase 3 plots
        "tica_plot": "tICA Slow Motions",
        "salt_bridges_plot": "Salt Bridges",
        "water_bridges_plot": "Water Bridges",
        "energy_plot": "Energy Decomposition",
        "prs_plot": "Perturbation Response Scanning",
        "nma_plot": "Normal Mode Analysis",
        "entropy_plot": "Configurational Entropy",
        "ifp_plot": "Interaction Fingerprints",
        "tunnel_plot": "Tunnels & Cavities",
        "vae_plot": "VAE Latent Space",
        "dynamic_network_plot": "Dynamic Network",
        # Phase 4 plots
        "convergence_plot": "Convergence Assessment",
        "binding_kinetics_plot": "Binding Kinetics",
        "network_graph_plot": "Allosteric Network Graph",
        "training_loss_plot": "Deep Learning Training Losses",
    }

    for plot_key, plot_title in plot_names_map.items():
        json_str = plots.get(plot_key, "")
        if json_str:
            div_id = plot_key.replace("_", "-")
            plot_html_sections.append(f"""
            <div class="plot-section">
                <h3>{plot_title}</h3>
                <div id="{div_id}" class="plot-container"></div>
                <script>
                    Plotly.newPlot('{div_id}', {json_str}.data, {json_str}.layout, {{responsive: true}});
                </script>
            </div>
            """)

    # Insights HTML
    insight_cards = []
    category_colors = {
        "structural": "#4ecdc4",
        "dynamic": "#a29bfe",
        "allosteric": "#ff6b6b",
        "binding": "#ffd93d",
        "transition": "#fd79a8",
    }
    for ins in insights:
        color = category_colors.get(ins.get("category", ""), "#00d4ff")
        confidence = ins.get("confidence", 0)
        conf_bar = int(confidence * 100)

        evidence_html = ""
        for ev in ins.get("evidence", []):
            evidence_html += f"<li>{ev}</li>"

        insight_cards.append(f"""
        <div class="insight-card" style="border-left: 4px solid {color};">
            <div class="insight-header">
                <span class="insight-type" style="color: {color};">{ins.get('type', '').replace('_', ' ').title()}</span>
                <span class="confidence-badge">Confidence: {confidence:.0%}</span>
            </div>
            <div class="confidence-bar"><div class="confidence-fill" style="width: {conf_bar}%; background: {color};"></div></div>
            <p class="insight-desc">{ins.get('description', '')}</p>
            {f'<div class="residue-tags">Residues: {", ".join(map(str, ins.get("residues", [])[:15]))}</div>' if ins.get('residues') else ''}
            <details><summary>Evidence</summary><ul>{evidence_html}</ul></details>
        </div>
        """)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MD AI Analysis Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: 'Inter', -apple-system, sans-serif; background: #0f0f23; color: #e0e0e0; line-height: 1.6; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #00d4ff; font-size: 2em; border-bottom: 2px solid #1a1a3e; padding-bottom: 15px; margin-bottom: 20px; }}
        h2 {{ color: #a29bfe; margin: 30px 0 15px; }}
        h3 {{ color: #ddd; margin: 20px 0 10px; }}
        .info-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .info-card {{ background: #1a1a3e; border-radius: 12px; padding: 15px; text-align: center; }}
        .info-card .value {{ font-size: 1.8em; font-weight: bold; color: #00d4ff; }}
        .info-card .label {{ color: #999; font-size: 0.85em; }}
        .plot-section {{ background: #1a1a2e; border-radius: 12px; padding: 20px; margin: 20px 0; }}
        .plot-container {{ width: 100%; min-height: 400px; }}
        .insight-card {{ background: #1a1a3e; border-radius: 12px; padding: 20px; margin: 15px 0; transition: transform .2s; }}
        .insight-card:hover {{ transform: translateX(5px); }}
        .insight-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
        .insight-type {{ font-weight: bold; font-size: 1.1em; text-transform: uppercase; letter-spacing: 1px; }}
        .confidence-badge {{ background: rgba(255,255,255,0.1); padding: 4px 12px; border-radius: 20px; font-size: 0.85em; }}
        .confidence-bar {{ height: 4px; background: rgba(255,255,255,0.1); border-radius: 2px; margin-bottom: 12px; }}
        .confidence-fill {{ height: 100%; border-radius: 2px; transition: width .5s; }}
        .insight-desc {{ color: #ccc; margin-bottom: 10px; }}
        .residue-tags {{ color: #888; font-size: 0.85em; margin-bottom: 10px; font-family: monospace; }}
        details {{ color: #888; font-size: 0.9em; }}
        details summary {{ cursor: pointer; color: #aaa; margin-bottom: 5px; }}
        details ul {{ padding-left: 20px; }}
        details li {{ margin: 4px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧬 MD AI Analysis Report</h1>

        <h2>📊 Trajectory Statistics</h2>
        <div class="info-grid">
            <div class="info-card">
                <div class="value">{info.get('n_frames', 'N/A')}</div>
                <div class="label">Total Frames</div>
            </div>
            <div class="info-card">
                <div class="value">{info.get('n_atoms', 'N/A')}</div>
                <div class="label">Atoms</div>
            </div>
            <div class="info-card">
                <div class="value">{info.get('n_residues', 'N/A')}</div>
                <div class="label">Residues</div>
            </div>
            <div class="info-card">
                <div class="value">{info.get('total_time_ns', 0):.1f} ns</div>
                <div class="label">Total Time</div>
            </div>
        </div>

        <h2>🧠 AI-Generated Biological Insights</h2>
        {''.join(insight_cards) if insight_cards else '<p>No insights generated.</p>'}

        <h2>📈 Analysis Plots</h2>
        {''.join(plot_html_sections) if plot_html_sections else '<p>No plots generated.</p>'}

        <footer style="text-align: center; color: #666; margin-top: 40px; padding: 20px; border-top: 1px solid #1a1a3e;">
            Generated by MD AI Analyzer | AI-Powered Molecular Dynamics Analysis Platform
        </footer>
    </div>
</body>
</html>"""

    report_path.write_text(html, encoding="utf-8")
    return report_path


def export_csv(result, output_dir: Path) -> Path:
    """Export ALL analysis metrics to CSV (item 61 — complete CSV)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metrics.csv"

    rows = []

    # RMSD
    rmsd = result.rmsd
    if isinstance(rmsd, dict) and "rmsd" in rmsd:
        for t, v in zip(rmsd.get("time", []), rmsd["rmsd"]):
            rows.append({"metric": "RMSD", "time": t, "value": v, "unit": "Å"})

    # RMSF
    rmsf = result.rmsf
    if isinstance(rmsf, dict) and "rmsf" in rmsf:
        for r, v in zip(rmsf.get("resids", []), rmsf["rmsf"]):
            rows.append({"metric": "RMSF", "residue": r, "value": v, "unit": "Å"})

    # Rg
    rg = result.rg
    if isinstance(rg, dict) and "rg" in rg:
        for t, v in zip(rg.get("time", []), rg["rg"]):
            rows.append({"metric": "Rg", "time": t, "value": v, "unit": "Å"})

    # SASA
    sasa = result.sasa
    if isinstance(sasa, dict) and "total_sasa" in sasa:
        for t, v in zip(sasa.get("time", []), sasa["total_sasa"]):
            rows.append({"metric": "SASA_total", "time": t, "value": v, "unit": "Å²"})

    # H-bonds
    hbonds = result.hbonds
    if isinstance(hbonds, dict) and "n_hbonds" in hbonds:
        times_hb = hbonds.get("time", list(range(len(hbonds["n_hbonds"]))))
        for t, v in zip(times_hb, hbonds["n_hbonds"]):
            rows.append({"metric": "HBonds_count", "time": t, "value": v, "unit": "count"})

    # Salt bridges
    sb = result.salt_bridges
    if isinstance(sb, dict) and "total_unique_pairs" in sb:
        rows.append({"metric": "SaltBridges_unique_pairs", "value": sb["total_unique_pairs"], "unit": "count"})
    if isinstance(sb, dict) and "mean_salt_bridges" in sb:
        rows.append({"metric": "SaltBridges_mean_per_frame", "value": sb["mean_salt_bridges"], "unit": "count"})

    # Secondary structure fractions
    ss = result.secondary_structure
    if isinstance(ss, dict):
        for key, mean_key in [("helix_fraction", "mean_helix"), ("sheet_fraction", "mean_sheet"), ("coil_fraction", "mean_coil")]:
            if mean_key in ss:
                rows.append({"metric": f"SS_{key}", "value": ss[mean_key], "unit": "fraction"})

    # Entropy
    ent = result.entropy
    if isinstance(ent, dict) and "total_entropy_kJ_mol_K" in ent:
        rows.append({"metric": "Entropy_total", "value": ent["total_entropy_kJ_mol_K"], "unit": "kJ/mol/K"})

    # Energy decomposition
    energy = result.energy_decomposition
    if isinstance(energy, dict) and "total_energy" in energy:
        for r, v in zip(energy.get("resids", []), energy["total_energy"]):
            rows.append({"metric": "Energy_per_residue", "residue": r, "value": v, "unit": "kJ/mol"})

    # NMA B-factors
    nma = result.nma
    if isinstance(nma, dict) and "bfactors" in nma:
        for r, v in zip(nma.get("resids", []), nma["bfactors"]):
            rows.append({"metric": "NMA_bfactor", "residue": r, "value": v, "unit": "norm"})

    # PRS effector/sensor scores
    prs = result.prs
    if isinstance(prs, dict) and "effector_scores" in prs:
        for r, v in zip(prs.get("resids", []), prs["effector_scores"]):
            rows.append({"metric": "PRS_effector", "residue": r, "value": v, "unit": "score"})
        for r, v in zip(prs.get("resids", []), prs.get("sensor_scores", [])):
            rows.append({"metric": "PRS_sensor", "residue": r, "value": v, "unit": "score"})

    # Water bridges
    wb = result.water_bridges
    if isinstance(wb, dict) and "bridges" in wb:
        for b in wb["bridges"]:
            rows.append({"metric": "WaterBridge", "residue": f"{b['resid_1']}-{b['resid_2']}", "value": b["occupancy"], "unit": "fraction"})

    # tICA timescales
    tica = result.tica
    if isinstance(tica, dict) and "timescales" in tica:
        for i, ts in enumerate(tica["timescales"]):
            if ts != float('inf'):
                rows.append({"metric": f"tICA_timescale_tIC{i+1}", "value": ts, "unit": "frames"})

    # Dynamic network community stability
    dn = result.dynamic_network
    if isinstance(dn, dict) and "community_stability" in dn:
        for r, v in zip(dn.get("resids", []), dn["community_stability"]):
            rows.append({"metric": "DynNetwork_stability", "residue": r, "value": v, "unit": "fraction"})

    # Interaction fingerprints consensus
    ifp = result.interaction_fingerprints
    if isinstance(ifp, dict) and "consensus_fingerprint" in ifp:
        for r, v in zip(ifp.get("resids", []), ifp["consensus_fingerprint"]):
            rows.append({"metric": "IFP_consensus", "residue": r, "value": v, "unit": "score"})

    # VAE reconstruction error
    vae = result.vae
    if isinstance(vae, dict) and "reconstruction_error" in vae:
        rows.append({"metric": "VAE_recon_error", "value": vae["reconstruction_error"], "unit": "MSE"})
        for i, v in enumerate(vae.get("latent_variance", [])):
            rows.append({"metric": f"VAE_latent_var_dim{i+1}", "value": v, "unit": "variance"})

    # PCA explained variance
    pca = result.pca
    if isinstance(pca, dict) and "explained_variance" in pca:
        for i, v in enumerate(pca["explained_variance"]):
            rows.append({"metric": f"PCA_variance_PC{i+1}", "value": v, "unit": "fraction"})

    # Convergence
    conv = result.convergence
    if isinstance(conv, dict) and "convergence_score" in conv:
        rows.append({"metric": "Convergence_score", "value": conv["convergence_score"], "unit": "0-1"})
        if "rmsd_drift" in conv:
            rows.append({"metric": "Convergence_RMSD_drift", "value": conv["rmsd_drift"], "unit": "fraction"})

    # Clustering
    clust = result.clustering
    if isinstance(clust, dict) and "n_clusters" in clust:
        rows.append({"metric": "Clustering_n_clusters", "value": clust["n_clusters"], "unit": "count"})

    # MSM
    msm = result.msm
    if isinstance(msm, dict) and "n_states" in msm:
        rows.append({"metric": "MSM_n_states", "value": msm["n_states"], "unit": "count"})

    # GNN importance
    gnn = result.gnn_results
    if isinstance(gnn, dict) and "importance_scores" in gnn:
        resids_gnn = gnn.get("resids", list(range(len(gnn["importance_scores"]))))
        for r, v in zip(resids_gnn, gnn["importance_scores"]):
            rows.append({"metric": "GNN_importance", "residue": r, "value": v, "unit": "score"})

    # Allosteric hub scores
    allo = result.allosteric
    if isinstance(allo, dict) and "hub_residues" in allo:
        for h in allo["hub_residues"]:
            rows.append({"metric": "Allosteric_hub", "residue": h.get("resid"), "value": h.get("score", 0), "unit": "score"})

    # Binding kinetics summary
    bk = result.binding_kinetics
    if isinstance(bk, dict) and "kon_estimate_per_ps" in bk:
        rows.append({"metric": "BindingKinetics_kon", "value": bk["kon_estimate_per_ps"], "unit": "1/ps"})
        rows.append({"metric": "BindingKinetics_koff", "value": bk["koff_estimate_per_ps"], "unit": "1/ps"})
        rows.append({"metric": "BindingKinetics_residence_continuous", "value": bk.get("residence_time_continuous_ps", 0), "unit": "ps"})

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=["metric", "time", "residue", "value", "unit"])
            writer.writeheader()
            writer.writerows(rows)

    return csv_path


def export_pdf(result, output_dir: Path) -> Path:
    """Export analysis report as PDF using kaleido (item 60)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "analysis_report.pdf"

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.io as pio
    except ImportError:
        raise RuntimeError("plotly required for PDF export")

    plots = result.plots or {}
    info = result.trajectory_info or {}
    insights = result.biological_insights or []

    # Collect all plot figures
    figs = []
    plot_order = [
        "rmsd_plot", "rmsf_plot", "rg_plot", "ss_plot", "hbond_plot",
        "contact_map", "pca_plot", "dccm_plot", "fel_plot", "clustering_plot",
        "sasa_plot", "dimensionality_plot", "dimensionality_3d_plot",
        "convergence_plot", "binding_kinetics_plot", "network_graph_plot",
        "gnn_plot", "transformer_plot", "training_loss_plot",
        "water_bridges_plot", "energy_plot", "prs_plot", "nma_plot",
        "entropy_plot", "tunnel_plot", "vae_plot", "dynamic_network_plot",
    ]

    for plot_key in plot_order:
        json_str = plots.get(plot_key, "")
        if json_str:
            try:
                fig_dict = json.loads(json_str)
                fig = go.Figure(fig_dict)
                figs.append(fig)
            except Exception:
                continue

    if not figs:
        raise RuntimeError("No plots available for PDF export")

    # Write each plot as a page-sized image and combine into a single PDF
    # Using kaleido for static image export
    images = []
    for fig in figs:
        try:
            img_bytes = pio.to_image(fig, format="png", width=1000, height=600, scale=2)
            images.append(img_bytes)
        except Exception:
            continue

    if not images:
        raise RuntimeError("Could not render any plots to images")

    # Build a simple HTML-to-PDF approach: write images into an HTML file,
    # then use a basic image-only PDF
    _write_image_pdf(images, info, insights, pdf_path)

    return pdf_path


def _write_image_pdf(images: list, info: dict, insights: list, pdf_path: Path):
    """Write a simple PDF from rendered plot images."""
    try:
        # Try using fpdf2 if available
        from fpdf import FPDF
        import tempfile
        import os

        pdf = FPDF(orientation='L', unit='mm', format='A4')
        pdf.set_auto_page_break(auto=False)

        # Title page
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 24)
        pdf.cell(0, 40, 'MD AI Analysis Report', ln=True, align='C')
        pdf.set_font('Helvetica', '', 14)
        pdf.cell(0, 10, f"Frames: {info.get('n_frames', 'N/A')}  |  "
                        f"Atoms: {info.get('n_atoms', 'N/A')}  |  "
                        f"Residues: {info.get('n_residues', 'N/A')}", ln=True, align='C')

        if insights:
            pdf.ln(10)
            pdf.set_font('Helvetica', 'B', 16)
            pdf.cell(0, 10, 'Key Insights:', ln=True)
            pdf.set_font('Helvetica', '', 11)
            for ins in insights[:10]:
                text = f"[{ins.get('category', '')}] {ins.get('description', '')}"
                pdf.multi_cell(0, 6, text)
                pdf.ln(2)

        # Plot pages
        tmp_files = []
        for i, img_bytes in enumerate(images):
            tmp_path = os.path.join(tempfile.gettempdir(), f"md_plot_{i}.png")
            with open(tmp_path, 'wb') as f:
                f.write(img_bytes)
            tmp_files.append(tmp_path)

            pdf.add_page()
            pdf.image(tmp_path, x=10, y=10, w=277)

        pdf.output(str(pdf_path))

        # Cleanup temp files
        for tmp in tmp_files:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    except ImportError:
        # Fallback: write a basic HTML file and note that fpdf2 is needed
        # Save plot images as standalone HTML with print-friendly CSS
        html_content = """<!DOCTYPE html><html><head><title>MD Report PDF</title>
        <style>@media print { img { page-break-after: always; max-width: 100%; } }</style>
        </head><body>
        <h1>MD AI Analysis Report</h1>
        <p>Install fpdf2 for native PDF: pip install fpdf2</p>
        </body></html>"""
        pdf_path.with_suffix('.html').write_text(html_content, encoding='utf-8')
        raise RuntimeError("fpdf2 not installed. Install with: pip install fpdf2")
