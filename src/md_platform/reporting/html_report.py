"""HTML Report Generation.

Generates a standalone HTML file from the AnalysisBundle and generated plot JSONs.
"""

from html import escape
from pathlib import Path
from typing import Dict, Optional

from ..aggregation.report_summary import build_report_summary
from ..ml.schemas import MLAnalysisBundle
from ..schemas.analysis_bundle import AnalysisBundle


def generate_html_report(
    bundle: AnalysisBundle,
    plots: Dict[str, str],
    output_dir: Path,
    narrative_report: Optional[str] = None,
    reviewer_signoff: Optional[str] = None,
    ml_bundle: Optional[MLAnalysisBundle] = None,
) -> Path:
    """Generate a standalone HTML report with embedded Plotly charts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "analysis_report.html"
    summary = build_report_summary(bundle, ml_bundle)

    # Generate plot divs
    plot_html_sections: list[str] = []
    plot_names_map: dict[str, str] = {
        "rmsd_plot": "RMSD Over Time",
        "rmsf_plot": "Per-Residue RMSF",
        "rg_plot": "Radius of Gyration",
        "ss_plot": "Secondary Structure Evolution",
        "hbond_plot": "Hydrogen Bonds",
        "contact_map": "Contact Map",
        "sasa_plot": "Solvent Accessible Surface Area",
        "salt_bridges_plot": "Salt Bridges",
        "ml_comparison_plot": "PCA vs TICA Comparison",
        "tica_plot": "TICA Embedding",
        "msm_timescales_plot": "MSM Implied Timescales",
    }

    for plot_key, plot_title in plot_names_map.items():
        json_str: str = plots.get(plot_key, "")
        if json_str:
            div_id: str = plot_key.replace("_", "-")
            plot_html_sections.append(f"""
            <div class="plot-section">
                <h3>{plot_title}</h3>
                <div id="{div_id}" class="plot-container"></div>
                <script>
                    Plotly.newPlot('{div_id}', {json_str}.data, {json_str}.layout, {{responsive: true}});
                </script>
            </div>
            """)

    meta = bundle.trajectory_metadata
    narrative_html = ""
    if narrative_report:
        narrative_html = f"""
        <section class="narrative-section">
            <h2>LLM Narrative</h2>
            <pre>{escape(narrative_report)}</pre>
        </section>
        """

    ml_html = ""
    ml_summary = summary.get("ml")
    if ml_summary:
        gating = ml_summary["gating"]
        baseline = ml_summary.get("baseline_comparison") or {}
        ml_html = f"""
        <section class="narrative-section">
            <h2>Phase 4 ML Summary</h2>
            <p><strong>Status:</strong> {escape(str(ml_summary['status']))}</p>
            <p><strong>Gate passed:</strong> {escape(str(gating['passed']))}</p>
            <p><strong>Baseline agreement:</strong> {escape(str(baseline.get('state_agreement_nmi', 'n/a')))}</p>
            <p><strong>CK cutoff:</strong> {gating['ck_cutoff']}</p>
        </section>
        """

    audit_html = ""
    if reviewer_signoff:
        audit_html = f"""
        <section class="audit-section">
            <h2>Audit Trail</h2>
            <p>Reviewer sign-off: {escape(reviewer_signoff)}</p>
            <p>Bundle run ID: {escape(bundle.run_id)}</p>
        </section>
        """

    html: str = f"""<!DOCTYPE html>
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
        .narrative-section, .audit-section {{ background: #141428; border-radius: 12px; padding: 20px; margin: 20px 0; }}
        .narrative-section pre {{ white-space: pre-wrap; font-family: inherit; color: #d6d6f2; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>MD AI Analysis Report</h1>

        <h2>Trajectory Statistics</h2>
        <div class="info-grid">
            <div class="info-card">
                <div class="value">{meta.n_frames_analyzed}</div>
                <div class="label">Analyzed Frames</div>
            </div>
            <div class="info-card">
                <div class="value">{meta.n_atoms}</div>
                <div class="label">Atoms</div>
            </div>
            <div class="info-card">
                <div class="value">{meta.n_residues}</div>
                <div class="label">Residues</div>
            </div>
            <div class="info-card">
                <div class="value">{meta.total_time_ns:.1f} ns</div>
                <div class="label">Analyzed Time</div>
            </div>
        </div>

        {narrative_html}
        {ml_html}

        <h2>Analysis Plots</h2>
        {''.join(plot_html_sections) if plot_html_sections else '<p>No plots generated.</p>'}

        {audit_html}

        <footer style="text-align: center; color: #666; margin-top: 40px; padding: 20px; border-top: 1px solid #1a1a3e;">
            Generated by MD AI Analyzer
        </footer>
    </div>
</body>
</html>"""

    report_path.write_text(html, encoding="utf-8")
    return report_path
