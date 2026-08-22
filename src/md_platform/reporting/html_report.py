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
        status_badge = "badge-teal" if ml_summary["status"] == "completed" else "badge-yellow"
        gate_badge = "badge-teal" if gating["passed"] else "badge-magenta"
        ml_html = f"""
        <section class="narrative-section">
            <h2>Phase 4 ML Summary</h2>
            <div>
                <span class="badge {status_badge}">Status: {escape(str(ml_summary['status']))}</span>
                <span class="badge {gate_badge}">Gate {'PASSED' if gating['passed'] else 'BLOCKED'}</span>
                <span class="badge badge-yellow">CK cutoff: {gating['ck_cutoff']}</span>
            </div>
            <p><strong>Baseline agreement (NMI):</strong> {escape(str(baseline.get('state_agreement_nmi', 'n/a')))}</p>
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
        :root {{
            --black: #000000;
            --white: #FFFFFF;
            --paper-beige: #F5F5DC;
            --accent-teal: #00C2CB;
            --accent-magenta: #FF00FF;
            --accent-yellow: #FFD700;
            --radius-1: 12px;
            --radius-2: 24px;
            --shadow-1: 4px 4px 0px var(--black);
            --shadow-2: 12px 12px 0px var(--black);
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Helvetica Neue', Arial, sans-serif;
            background: var(--paper-beige); color: var(--black);
            font-size: 18px; line-height: 1.6;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
        h1 {{
            font-family: 'Arial Black', 'Helvetica Neue', Arial, sans-serif;
            font-size: 64px; font-weight: 900; text-transform: uppercase;
            letter-spacing: -1px; line-height: 1.1; padding: 24px 0;
        }}
        .title-bar {{ display: block; width: 96px; height: 12px; background: var(--accent-yellow); margin-bottom: 24px; }}
        h2 {{
            font-family: 'Arial Black', 'Helvetica Neue', Arial, sans-serif;
            font-size: 32px; text-transform: uppercase; margin: 48px 0 16px;
            display: flex; align-items: center; gap: 12px;
        }}
        h2::before {{ content: ''; width: 16px; height: 32px; background: var(--accent-teal); flex: none; }}
        h3 {{ font-family: 'Arial Black', 'Helvetica Neue', Arial, sans-serif; font-size: 18px; margin: 0 0 12px; }}
        .info-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 24px; margin: 24px 0; }}
        .info-card {{
            background: var(--white); border: 3px solid var(--black);
            border-radius: var(--radius-1); box-shadow: var(--shadow-1);
            padding: 24px; text-align: center;
        }}
        .info-card .value {{ font-family: 'Arial Black', Arial, sans-serif; font-size: 32px; font-weight: 900; }}
        .info-card:first-child .value {{ color: var(--accent-teal); -webkit-text-stroke: 1px var(--black); }}
        .info-card .label {{ font-size: 14px; text-transform: uppercase; font-weight: bold; }}
        .plot-section {{
            background: var(--white); border: 3px solid var(--black);
            border-radius: var(--radius-1); box-shadow: var(--shadow-1);
            padding: 24px; margin: 24px 0;
        }}
        .plot-container {{ width: 100%; min-height: 400px; }}
        .narrative-section {{
            background: var(--white); border: 3px solid var(--black);
            border-radius: var(--radius-1); box-shadow: var(--shadow-1);
            padding: 24px; margin: 24px 0;
        }}
        .narrative-section pre {{ white-space: pre-wrap; font-family: inherit; font-size: 18px; }}
        .audit-section {{
            background: var(--black); color: var(--white);
            border: 3px solid var(--black); border-radius: var(--radius-1);
            box-shadow: var(--shadow-1); padding: 24px; margin: 24px 0;
        }}
        .audit-section h2 {{
            color: var(--accent-yellow);
        }}
        .audit-section h2::before {{ background: var(--accent-yellow); }}
        .badge {{
            display: inline-block; padding: 4px 12px; margin: 0 8px 8px 0;
            border: 3px solid var(--black); border-radius: var(--radius-1);
            font-family: 'Arial Black', Arial, sans-serif; font-size: 14px;
            font-weight: 900; text-transform: uppercase;
        }}
        .badge-teal {{ background: var(--accent-teal); color: var(--black); }}
        .badge-yellow {{ background: var(--accent-yellow); color: var(--black); }}
        .badge-magenta {{ background: var(--accent-magenta); color: var(--black); }}
        footer {{
            text-align: center; font-size: 14px; text-transform: uppercase; font-weight: bold;
            margin-top: 48px; padding: 24px; border-top: 3px solid var(--black);
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>MD AI Analysis Report</h1>
        <span class="title-bar"></span>

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

        <footer>
            Generated by MD AI Analyzer
        </footer>
    </div>
</body>
</html>"""

    report_path.write_text(html, encoding="utf-8")
    return report_path
