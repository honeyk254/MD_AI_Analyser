"""HTML Report Generation.

Generates a standalone HTML file from the AnalysisBundle and generated plot JSONs.
The deterministic summary (QC, per-module results, ML gating) is rendered from
`build_report_summary`, so the page stands on its own even when the narrative
falls back to the template mode.
"""

import re
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        "dihedrals_plot": "Backbone Dihedral Flexibility",
        "com_plot": "Center-of-Mass Drift",
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

    identity_html = f"""
        <p class="identity-line">
            Run <strong>{escape(bundle.run_id)}</strong>
            &middot; {escape(meta.original_format)} &middot; force field:
            {escape(meta.force_field)} &middot; {meta.timestep_ps:g} ps/frame
        </p>
    """

    qc_html = _qc_section_html(summary)

    modules_html = _modules_section_html(summary)

    narrative_html = ""
    if narrative_report:
        narrative_body = _markdown_to_html(narrative_report)
        narrative_html = f"""
        <section class="narrative-section">
            <h2>LLM Narrative</h2>
            <div class="narrative-body">{narrative_body}</div>
        </section>
        """

    ml_html = _ml_section_html(summary)

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
    <title>{escape(bundle.run_id)} — MD AI Analysis Report</title>
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
        .identity-line {{ font-size: 16px; margin: -8px 0 8px; }}
        .identity-line strong {{
            font-family: 'Arial Black', Arial, sans-serif;
            font-size: 18px;
        }}
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
        table.data-table {{
            width: 100%; border-collapse: collapse; background: var(--white);
            border: 3px solid var(--black); border-radius: var(--radius-1);
            box-shadow: var(--shadow-1); overflow: hidden; margin: 24px 0;
        }}
        table.data-table th, table.data-table td {{
            border: 2px solid var(--black); padding: 10px 14px;
            text-align: left; vertical-align: top; font-size: 15px;
        }}
        table.data-table th {{
            background: var(--black); color: var(--white);
            font-family: 'Arial Black', Arial, sans-serif;
            font-size: 13px; text-transform: uppercase;
        }}
        table.data-table td.module-name {{
            font-family: 'Arial Black', Arial, sans-serif; white-space: nowrap;
        }}
        .pass {{ background: var(--accent-teal); font-weight: bold; }}
        .fail {{ background: var(--accent-magenta); font-weight: bold; }}
        .ref-note {{ font-size: 14px; }}
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
        .narrative-body h3 {{
            font-size: 22px; margin: 20px 0 8px; border-bottom: 3px solid var(--black);
            padding-bottom: 4px;
        }}
        .narrative-body h4 {{ font-size: 18px; margin: 16px 0 6px; }}
        .narrative-body ul {{ padding-left: 28px; margin: 8px 0; }}
        .narrative-body p {{ margin: 8px 0; }}
        .ml-reasons {{ margin: 8px 0 0 20px; }}
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

        {identity_html}

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

        {qc_html}
        {modules_html}
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


def _qc_section_html(summary: Dict[str, Any]) -> str:
    """Render the QC flag table from the aggregated summary."""
    qc = summary.get("qc") or {}
    flag_rows: List[str] = []
    for flag in qc.get("flags", []):
        result_class = "pass" if flag["passed"] else "fail"
        result_text = "PASSED" if flag["passed"] else "FAILED"
        flag_rows.append(
            f"<tr><td class=\"module-name\">{escape(str(flag['check_name']))}</td>"
            f"<td class=\"{result_class}\">{result_text}</td>"
            f"<td>{escape(str(flag['details']))}</td></tr>"
        )
    flags_table = (
        f"<table class=\"data-table\">"
        "<tr><th>Check</th><th>Result</th><th>Details</th></tr>"
        f"{''.join(flag_rows)}</table>"
        if flag_rows
        else "<p>No QC checks recorded.</p>"
    )

    equilibrated = bool(qc.get("is_equilibrated"))
    sufficient = bool(qc.get("sufficient_frames"))
    top_line = (
        f"<p><span class=\"badge {'badge-teal' if equilibrated else 'badge-magenta'}\">"
        f"Equilibrated: {'YES' if equilibrated else 'NO'}</span>"
        f"<span class=\"badge {'badge-teal' if sufficient else 'badge-yellow'}\">"
        f"Sufficient frames: {'YES' if sufficient else 'NO'}</span></p>"
    )
    return (
        "<section><h2>Quality Control</h2>"
        f"{top_line}{flags_table}</section>"
    )


def _modules_section_html(summary: Dict[str, Any]) -> str:
    """Render one row per classical module using the aggregated takeaways."""
    modules = summary.get("modules") or {}
    reference_ranges = summary.get("reference_ranges") or {}
    if not modules:
        return ""

    rows: List[str] = []
    for name, mod in modules.items():
        metric_parts = []
        for metric_name, metric in (mod.get("metrics") or {}).items():
            metric_parts.append(
                f"{escape(metric_name)}: {metric['mean']:.2f} &plusmn; {metric['std']:.2f} "
                f"{escape(str(metric['unit']))}"
            )
        key_result = "<br>".join(metric_parts) if metric_parts else "&mdash;"
        ref_note = escape(reference_ranges.get(name, ""))
        ref_cell = f'<span class="ref-note">{ref_note}</span>' if ref_note else "&mdash;"
        takeaway = escape(str(mod.get("takeaway") or ""))
        runtime = float(mod.get("runtime_seconds") or 0.0)
        version = escape(str(mod.get("version") or ""))
        rows.append(
            f"<tr>"
            f"<td class=\"module-name\">{escape(name)}<br><small>v{version}, {runtime:.2f}s</small></td>"
            f"<td>{key_result}</td>"
            f"<td>{takeaway}</td>"
            f"<td>{ref_cell}</td>"
            f"</tr>"
        )

    table = (
        "<table class=\"data-table\">"
        "<tr><th>Module</th><th>Key metrics (mean &plusmn; std)</th><th>Takeaway</th>"
        "<th>Literature reference</th></tr>"
        f"{''.join(rows)}"
        "</table>"
    )
    return f"<section><h2>Module Results ({len(modules)})</h2>{table}</section>"


def _ml_section_html(summary: Dict[str, Any]) -> str:
    """Render the Phase 4 ML panel: full detail when completed, the refusal reasons when blocked."""
    ml_summary = summary.get("ml")
    if not ml_summary:
        return ""

    gating = ml_summary["gating"]
    baseline = ml_summary.get("baseline_comparison") or {}
    status_badge = "badge-teal" if ml_summary["status"] == "completed" else "badge-yellow"
    gate_badge = "badge-teal" if gating["passed"] else "badge-magenta"

    detail_lines: List[str] = []
    msm = ml_summary.get("msm")
    if msm:
        markovian_text = "passed" if msm.get("is_markovian") else "FAILED"
        cis = msm.get("implied_timescales_ci_ps")
        ts_bits = ", ".join(f"{t:.1f} ps" for t in msm.get("implied_timescales_ps") or [])
        ci_bits = ""
        if cis:
            ci_bits = "; ".join(f"[{lo:.1f}, {hi:.1f}] ps" for lo, hi in cis)
        detail_lines.append(
            f"MSM lag {msm['lag_frames']} frames ({msm['lag_ps']:.2f} ps), "
            f"{msm['n_states']} states; CK deviation {msm['ck_deviation']:.3f} ({markovian_text})."
        )
        if ts_bits:
            detail_lines.append(
                f"Implied timescales: {ts_bits}" + (f" &middot; bootstrap 90% CI {ci_bits}" if ci_bits else "")
            )
    if baseline.get("state_agreement_nmi") is not None:
        detail_lines.append(
            f"Baseline agreement (PCA vs TICA states): "
            f"{baseline['state_agreement_nmi']:.2f} NMI."
        )
    vampnet = ml_summary.get("vampnet_ablation")
    if vampnet and vampnet.get("summary"):
        detail_lines.append(escape(str(vampnet["summary"])))

    if gating["passed"]:
        detail_html = "".join(f"<li>{line}</li>" for line in detail_lines)
        details_block = f"<ul class=\"ml-reasons\">{detail_html}</ul>" if detail_html else ""
    else:
        threshold_line = (
            f"Gating thresholds: observed {gating['observed_frames']} frames vs required minimum "
            f"{gating['minimum_frames_required']}; observed minimum state-pair transitions "
            f"{gating['observed_min_transition_count']} vs required minimum "
            f"{gating['minimum_transition_count_required']}; lag {gating['lag_frames']} frames "
            f"({gating['lag_ps']:.2f} ps); CK cutoff {gating['ck_cutoff']}."
        )
        reason_items = [escape(str(r)) for r in (gating.get("reasons") or [])]
        refusal_reason = ml_summary.get("refusal_reason")
        if refusal_reason:
            reason_items.append(escape(str(refusal_reason)))
        reasons_block = (
            "".join(f"<li>{r}</li>" for r in reason_items) if reason_items else ""
        )
        details_block = (
            f"<p>{threshold_line}</p>"
            + (f"<ul class=\"ml-reasons\">{reasons_block}</ul>" if reasons_block else "")
        )

    analysis_card = ml_summary.get("analysis_card") or {}
    card_note = (
        f"<p class=\"ref-note\">Analysis card: {escape(str(analysis_card.get('title', '')))} — "
        f"{escape(str(analysis_card.get('baseline_protocol', '')))}</p>"
        if analysis_card
        else ""
    )

    return f"""
        <section class="narrative-section">
            <h2>Phase 4 ML Summary</h2>
            <div>
                <span class="badge {status_badge}">Status: {escape(str(ml_summary['status']))}</span>
                <span class="badge {gate_badge}">Gate {'PASSED' if gating['passed'] else 'BLOCKED'}</span>
                <span class="badge badge-yellow">CK cutoff: {gating['ck_cutoff']}</span>
            </div>
            {details_block}
            {card_note}
        </section>
        """


def _markdown_to_html(markdown_text: str) -> str:
    """Minimal markdown rendering (headings, bullets, bold, code) for the narrative.

    The narrative is LLM/template output constrained to a small subset of
    markdown, so this avoids shipping a markdown dependency or dumping a raw
    `<pre>` blob into the page.
    """
    lines = escape(markdown_text).splitlines()
    out: List[str] = []
    in_list = False

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            out.append("</ul>")
            in_list = False

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            close_list()
            continue
        line = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line)
        line = re.sub(r"`([^`]+)`", r"<code>\1</code>", line)
        if line.startswith("### "):
            close_list()
            out.append(f"<h4>{line[4:]}</h4>")
        elif line.startswith("## "):
            close_list()
            out.append(f"<h4>{line[3:]}</h4>")
        elif line.startswith("# "):
            close_list()
            out.append(f"<h3>{line[2:]}</h3>")
        elif line.startswith("- ") or line.startswith("* "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{line[2:]}</li>")
        else:
            close_list()
            out.append(f"<p>{line}</p>")
    close_list()
    return "\n".join(out)
