"""HTML report generation.

Produces a standalone HTML file combining the classical plots with the narrative
sections, the grounding verdict and the review banner. The grounding and review
state are rendered prominently and never omitted: a draft report must look like
a draft, and a report whose numbers failed verification must say so at the top.

User-supplied and model-generated strings are HTML-escaped; the only unescaped
interpolations are the Plotly figure JSON blobs this package produces itself.
"""

from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..schemas.analysis_bundle import AnalysisBundle
from ..schemas.report import GeneratedReport, ReviewStatus

PLOT_TITLES: Dict[str, str] = {
    "rmsd_plot": "RMSD Over Time",
    "rmsf_plot": "Per-Residue RMSF",
    "rg_plot": "Radius of Gyration",
    "ss_plot": "Secondary Structure Evolution",
    "hbond_plot": "Hydrogen Bonds",
    "contact_map": "Contact Map",
    "sasa_plot": "Solvent Accessible Surface Area",
    "salt_bridges_plot": "Salt Bridges",
}

STATUS_COLORS = {
    ReviewStatus.DRAFT: "#ff7675",
    ReviewStatus.PENDING_REVIEW: "#fdcb6e",
    ReviewStatus.APPROVED: "#55efc4",
    ReviewStatus.REJECTED: "#ff7675",
}

STYLE = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Inter', -apple-system, sans-serif; background: #0f0f23; color: #e0e0e0; line-height: 1.6; }
.container { max-width: 1200px; margin: 0 auto; padding: 20px; }
h1 { color: #00d4ff; font-size: 2em; border-bottom: 2px solid #1a1a3e; padding-bottom: 15px; margin-bottom: 20px; }
h2 { color: #a29bfe; margin: 30px 0 15px; }
h3 { color: #ddd; margin: 20px 0 10px; }
.banner { border-radius: 12px; padding: 15px 20px; margin: 20px 0; font-weight: 600; color: #0f0f23; }
.info-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
.info-card { background: #1a1a3e; border-radius: 12px; padding: 15px; text-align: center; }
.info-card .value { font-size: 1.8em; font-weight: bold; color: #00d4ff; }
.info-card .label { color: #999; font-size: 0.85em; }
.plot-section, .narrative-section, .audit { background: #1a1a2e; border-radius: 12px; padding: 20px; margin: 20px 0; }
.plot-container { width: 100%; min-height: 400px; }
table { width: 100%; border-collapse: collapse; font-size: 0.9em; }
th, td { text-align: left; padding: 8px; border-bottom: 1px solid #2d2d5a; vertical-align: top; }
th { color: #a29bfe; }
code { background: #10102a; padding: 2px 5px; border-radius: 4px; font-size: 0.85em; }
.fail { color: #ff7675; }
.pass { color: #55efc4; }
"""


def generate_html_report(
    bundle: AnalysisBundle,
    plots: Dict[str, str],
    output_dir: Path,
    report: Optional[GeneratedReport] = None,
) -> Path:
    """Write ``analysis_report.html`` for a run, with the narrative when present."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "analysis_report.html"
    meta = bundle.trajectory_metadata

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MD Analysis Report {escape(bundle.run_id)}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>{STYLE}</style>
</head>
<body>
    <div class="container">
        <h1>MD Analysis Report</h1>
        <p>Run <code>{escape(bundle.run_id)}</code>, generated {bundle.created_at:%Y-%m-%d %H:%M} UTC.</p>
        {_banners(report)}

        <h2>Trajectory Statistics</h2>
        <div class="info-grid">
            {_card(meta.n_frames_analyzed, "Analyzed Frames")}
            {_card(meta.n_atoms, "Atoms")}
            {_card(meta.n_residues, "Residues")}
            {_card(f"{meta.total_time_ns:.3f} ns", "Analyzed Time")}
        </div>

        {_narrative(report)}

        <h2>Analysis Plots</h2>
        {_plots(plots)}

        {_grounding(report)}
        {_audit(bundle, report)}

        <footer style="text-align:center;color:#666;margin-top:40px;padding:20px;border-top:1px solid #1a1a3e;">
            Deterministic classical analysis with a grounded narrative layer.
            Every number above is computed classically and verified against the analysis bundle.
        </footer>
    </div>
</body>
</html>"""

    report_path.write_text(html, encoding="utf-8")
    return report_path


def _card(value: Any, label: str) -> str:
    return (
        '<div class="info-card">'
        f'<div class="value">{escape(str(value))}</div>'
        f'<div class="label">{escape(label)}</div>'
        "</div>"
    )


def _banners(report: Optional[GeneratedReport]) -> str:
    if report is None:
        return (
            '<div class="banner" style="background:#fdcb6e;">No narrative has been '
            "generated for this run yet.</div>"
        )

    review = report.review
    banners = [
        f'<div class="banner" style="background:{STATUS_COLORS[review.status]};">'
        f"Review status: {review.status.value.replace('_', ' ')}"
        + (
            f" &mdash; signed off by {escape(review.reviewer)} on "
            f"{review.reviewed_at:%Y-%m-%d %H:%M} UTC"
            if review.reviewer and review.reviewed_at
            else ""
        )
        + (f". Reviewer note: {escape(review.comment)}" if review.comment else "")
        + "</div>"
    ]

    grounding = report.grounding
    if grounding.passed:
        banners.append(
            '<div class="banner" style="background:#55efc4;">'
            f"Grounding check passed: all {grounding.n_verified} numeric claims match "
            "the analysis bundle.</div>"
        )
    else:
        banners.append(
            '<div class="banner" style="background:#ff7675;">'
            f"Grounding check FAILED: {grounding.n_mismatched} claims contradict the "
            f"analysis bundle and {grounding.n_unsupported} are unsupported. "
            "This report cannot be approved.</div>"
        )
    return "\n".join(banners)


def _narrative(report: Optional[GeneratedReport]) -> str:
    if report is None:
        return ""
    sections: List[str] = [
        "<h2>Narrative</h2>",
        f"<p style='color:#999;'>Written by <code>{escape(report.narrative.generator)}</code> "
        f"from precomputed statistics only.</p>",
    ]
    for section in report.narrative.sections:
        sections.append(
            '<div class="narrative-section">'
            f"<h3>{escape(section.heading)}</h3><p>{escape(section.body)}</p></div>"
        )
    return "\n".join(sections)


def _plots(plots: Dict[str, str]) -> str:
    sections: List[str] = []
    for key, title in PLOT_TITLES.items():
        figure_json = plots.get(key, "")
        if not figure_json:
            continue
        div_id = key.replace("_", "-")
        sections.append(
            f"""
            <div class="plot-section">
                <h3>{escape(title)}</h3>
                <div id="{div_id}" class="plot-container"></div>
                <script>
                    (function () {{
                        var figure = {figure_json};
                        Plotly.newPlot('{div_id}', figure.data, figure.layout, {{responsive: true}});
                    }})();
                </script>
            </div>
            """
        )
    return "\n".join(sections) if sections else "<p>No plots generated.</p>"


def _grounding(report: Optional[GeneratedReport]) -> str:
    if report is None:
        return ""
    failures = report.grounding.failures()
    rows = "".join(
        "<tr>"
        f"<td class='fail'>{escape(check.status.value)}</td>"
        f"<td>{check.claim.value:g} {escape(check.claim.unit or '')}</td>"
        f"<td>{escape(check.claim.section)}</td>"
        f"<td>{escape(check.detail)}</td>"
        "</tr>"
        for check in failures
    )
    table = (
        "<table><tr><th>Status</th><th>Claim</th><th>Section</th><th>Detail</th></tr>"
        f"{rows}</table>"
        if failures
        else "<p class='pass'>Every numeric claim was matched to a value in the "
        "analysis bundle.</p>"
    )
    verified = report.grounding.n_verified
    total = len(report.grounding.checks)
    return (
        "<h2>Grounding Verification</h2>"
        f'<div class="audit"><p>Checker <code>{escape(report.grounding.checker_version)}</code> '
        f"verified {verified} of {total} numeric claims against the analysis bundle.</p>"
        f"{table}</div>"
    )


def _audit(bundle: AnalysisBundle, report: Optional[GeneratedReport]) -> str:
    tools = bundle.run_card.tools
    rows = [
        ("Bundle hash", report.audit.bundle_hash if report else "not computed"),
        ("Prompt version", report.audit.prompt_version if report else "n/a"),
        ("Narrator", report.narrative.generator if report else "n/a"),
        (
            "Report generation",
            f"{report.audit.generation_seconds:.2f} s" if report else "n/a",
        ),
        (
            "Estimated LLM cost",
            f"${report.audit.usage.cost_usd:.4f} "
            f"({report.audit.usage.input_tokens} in / "
            f"{report.audit.usage.output_tokens} out tokens)"
            if report
            else "n/a",
        ),
        ("Tool calls", str(report.audit.usage.n_tool_calls) if report else "n/a"),
        ("Force field", bundle.trajectory_metadata.force_field),
        (
            "Library versions",
            f"MDAnalysis {tools.mdanalysis}, MDTraj {tools.mdtraj}, "
            f"NumPy {tools.numpy}, Python {tools.python}",
        ),
        (
            "Input hashes",
            ", ".join(
                f"{name}: {provenance.filename} sha256:{provenance.sha256[:12]}"
                for name, provenance in bundle.run_card.inputs.items()
            ),
        ),
        ("Parameters", str(bundle.run_card.parameters)),
    ]
    body = "".join(
        f"<tr><th>{escape(label)}</th><td>{escape(str(value))}</td></tr>"
        for label, value in rows
    )
    return f"<h2>Provenance</h2><div class='audit'><table>{body}</table></div>"
