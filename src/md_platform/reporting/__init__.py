"""Reporting init."""

from .html_report import generate_html_report
from .plots import generate_all_plots

__all__ = ["generate_all_plots", "generate_html_report"]
