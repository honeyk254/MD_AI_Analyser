"""
Shared Plotly theming and plotting utilities.

Consolidates the repeated ``_dark_layout`` call and colour constants
used across all plot generators.

Constants
---------
PAPER_BG, PLOT_BG, ACCENT_*, TEXT_COLOR
    Consistent dark-theme colour palette.

Functions
---------
apply_dark_theme
    Apply the standard MD AI Analyzer dark theme to any Plotly figure.
safe_plot
    Decorator/wrapper to safely execute a plot generator and return None on error.
"""
import functools
import logging
from typing import Optional, Callable

import plotly.graph_objects as go

logger = logging.getLogger("md_ai_analyzer")

# ── Colour palette ────────────────────────────────────────────
PAPER_BG = "#1a1a2e"
PLOT_BG = "#16213e"
TEXT_COLOR = "#E0E0E0"

ACCENT_CYAN = "#00d4ff"
ACCENT_RED = "#ff6b6b"
ACCENT_TEAL = "#4ecdc4"
ACCENT_YELLOW = "#ffd93d"
ACCENT_PURPLE = "#a29bfe"
ACCENT_PINK = "#fd79a8"
ACCENT_ORANGE = "#e17055"
ACCENT_GREEN = "#55efc4"
ACCENT_LIGHT_BLUE = "#74b9ff"
ACCENT_DARK_PURPLE = "#6c5ce7"
ACCENT_DARK_TEAL = "#00cec9"
ACCENT_SAGE = "#00b894"

COMMUNITY_COLORS = [
    ACCENT_GREEN, ACCENT_PURPLE, ACCENT_RED, ACCENT_YELLOW,
    ACCENT_PINK, ACCENT_CYAN, ACCENT_ORANGE, ACCENT_LIGHT_BLUE,
    ACCENT_DARK_TEAL, ACCENT_DARK_PURPLE,
]


def apply_dark_theme(
    fig: go.Figure,
    title: str,
    xaxis: str = "",
    yaxis: str = "",
    height: Optional[int] = None,
) -> go.Figure:
    """Apply the standard dark theme to a Plotly figure.

    Parameters
    ----------
    fig : go.Figure
    title : str
    xaxis, yaxis : str
        Axis labels.
    height : int, optional
        Override figure height.

    Returns
    -------
    go.Figure
        The same figure, mutated in-place.
    """
    layout_kwargs = dict(
        title=dict(text=title, font=dict(size=18, color=TEXT_COLOR)),
        template="plotly_dark",
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT_COLOR),
        xaxis_title=xaxis,
        yaxis_title=yaxis,
        margin=dict(l=60, r=30, t=60, b=50),
    )
    if height is not None:
        layout_kwargs["height"] = height
    fig.update_layout(**layout_kwargs)
    return fig


def safe_plot(func: Callable) -> Callable:
    """Decorator that catches exceptions and returns *None* on failure.

    Also logs a warning with the plot function name and error message.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            logger.warning("Plot '%s' failed: %s", func.__name__, exc)
            return None
    return wrapper
