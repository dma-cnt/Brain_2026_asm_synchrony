"""Shared visual style for Brain manuscript figures."""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

BLUE = "#2b7bba"
BLUE_LIGHT = "#a4c8e1"
RED = "#cf4446"
NEUTRAL = "#34495e"
DARK = "#222222"
MID_GRAY = "#4d4d4d"
POINT_GRAY = "#595959"
SOFT_GRAY = "#7f8c8d"
LINK_GRAY = "#b8b8b8"
ROW_BAND = "#f0f0f0"
GRID_GRAY = "#d9d9d9"
BOX_EDGE = "#cccccc"

TITLE_SIZE = 16
AXIS_LABEL_SIZE = 14
TICK_LABEL_SIZE = 12.5
COMPACT_TICK_LABEL_SIZE = 11.5
ANNOTATION_SIZE = 11.5
SMALL_ANNOTATION_SIZE = 10.5
PANEL_LABEL_SIZE = 24

AXIS_LABEL_WEIGHT = "normal"
AXIS_LINE_WIDTH = 1.1
REFERENCE_LINE_WIDTH = 1.1
GRID_LINE_WIDTH = 0.9
GRID_ALPHA = 0.42


def set_brain_theme() -> None:
    """Apply typography and line defaults shared across manuscript figures."""
    sns.set_theme(style="ticks", context="paper", font_scale=1.0)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": TICK_LABEL_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "axes.titleweight": "bold",
            "axes.labelsize": AXIS_LABEL_SIZE,
            "axes.labelweight": AXIS_LABEL_WEIGHT,
            "axes.linewidth": AXIS_LINE_WIDTH,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": TICK_LABEL_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axes(
    ax: plt.Axes,
    *,
    grid_axis: str | None = None,
    hide_left: bool = False,
    hide_bottom: bool = False,
    grid_alpha: float = GRID_ALPHA,
) -> None:
    """Use common spine and grid treatment for clean journal panels."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(not hide_left)
    ax.spines["bottom"].set_visible(not hide_bottom)
    for side in ("left", "bottom"):
        if ax.spines[side].get_visible():
            ax.spines[side].set_linewidth(AXIS_LINE_WIDTH)
    if grid_axis is not None:
        ax.grid(
            axis=grid_axis,
            linestyle=":",
            linewidth=GRID_LINE_WIDTH,
            alpha=grid_alpha,
            color=GRID_GRAY,
        )


def add_row_bands(ax: plt.Axes, n_rows: int) -> None:
    """Add light alternating row bands behind horizontal dot/interval plots."""
    for i in range(n_rows):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color=ROW_BAND, alpha=0.55, zorder=0)


def add_panel_label(
    ax: plt.Axes,
    label: str,
    *,
    x: float = -0.12,
    y: float = 1.05,
) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        color=DARK,
        va="bottom",
        ha="right",
    )
