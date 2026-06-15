"""Shared visual style for revised Brain figure scripts."""

from __future__ import annotations


# Okabe-Ito palette plus neutral colors. These colors are widely used for
# colorblind-accessible scientific figures.
CARB_COLOR = "#0072B2"  # blue
LEV_COLOR = "#CC79A7"  # reddish purple
SEIZURE_COLOR = "#D55E00"  # vermillion
IQR_FILL_COLOR = "#BDBDBD"  # neutral light gray
SYNC_COLOR = "#4D4D4D"
LINE_COLOR = "#111111"
AXIS_COLOR = "#333333"
EDGE_COLOR = "#5C5C5C"

BODY_FONT_SIZE = 12
TICK_FONT_SIZE = 10
LEGEND_FONT_SIZE = 11
TITLE_FONT_SIZE = 15
PANEL_LABEL_FONT_SIZE = 15

AXIS_LINE_WIDTH = 0.8
TIMESERIES_LINE_WIDTH = 1.15
ASM_LINE_WIDTH = 2.1
ASM_LEGEND_LINE_WIDTH = 3.1
EVENT_LINE_WIDTH = 1.5
ROBUSTNESS_LINE_WIDTH = 2.0
ROBUSTNESS_MARKER_SIZE = 7.5
EXPORT_PAD_INCHES = 0.14


def apply_brain_style(mpl_module) -> None:
    """Apply journal-style matplotlib settings shared by revised figures."""
    mpl_module.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": BODY_FONT_SIZE,
            "axes.linewidth": AXIS_LINE_WIDTH,
            "axes.edgecolor": AXIS_COLOR,
            "axes.labelsize": BODY_FONT_SIZE,
            "axes.titlesize": TITLE_FONT_SIZE,
            "xtick.labelsize": TICK_FONT_SIZE,
            "ytick.labelsize": TICK_FONT_SIZE,
            "xtick.major.width": AXIS_LINE_WIDTH,
            "ytick.major.width": AXIS_LINE_WIDTH,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax, *, hide_top_right: bool = True) -> None:
    """Use one consistent axis treatment across revised figures."""
    if hide_top_right:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color(AXIS_COLOR)
        spine.set_linewidth(AXIS_LINE_WIDTH)
    ax.tick_params(
        axis="both",
        width=AXIS_LINE_WIDTH,
        length=3,
        labelsize=TICK_FONT_SIZE,
        color=AXIS_COLOR,
        labelcolor=LINE_COLOR,
    )
