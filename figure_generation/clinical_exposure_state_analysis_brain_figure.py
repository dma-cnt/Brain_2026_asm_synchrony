#!/usr/bin/env python3
"""Create revised Figure 2 for Brain submission.

This keeps the same patient-level summaries as clinical_exposure_state_analysis.py
but uses manuscript-facing titles, labels, annotations, and sensitivity rows. It
writes a high-resolution TIFF plus a PNG preview to ``figure_generation/outputs``
by default.

Run with:
    conda run -n cnt python figure_generation/clinical_exposure_state_analysis_brain_figure.py
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import warnings
from pathlib import Path

_cache_root = Path(tempfile.gettempdir()) / "asm_synchrony_cache"
_cache_root.mkdir(parents=True, exist_ok=True)
(_cache_root / "mpl").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root))
os.environ.setdefault("MPLCONFIGDIR", str(_cache_root / "mpl"))

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figure_style import (
    AXIS_LINE_WIDTH,
    BODY_FONT_SIZE,
    CARB_COLOR,
    EDGE_COLOR,
    EXPORT_PAD_INCHES,
    LEGEND_FONT_SIZE,
    LEV_COLOR,
    LINE_COLOR,
    PANEL_LABEL_FONT_SIZE,
    SYNC_COLOR,
    TICK_FONT_SIZE,
    TITLE_FONT_SIZE,
    apply_brain_style,
    style_axis,
)
from matplotlib.transforms import blended_transform_factory
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
BRAIN_R2_DIR = REPO_ROOT / "brain_r2"
sys.path.insert(0, str(BRAIN_R2_DIR))

from clinical_exposure_state_analysis import (
    HIGH_THRESHOLD,
    LOW_THRESHOLD,
    MIN_ROWS_PER_STATE,
    build_patient_summaries,
    paired_stats,
)

warnings.filterwarnings("ignore", category=UserWarning)

HIGH_COLOR = CARB_COLOR
LOW_COLOR = LEV_COLOR
MEDIAN_COLOR = CARB_COLOR
POINT_COLOR = SYNC_COLOR
LINK_COLOR = "#D0D0D0"
ROW_BAND_COLOR = "#F1F1F1"
GRID_COLOR = "#D9D9D9"
REFERENCE_LINE_WIDTH = AXIS_LINE_WIDTH
SMALL_FONT_SIZE = TICK_FONT_SIZE
PANEL_C_PVALUE_TEST = "Wilcoxon signed-rank test"
DEFAULT_TIFF_DPI = 560
SCHOLARONE_MAX_PIXELS = 40_000_000

OUT_TIFF_FILENAME = "figure_2.tiff"
OUT_PREVIEW_FILENAME = "figure_2_preview.png"
OUT_PATIENT_CSV_FILENAME = "figure_2_patient_summaries.csv"
OUT_STATS_CSV_FILENAME = "figure_2_stats.csv"
OUT_LEGEND_TXT_FILENAME = "figure_2_legend_note.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for figure_2.tiff, figure_2_preview.png, and companion CSV/text outputs.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_TIFF_DPI,
        help=(
            "TIFF resolution in dots per inch. The default is set below "
            "ScholarOne's 40 megapixel conversion limit for this figure."
        ),
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
    apply_brain_style(mpl)


def format_axis(ax: plt.Axes, *, grid_axis: str | None = None) -> None:
    style_axis(ax)
    if grid_axis is not None:
        ax.grid(
            axis=grid_axis,
            linestyle=":",
            linewidth=0.75,
            alpha=0.42,
            color=GRID_COLOR,
        )


def add_row_bands(ax: plt.Axes, n_rows: int) -> None:
    for i in range(n_rows):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color=ROW_BAND_COLOR, alpha=0.75, zorder=0)


def add_panel_label(ax: plt.Axes, label: str, *, x: float = -0.10, y: float = 1.06) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_FONT_SIZE,
        fontweight="bold",
        color=LINE_COLOR,
        va="bottom",
        ha="right",
    )


def format_pvalue_math(p_value: float) -> str:
    """Format P values for figure text using the manuscript convention."""
    if not np.isfinite(p_value):
        return r"$P$ = NA"
    if p_value < 1e-3:
        return r"$P < 0.001$"
    return rf"$P = {p_value:.3f}$"


def format_pvalue_text(p_value: float) -> str:
    """Plain-text version for the companion legend note."""
    if not np.isfinite(p_value):
        return "P = NA"
    if p_value < 1e-3:
        return "P < 0.001"
    return f"P = {p_value:.3f}"


def compact_row_label(label: str) -> str:
    return label.replace("\n", "; ")


def sensitivity_grid_for_brain() -> pd.DataFrame:
    variants = [
        (
            "Primary: ±12 h exclusion",
            "Primary: low ASM load ≤50%, high ASM load ≥80%\nexclude windows within 12 h of seizures",
            "Main definition: low exposure ≤50%, high exposure ≥80%, excluding ±12 h around seizures.",
            LOW_THRESHOLD,
            HIGH_THRESHOLD,
            12 * 60,
        ),
        (
            "±1 h exclusion",
            "Low ASM load ≤50%, high ASM load ≥80%\nexclude windows within 1 h of seizures",
            "Less conservative peri-ictal exclusion with the same exposure thresholds.",
            LOW_THRESHOLD,
            HIGH_THRESHOLD,
            60,
        ),
        (
            "No seizure exclusion",
            "Low ASM load ≤50%, high ASM load ≥80%\nno seizure-window exclusion",
            "Same exposure thresholds without peri-ictal seizure exclusion.",
            LOW_THRESHOLD,
            HIGH_THRESHOLD,
            0,
        ),
        (
            "Low ≤33%, high ≥67%",
            "Low ASM load ≤33%, high ASM load ≥67%\nexclude windows within 12 h of seizures",
            "Alternative exposure threshold with ±12 h peri-ictal exclusion.",
            0.33,
            0.67,
            12 * 60,
        ),
        (
            "Low ≤25%, high ≥75%",
            "Low ASM load ≤25%, high ASM load ≥75%\nexclude windows within 12 h of seizures",
            "Stricter exposure separation with ±12 h peri-ictal exclusion.",
            0.25,
            0.75,
            12 * 60,
        ),
    ]

    rows = []
    for plot_label, row_label, purpose, low_threshold, high_threshold, exclusion in variants:
        summary = build_patient_summaries(
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            peri_ictal_exclusion_min=exclusion,
            min_rows_per_state=MIN_ROWS_PER_STATE,
        )
        stats = paired_stats(summary, label=plot_label)
        stats.update(
            {
                "plot_label": plot_label,
                "row_label": row_label,
                "purpose": purpose,
                "low_threshold": low_threshold,
                "high_threshold": high_threshold,
                "peri_ictal_exclusion_min": exclusion,
            }
        )
        rows.append(stats)
    return pd.DataFrame(rows)


def plot_primary_pair(ax: plt.Axes, df: pd.DataFrame) -> None:
    for _, row in df.iterrows():
        ax.plot(
            [0, 1],
            [row["high_asm_median_synchrony"], row["low_asm_median_synchrony"]],
            color=LINK_COLOR,
            alpha=0.34,
            linewidth=0.70,
            zorder=1,
        )

    ax.scatter(
        np.zeros(len(df)),
        df["high_asm_median_synchrony"],
        s=58,
        color=HIGH_COLOR,
        edgecolor="black",
        linewidth=0.55,
        zorder=5,
        label="High exposure",
    )
    ax.scatter(
        np.ones(len(df)),
        df["low_asm_median_synchrony"],
        s=58,
        color=LOW_COLOR,
        edgecolor="black",
        linewidth=0.55,
        zorder=5,
        label="Low exposure",
    )

    ax.set_xlim(-0.25, 1.18)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        ["High exposure\n≥80% medication load", "Low exposure\n≤50% medication load"],
        fontsize=TICK_FONT_SIZE,
    )
    ax.set_ylabel(
        "Patient median synchrony",
        fontsize=BODY_FONT_SIZE,
        fontweight="bold",
    )
    ax.set_title(
        "High and low medication exposure",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
        pad=10,
    )
    format_axis(ax, grid_axis="y")
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    ax.legend(frameon=False, loc="upper left", fontsize=LEGEND_FONT_SIZE)


def plot_patient_difference(ax: plt.Axes, df: pd.DataFrame, stats: dict) -> None:
    plot_df = df.sort_values("delta_low_minus_high").reset_index(drop=True)
    y = np.arange(len(plot_df))
    delta = plot_df["delta_low_minus_high"]

    ax.axvline(0, color=LINE_COLOR, linestyle="--", linewidth=REFERENCE_LINE_WIDTH, zorder=1)
    ax.scatter(
        delta,
        y,
        s=55,
        color=POINT_COLOR,
        edgecolor="black",
        linewidth=0.50,
        zorder=5,
    )
    median_delta = stats["median_delta"]
    ax.axvline(median_delta, color=MEDIAN_COLOR, linewidth=1.9, zorder=2)

    ax.set_yticks([])
    ax.set_xlim(-0.024, 0.070)
    ax.set_xlabel(
        "Δ synchrony: low exposure − high exposure",
        fontsize=BODY_FONT_SIZE,
        fontweight="bold",
    )
    ax.set_title(
        "Within-patient synchrony difference",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
        pad=10,
    )
    format_axis(ax, grid_axis="x")
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)

    data_axes = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(
        0.0012,
        0.66,
        "No difference",
        transform=data_axes,
        ha="left",
        va="center",
        rotation=90,
        fontsize=SMALL_FONT_SIZE,
        bbox=dict(boxstyle="round,pad=0.20", facecolor="white", alpha=0.92, edgecolor="none"),
        zorder=20,
    )
    ax.annotate(
        "Higher synchrony during low exposure",
        xy=(0.058, 3.2),
        xytext=(0.016, 3.2),
        ha="left",
        va="center",
        fontsize=SMALL_FONT_SIZE,
        arrowprops=dict(arrowstyle="->", color=SYNC_COLOR, linewidth=0.9),
        zorder=20,
    )

    annotation = (
        "Each dot = 1 patient\n"
        f"n = {stats['n_patients']} patients\n"
        f"{stats['n_positive_delta']}/{stats['n_patients']} higher during low exposure\n"
        f"median Δ = {median_delta:.4f}\n"
        f"95% CI ({stats['median_delta_ci_low']:.4f}, {stats['median_delta_ci_high']:.4f})"
    )
    ax.text(
        0.98,
        0.50,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=LEGEND_FONT_SIZE,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.96, edgecolor=EDGE_COLOR),
        zorder=20,
    )


def plot_sensitivity(ax: plt.Axes, sens: pd.DataFrame) -> None:
    plot_df = sens.reset_index(drop=True)
    y = np.arange(len(plot_df))
    err_low = plot_df["median_delta"] - plot_df["median_delta_ci_low"]
    err_high = plot_df["median_delta_ci_high"] - plot_df["median_delta"]

    add_row_bands(ax, len(plot_df))
    ax.axvline(0, color=LINE_COLOR, linestyle="--", linewidth=REFERENCE_LINE_WIDTH, zorder=1)
    for i, row in plot_df.iterrows():
        is_primary = row["plot_label"].startswith("Primary")
        color = MEDIAN_COLOR if is_primary else SYNC_COLOR
        ax.errorbar(
            row["median_delta"],
            i,
            xerr=[[err_low.iloc[i]], [err_high.iloc[i]]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            markersize=7,
            linewidth=1.45,
            zorder=5,
        )
        ax.text(
            row["median_delta_ci_high"] + 0.001,
            i,
            f"n={int(row['n_patients'])}, Wilcoxon {format_pvalue_math(row['wilcoxon_p_two_sided'])}",
            va="center",
            ha="left",
            fontsize=SMALL_FONT_SIZE,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["row_label"].tolist(), fontsize=TICK_FONT_SIZE)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlabel(
        "Median Δ synchrony: low exposure − high exposure",
        fontsize=BODY_FONT_SIZE,
        fontweight="bold",
    )
    ax.set_title(
        "Robustness to exposure thresholds and peri-ictal exclusions",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
        pad=10,
    )
    format_axis(ax, grid_axis="x")
    ax.tick_params(axis="x", labelsize=TICK_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE, pad=8)
    x_min = min(0.0, float(plot_df["median_delta_ci_low"].min()) - 0.0015)
    x_max = float(plot_df["median_delta_ci_high"].max()) + 0.010
    ax.set_xlim(x_min, x_max)


def create_figure(primary: pd.DataFrame, stats: dict, sens: pd.DataFrame) -> plt.Figure:
    configure_matplotlib()
    fig = plt.figure(figsize=(14.1, 8.2), facecolor="white", constrained_layout=True)
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.08, 0.92],
        hspace=0.40,
    )
    top = outer[0].subgridspec(1, 3, width_ratios=[1.02, 0.18, 1.80], wspace=0.0)
    bottom = outer[1].subgridspec(1, 3, width_ratios=[0.55, 1.86, 0.55], wspace=0.02)
    ax_a = fig.add_subplot(top[0, 0])
    ax_b = fig.add_subplot(top[0, 2])
    ax_c = fig.add_subplot(bottom[0, 1])

    plot_primary_pair(ax_a, primary)
    plot_patient_difference(ax_b, primary, stats)
    plot_sensitivity(ax_c, sens)

    for label, ax, x in zip(["A", "B", "C"], [ax_a, ax_b, ax_c], [-0.12, -0.02, -0.12]):
        add_panel_label(ax, label, x=x, y=1.06)

    return fig


def flatten_tiff_to_rgb(path: Path, dpi: int) -> None:
    image = Image.open(path)
    if image.mode == "RGBA":
        flattened = Image.new("RGB", image.size, "white")
        flattened.paste(image, mask=image.getchannel("A"))
        image = flattened
    elif image.mode != "RGB":
        image = image.convert("RGB")
    image.save(path, format="TIFF", dpi=(dpi, dpi), compression="tiff_lzw")


def validate_submission_pixel_count(path: Path) -> tuple[int, int, int]:
    image = Image.open(path)
    width, height = image.size
    total_pixels = width * height
    if total_pixels >= SCHOLARONE_MAX_PIXELS:
        raise ValueError(
            f"{path} is {width} x {height} = {total_pixels:,} pixels, "
            f"which exceeds the ScholarOne limit of less than {SCHOLARONE_MAX_PIXELS:,} pixels. "
            "Lower --dpi or reduce the figure size."
        )
    return width, height, total_pixels


def write_legend_note(path: Path, stats: dict, sens: pd.DataFrame) -> None:
    row_notes = "\n".join(f"- {compact_row_label(row.row_label)}." for row in sens.itertuples())
    legend = (
        "Suggested Figure 2 legend note for the revised clinical exposure-state figure.\n\n"
        "ASM, anti-seizure medication. Panel A shows each patient's median synchrony during high "
        "and low medication exposure. Panel B shows the within-patient difference, calculated as "
        "low exposure minus high exposure. Positive values indicate higher synchrony during low "
        "medication exposure. Exact sign test, "
        f"{format_pvalue_text(stats['sign_test_p_two_sided'])}; Wilcoxon signed-rank test, "
        f"{format_pvalue_text(stats['wilcoxon_p_two_sided'])}. Panel C shows robustness analyses "
        f"with displayed P-values from the two-sided {PANEL_C_PVALUE_TEST}, "
        "using the following definitions:\n"
        f"{row_notes}\n"
    )
    with path.open("w", encoding="utf-8") as handle:
        handle.write(legend)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tiff_path = output_dir / OUT_TIFF_FILENAME
    preview_path = output_dir / OUT_PREVIEW_FILENAME
    patient_csv_path = output_dir / OUT_PATIENT_CSV_FILENAME
    stats_csv_path = output_dir / OUT_STATS_CSV_FILENAME
    legend_txt_path = output_dir / OUT_LEGEND_TXT_FILENAME

    print("Building revised Figure 2 clinical exposure-state figure")
    primary = build_patient_summaries()
    stats = paired_stats(primary, label="primary")
    sens = sensitivity_grid_for_brain()

    primary.to_csv(patient_csv_path, index=False)
    sens.to_csv(stats_csv_path, index=False)
    write_legend_note(legend_txt_path, stats, sens)

    fig = create_figure(primary, stats, sens)
    fig.savefig(
        tiff_path,
        dpi=args.dpi,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=EXPORT_PAD_INCHES,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    flatten_tiff_to_rgb(tiff_path, args.dpi)
    tiff_width, tiff_height, tiff_pixels = validate_submission_pixel_count(tiff_path)
    fig.savefig(
        preview_path,
        dpi=220,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=EXPORT_PAD_INCHES,
    )
    plt.close(fig)

    print("\nPRIMARY EXPOSURE-STATE CONTRAST")
    print(
        f"  n = {stats['n_patients']} patients; "
        f"{stats['n_positive_delta']}/{stats['n_patients']} higher during low exposure"
    )
    print(
        f"  median delta = {stats['median_delta']:.6f} "
        f"(bootstrap 95% CI {stats['median_delta_ci_low']:.6f}, "
        f"{stats['median_delta_ci_high']:.6f})"
    )
    print(
        f"  sign test: {format_pvalue_text(stats['sign_test_p_two_sided'])}; "
        f"Wilcoxon signed-rank: {format_pvalue_text(stats['wilcoxon_p_two_sided'])}"
    )
    print(
        f"  TIFF dimensions = {tiff_width} x {tiff_height} "
        f"({tiff_pixels:,} pixels; ScholarOne limit < {SCHOLARONE_MAX_PIXELS:,})"
    )
    print("\nSaved:")
    print(f"  {tiff_path}")
    print(f"  {preview_path}")
    print(f"  {patient_csv_path}")
    print(f"  {stats_csv_path}")
    print(f"  {legend_txt_path}")


if __name__ == "__main__":
    main()
