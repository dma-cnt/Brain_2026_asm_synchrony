"""Brain manuscript version of the clinical exposure-state figure.

This keeps the same patient-level summaries as clinical_exposure_state_analysis.py
but uses manuscript-facing titles, labels, annotations, and sensitivity rows.

Run with:
    conda run -n cnt python brain_r2/clinical_exposure_state_analysis_brain_figure.py
"""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/asm_synchrony_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory

import brain_figure_style as bfs
from clinical_exposure_state_analysis import (
    HIGH_THRESHOLD,
    LOW_THRESHOLD,
    MIN_ROWS_PER_STATE,
    SCRIPT_DIR,
    build_patient_summaries,
    paired_stats,
)

warnings.filterwarnings("ignore", category=UserWarning)


OUT_PNG = os.path.join(SCRIPT_DIR, "clinical_exposure_state_analysis_brain_figure.png")
OUT_PDF = os.path.join(SCRIPT_DIR, "clinical_exposure_state_analysis_brain_figure.pdf")
OUT_PATIENT_CSV = os.path.join(SCRIPT_DIR, "clinical_exposure_state_brain_figure_patient_summaries.csv")
OUT_STATS_CSV = os.path.join(SCRIPT_DIR, "clinical_exposure_state_brain_figure_stats.csv")
OUT_LEGEND_TXT = os.path.join(SCRIPT_DIR, "clinical_exposure_state_brain_figure_legend.txt")


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
            color=bfs.LINK_GRAY,
            alpha=0.34,
            linewidth=0.70,
            zorder=1,
        )

    ax.scatter(
        np.zeros(len(df)),
        df["high_asm_median_synchrony"],
        s=58,
        color=bfs.BLUE,
        edgecolor="black",
        linewidth=0.55,
        zorder=5,
        label="High exposure",
    )
    ax.scatter(
        np.ones(len(df)),
        df["low_asm_median_synchrony"],
        s=58,
        color=bfs.RED,
        edgecolor="black",
        linewidth=0.55,
        zorder=5,
        label="Low exposure",
    )

    ax.set_xlim(-0.25, 1.18)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        ["High exposure\n≥80% medication load", "Low exposure\n≤50% medication load"],
        fontsize=bfs.TICK_LABEL_SIZE,
    )
    ax.set_ylabel(
        "Patient median synchrony",
        fontsize=bfs.AXIS_LABEL_SIZE,
        fontweight="bold",
    )
    ax.set_title(
        "Median synchrony during high and low medication exposure",
        fontsize=bfs.TITLE_SIZE,
        fontweight="bold",
        pad=10,
    )
    bfs.style_axes(ax, grid_axis="y")
    ax.tick_params(axis="both", labelsize=bfs.TICK_LABEL_SIZE)
    ax.legend(frameon=False, loc="upper left", fontsize=bfs.TICK_LABEL_SIZE)


def plot_patient_difference(ax: plt.Axes, df: pd.DataFrame, stats: dict) -> None:
    plot_df = df.sort_values("delta_low_minus_high").reset_index(drop=True)
    y = np.arange(len(plot_df))
    delta = plot_df["delta_low_minus_high"]

    ax.axvline(0, color=bfs.DARK, linestyle="--", linewidth=bfs.REFERENCE_LINE_WIDTH, zorder=1)
    ax.scatter(
        delta,
        y,
        s=55,
        color=bfs.POINT_GRAY,
        edgecolor="black",
        linewidth=0.50,
        zorder=5,
    )
    median_delta = stats["median_delta"]
    ax.axvline(median_delta, color=bfs.BLUE, linewidth=1.9, zorder=2)

    ax.set_yticks([])
    ax.set_xlim(-0.024, 0.070)
    ax.set_xlabel(
        "Δ synchrony: low exposure − high exposure",
        fontsize=bfs.AXIS_LABEL_SIZE,
        fontweight="bold",
    )
    ax.set_title(
        "Within-patient synchrony difference",
        fontsize=bfs.TITLE_SIZE,
        fontweight="bold",
        pad=10,
    )
    bfs.style_axes(ax, grid_axis="x")
    ax.tick_params(axis="both", labelsize=bfs.TICK_LABEL_SIZE)

    data_axes = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(
        0.0012,
        0.66,
        "No difference",
        transform=data_axes,
        ha="left",
        va="center",
        rotation=90,
        fontsize=bfs.SMALL_ANNOTATION_SIZE,
        bbox=dict(boxstyle="round,pad=0.20", facecolor="white", alpha=0.92, edgecolor="none"),
        zorder=20,
    )
    ax.annotate(
        "Higher synchrony during low exposure",
        xy=(0.058, 3.2),
        xytext=(0.016, 3.2),
        ha="left",
        va="center",
        fontsize=bfs.SMALL_ANNOTATION_SIZE,
        arrowprops=dict(arrowstyle="->", color=bfs.MID_GRAY, linewidth=0.9),
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
        fontsize=bfs.ANNOTATION_SIZE,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.96, edgecolor=bfs.BOX_EDGE),
        zorder=20,
    )


def plot_sensitivity(ax: plt.Axes, sens: pd.DataFrame) -> None:
    plot_df = sens.reset_index(drop=True)
    y = np.arange(len(plot_df))
    err_low = plot_df["median_delta"] - plot_df["median_delta_ci_low"]
    err_high = plot_df["median_delta_ci_high"] - plot_df["median_delta"]

    bfs.add_row_bands(ax, len(plot_df))
    ax.axvline(0, color=bfs.DARK, linestyle="--", linewidth=bfs.REFERENCE_LINE_WIDTH, zorder=1)
    for i, row in plot_df.iterrows():
        is_primary = row["plot_label"].startswith("Primary")
        color = bfs.BLUE if is_primary else bfs.MID_GRAY
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
            f"n={int(row['n_patients'])}, {format_pvalue_math(row['wilcoxon_p_two_sided'])}",
            va="center",
            ha="left",
            fontsize=bfs.SMALL_ANNOTATION_SIZE,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["row_label"].tolist(), fontsize=bfs.COMPACT_TICK_LABEL_SIZE)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlabel(
        "Median Δ synchrony: low exposure − high exposure",
        fontsize=bfs.AXIS_LABEL_SIZE,
        fontweight="bold",
    )
    ax.set_title(
        "Robustness to exposure thresholds and peri-ictal exclusions",
        fontsize=bfs.TITLE_SIZE,
        fontweight="bold",
        pad=10,
    )
    bfs.style_axes(ax, grid_axis="x")
    ax.tick_params(axis="x", labelsize=bfs.TICK_LABEL_SIZE)
    ax.tick_params(axis="y", labelsize=bfs.COMPACT_TICK_LABEL_SIZE, pad=8)
    x_min = min(0.0, float(plot_df["median_delta_ci_low"].min()) - 0.0015)
    x_max = float(plot_df["median_delta_ci_high"].max()) + 0.010
    ax.set_xlim(x_min, x_max)


def plot_figure(primary: pd.DataFrame, stats: dict, sens: pd.DataFrame) -> None:
    bfs.set_brain_theme()
    fig = plt.figure(figsize=(16, 9.3), dpi=300, constrained_layout=True)
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.08, 0.92],
        hspace=0.18,
    )
    top = outer[0].subgridspec(1, 3, width_ratios=[1.34, 0.13, 1.49], wspace=0.0)
    bottom = outer[1].subgridspec(1, 3, width_ratios=[0.55, 1.86, 0.55], wspace=0.02)
    ax_a = fig.add_subplot(top[0, 0])
    ax_b = fig.add_subplot(top[0, 2])
    ax_c = fig.add_subplot(bottom[0, 1])

    plot_primary_pair(ax_a, primary)
    plot_patient_difference(ax_b, primary, stats)
    plot_sensitivity(ax_c, sens)

    for label, ax, x in zip(["A", "B", "C"], [ax_a, ax_b, ax_c], [-0.12, -0.02, -0.12]):
        bfs.add_panel_label(ax, label, x=x, y=1.06)

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)


def write_legend_note(stats: dict, sens: pd.DataFrame) -> None:
    row_notes = "\n".join(f"- {compact_row_label(row.row_label)}." for row in sens.itertuples())
    legend = (
        "Suggested Figure 2 legend note for the revised clinical exposure-state figure.\n\n"
        "ASM, anti-seizure medication. Panel A shows each patient's median synchrony during high "
        "and low medication exposure. Panel B shows the within-patient difference, calculated as "
        "low exposure minus high exposure. Positive values indicate higher synchrony during low "
        "medication exposure. Exact sign test, "
        f"{format_pvalue_text(stats['sign_test_p_two_sided'])}; Wilcoxon signed-rank test, "
        f"{format_pvalue_text(stats['wilcoxon_p_two_sided'])}. Panel C shows robustness analyses "
        "using the following definitions:\n"
        f"{row_notes}\n"
    )
    with open(OUT_LEGEND_TXT, "w", encoding="utf-8") as handle:
        handle.write(legend)


def main() -> None:
    print("Building Brain manuscript version of clinical exposure-state figure")
    primary = build_patient_summaries()
    stats = paired_stats(primary, label="primary")
    sens = sensitivity_grid_for_brain()

    primary.to_csv(OUT_PATIENT_CSV, index=False)
    sens.to_csv(OUT_STATS_CSV, index=False)
    write_legend_note(stats, sens)
    plot_figure(primary, stats, sens)

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
    print("\nSaved:")
    print(f"  {OUT_PNG}")
    print(f"  {OUT_PDF}")
    print(f"  {OUT_PATIENT_CSV}")
    print(f"  {OUT_STATS_CSV}")
    print(f"  {OUT_LEGEND_TXT}")


if __name__ == "__main__":
    main()
