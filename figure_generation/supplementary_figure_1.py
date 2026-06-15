#!/usr/bin/env python3
"""Create Supplementary Figure 1 for the Brain submission.

Panel A shows fixed-effect estimates from the linear mixed-effects model.
Panel B shows likelihood-ratio tests for adding synchrony under the two
mixed-effects random-effects structures.
"""

from __future__ import annotations

import argparse
import os
import tempfile
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
from figure_3 import (
    HIGHLIGHT,
    HIGHLIGHT_LIGHT,
    LINE_COLOR,
    NEUTRAL,
    add_panel_label,
    find_data_root,
    fit_continuous_results,
    fixed_effect_coefficients,
    flatten_tiff_to_rgb,
    format_lr_pvalue_math,
)
from figure_style import (
    AXIS_LINE_WIDTH,
    BODY_FONT_SIZE,
    CARB_COLOR,
    EXPORT_PAD_INCHES,
    LINE_COLOR as STYLE_LINE_COLOR,
    ROBUSTNESS_LINE_WIDTH,
    ROBUSTNESS_MARKER_SIZE,
    TICK_FONT_SIZE,
    TITLE_FONT_SIZE,
    apply_brain_style,
    style_axis,
)


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Path to cnt-data. If omitted, common local paths are searched.",
    )
    parser.add_argument(
        "--perm-csv",
        type=Path,
        default=None,
        help="Optional circular-shift CSV path; accepted for shared model fitting.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "outputs",
        help=(
            "Directory for supplementary_figure_1.tiff and "
            "supplementary_figure_1_preview.png."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="TIFF resolution in dots per inch.",
    )
    return parser.parse_args()


def draw_lmm_panel(ax: plt.Axes, res_lme_ri) -> None:
    coefs = fixed_effect_coefficients(res_lme_ri)
    y_positions = np.arange(len(coefs))

    for i in y_positions:
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#F2F2F2", alpha=0.7, zorder=0)

    err_low = coefs["estimate"] - coefs["ci_lower"]
    err_high = coefs["ci_upper"] - coefs["estimate"]
    sync_mask = coefs["is_synchrony"].values
    other_mask = ~sync_mask
    other_idx = np.where(other_mask)[0]
    sync_idx = np.where(sync_mask)[0]

    if other_idx.size:
        ax.errorbar(
            coefs.loc[other_mask, "estimate"],
            other_idx,
            xerr=[err_low[other_mask], err_high[other_mask]],
            fmt="o",
            color=STYLE_LINE_COLOR,
            ecolor=STYLE_LINE_COLOR,
            capsize=3,
            markersize=ROBUSTNESS_MARKER_SIZE,
            linewidth=ROBUSTNESS_LINE_WIDTH,
            zorder=10,
        )
    if sync_idx.size:
        ax.errorbar(
            coefs.loc[sync_mask, "estimate"],
            sync_idx,
            xerr=[err_low[sync_mask], err_high[sync_mask]],
            fmt="o",
            color=CARB_COLOR,
            ecolor=CARB_COLOR,
            capsize=3,
            markersize=ROBUSTNESS_MARKER_SIZE + 1.0,
            linewidth=ROBUSTNESS_LINE_WIDTH,
            zorder=11,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(coefs["label"], fontsize=TICK_FONT_SIZE)
    for i, row in coefs.iterrows():
        if row["is_synchrony"]:
            ax.get_yticklabels()[i].set_fontweight("bold")
            ax.get_yticklabels()[i].set_color(CARB_COLOR)

    x_min = float(coefs["ci_lower"].min())
    x_max = float(coefs["ci_upper"].max())
    span = max(0.1, x_max - x_min)
    ax.set_xlim(x_min - 0.08 * span, x_max + 0.10 * span)
    ax.axvline(0, color=LINE_COLOR, linestyle="--", linewidth=AXIS_LINE_WIDTH, zorder=5)
    ax.set_xlabel("Fixed-effect estimate (95% CI)", fontsize=BODY_FONT_SIZE)
    ax.set_title(
        "Linear mixed-effects model",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
        pad=8,
    )
    ax.grid(axis="x", linestyle=":", alpha=0.45)
    ax.tick_params(axis="y", length=0)
    style_axis(ax)
    ax.spines["left"].set_visible(False)
    add_panel_label(ax, "A", x=-0.18)


def draw_lrt_panel(ax: plt.Axes, lrt_ri: dict[str, object], lrt_rs: dict[str, object]) -> None:
    rows = [
        {
            "label": "Random intercept\n(per patient)",
            "lr_stat": float(lrt_ri["lr_stat"]),
            "p": float(lrt_ri["p"]),
        },
        {
            "label": "Random intercept +\nrandom synchrony slope",
            "lr_stat": float(lrt_rs["lr_stat"]),
            "p": float(lrt_rs["p"]),
        },
    ]
    df = pd.DataFrame(rows)
    positions = np.arange(len(df))
    colors = [HIGHLIGHT, HIGHLIGHT_LIGHT]

    ax.vlines(
        positions,
        ymin=1,
        ymax=df["lr_stat"],
        color=colors,
        linewidth=3.2,
        zorder=9,
    )
    ax.scatter(
        positions,
        df["lr_stat"],
        s=170,
        color=colors,
        edgecolor=STYLE_LINE_COLOR,
        linewidth=1.0,
        zorder=10,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(df["label"], fontsize=TICK_FONT_SIZE)
    ax.set_yscale("log")
    ax.set_xlim(-0.55, len(df) - 0.45)
    ax.set_ylim(1, float(df["lr_stat"].max()) * 8.0)

    crit = 3.841
    ax.axhline(crit, color=STYLE_LINE_COLOR, linestyle=":", linewidth=1.2, zorder=5)
    ax.text(
        0.04,
        crit,
        "χ²(1) critical value\nat α = 0.05 (3.84)",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="center",
        fontsize=TICK_FONT_SIZE - 1,
        color=NEUTRAL,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.92, edgecolor="none"),
        zorder=6,
    )

    for x, row in zip(positions, rows):
        height = float(row["lr_stat"])
        ax.text(
            x,
            height * 1.18,
            f"LR = {height:.2f}\n{format_lr_pvalue_math(float(row['p']))}",
            ha="center",
            va="bottom",
            fontsize=TICK_FONT_SIZE,
            fontweight="bold",
        )

    ax.set_ylabel(
        "Likelihood-ratio statistic\n(2 × Δ log-likelihood, log scale)",
        fontsize=BODY_FONT_SIZE,
    )
    ax.set_title(
        "Synchrony improves model fit beyond covariates",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
        pad=8,
    )
    ax.grid(axis="y", linestyle=":", alpha=0.4, which="both")
    style_axis(ax)
    add_panel_label(ax, "B")


def create_figure(results: dict[str, object]) -> plt.Figure:
    apply_brain_style(mpl)
    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(14.5, 6.2),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.1, 1.0], "wspace": 0.08},
        facecolor="white",
    )
    draw_lmm_panel(ax_a, results["res_lme_ri"])
    draw_lrt_panel(ax_b, results["lrt_ri"], results["lrt_rs"])
    return fig


def save_figure(fig: plt.Figure, output_dir: Path, dpi: int) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tiff_path = output_dir / "supplementary_figure_1.tiff"
    preview_path = output_dir / "supplementary_figure_1_preview.png"
    fig.savefig(
        tiff_path,
        dpi=dpi,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=EXPORT_PAD_INCHES,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    flatten_tiff_to_rgb(tiff_path, dpi)
    fig.savefig(
        preview_path,
        dpi=220,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=EXPORT_PAD_INCHES,
    )
    return tiff_path, preview_path


def main() -> None:
    args = parse_args()
    data_root = find_data_root(args.data_root)
    results = fit_continuous_results(data_root, args.perm_csv)
    fig = create_figure(results)
    tiff_path, preview_path = save_figure(fig, args.output_dir.resolve(), args.dpi)
    plt.close(fig)
    print(f"Wrote {tiff_path}")
    print(f"Wrote {preview_path}")


if __name__ == "__main__":
    main()
