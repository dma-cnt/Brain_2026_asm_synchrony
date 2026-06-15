#!/usr/bin/env python3
"""Create a Brain journal thumbnail from revised Figure 1 and Figure 4 assets."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figure_style import (
    CARB_COLOR,
    EDGE_COLOR,
    IQR_FILL_COLOR,
    LEV_COLOR,
    LINE_COLOR,
    SEIZURE_COLOR,
    SYNC_COLOR,
)
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
from PIL import Image


BACKGROUND = "#FAFAF7"
PANEL_FACE = "#FFFFFF"
SUBTLE_EDGE = "#D9D6CF"
TEXT_COLOR = "#161616"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "brain_thumbnail.jpg",
        help="JPEG output path.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=320,
        help="Output resolution. The default produces a 1600 x 1600 px JPEG.",
    )
    return parser.parse_args()


def crop_near_white(image_path: Path, threshold: int = 248, pad: int = 4) -> Image.Image:
    image = Image.open(image_path).convert("RGBA")
    arr = np.asarray(image)
    alpha_mask = arr[:, :, 3] > 8
    rgb_mask = np.any(arr[:, :, :3] < threshold, axis=2)
    mask = alpha_mask & rgb_mask
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return image
    left = max(int(xs.min()) - pad, 0)
    right = min(int(xs.max()) + pad + 1, image.width)
    top = max(int(ys.min()) - pad, 0)
    bottom = min(int(ys.max()) + pad + 1, image.height)
    return image.crop((left, top, right, bottom))


def add_panel(
    fig: plt.Figure,
    bounds: tuple[float, float, float, float],
    *,
    radius: float = 0.025,
    zorder: int = -10,
) -> None:
    patch = FancyBboxPatch(
        (bounds[0], bounds[1]),
        bounds[2],
        bounds[3],
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        transform=fig.transFigure,
        facecolor=PANEL_FACE,
        edgecolor=SUBTLE_EDGE,
        linewidth=1.0,
        zorder=zorder,
    )
    fig.patches.append(patch)


def add_image_axis(
    fig: plt.Figure,
    image: Image.Image,
    bounds: tuple[float, float, float, float],
    *,
    alpha: float = 1.0,
) -> plt.Axes:
    ax = fig.add_axes(bounds)
    ax.set_zorder(2)
    ax.imshow(image, interpolation="lanczos", alpha=alpha)
    ax.axis("off")
    return ax


def medication_curves(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dose_times = np.array([0.55, 0.95, 1.45, 1.95, 2.45, 6.65, 7.05, 7.45, 7.85])
    dose_sizes = np.array([0.95, 0.90, 0.82, 0.70, 0.42, 0.56, 0.74, 0.82, 0.88])
    lev = np.zeros_like(x)
    for dose_time, dose_size in zip(dose_times, dose_sizes):
        lev += dose_size * np.exp(-(x - dose_time) / 0.34) * (x >= dose_time)
    lev = np.clip(lev, 0, 1)

    carb = np.zeros_like(x)
    early = (x >= 0.52) & (x <= 3.75)
    carb[early] = (
        0.88
        - 0.23 * np.clip((x[early] - 1.75) / 1.9, 0, 1)
        + 0.055 * np.sin(2 * np.pi * (x[early] - 0.55) / 0.7)
    )
    taper = (x > 3.75) & (x <= 5.2)
    carb[taper] = 0.62 * np.exp(-(x[taper] - 3.75) / 0.55)
    restart = x >= 6.75
    carb[restart] = np.minimum(
        0.86,
        0.24
        + 0.58 * (1 - np.exp(-(x[restart] - 6.75) / 0.65))
        + 0.04 * np.sin(2 * np.pi * x[restart] / 0.55),
    )
    carb = np.clip(carb, 0, 1)
    total = np.clip(np.maximum(carb, lev), 0, 1)
    return carb, lev, total


def draw_bolt(ax: plt.Axes, x: float, y: float, width: float, height: float) -> None:
    verts = np.array(
        [
            [x - 0.24 * width, y + 0.50 * height],
            [x + 0.22 * width, y + 0.50 * height],
            [x - 0.03 * width, y + 0.02 * height],
            [x + 0.30 * width, y + 0.02 * height],
            [x - 0.22 * width, y - 0.55 * height],
            [x - 0.04 * width, y - 0.06 * height],
            [x - 0.33 * width, y - 0.06 * height],
        ]
    )
    ax.add_patch(
        Polygon(
            verts,
            closed=True,
            facecolor=SEIZURE_COLOR,
            edgecolor="#9A4200",
            linewidth=0.4,
            clip_on=False,
            zorder=6,
        )
    )


def draw_timeseries(ax: plt.Axes) -> None:
    x = np.linspace(0, 8.25, 900)
    carb, lev, total = medication_curves(x)
    wiggle = 0.025 * np.sin(2 * np.pi * x / 0.72) + 0.014 * np.sin(2 * np.pi * x / 0.19)
    synchrony = 0.70 + 0.26 * (1 - total) + wiggle
    synchrony += 0.06 * np.exp(-((x - 3.65) / 0.28) ** 2)

    ax.plot(x, synchrony, color=SYNC_COLOR, lw=2.5, solid_capstyle="round", zorder=3)
    ax.plot(x, 0.10 + 0.34 * carb, color=CARB_COLOR, lw=3.3, solid_capstyle="round")
    ax.plot(x, 0.10 + 0.34 * lev, color=LEV_COLOR, lw=3.3, solid_capstyle="round")

    seizure_x = [3.37, 3.50, 4.86]
    for value in seizure_x:
        ax.axvline(value, ymin=0.05, ymax=0.97, color=SEIZURE_COLOR, lw=1.7, ls=(0, (5, 4)))
    draw_bolt(ax, 3.34, -0.060, 0.16, 0.080)
    draw_bolt(ax, 3.55, -0.060, 0.16, 0.080)
    draw_bolt(ax, 4.86, -0.060, 0.18, 0.085)

    ax.text(
        5.18,
        -0.075,
        "seizures",
        color=SEIZURE_COLOR,
        fontsize=7.7,
        fontweight="bold",
    )

    ax.set_xlim(0, 8.25)
    ax.set_ylim(-0.14, 1.05)
    ax.set_xticks([1, 3, 5, 7])
    ax.set_xticklabels(["Day 1", "3", "5", "7"], fontsize=7.6, color=LINE_COLOR)
    ax.set_yticks([])
    for spine in ["left", "right", "top"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#8A8A8A")
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="x", length=2.5, color="#8A8A8A")


def draw_sampling_curve(ax: plt.Axes, data_path: Path) -> None:
    df = pd.read_csv(data_path)
    x = df["sampling_percentage"].to_numpy()
    med = df["median_correlation"].to_numpy()
    q1 = df["q1_correlation"].to_numpy()
    q3 = df["q3_correlation"].to_numpy()

    ax.fill_between(x, q1, q3, color=IQR_FILL_COLOR, alpha=0.42, linewidth=0)
    ax.plot(x, med, color=LINE_COLOR, marker="o", markersize=4.3, lw=2.2)
    ax.scatter([40], [med[list(x).index(40)]], color=CARB_COLOR, s=36, zorder=4)

    ax.set_xlim(8, 102)
    ax.set_ylim(0.30, 1.02)
    ax.set_xticks([10, 40, 100])
    ax.set_xticklabels(["10%", "40%", "100%"], fontsize=6.9)
    ax.set_yticks([0.4, 1.0])
    ax.set_yticklabels(["0.4", "1.0"], fontsize=6.9)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#6F6F6F")
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(width=0.7, length=2.4, color="#6F6F6F", pad=1)


def create_thumbnail(output_path: Path, dpi: int) -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    brain = crop_near_white(repo_root / "brain_r1" / "figure_1_component_1.jpg", pad=1)
    traces = crop_near_white(repo_root / "brain_r1" / "figure_1_component_2.jpg", pad=1)
    sampled_brain = crop_near_white(script_dir / "figure_4_component_1.png", pad=1)
    full_brain = crop_near_white(script_dir / "figure_4_component_2.png", pad=1)
    strip = crop_near_white(script_dir / "figure_4_component_3.png", pad=1)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(5, 5), dpi=dpi, facecolor=BACKGROUND)
    fig.text(
        0.075,
        0.955,
        "Synchrony tracks ASM load",
        ha="left",
        va="top",
        color=TEXT_COLOR,
        fontsize=20.5,
        fontweight="bold",
    )

    add_panel(fig, (0.055, 0.545, 0.47, 0.325), radius=0.03)
    add_panel(fig, (0.555, 0.545, 0.39, 0.325), radius=0.03)
    add_panel(fig, (0.055, 0.135, 0.89, 0.330), radius=0.03)

    add_image_axis(fig, brain, (0.078, 0.635, 0.238, 0.185))
    add_image_axis(fig, traces, (0.338, 0.660, 0.135, 0.135), alpha=0.88)
    fig.text(0.086, 0.595, "Intracranial EEG", fontsize=7.8, color=TEXT_COLOR, fontweight="bold")
    fig.text(0.338, 0.595, "Kuramoto\nsynchrony", fontsize=7.8, color=SYNC_COLOR, fontweight="bold")
    fig.add_artist(
        FancyArrowPatch(
            (0.300, 0.722),
            (0.348, 0.722),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            color=EDGE_COLOR,
            zorder=3,
        )
    )

    add_image_axis(fig, sampled_brain, (0.575, 0.700, 0.145, 0.108), alpha=0.92)
    add_image_axis(fig, strip, (0.724, 0.684, 0.036, 0.142), alpha=0.92)
    add_image_axis(fig, full_brain, (0.780, 0.694, 0.126, 0.099), alpha=0.88)
    fig.text(
        0.573,
        0.850,
        "Electrode subsampling",
        fontsize=7.9,
        color=TEXT_COLOR,
        fontweight="bold",
        va="top",
    )
    curve_ax = fig.add_axes([0.595, 0.575, 0.295, 0.100], facecolor="none")
    curve_ax.set_zorder(2)
    draw_sampling_curve(curve_ax, script_dir / "figure_4_panel_b_quartiles.csv")

    fig.text(
        0.090,
        0.430,
        "ASM taper exposes increased network synchrony",
        ha="left",
        va="center",
        fontsize=9.4,
        color=TEXT_COLOR,
        fontweight="bold",
    )
    legend_y = 0.408
    legend_items = [
        (0.095, 0.170, SYNC_COLOR, "Synchrony"),
        (0.278, 0.353, CARB_COLOR, "ASM load"),
        (0.455, 0.530, SEIZURE_COLOR, "Seizures"),
    ]
    for x0, x1, color, label in legend_items:
        fig.add_artist(
            mpl.lines.Line2D(
                [x0, x1],
                [legend_y, legend_y],
                transform=fig.transFigure,
                color=color,
                linewidth=2.4,
                solid_capstyle="round",
                linestyle=(0, (4, 3)) if label == "Seizures" else "-",
                zorder=3,
            )
        )
        fig.text(
            x1 + 0.008,
            legend_y,
            label,
            ha="left",
            va="center",
            fontsize=7.3,
            color="#4E4E4E",
        )

    time_ax = fig.add_axes([0.095, 0.185, 0.815, 0.190], facecolor="none")
    time_ax.set_zorder(2)
    draw_timeseries(time_ax)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=dpi,
        format="jpg",
        facecolor=BACKGROUND,
        bbox_inches=None,
        pil_kwargs={"quality": 95, "optimize": True, "progressive": True},
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    create_thumbnail(args.output.resolve(), args.dpi)
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
