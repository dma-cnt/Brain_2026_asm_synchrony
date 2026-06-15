#!/usr/bin/env python3
"""Create revised Figure 1 for Brain submission.

The script recreates the composed Figure 1 using the two static panel assets in
``brain_r1`` and the HUP 142 time-series curves used by ``brain_r1/figure_1.ipynb``.
It writes a high-resolution TIFF plus a PNG preview to ``figure_generation/outputs``
by default.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import tempfile

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
    ASM_LEGEND_LINE_WIDTH,
    ASM_LINE_WIDTH,
    BODY_FONT_SIZE,
    CARB_COLOR,
    EDGE_COLOR,
    EVENT_LINE_WIDTH,
    EXPORT_PAD_INCHES,
    LEGEND_FONT_SIZE,
    LEV_COLOR,
    PANEL_LABEL_FONT_SIZE,
    SEIZURE_COLOR,
    SYNC_COLOR,
    TICK_FONT_SIZE,
    TIMESERIES_LINE_WIDTH,
    TITLE_FONT_SIZE,
    apply_brain_style,
    style_axis,
)
from matplotlib.legend import Legend
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
from PIL import Image


PATIENT_HUP_ID = 142
NOTEBOOK_XTICK_MINUTES = 2500
TOP_CONTENT_CENTER_Y = 0.775
TOP_TEXT_COL_WIDTH = 0.12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Path to cnt-data. If omitted, common local paths are searched.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for figure_1.tiff and figure_1_preview.png.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="TIFF resolution in dots per inch.",
    )
    return parser.parse_args()


def find_data_root(repo_root: Path, explicit: Path | None = None) -> Path:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    env_path = os.environ.get("CNT_DATA_PATH")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            repo_root.parent / "cnt-data",
            repo_root.parent.parent / "cnt-data",
            Path.cwd().parent / "cnt-data",
            Path("/Users/devinma/Documents/cnt-data"),
        ]
    )

    for candidate in candidates:
        table = candidate / "giant_new_tables" / f"HUP_{PATIENT_HUP_ID}.csv"
        if table.exists():
            return candidate

    checked = "\n".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Could not locate HUP_{PATIENT_HUP_ID}.csv. Checked:\n{checked}")


def crop_near_white(image_path: Path, threshold: int = 246, pad: int = 4) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    arr = np.asarray(image)
    mask = np.any(arr < threshold, axis=2)
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return image
    left = max(int(xs.min()) - pad, 0)
    right = min(int(xs.max()) + pad + 1, image.width)
    top = max(int(ys.min()) - pad, 0)
    bottom = min(int(ys.max()) + pad + 1, image.height)
    return image.crop((left, top, right, bottom))


def normalized(series: pd.Series) -> pd.Series:
    max_value = series.max(skipna=True)
    if not np.isfinite(max_value) or max_value == 0:
        return series * np.nan
    return series / max_value


def display_days(df: pd.DataFrame) -> np.ndarray:
    # The notebook plots against emu_minute; convert minutes to EMU days.
    return df["emu_minute"].to_numpy(dtype=float) / 1440


def add_round_box(fig: plt.Figure, bounds: tuple[float, float, float, float]) -> None:
    box = FancyBboxPatch(
        (bounds[0], bounds[1]),
        bounds[2],
        bounds[3],
        boxstyle="round,pad=0.012,rounding_size=0.07",
        transform=fig.transFigure,
        facecolor="white",
        edgecolor=EDGE_COLOR,
        linewidth=1.0,
        zorder=-1,
    )
    fig.patches.append(box)


def add_down_arrow(fig: plt.Figure, x: float, y0: float, y1: float, scale: int = 18) -> None:
    arrow = FancyArrowPatch(
        (x, y0),
        (x, y1),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=scale,
        linewidth=0.8,
        color=EDGE_COLOR,
        shrinkA=0,
        shrinkB=0,
    )
    fig.patches.append(arrow)


def add_flow_arrows(
    fig: plt.Figure,
    flow_text_groups: list[list[plt.Text]],
    pad: float = 0.006,
    scale: int = 15,
) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inverse = fig.transFigure.inverted()
    for text_group in flow_text_groups:
        for upper_text, lower_text in zip(text_group[:-1], text_group[1:]):
            upper_box = upper_text.get_window_extent(renderer=renderer).transformed(inverse)
            lower_box = lower_text.get_window_extent(renderer=renderer).transformed(inverse)
            gap_top = upper_box.y0
            gap_bottom = lower_box.y1
            gap = gap_top - gap_bottom
            if gap <= 0:
                continue
            arrow_length = max(0.012, gap - 2 * pad)
            arrow_length = min(arrow_length, gap * 0.72)
            center = (gap_top + gap_bottom) / 2
            x = upper_text.get_position()[0]
            add_down_arrow(
                fig,
                x=x,
                y0=center + arrow_length / 2,
                y1=center - arrow_length / 2,
                scale=scale,
            )


def distribute_texts_with_equal_gaps(
    fig: plt.Figure,
    texts: list[plt.Text],
    center_y: float,
    total_height: float,
) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inverse = fig.transFigure.inverted()
    heights = [
        text.get_window_extent(renderer=renderer).transformed(inverse).height
        for text in texts
    ]
    gap = (total_height - sum(heights)) / (len(texts) - 1)
    if gap < 0:
        total_height = sum(heights) + 0.006 * (len(texts) - 1)
        gap = 0.006

    current_top = center_y + total_height / 2
    for text, height in zip(texts, heights):
        x, _ = text.get_position()
        text.set_position((x, current_top - height / 2))
        current_top -= height + gap


def match_flow_gap_to_reference(
    fig: plt.Figure,
    target_texts: list[plt.Text],
    reference_texts: list[plt.Text],
    center_y: float,
) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inverse = fig.transFigure.inverted()

    reference_gaps = []
    for upper_text, lower_text in zip(reference_texts[:-1], reference_texts[1:]):
        upper_box = upper_text.get_window_extent(renderer=renderer).transformed(inverse)
        lower_box = lower_text.get_window_extent(renderer=renderer).transformed(inverse)
        reference_gaps.append(upper_box.y0 - lower_box.y1)
    target_heights = [
        text.get_window_extent(renderer=renderer).transformed(inverse).height
        for text in target_texts
    ]
    gap = float(np.mean(reference_gaps))
    total_height = sum(target_heights) + gap * (len(target_texts) - 1)

    current_top = center_y + total_height / 2
    for text, height in zip(target_texts, target_heights):
        x, _ = text.get_position()
        text.set_position((x, current_top - height / 2))
        current_top -= height + gap


def center_panel_b_content(
    fig: plt.Figure,
    plot_ax: plt.Axes,
    flow_texts: list[plt.Text],
    legend: Legend,
    panel_bounds: tuple[float, float, float, float],
    legend_anchor_y: float,
) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inverse = fig.transFigure.inverted()
    boxes = [plot_ax.get_tightbbox(renderer=renderer).transformed(inverse)]
    boxes.extend(
        text.get_window_extent(renderer=renderer).transformed(inverse)
        for text in flow_texts
    )
    boxes.append(legend.get_window_extent(renderer=renderer).transformed(inverse))

    content_left = min(box.x0 for box in boxes)
    content_right = max(box.x1 for box in boxes)
    content_center_x = (content_left + content_right) / 2
    panel_center_x = panel_bounds[0] + panel_bounds[2] / 2
    dx = panel_center_x - content_center_x

    position = plot_ax.get_position()
    plot_ax.set_position(
        [position.x0 + dx, position.y0, position.width, position.height]
    )
    for text in flow_texts:
        x, y = text.get_position()
        text.set_position((x + dx, y))

    shifted_position = plot_ax.get_position()
    shifted_plot_center = shifted_position.x0 + shifted_position.width / 2
    legend.set_bbox_to_anchor((shifted_plot_center, legend_anchor_y), transform=fig.transFigure)


def add_outer_arrow(
    fig: plt.Figure,
    start: tuple[float, float],
    end: tuple[float, float],
    rad: float,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        transform=fig.transFigure,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="Simple,tail_width=0.75,head_width=8,head_length=10",
        facecolor="#d8d8d8",
        edgecolor=EDGE_COLOR,
        linewidth=0.9,
        mutation_scale=1.0,
        clip_on=False,
        zorder=-2,
    )
    fig.patches.append(arrow)


def add_image_axis(
    fig: plt.Figure,
    image: Image.Image,
    center_x: float,
    center_y: float,
    width: float,
) -> plt.Axes:
    fig_aspect = fig.get_figwidth() / fig.get_figheight()
    height = width * fig_aspect * (image.height / image.width)
    ax = fig.add_axes([center_x - width / 2, center_y - height / 2, width, height])
    ax.imshow(image, interpolation="lanczos")
    ax.axis("off")
    return ax


def format_timeseries_axis(ax: plt.Axes) -> None:
    style_axis(ax)


def draw_seizure_markers(ax: plt.Axes, seizure_x: np.ndarray) -> None:
    for x_value in seizure_x:
        ax.axvline(
            x_value,
            color=SEIZURE_COLOR,
            linestyle=(0, (5, 4)),
            linewidth=EVENT_LINE_WIDTH,
            zorder=1,
        )


def add_lightning_bolt(ax: plt.Axes, x: float, y: float, width: float) -> None:
    height = 0.12
    verts = np.array(
        [
            [x - 0.30 * width, y + 0.50 * height],
            [x + 0.24 * width, y + 0.50 * height],
            [x - 0.02 * width, y + 0.05 * height],
            [x + 0.36 * width, y + 0.05 * height],
            [x - 0.28 * width, y - 0.55 * height],
            [x - 0.06 * width, y - 0.08 * height],
            [x - 0.42 * width, y - 0.08 * height],
        ]
    )
    ax.add_patch(
        Polygon(
            verts,
            closed=True,
            facecolor=SEIZURE_COLOR,
            edgecolor=SEIZURE_COLOR,
            linewidth=0,
            transform=ax.get_xaxis_transform(),
            clip_on=False,
            zorder=6,
        )
    )


def add_seizure_icons(ax: plt.Axes, first_pair: float, third: float) -> None:
    pair_offset = NOTEBOOK_XTICK_MINUTES * 0.045
    bolt_width = NOTEBOOK_XTICK_MINUTES * 0.08
    add_lightning_bolt(ax, first_pair - pair_offset, -0.22, width=bolt_width)
    add_lightning_bolt(ax, first_pair + pair_offset, -0.22, width=bolt_width)
    add_lightning_bolt(ax, third, -0.22, width=bolt_width)


def add_top_panel_a(fig: plt.Figure, repo_root: Path) -> list[plt.Text]:
    add_round_box(fig, (0.055, 0.62, 0.425, 0.33))
    fig.text(
        0.075,
        0.92,
        "A. Intracranial EEG synchrony",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
    )

    brain = crop_near_white(repo_root / "brain_r1" / "figure_1_component_1.jpg")
    traces = crop_near_white(repo_root / "brain_r1" / "figure_1_component_2.jpg")

    add_image_axis(
        fig,
        brain,
        center_x=0.135,
        center_y=TOP_CONTENT_CENTER_Y,
        width=TOP_TEXT_COL_WIDTH,
    )
    add_image_axis(
        fig,
        traces,
        center_x=0.275,
        center_y=TOP_CONTENT_CENTER_Y,
        width=TOP_TEXT_COL_WIDTH,
    )

    flow_x = 0.415
    flow_step = 0.085
    flow_centers = [
        TOP_CONTENT_CENTER_Y + 1.5 * flow_step,
        TOP_CONTENT_CENTER_Y + 0.5 * flow_step,
        TOP_CONTENT_CENTER_Y - 0.5 * flow_step,
        TOP_CONTENT_CENTER_Y - 1.5 * flow_step,
    ]
    flow_labels = [
        "2-minute iEEG\nsegments",
        "Common average\nreferencing with\nnotch filters",
        "Kuramoto model",
        "iEEG synchrony\nmeasure",
    ]
    flow_texts = [
        fig.text(
            flow_x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=BODY_FONT_SIZE,
            linespacing=1.15,
        )
        for y, label in zip(flow_centers, flow_labels)
    ]
    distribute_texts_with_equal_gaps(
        fig,
        flow_texts,
        center_y=TOP_CONTENT_CENTER_Y,
        total_height=0.275,
    )
    return flow_texts


def add_top_panel_b(
    fig: plt.Figure, days: np.ndarray, df: pd.DataFrame
) -> tuple[list[plt.Text], plt.Axes, Legend, tuple[float, float, float, float], float]:
    panel_b_bounds = (0.525, 0.62, 0.41, 0.33)
    add_round_box(fig, panel_b_bounds)
    fig.text(
        0.545,
        0.92,
        "B. Pharmacokinetic model",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
    )

    panel_b_center_x = panel_b_bounds[0] + panel_b_bounds[2] / 2
    plot_width = 0.145
    flow_width = 0.16
    component_gap = 0.02
    group_width = plot_width + component_gap + flow_width
    plot_left = panel_b_center_x - group_width / 2
    plot_center_x = plot_left + plot_width / 2
    flow_x = plot_left + plot_width + component_gap + flow_width / 2

    panel_b_plot_height = 0.17
    panel_b_plot_bottom = TOP_CONTENT_CENTER_Y - panel_b_plot_height / 2
    mini_ax = fig.add_axes([plot_left, panel_b_plot_bottom, plot_width, panel_b_plot_height])
    emu_minutes = df["emu_minute"].to_numpy(dtype=float)
    carb = normalized(df["med_carbamazepine_raw"])
    lev = normalized(df["med_levetiracetam_raw"])
    full_x_min = float(np.nanmin(emu_minutes))
    full_x_max = float(np.nanmax(emu_minutes))
    full_x_margin = (full_x_max - full_x_min) * 0.05
    mini_ax.plot(emu_minutes, carb, color=CARB_COLOR, lw=ASM_LINE_WIDTH)
    mini_ax.plot(emu_minutes, lev, color=LEV_COLOR, lw=ASM_LINE_WIDTH)
    mini_ax.set_xlim(full_x_min - full_x_margin, NOTEBOOK_XTICK_MINUTES * 2)
    mini_ax.set_ylim(0, 1.3)
    mini_ax.set_yticks([0, 0.5, 1.0])
    mini_ax.set_yticklabels([])
    mini_ax.set_xticks([NOTEBOOK_XTICK_MINUTES])
    mini_ax.set_xticklabels(["Day 1"], fontsize=TICK_FONT_SIZE)
    mini_ax.set_ylabel("ASM load", fontsize=BODY_FONT_SIZE)
    mini_ax.set_xlabel(
        "Time (day of EMU admission)", fontsize=BODY_FONT_SIZE, labelpad=2
    )
    style_axis(mini_ax)

    legend_handles = [
        mpl.lines.Line2D(
            [0],
            [0],
            color=CARB_COLOR,
            lw=ASM_LEGEND_LINE_WIDTH,
            label="Carbamazepine",
        ),
        mpl.lines.Line2D(
            [0],
            [0],
            color=LEV_COLOR,
            lw=ASM_LEGEND_LINE_WIDTH,
            label="Levetiracetam",
        ),
    ]
    legend_anchor_y = TOP_CONTENT_CENTER_Y + panel_b_plot_height / 2 + 0.006
    legend = fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(plot_center_x, legend_anchor_y),
        ncol=2,
        frameon=True,
        framealpha=1,
        edgecolor=EDGE_COLOR,
        facecolor="white",
        fontsize=LEGEND_FONT_SIZE,
        handlelength=1.8,
        columnspacing=0.9,
        borderpad=0.35,
        labelspacing=0.25,
    )

    upper_text = fig.text(
        flow_x,
        TOP_CONTENT_CENTER_Y + 0.085,
        "Validated\npharmacokinetic model\nfor ASM",
        ha="center",
        va="center",
        fontsize=LEGEND_FONT_SIZE,
        linespacing=1.15,
    )
    lower_text = fig.text(
        flow_x,
        TOP_CONTENT_CENTER_Y - 0.085,
        "Continuous ASM load\nestimates throughout\nthe EMU stay",
        ha="center",
        va="center",
        fontsize=BODY_FONT_SIZE,
        linespacing=1.15,
    )
    return [upper_text, lower_text], mini_ax, legend, panel_b_bounds, legend_anchor_y


def add_timeseries_panels(fig: plt.Figure, df: pd.DataFrame) -> tuple[plt.Axes, plt.Axes]:
    sync_ax = fig.add_axes([0.105, 0.355, 0.83, 0.17])
    asm_ax = fig.add_axes([0.105, 0.12, 0.83, 0.17], sharex=sync_ax)
    emu_minutes = df["emu_minute"].to_numpy(dtype=float)

    sync_ax.plot(
        emu_minutes,
        df["synchrony_broadband"],
        color=SYNC_COLOR,
        lw=TIMESERIES_LINE_WIDTH,
        alpha=0.95,
    )
    sync_ax.set_ylim(0.06, 0.405)
    sync_ax.set_ylabel("Synchrony", fontsize=BODY_FONT_SIZE)
    sync_ax.set_title("Synchrony", fontsize=TITLE_FONT_SIZE, fontweight="bold", pad=6)
    sync_ax.tick_params(labelbottom=False)
    format_timeseries_axis(sync_ax)
    fig.text(
        0.52,
        0.585,
        "Illustrative representative patient example (Panels C-D)",
        ha="center",
        va="center",
        fontsize=LEGEND_FONT_SIZE,
        fontstyle="italic",
        bbox={
            "boxstyle": "round,pad=0.35,rounding_size=0.12",
            "facecolor": "#F2F2F2",
            "edgecolor": "#D0D0D0",
            "linewidth": 0.6,
        },
    )

    carb = normalized(df["med_carbamazepine_raw"])
    lev = normalized(df["med_levetiracetam_raw"])
    asm_ax.plot(
        emu_minutes,
        carb,
        color=CARB_COLOR,
        lw=ASM_LINE_WIDTH,
        label="Carbamazepine",
    )
    asm_ax.plot(
        emu_minutes,
        lev,
        color=LEV_COLOR,
        lw=ASM_LINE_WIDTH,
        label="Levetiracetam",
    )
    asm_ax.set_ylim(0, 1.3)
    asm_ax.set_ylabel("ASM load", fontsize=BODY_FONT_SIZE)
    asm_ax.set_title(
        "Antiseizure medication load",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
        pad=6,
    )
    x_min = float(np.nanmin(emu_minutes))
    x_max = float(np.nanmax(emu_minutes))
    x_margin = (x_max - x_min) * 0.05
    xlim = (x_min - x_margin, x_max + x_margin)
    asm_ax.set_xlim(*xlim)
    tick_start = np.ceil(xlim[0] / NOTEBOOK_XTICK_MINUTES) * NOTEBOOK_XTICK_MINUTES
    tick_stop = np.floor(xlim[1] / NOTEBOOK_XTICK_MINUTES) * NOTEBOOK_XTICK_MINUTES
    day_ticks = np.arange(tick_start, tick_stop + 1, NOTEBOOK_XTICK_MINUTES)
    tick_labels = []
    day_number = 1
    for tick in day_ticks:
        if tick <= x_max:
            tick_labels.append("Day 1" if day_number == 1 else f"{day_number}")
            day_number += 1
        else:
            tick_labels.append("")
    asm_ax.set_xticks(day_ticks)
    asm_ax.set_xticklabels(tick_labels, fontsize=TICK_FONT_SIZE)
    format_timeseries_axis(asm_ax)

    seizure_x = df.loc[df["had_seizure"].to_numpy() >= 1, "emu_minute"].to_numpy(
        dtype=float
    )
    draw_seizure_markers(sync_ax, seizure_x)
    draw_seizure_markers(asm_ax, seizure_x)

    legend = asm_ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.005, 1.3),
        frameon=True,
        framealpha=1,
        edgecolor=EDGE_COLOR,
        facecolor="white",
        fontsize=BODY_FONT_SIZE,
        handlelength=2.5,
        borderpad=0.45,
        labelspacing=0.25,
    )
    legend.get_frame().set_linewidth(0.8)
    for legend_line in legend.get_lines():
        legend_line.set_linewidth(ASM_LEGEND_LINE_WIDTH)

    first_pair = seizure_x[:2].mean() if len(seizure_x) >= 2 else seizure_x[0]
    third = seizure_x[2] if len(seizure_x) >= 3 else seizure_x[-1]
    add_seizure_icons(asm_ax, first_pair, third)
    asm_ax.text(
        seizure_x.mean(),
        -0.43,
        "Seizures",
        color=SEIZURE_COLOR,
        ha="center",
        va="center",
        fontsize=BODY_FONT_SIZE,
        fontweight="bold",
        fontstyle="italic",
        transform=asm_ax.get_xaxis_transform(),
        clip_on=False,
    )

    fig.text(0.055, 0.52, "C", fontsize=PANEL_LABEL_FONT_SIZE, fontweight="bold")
    fig.text(0.055, 0.29, "D", fontsize=PANEL_LABEL_FONT_SIZE, fontweight="bold")
    return sync_ax, asm_ax


def create_figure(data_root: Path, repo_root: Path) -> plt.Figure:
    table_path = data_root / "giant_new_tables" / f"HUP_{PATIENT_HUP_ID}.csv"
    df = pd.read_csv(table_path)
    days = display_days(df)

    apply_brain_style(mpl)

    fig = plt.figure(figsize=(14.1, 7.95), facecolor="white")
    add_outer_arrow(fig, (0.052, 0.73), (0.055, 0.43), rad=0.34)
    add_outer_arrow(fig, (0.945, 0.73), (0.945, 0.205), rad=-0.22)
    panel_a_flow_texts = add_top_panel_a(fig, repo_root)
    (
        panel_b_flow_texts,
        panel_b_plot_ax,
        panel_b_legend,
        panel_b_bounds,
        panel_b_legend_anchor_y,
    ) = add_top_panel_b(fig, days, df)
    match_flow_gap_to_reference(
        fig,
        panel_b_flow_texts,
        reference_texts=panel_a_flow_texts,
        center_y=TOP_CONTENT_CENTER_Y,
    )
    center_panel_b_content(
        fig,
        panel_b_plot_ax,
        panel_b_flow_texts,
        panel_b_legend,
        panel_b_bounds,
        panel_b_legend_anchor_y,
    )
    flow_text_groups = [panel_a_flow_texts, panel_b_flow_texts]
    add_timeseries_panels(fig, df)
    add_flow_arrows(fig, flow_text_groups)
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


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    data_root = find_data_root(repo_root, args.data_root)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = create_figure(data_root=data_root, repo_root=repo_root)

    tiff_path = output_dir / "figure_1.tiff"
    preview_path = output_dir / "figure_1_preview.png"
    fig.savefig(
        tiff_path,
        dpi=args.dpi,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=EXPORT_PAD_INCHES,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    flatten_tiff_to_rgb(tiff_path, args.dpi)
    fig.savefig(
        preview_path,
        dpi=220,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=EXPORT_PAD_INCHES,
    )
    plt.close(fig)

    print(f"Wrote {tiff_path}")
    print(f"Wrote {preview_path}")


if __name__ == "__main__":
    main()
