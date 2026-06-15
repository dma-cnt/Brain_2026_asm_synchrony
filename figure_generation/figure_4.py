#!/usr/bin/env python3
"""Create revised Figure 4 for Brain submission.

Panel A recreates the RNS co-located synchrony comparison from the original
Figure 4 analysis notebook. Panel B reads the median and
interquartile synchrony correlations from ``figure_4_panel_b_quartiles.csv``.
"""

from __future__ import annotations

import argparse
import csv
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

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from figure_style import (
    EDGE_COLOR,
    BODY_FONT_SIZE,
    CARB_COLOR,
    EXPORT_PAD_INCHES,
    IQR_FILL_COLOR,
    LEGEND_FONT_SIZE,
    LINE_COLOR,
    PANEL_LABEL_FONT_SIZE,
    ROBUSTNESS_LINE_WIDTH,
    ROBUSTNESS_MARKER_SIZE,
    SYNC_COLOR,
    TIMESERIES_LINE_WIDTH,
    TITLE_FONT_SIZE,
    apply_brain_style,
    style_axis,
)
from PIL import Image


RNS_HUP_ID = 143
RNS_SEGMENT_START = 1000
RNS_SEGMENT_STOP = 1300
PANEL_B_DATA_FILENAME = "figure_4_panel_b_quartiles.csv"
PANEL_B_REQUIRED_COLUMNS = (
    "sampling_percentage",
    "median_correlation",
    "q1_correlation",
    "q3_correlation",
)
PANEL_GRID_WSPACE = 0.045
COMPONENT_FILENAMES = (
    "figure_4_component_1.png",
    "figure_4_component_2.png",
    "figure_4_component_3.png",
)

IQR_COLOR = IQR_FILL_COLOR
RNS_COLOR = CARB_COLOR
FULL_COLOR = SYNC_COLOR


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
        help="Directory for figure_4.tiff and figure_4_preview.png.",
    )
    parser.add_argument(
        "--component-dir",
        type=Path,
        default=None,
        help="Directory containing figure_4_component_1/2/3.png.",
    )
    parser.add_argument(
        "--panel-b-data",
        type=Path,
        default=Path(__file__).resolve().parent / PANEL_B_DATA_FILENAME,
        help=(
            "CSV containing sampling_percentage, median_correlation, "
            "q1_correlation, and q3_correlation."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="TIFF resolution in dots per inch.",
    )
    return parser.parse_args()


def find_component_paths(explicit: Path | None = None) -> tuple[Path, Path, Path]:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.extend(
        [
            Path(__file__).resolve().parent,
            Path("/Users/devinma/Downloads"),
        ]
    )

    for candidate in candidates:
        paths = tuple(candidate / filename for filename in COMPONENT_FILENAMES)
        if all(path.exists() for path in paths):
            return paths

    checked = "\n".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Could not locate Figure 4 component images. Checked:\n" + checked
    )


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
        rns_path = (
            candidate
            / "synchrony"
            / "all"
            / "robustness"
            / "rns_full"
            / f"HUP_{RNS_HUP_ID}_rns.npy"
        )
        full_path = (
            candidate
            / "synchrony"
            / "all"
            / "broadband"
            / f"HUP_{RNS_HUP_ID}.npy"
        )
        if rns_path.exists() and full_path.exists():
            return candidate

    checked = "\n".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Could not locate HUP_{RNS_HUP_ID} synchrony files. Checked:\n{checked}"
    )


def configure_matplotlib() -> None:
    apply_brain_style(mpl)


def format_axis(ax: plt.Axes) -> None:
    style_axis(ax)


def load_rns_synchrony(data_root: Path) -> tuple[np.ndarray, np.ndarray]:
    rns_path = (
        data_root
        / "synchrony"
        / "all"
        / "robustness"
        / "rns_full"
        / f"HUP_{RNS_HUP_ID}_rns.npy"
    )
    full_path = (
        data_root / "synchrony" / "all" / "broadband" / f"HUP_{RNS_HUP_ID}.npy"
    )
    return np.load(rns_path), np.load(full_path)


def load_panel_b_quartiles(
    csv_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = {column: [] for column in PANEL_B_REQUIRED_COLUMNS}
    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        missing_columns = [
            column for column in PANEL_B_REQUIRED_COLUMNS if column not in fieldnames
        ]
        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(f"{csv_path} is missing required columns: {missing}")

        for row in reader:
            for column in PANEL_B_REQUIRED_COLUMNS:
                data[column].append(float(row[column]))

    if not data["sampling_percentage"]:
        raise ValueError(f"{csv_path} does not contain any Panel B data rows")

    sampling_percentages = np.asarray(data["sampling_percentage"])
    median_correlations = np.asarray(data["median_correlation"])
    q1_correlations = np.asarray(data["q1_correlation"])
    q3_correlations = np.asarray(data["q3_correlation"])

    if not np.all(np.diff(sampling_percentages) > 0):
        raise ValueError(f"{csv_path} sampling percentages must be strictly increasing")
    if not np.all(
        (q1_correlations <= median_correlations)
        & (median_correlations <= q3_correlations)
    ):
        raise ValueError(f"{csv_path} quartiles must satisfy q1 <= median <= q3")

    return sampling_percentages, median_correlations, q1_correlations, q3_correlations


def crop_near_white(image_path: Path, threshold: int = 248, pad: int = 2) -> Image.Image:
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


def add_panel_label(ax: plt.Axes, label: str) -> None:
    position = ax.get_position()
    ax.figure.text(
        position.x0 - 0.035,
        position.y1 + 0.015,
        label,
        fontsize=PANEL_LABEL_FONT_SIZE,
        fontweight="bold",
        ha="left",
        va="top",
    )


def add_component_image(
    parent_ax: plt.Axes,
    image: Image.Image,
    bounds: tuple[float, float, float, float],
) -> None:
    image_ax = parent_ax.inset_axes(bounds)
    image_ax.imshow(image, interpolation="lanczos", aspect="equal")
    image_ax.axis("off")


def draw_panel_a_components(
    ax: plt.Axes,
    component_paths: tuple[Path, Path, Path],
) -> None:
    ax.axis("off")
    rns_brain = crop_near_white(component_paths[0])
    full_brain = crop_near_white(component_paths[1])
    rns_strip = crop_near_white(component_paths[2])

    add_component_image(ax, rns_brain, (0.00, 0.56, 0.45, 0.38))
    add_component_image(ax, rns_strip, (0.50, 0.54, 0.09, 0.43))
    add_component_image(ax, full_brain, (0.00, 0.06, 0.45, 0.38))

    ax.text(
        0.70,
        0.76,
        "}",
        color=EDGE_COLOR,
        fontsize=44,
        ha="center",
        va="center",
    )
    ax.text(
        0.70,
        0.27,
        "}",
        color=EDGE_COLOR,
        fontsize=44,
        ha="center",
        va="center",
    )
    add_panel_label(ax, "A")


def draw_panel_a_trace(ax: plt.Axes, data_root: Path) -> None:
    rns_synchrony, full_synchrony = load_rns_synchrony(data_root)
    rns_segment = rns_synchrony[RNS_SEGMENT_START:RNS_SEGMENT_STOP]
    full_segment = full_synchrony[RNS_SEGMENT_START:RNS_SEGMENT_STOP]
    seconds = np.arange(RNS_SEGMENT_STOP - RNS_SEGMENT_START)

    ax.plot(
        seconds,
        rns_segment,
        label="Subsampled",
        color=RNS_COLOR,
        linewidth=TIMESERIES_LINE_WIDTH,
    )
    ax.plot(
        seconds,
        full_segment,
        label="Full-array",
        color=FULL_COLOR,
        linewidth=TIMESERIES_LINE_WIDTH,
    )

    ax.set_title(
        "Illustrative subsampled vs. full-array iEEG synchrony\n"
        f"({RNS_SEGMENT_STOP - RNS_SEGMENT_START}s window, HUP {RNS_HUP_ID})",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
        pad=7,
    )
    ax.set_xlabel("Time (s)", fontsize=BODY_FONT_SIZE)
    ax.set_ylabel("Synchrony", fontsize=BODY_FONT_SIZE)
    legend = ax.legend(
        loc="center right",
        frameon=True,
        framealpha=1,
        edgecolor=SYNC_COLOR,
        facecolor="white",
        fontsize=LEGEND_FONT_SIZE,
        borderpad=0.45,
        labelspacing=0.35,
    )
    legend.get_frame().set_linewidth(0.8)
    format_axis(ax)


def draw_panel_b(
    ax: plt.Axes,
    sampling_percentages: np.ndarray,
    median_correlations: np.ndarray,
    q1_correlations: np.ndarray,
    q3_correlations: np.ndarray,
) -> None:
    ax.plot(
        sampling_percentages,
        median_correlations,
        marker="o",
        linestyle="-",
        linewidth=ROBUSTNESS_LINE_WIDTH,
        markersize=ROBUSTNESS_MARKER_SIZE,
        color=LINE_COLOR,
        zorder=3,
    )
    ax.fill_between(
        sampling_percentages,
        q1_correlations,
        q3_correlations,
        alpha=0.40,
        color=IQR_COLOR,
        linewidth=0,
        zorder=1,
    )

    ax.set_title(
        "Subsampled synchrony correlated with full-array synchrony\n"
        "(67-patient continuous cohort)",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
    )
    ax.set_xlabel(
        "Sampling percentage (of all intracranial EEG electrodes)",
        fontsize=BODY_FONT_SIZE,
    )
    ax.set_ylabel("Median Pearson Correlation", fontsize=BODY_FONT_SIZE)
    ax.set_xticks(sampling_percentages)
    ax.set_xticklabels([f"{value:g}%" for value in sampling_percentages])
    ax.set_ylim(0, 1)

    format_axis(ax)
    add_panel_label(ax, "B")


def create_figure(
    data_root: Path,
    component_paths: tuple[Path, Path, Path],
    panel_b_data_path: Path,
) -> plt.Figure:
    configure_matplotlib()
    panel_b_data = load_panel_b_quartiles(panel_b_data_path)

    fig = plt.figure(figsize=(14.1, 9.6), facecolor="white")
    outer_gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.0, 1.18],
        left=0.115,
        right=0.93,
        top=0.93,
        bottom=0.08,
        hspace=0.38,
    )
    panel_a_gs = gridspec.GridSpecFromSubplotSpec(
        1,
        3,
        subplot_spec=outer_gs[0],
        width_ratios=[0.30, 0.52, 0.18],
        wspace=PANEL_GRID_WSPACE,
    )
    component_ax = fig.add_subplot(panel_a_gs[0])
    trace_ax = fig.add_subplot(panel_a_gs[1])
    panel_b_gs = gridspec.GridSpecFromSubplotSpec(
        1,
        3,
        subplot_spec=outer_gs[1],
        width_ratios=[0.30, 0.52, 0.18],
        wspace=PANEL_GRID_WSPACE,
    )
    panel_b_ax = fig.add_subplot(panel_b_gs[0, :2])

    draw_panel_a_components(component_ax, component_paths)
    draw_panel_a_trace(trace_ax, data_root)
    draw_panel_b(panel_b_ax, *panel_b_data)
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
    component_paths = find_component_paths(args.component_dir)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = create_figure(data_root, component_paths, args.panel_b_data)

    tiff_path = output_dir / "figure_4.tiff"
    preview_path = output_dir / "figure_4_preview.png"
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
