"""Figure 2 revision: patient-level ASM taper/re-initiation event analysis.

This script replaces the previous window-level Mann-Whitney inference with
paired patient-level summaries. Each patient contributes one median synchrony
value from the 12 hours before the medication-change threshold and one median
synchrony value from the 12 hours after the threshold.

Run with:
    conda run -n cnt python brain_r2/figure_2.py
or, with the env active:
    python brain_r2/figure_2.py
"""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/asm_synchrony_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binomtest, wilcoxon

warnings.filterwarnings("ignore", category=UserWarning)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "cnt-data"))
TABLES_PATH = os.path.join(DATA_PATH, "giant_new_tables")
ANNOTATION_CSV = os.path.join(DATA_PATH, "taper_annotations.csv")

WINDOW_MINUTES = 12 * 60
N_BOOT = 10000
RNG_SEED = 20260430

OUT_PNG = os.path.join(SCRIPT_DIR, "figure_2.png")
OUT_PDF = os.path.join(SCRIPT_DIR, "figure_2.pdf")
OUT_SUMMARY_CSV = os.path.join(SCRIPT_DIR, "figure_2_patient_summaries.csv")
OUT_STATS_CSV = os.path.join(SCRIPT_DIR, "figure_2_stats.csv")


def format_pvalue(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "P = NA"
    if p_value < 1e-4:
        return f"P = {p_value:.1e}"
    if p_value < 1e-3:
        return f"P = {p_value:.4f}"
    return f"P = {p_value:.3f}"


def load_annotations(event_type: str) -> pd.DataFrame:
    annotations = pd.read_csv(ANNOTATION_CSV)
    if event_type == "taper":
        out = annotations.dropna(subset=["taper_start_hr"]).copy()
        out = out[["hup_id", "taper_start_hr"]].rename(
            columns={"taper_start_hr": "event_start_hr"}
        )
    elif event_type == "reinitiation":
        out = annotations.dropna(subset=["reinit_start_hr"]).copy()
        # Mirrors the R1 analysis cohort: HUP 152 was excluded from the
        # re-initiation panel, yielding the manuscript's nine-patient sample.
        out = out[out["hup_id"] != 152]
        out = out[["hup_id", "reinit_start_hr"]].rename(
            columns={"reinit_start_hr": "event_start_hr"}
        )
    else:
        raise ValueError(f"Unknown event_type: {event_type}")
    out["hup_id"] = out["hup_id"].astype(int)
    return out.sort_values("hup_id").reset_index(drop=True)


def patient_table_path(hup_id: int) -> str:
    return os.path.join(TABLES_PATH, f"HUP_{hup_id}.csv")


def load_patient_table(hup_id: int) -> pd.DataFrame:
    path = patient_table_path(hup_id)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    cols = ["emu_minute", "synchrony_broadband", "had_seizure"]
    df = pd.read_csv(path, usecols=cols, low_memory=False)
    return df.sort_values("emu_minute").reset_index(drop=True)


def summarize_event(row: pd.Series, event_type: str) -> dict[str, float | int | str]:
    hup_id = int(row["hup_id"])
    event_minute = float(row["event_start_hr"]) * 60.0
    df = load_patient_table(hup_id)

    pre = df[
        (df["emu_minute"] >= event_minute - WINDOW_MINUTES)
        & (df["emu_minute"] < event_minute)
    ].copy()
    post = df[
        (df["emu_minute"] >= event_minute)
        & (df["emu_minute"] < event_minute + WINDOW_MINUTES)
    ].copy()

    pre_sync = pre["synchrony_broadband"].dropna().to_numpy(dtype=float)
    post_sync = post["synchrony_broadband"].dropna().to_numpy(dtype=float)
    if len(pre_sync) == 0 or len(post_sync) == 0:
        raise ValueError(
            f"HUP {hup_id} has insufficient synchrony data around {event_type}"
        )

    pre_median = float(np.median(pre_sync))
    post_median = float(np.median(post_sync))
    return {
        "event_type": event_type,
        "hup_id": hup_id,
        "event_start_hr": float(row["event_start_hr"]),
        "event_start_minute": event_minute,
        "pre_median_synchrony": pre_median,
        "post_median_synchrony": post_median,
        "delta_synchrony": post_median - pre_median,
        "n_pre_segments": int(len(pre_sync)),
        "n_post_segments": int(len(post_sync)),
        "pre_seizure_segments": int((pre["had_seizure"] >= 1).sum()),
        "post_seizure_segments": int((post["had_seizure"] >= 1).sum()),
    }


def build_event_summaries(event_type: str) -> pd.DataFrame:
    rows = [
        summarize_event(row, event_type)
        for _, row in load_annotations(event_type).iterrows()
    ]
    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray, statistic=np.median) -> tuple[float, float]:
    rng = np.random.default_rng(RNG_SEED)
    values = np.asarray(values, dtype=float)
    boot = np.empty(N_BOOT, dtype=float)
    for i in range(N_BOOT):
        boot[i] = statistic(rng.choice(values, size=len(values), replace=True))
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def event_stats(df: pd.DataFrame, expected_direction: str) -> dict[str, float | int | str]:
    deltas = df["delta_synchrony"].to_numpy(dtype=float)
    n_positive = int(np.sum(deltas > 0))
    n_negative = int(np.sum(deltas < 0))
    n_zero = int(np.sum(deltas == 0))
    n_nonzero = n_positive + n_negative

    if expected_direction == "increase":
        n_expected = n_positive
        n_opposite = n_negative
    elif expected_direction == "decrease":
        n_expected = n_negative
        n_opposite = n_positive
    else:
        raise ValueError(f"Unknown direction: {expected_direction}")

    sign_p = (
        float(binomtest(n_expected, n_nonzero, p=0.5).pvalue)
        if n_nonzero > 0
        else float("nan")
    )

    try:
        wilcoxon_result = wilcoxon(
            deltas,
            zero_method="wilcox",
            alternative="two-sided",
            method="exact",
        )
        wilcoxon_p = float(wilcoxon_result.pvalue)
        wilcoxon_stat = float(wilcoxon_result.statistic)
    except ValueError:
        wilcoxon_p = float("nan")
        wilcoxon_stat = float("nan")

    med_ci_low, med_ci_high = bootstrap_ci(deltas, statistic=np.median)
    mean_ci_low, mean_ci_high = bootstrap_ci(deltas, statistic=np.mean)

    return {
        "event_type": str(df["event_type"].iloc[0]),
        "n_patients": int(len(df)),
        "expected_direction": expected_direction,
        "n_expected_direction": n_expected,
        "n_opposite_direction": n_opposite,
        "n_positive_delta": n_positive,
        "n_negative_delta": n_negative,
        "n_zero_delta": n_zero,
        "median_delta": float(np.median(deltas)),
        "mean_delta": float(np.mean(deltas)),
        "median_delta_ci_low": med_ci_low,
        "median_delta_ci_high": med_ci_high,
        "mean_delta_ci_low": mean_ci_low,
        "mean_delta_ci_high": mean_ci_high,
        "sign_test_p_two_sided": sign_p,
        "wilcoxon_statistic": wilcoxon_stat,
        "wilcoxon_p_two_sided": wilcoxon_p,
    }


def plot_paired(ax: plt.Axes, df: pd.DataFrame, stats: dict, title: str) -> None:
    before_color = "#2b7bba"
    after_color = "#cf4446"
    line_color = "#777777"

    for i, row in df.reset_index(drop=True).iterrows():
        y = [row["pre_median_synchrony"], row["post_median_synchrony"]]
        ax.plot([0, 1], y, color=line_color, linewidth=1.2, alpha=0.65, zorder=1)
        ax.scatter(0, y[0], s=70, color=before_color, edgecolor="black", linewidth=0.6, zorder=5)
        ax.scatter(1, y[1], s=70, color=after_color, edgecolor="black", linewidth=0.6, zorder=5)

    ax.set_xlim(-0.35, 1.35)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Before\n12 h", "After\n12 h"])
    ax.set_ylabel("Median synchrony")
    ax.set_title(title, fontweight="bold", pad=10)
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    sign_p = format_pvalue(float(stats["sign_test_p_two_sided"]))
    wilcoxon_p = format_pvalue(float(stats["wilcoxon_p_two_sided"]))
    annotation = (
        f"n = {stats['n_patients']} patients\n"
        f"{stats['n_expected_direction']}/{stats['n_patients']} in expected direction\n"
        f"sign test {sign_p}\n"
        f"Wilcoxon {wilcoxon_p}"
    )
    ax.text(
        0.03,
        0.97,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.92, edgecolor="#cccccc"),
    )


def plot_deltas(ax: plt.Axes, df: pd.DataFrame, stats: dict, title: str, expected: str) -> None:
    plot_df = df.sort_values("delta_synchrony").reset_index(drop=True)
    y = np.arange(len(plot_df))
    if expected == "increase":
        colors = np.where(plot_df["delta_synchrony"] > 0, "#2e7d32", "#b45f06")
        expected_label = "Increase"
    else:
        colors = np.where(plot_df["delta_synchrony"] < 0, "#2e7d32", "#b45f06")
        expected_label = "Decrease"

    ax.axvline(0, color="black", linestyle="--", linewidth=1.1, zorder=1)
    ax.scatter(
        plot_df["delta_synchrony"],
        y,
        s=68,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        zorder=5,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([f"HUP {pid}" for pid in plot_df["hup_id"]], fontsize=8)
    ax.set_xlabel("Post-event median minus pre-event median")
    ax.set_title(title, fontweight="bold", pad=10)
    ax.grid(axis="x", linestyle=":", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    median_delta = float(stats["median_delta"])
    ci_low = float(stats["median_delta_ci_low"])
    ci_high = float(stats["median_delta_ci_high"])
    ax.axvline(median_delta, color="#6a1b9a", linewidth=2.0, zorder=2)
    ax.text(
        0.98,
        0.04,
        f"{expected_label} expected\n"
        f"median Δ = {median_delta:.4f}\n"
        f"bootstrap 95% CI [{ci_low:.4f}, {ci_high:.4f}]",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.92, edgecolor="#cccccc"),
    )


def plot_figure(taper_df: pd.DataFrame, reinit_df: pd.DataFrame, stats_df: pd.DataFrame) -> None:
    sns.set_theme(style="ticks", context="paper", font_scale=1.35)
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15, 12),
        dpi=300,
        constrained_layout=True,
        gridspec_kw={"width_ratios": [0.92, 1.08]},
    )

    taper_stats = stats_df[stats_df["event_type"] == "taper"].iloc[0].to_dict()
    reinit_stats = stats_df[stats_df["event_type"] == "reinitiation"].iloc[0].to_dict()

    plot_paired(axes[0, 0], taper_df, taper_stats, "ASM taper")
    plot_deltas(axes[0, 1], taper_df, taper_stats, "Patient-level taper deltas", "increase")
    plot_paired(axes[1, 0], reinit_df, reinit_stats, "ASM re-initiation")
    plot_deltas(
        axes[1, 1],
        reinit_df,
        reinit_stats,
        "Patient-level re-initiation deltas",
        "decrease",
    )

    for label, ax in zip(["A", "B", "C", "D"], axes.ravel()):
        ax.text(
            -0.14,
            1.06,
            label,
            transform=ax.transAxes,
            fontsize=22,
            fontweight="bold",
            va="bottom",
            ha="right",
        )

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")


def print_stats(stats_df: pd.DataFrame) -> None:
    for _, row in stats_df.iterrows():
        print(f"\n{row['event_type'].upper()}")
        print(
            f"  n = {int(row['n_patients'])}; "
            f"{int(row['n_expected_direction'])}/{int(row['n_patients'])} "
            f"in expected direction ({row['expected_direction']})"
        )
        print(
            f"  median delta = {row['median_delta']:.6f} "
            f"(bootstrap 95% CI {row['median_delta_ci_low']:.6f}, "
            f"{row['median_delta_ci_high']:.6f})"
        )
        print(
            f"  sign test: {format_pvalue(row['sign_test_p_two_sided'])}; "
            f"Wilcoxon signed-rank: {format_pvalue(row['wilcoxon_p_two_sided'])}"
        )


def main() -> None:
    print(f"Tables: {TABLES_PATH}")
    print(f"Annotations: {ANNOTATION_CSV}")
    print(f"Window: {WINDOW_MINUTES} minutes before/after event")

    taper_df = build_event_summaries("taper")
    reinit_df = build_event_summaries("reinitiation")
    summary_df = pd.concat([taper_df, reinit_df], ignore_index=True)

    stats_df = pd.DataFrame(
        [
            event_stats(taper_df, expected_direction="increase"),
            event_stats(reinit_df, expected_direction="decrease"),
        ]
    )

    summary_df.to_csv(OUT_SUMMARY_CSV, index=False)
    stats_df.to_csv(OUT_STATS_CSV, index=False)
    plot_figure(taper_df, reinit_df, stats_df)

    print_stats(stats_df)
    print("\nSaved:")
    print(f"  {OUT_PNG}")
    print(f"  {OUT_PDF}")
    print(f"  {OUT_SUMMARY_CSV}")
    print(f"  {OUT_STATS_CSV}")


if __name__ == "__main__":
    main()
