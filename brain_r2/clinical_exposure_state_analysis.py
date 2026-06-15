"""Clinically interpretable ASM exposure-state analysis.

Question:
    Within the same patient, is synchrony higher when ASM load is low than
    when ASM load is high?

This is intended as a clinician-facing complement to the full longitudinal
mixed-effects model. It avoids dense-window inference by giving each patient
one high-ASM synchrony summary and one low-ASM synchrony summary.

Primary contrast:
    - high ASM coverage: normalized non-lorazepam ASM load >= 0.80
    - low ASM coverage: normalized non-lorazepam ASM load <= 0.50
    - peri-ictal exclusion: remove windows within 12 h before or after seizures
    - minimum data per state: at least 6 ten-minute rows (1 h) per patient/state

Run with:
    conda run -n cnt python brain_r2/clinical_exposure_state_analysis.py
or, with the env active:
    python brain_r2/clinical_exposure_state_analysis.py
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
TABLES_PATH = os.path.join(DATA_PATH, "revised_tables_step_4")

SYNC = "synchrony_broadband"
ASM_NORM = "med_sum_no_lorazepam_norm"
TIME_SINCE = "time_since_last_seizure"
TIME_UNTIL = "time_until_next_seizure"

LOW_THRESHOLD = 0.50
HIGH_THRESHOLD = 0.80
PERI_ICTAL_EXCLUSION_MIN = 12 * 60
MIN_ROWS_PER_STATE = 6
N_BOOT = 10000
RNG_SEED = 20260430

OUT_PNG = os.path.join(SCRIPT_DIR, "clinical_exposure_state_analysis.png")
OUT_PDF = os.path.join(SCRIPT_DIR, "clinical_exposure_state_analysis.pdf")
OUT_PATIENT_CSV = os.path.join(SCRIPT_DIR, "clinical_exposure_state_patient_summaries.csv")
OUT_STATS_CSV = os.path.join(SCRIPT_DIR, "clinical_exposure_state_stats.csv")


def format_pvalue(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "P = NA"
    if p_value < 1e-4:
        return f"P = {p_value:.1e}"
    if p_value < 1e-3:
        return f"P = {p_value:.4f}"
    return f"P = {p_value:.3f}"


def patient_ids() -> list[int]:
    return sorted(
        int(filename.split("_")[1].split(".")[0])
        for filename in os.listdir(TABLES_PATH)
        if filename.startswith("HUP_") and filename.endswith(".csv")
    )


def load_patient(hup_id: int) -> pd.DataFrame:
    cols = [SYNC, ASM_NORM, TIME_SINCE, TIME_UNTIL]
    path = os.path.join(TABLES_PATH, f"HUP_{hup_id}.csv")
    return pd.read_csv(path, usecols=cols, low_memory=False)


def clean_rows(df: pd.DataFrame, peri_ictal_exclusion_min: int) -> pd.DataFrame:
    out = df.dropna(subset=[SYNC, ASM_NORM]).copy()
    out = out[out[SYNC] < 0.8]
    if peri_ictal_exclusion_min > 0:
        far_from_prior = out[TIME_SINCE].isna() | (out[TIME_SINCE] >= peri_ictal_exclusion_min)
        far_from_next = out[TIME_UNTIL].isna() | (out[TIME_UNTIL] >= peri_ictal_exclusion_min)
        out = out[far_from_prior & far_from_next].copy()
    return out


def summarize_patient(
    hup_id: int,
    low_threshold: float,
    high_threshold: float,
    peri_ictal_exclusion_min: int,
    min_rows_per_state: int,
) -> dict[str, float | int] | None:
    df = clean_rows(load_patient(hup_id), peri_ictal_exclusion_min)
    high = df[df[ASM_NORM] >= high_threshold][SYNC].dropna()
    low = df[df[ASM_NORM] <= low_threshold][SYNC].dropna()
    if len(high) < min_rows_per_state or len(low) < min_rows_per_state:
        return None

    high_median = float(high.median())
    low_median = float(low.median())
    delta = low_median - high_median
    return {
        "hup_id": hup_id,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "peri_ictal_exclusion_min": peri_ictal_exclusion_min,
        "n_high_rows": int(len(high)),
        "n_low_rows": int(len(low)),
        "high_asm_median_synchrony": high_median,
        "low_asm_median_synchrony": low_median,
        "delta_low_minus_high": delta,
        "relative_delta_percent": float(100.0 * delta / high_median),
    }


def build_patient_summaries(
    low_threshold: float = LOW_THRESHOLD,
    high_threshold: float = HIGH_THRESHOLD,
    peri_ictal_exclusion_min: int = PERI_ICTAL_EXCLUSION_MIN,
    min_rows_per_state: int = MIN_ROWS_PER_STATE,
) -> pd.DataFrame:
    rows = []
    for hup_id in patient_ids():
        row = summarize_patient(
            hup_id,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            peri_ictal_exclusion_min=peri_ictal_exclusion_min,
            min_rows_per_state=min_rows_per_state,
        )
        if row is not None:
            rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray, statistic=np.median) -> tuple[float, float]:
    rng = np.random.default_rng(RNG_SEED)
    values = np.asarray(values, dtype=float)
    boot = np.empty(N_BOOT, dtype=float)
    for i in range(N_BOOT):
        boot[i] = statistic(rng.choice(values, size=len(values), replace=True))
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def paired_stats(df: pd.DataFrame, label: str) -> dict[str, float | int | str]:
    delta = df["delta_low_minus_high"].to_numpy(dtype=float)
    n_positive = int(np.sum(delta > 0))
    n_negative = int(np.sum(delta < 0))
    n_nonzero = n_positive + n_negative
    sign_p = (
        float(binomtest(n_positive, n_nonzero, p=0.5).pvalue)
        if n_nonzero > 0
        else float("nan")
    )
    wilc = wilcoxon(delta, zero_method="wilcox", alternative="two-sided", method="exact")
    med_ci_low, med_ci_high = bootstrap_ci(delta, np.median)
    mean_ci_low, mean_ci_high = bootstrap_ci(delta, np.mean)
    rel_ci_low, rel_ci_high = bootstrap_ci(df["relative_delta_percent"].to_numpy(float), np.median)
    return {
        "label": label,
        "n_patients": int(len(df)),
        "n_positive_delta": n_positive,
        "n_negative_delta": n_negative,
        "median_delta": float(np.median(delta)),
        "mean_delta": float(np.mean(delta)),
        "median_delta_ci_low": med_ci_low,
        "median_delta_ci_high": med_ci_high,
        "mean_delta_ci_low": mean_ci_low,
        "mean_delta_ci_high": mean_ci_high,
        "median_relative_delta_percent": float(np.median(df["relative_delta_percent"])),
        "median_relative_delta_percent_ci_low": rel_ci_low,
        "median_relative_delta_percent_ci_high": rel_ci_high,
        "sign_test_p_two_sided": sign_p,
        "wilcoxon_statistic": float(wilc.statistic),
        "wilcoxon_p_two_sided": float(wilc.pvalue),
    }


def sensitivity_grid() -> pd.DataFrame:
    variants = [
        (
            "Main: low <= 50%, high >= 80%, exclude +/-12 h",
            "Main threshold\nexclude +/-12 h",
            0.50,
            0.80,
            12 * 60,
        ),
        ("Exclude +/-5 h", "Main threshold\nexclude +/-5 h", 0.50, 0.80, 5 * 60),
        ("Exclude +/-1 h", "Main threshold\nexclude +/-1 h", 0.50, 0.80, 60),
        ("No peri-ictal exclusion", "Main threshold\nno peri-ictal exclusion", 0.50, 0.80, 0),
        (
            "Stricter low: <= 33%, high >= 67%, exclude +/-12 h",
            "Low <=33%, high >=67%\nexclude +/-12 h",
            0.33,
            0.67,
            12 * 60,
        ),
        (
            "Stricter extremes: <= 25%, >= 75%, exclude +/-12 h",
            "Low <=25%, high >=75%\nexclude +/-12 h",
            0.25,
            0.75,
            12 * 60,
        ),
        (
            "Low <= 40%, high >= 80%, exclude +/-12 h",
            "Low <=40%, high >=80%\nexclude +/-12 h",
            0.40,
            0.80,
            12 * 60,
        ),
        (
            "Low <= 50%, high >= 90%, exclude +/-12 h",
            "Low <=50%, high >=90%\nexclude +/-12 h",
            0.50,
            0.90,
            12 * 60,
        ),
    ]

    rows = []
    for label, plot_label, low_threshold, high_threshold, exclusion in variants:
        summary = build_patient_summaries(
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            peri_ictal_exclusion_min=exclusion,
            min_rows_per_state=MIN_ROWS_PER_STATE,
        )
        stats = paired_stats(summary, label=label)
        stats.update(
            {
                "plot_label": plot_label,
                "low_threshold": low_threshold,
                "high_threshold": high_threshold,
                "peri_ictal_exclusion_min": exclusion,
            }
        )
        rows.append(stats)
    return pd.DataFrame(rows)


def plot_primary_pair(ax: plt.Axes, df: pd.DataFrame, stats: dict) -> None:
    for _, row in df.iterrows():
        ax.plot(
            [0, 1],
            [row["high_asm_median_synchrony"], row["low_asm_median_synchrony"]],
            color="#b8b8b8",
            alpha=0.38,
            linewidth=0.75,
            zorder=1,
        )
    ax.scatter(
        np.zeros(len(df)),
        df["high_asm_median_synchrony"],
        s=62,
        color="#2b7bba",
        edgecolor="black",
        linewidth=0.6,
        zorder=5,
        label="High ASM",
    )
    ax.scatter(
        np.ones(len(df)),
        df["low_asm_median_synchrony"],
        s=62,
        color="#cf4446",
        edgecolor="black",
        linewidth=0.6,
        zorder=5,
        label="Low ASM",
    )
    ax.set_xlim(-0.35, 2.25)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["High ASM\n>= 80%", "Low ASM\n<= 50%"])
    ax.set_ylabel("Patient median synchrony")
    ax.set_title("Within-patient ASM exposure state", fontweight="bold", pad=10)
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left")

    annotation = (
        f"n = {stats['n_patients']} patients\n"
        f"{stats['n_positive_delta']}/{stats['n_patients']} higher in low ASM\n"
        f"median relative increase = {stats['median_relative_delta_percent']:.1f}%\n"
        f"sign test {format_pvalue(stats['sign_test_p_two_sided'])}\n"
        f"Wilcoxon {format_pvalue(stats['wilcoxon_p_two_sided'])}"
    )
    ax.text(
        0.98,
        0.97,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=1.0, edgecolor="#cccccc"),
        zorder=20,
    )


def plot_delta_rank(ax: plt.Axes, df: pd.DataFrame, stats: dict) -> None:
    plot_df = df.sort_values("delta_low_minus_high").reset_index(drop=True)
    y = np.arange(len(plot_df))
    colors = np.where(plot_df["delta_low_minus_high"] > 0, "#2e7d32", "#b45f06")
    ax.axvline(0, color="black", linestyle="--", linewidth=1.0)
    ax.scatter(
        plot_df["delta_low_minus_high"],
        y,
        s=58,
        color=colors,
        edgecolor="black",
        linewidth=0.55,
        zorder=5,
    )
    median_delta = stats["median_delta"]
    ax.axvline(median_delta, color="#6a1b9a", linewidth=2.0)
    ax.set_yticks([])
    ax.set_xlabel("Low-ASM median synchrony minus high-ASM median synchrony")
    ax.set_title("Patient-level deltas", fontweight="bold", pad=10)
    ax.grid(axis="x", linestyle=":", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.98,
        0.05,
        f"median Δ = {median_delta:.4f}\n"
        f"bootstrap 95% CI [{stats['median_delta_ci_low']:.4f}, "
        f"{stats['median_delta_ci_high']:.4f}]",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.92, edgecolor="#cccccc"),
    )


def plot_sensitivity(ax: plt.Axes, sens: pd.DataFrame) -> None:
    plot_df = sens.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(plot_df))
    colors = ["#6a1b9a" if row["label"].startswith("Main") else "#4d4d4d" for _, row in plot_df.iterrows()]
    err_low = plot_df["median_delta"] - plot_df["median_delta_ci_low"]
    err_high = plot_df["median_delta_ci_high"] - plot_df["median_delta"]
    ax.axvline(0, color="black", linestyle="--", linewidth=1.0)
    for i, row in plot_df.iterrows():
        ax.errorbar(
            row["median_delta"],
            i,
            xerr=[[err_low.iloc[i]], [err_high.iloc[i]]],
            fmt="o",
            color=colors[i],
            ecolor=colors[i],
            capsize=3,
            markersize=7,
            linewidth=1.6,
        )
        ax.text(
            row["median_delta_ci_high"] + 0.001,
            i,
            f"n={int(row['n_patients'])}, {format_pvalue(row['wilcoxon_p_two_sided'])}",
            va="center",
            ha="left",
            fontsize=9.2,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["plot_label"].tolist(), fontsize=9.2)
    ax.set_xlabel("Median patient-level Δ synchrony (bootstrap 95% CI)")
    ax.set_title("Sensitivity checks", fontweight="bold", pad=10)
    ax.grid(axis="x", linestyle=":", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    x_max = max(plot_df["median_delta_ci_high"]) + 0.012
    x_min = min(0.0, min(plot_df["median_delta_ci_low"]) - 0.002)
    ax.set_xlim(x_min, x_max)


def plot_figure(primary: pd.DataFrame, stats: dict, sens: pd.DataFrame) -> None:
    sns.set_theme(style="ticks", context="paper", font_scale=1.35)
    fig = plt.figure(figsize=(16, 10), dpi=300, constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[0.9, 1.1], height_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    plot_primary_pair(ax_a, primary, stats)
    plot_delta_rank(ax_b, primary, stats)
    plot_sensitivity(ax_c, sens)

    for label, ax in zip(["A", "B", "C"], [ax_a, ax_b, ax_c]):
        ax.text(
            -0.12,
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


def main() -> None:
    print(f"Tables: {TABLES_PATH}")
    print(
        "Primary contrast: high ASM >= 80%, low ASM <= 50%, "
        "excluding +/-12 h around seizures"
    )

    primary = build_patient_summaries()
    stats = paired_stats(primary, label="primary")
    sens = sensitivity_grid()

    primary.to_csv(OUT_PATIENT_CSV, index=False)
    sens.to_csv(OUT_STATS_CSV, index=False)
    plot_figure(primary, stats, sens)

    print("\nPRIMARY EXPOSURE-STATE CONTRAST")
    print(
        f"  n = {stats['n_patients']} patients; "
        f"{stats['n_positive_delta']}/{stats['n_patients']} had higher synchrony "
        "during low ASM state"
    )
    print(
        f"  median delta = {stats['median_delta']:.6f} "
        f"(bootstrap 95% CI {stats['median_delta_ci_low']:.6f}, "
        f"{stats['median_delta_ci_high']:.6f})"
    )
    print(
        f"  median relative increase = {stats['median_relative_delta_percent']:.2f}% "
        f"(bootstrap 95% CI {stats['median_relative_delta_percent_ci_low']:.2f}%, "
        f"{stats['median_relative_delta_percent_ci_high']:.2f}%)"
    )
    print(
        f"  sign test: {format_pvalue(stats['sign_test_p_two_sided'])}; "
        f"Wilcoxon signed-rank: {format_pvalue(stats['wilcoxon_p_two_sided'])}"
    )

    print("\nSaved:")
    print(f"  {OUT_PNG}")
    print(f"  {OUT_PDF}")
    print(f"  {OUT_PATIENT_CSV}")
    print(f"  {OUT_STATS_CSV}")


if __name__ == "__main__":
    main()
