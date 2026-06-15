"""Figure 3 candidate: patient-level detection of low ASM coverage from synchrony.

Clinical question:
    Within a patient, do higher synchrony values identify windows in which ASM
    coverage is low?

This is a translational companion to the exposure-state analysis. The unit of
inference is the patient: for each patient, we compute an AUC for using
synchrony to discriminate low-ASM windows from high-ASM windows. AUC > 0.5
means higher synchrony tends to occur during the low-ASM state.

Primary definition:
    - low ASM coverage: normalized non-lorazepam ASM load <= 0.50
    - high ASM coverage: normalized non-lorazepam ASM load >= 0.80
    - exclude windows within 12 h before/after seizures
    - require at least 6 ten-minute rows in each state

Run with:
    conda run -n cnt python brain_r2/figure_3_low_asm_detection.py
or, with the env active:
    python brain_r2/figure_3_low_asm_detection.py
"""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/asm_synchrony_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binomtest, rankdata, wilcoxon

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

OUT_PNG = os.path.join(SCRIPT_DIR, "figure_3_low_asm_detection.png")
OUT_PDF = os.path.join(SCRIPT_DIR, "figure_3_low_asm_detection.pdf")
OUT_PATIENT_CSV = os.path.join(SCRIPT_DIR, "figure_3_low_asm_detection_patient_auc.csv")
OUT_STATS_CSV = os.path.join(SCRIPT_DIR, "figure_3_low_asm_detection_stats.csv")


def format_pvalue(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "P = NA"
    if p_value < 1e-4:
        return f"P = {p_value:.1e}"
    if p_value < 1e-3:
        return f"P = {p_value:.4f}"
    return f"P = {p_value:.3f}"


def format_auc(value: float) -> str:
    return f"{value:.3f}"


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


def auc_from_scores(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUC for positive label 1 using rank-sum/Mann-Whitney equivalence."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    rank_sum_pos = float(np.sum(ranks[labels == 1]))
    u_pos = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u_pos / (n_pos * n_neg))


def summarize_patient(
    hup_id: int,
    low_threshold: float,
    high_threshold: float,
    peri_ictal_exclusion_min: int,
    min_rows_per_state: int,
) -> dict[str, float | int] | None:
    df = clean_rows(load_patient(hup_id), peri_ictal_exclusion_min)
    high = df[df[ASM_NORM] >= high_threshold].copy()
    low = df[df[ASM_NORM] <= low_threshold].copy()
    if len(high) < min_rows_per_state or len(low) < min_rows_per_state:
        return None

    high = high.assign(low_asm_state=0)
    low = low.assign(low_asm_state=1)
    both = pd.concat([high, low], axis=0, ignore_index=True)
    auc = auc_from_scores(
        labels=both["low_asm_state"].to_numpy(),
        scores=both[SYNC].to_numpy(),
    )
    high_median = float(high[SYNC].median())
    low_median = float(low[SYNC].median())
    return {
        "hup_id": hup_id,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "peri_ictal_exclusion_min": peri_ictal_exclusion_min,
        "n_high_rows": int(len(high)),
        "n_low_rows": int(len(low)),
        "auc": auc,
        "auc_minus_chance": auc - 0.5,
        "high_asm_median_synchrony": high_median,
        "low_asm_median_synchrony": low_median,
        "delta_low_minus_high": low_median - high_median,
    }


def build_patient_auc(
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


def auc_stats(df: pd.DataFrame, label: str) -> dict[str, float | int | str]:
    auc = df["auc"].to_numpy(dtype=float)
    auc_minus_chance = auc - 0.5
    n_above = int(np.sum(auc > 0.5))
    n_below = int(np.sum(auc < 0.5))
    n_nonchance = n_above + n_below
    sign_p = (
        float(binomtest(n_above, n_nonchance, p=0.5).pvalue)
        if n_nonchance > 0
        else float("nan")
    )
    wilc = wilcoxon(
        auc_minus_chance,
        zero_method="wilcox",
        alternative="two-sided",
        method="exact",
    )
    med_ci_low, med_ci_high = bootstrap_ci(auc, np.median)
    mean_ci_low, mean_ci_high = bootstrap_ci(auc, np.mean)
    return {
        "label": label,
        "n_patients": int(len(df)),
        "n_auc_above_chance": n_above,
        "n_auc_below_chance": n_below,
        "median_auc": float(np.median(auc)),
        "mean_auc": float(np.mean(auc)),
        "median_auc_ci_low": med_ci_low,
        "median_auc_ci_high": med_ci_high,
        "mean_auc_ci_low": mean_ci_low,
        "mean_auc_ci_high": mean_ci_high,
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
            "Low <= 33%, high >= 67%, exclude +/-12 h",
            "Low <=33%, high >=67%\nexclude +/-12 h",
            0.33,
            0.67,
            12 * 60,
        ),
        (
            "Low <= 25%, high >= 75%, exclude +/-12 h",
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
        patient_auc = build_patient_auc(
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            peri_ictal_exclusion_min=exclusion,
        )
        stats = auc_stats(patient_auc, label=label)
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


def plot_auc_distribution(ax: plt.Axes, df: pd.DataFrame, stats: dict) -> None:
    rng = np.random.default_rng(RNG_SEED)
    x = rng.normal(0, 0.045, size=len(df))
    colors = np.where(df["auc"] > 0.5, "#2e7d32", "#b45f06")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.1)
    ax.scatter(
        x,
        df["auc"],
        s=76,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.9,
        zorder=5,
    )
    med = float(stats["median_auc"])
    ax.errorbar(
        0.2,
        med,
        yerr=[
            [med - float(stats["median_auc_ci_low"])],
            [float(stats["median_auc_ci_high"]) - med],
        ],
        fmt="o",
        color="#6a1b9a",
        ecolor="#6a1b9a",
        capsize=5,
        markersize=9,
        linewidth=2.0,
        label="Patient median",
    )
    ax.set_xlim(-0.22, 0.34)
    ax.set_xticks([])
    ax.set_ylabel("Patient-level AUC")
    ax.set_title("Synchrony detection of low ASM state", fontweight="bold", pad=10)
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    annotation = (
        f"n = {stats['n_patients']} patients\n"
        f"{stats['n_auc_above_chance']}/{stats['n_patients']} AUC > 0.5\n"
        f"median AUC = {format_auc(stats['median_auc'])}\n"
        f"sign test {format_pvalue(stats['sign_test_p_two_sided'])}\n"
        f"Wilcoxon {format_pvalue(stats['wilcoxon_p_two_sided'])}"
    )
    ax.text(
        0.96,
        0.04,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.92, edgecolor="#cccccc"),
    )


def plot_auc_rank(ax: plt.Axes, df: pd.DataFrame) -> None:
    plot_df = df.sort_values("auc").reset_index(drop=True)
    y = np.arange(len(plot_df))
    colors = np.where(plot_df["auc"] > 0.5, "#2e7d32", "#b45f06")
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.1)
    ax.scatter(
        plot_df["auc"],
        y,
        s=58,
        color=colors,
        edgecolor="black",
        linewidth=0.55,
        zorder=5,
    )
    ax.set_yticks([])
    ax.set_xlabel("Patient-level AUC")
    ax.set_title("Patient heterogeneity", fontweight="bold", pad=10)
    ax.grid(axis="x", linestyle=":", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.04,
        0.96,
        "AUC > 0.5 means higher synchrony\nidentifies low ASM coverage.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.92, edgecolor="#cccccc"),
    )


def plot_sensitivity(ax: plt.Axes, sens: pd.DataFrame) -> None:
    plot_df = sens.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(plot_df))
    colors = ["#6a1b9a" if row["label"].startswith("Main") else "#4d4d4d" for _, row in plot_df.iterrows()]
    err_low = plot_df["median_auc"] - plot_df["median_auc_ci_low"]
    err_high = plot_df["median_auc_ci_high"] - plot_df["median_auc"]
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.0)
    for i, row in plot_df.iterrows():
        ax.errorbar(
            row["median_auc"],
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
            row["median_auc_ci_high"] + 0.02,
            i,
            f"n={int(row['n_patients'])}, {format_pvalue(row['wilcoxon_p_two_sided'])}",
            va="center",
            ha="left",
            fontsize=9.2,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["plot_label"].tolist(), fontsize=9.2)
    ax.set_xlabel("Median patient-level AUC (bootstrap 95% CI)")
    ax.set_title("Sensitivity checks", fontweight="bold", pad=10)
    ax.grid(axis="x", linestyle=":", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    x_min = min(0.45, float(plot_df["median_auc_ci_low"].min()) - 0.03)
    x_max = max(0.75, float(plot_df["median_auc_ci_high"].max()) + 0.15)
    ax.set_xlim(x_min, x_max)


def plot_figure(primary: pd.DataFrame, stats: dict, sens: pd.DataFrame) -> None:
    sns.set_theme(style="ticks", context="paper", font_scale=1.35)
    fig = plt.figure(figsize=(16, 10), dpi=300, constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[0.8, 1.2], height_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])
    plot_auc_distribution(ax_a, primary, stats)
    plot_auc_rank(ax_b, primary)
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
        "Primary detection contrast: synchrony as score for low ASM "
        "(low <= 50%, high >= 80%, excluding +/-12 h around seizures)"
    )
    primary = build_patient_auc()
    stats = auc_stats(primary, label="primary")
    sens = sensitivity_grid()
    primary.to_csv(OUT_PATIENT_CSV, index=False)
    sens.to_csv(OUT_STATS_CSV, index=False)
    plot_figure(primary, stats, sens)

    print("\nPRIMARY LOW-ASM DETECTION ANALYSIS")
    print(
        f"  n = {stats['n_patients']} patients; "
        f"{stats['n_auc_above_chance']}/{stats['n_patients']} had AUC > 0.5"
    )
    print(
        f"  median AUC = {stats['median_auc']:.6f} "
        f"(bootstrap 95% CI {stats['median_auc_ci_low']:.6f}, "
        f"{stats['median_auc_ci_high']:.6f})"
    )
    print(
        f"  sign test: {format_pvalue(stats['sign_test_p_two_sided'])}; "
        f"Wilcoxon signed-rank vs 0.5: {format_pvalue(stats['wilcoxon_p_two_sided'])}"
    )
    print("\nSaved:")
    print(f"  {OUT_PNG}")
    print(f"  {OUT_PDF}")
    print(f"  {OUT_PATIENT_CSV}")
    print(f"  {OUT_STATS_CSV}")


if __name__ == "__main__":
    main()
