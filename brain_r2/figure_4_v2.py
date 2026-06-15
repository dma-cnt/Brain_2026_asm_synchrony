"""Figure 4 v2 — Four-panel layout (LMM coefficients + sensitivity analyses).

Four panels:
  A. Linear mixed-effects model coefficients (random intercept on patient).
     Standard 10-predictor forest plot; synchrony highlighted in red.
  B. Convergence forest plot of the synchrony fixed-effect coefficient (β
     with 95% CI) in the primary LMM and the two coefficient-estimating
     sensitivity analyses: GEE with Independence working correlation +
     robust sandwich SEs, and GEE with Unstructured working correlation on
     the 30-min binned panel.
  C. Within-patient circular-shift permutation null with the observed
     synchrony coefficient marked. Preserves each patient's autocorrelation
     under the null.
  D. Likelihood-ratio tests for incremental fit (with vs. without the
     synchrony fixed effect), under both random-effects structures.

Run with:
    /opt/miniconda3/envs/cnt/bin/python brain_r2/figure_4_v2.py
or, with the env active:
    python brain_r2/figure_4_v2.py
"""

import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler
from statsmodels.genmod.cov_struct import Independence, Unstructured
from statsmodels.genmod.families import Gaussian

from circular_shift_pvalues import circular_shift_pvalues

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Paths and config (mirrored from figure_4.py / appeal_analysis.ipynb)
# -----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "cnt-data"))
TABLES_PATH = os.path.join(DATA_PATH, "revised_tables_step_4")
PERM_CSV = os.path.join(SCRIPT_DIR, "perm_betas_circular_shift_full_5000perm_seed0.csv")
OUT_PNG = os.path.join(SCRIPT_DIR, "figure_4_v2.png")
OUT_PDF = os.path.join(SCRIPT_DIR, "figure_4_v2.pdf")

Y = "med_sum_no_lorazepam_raw"
GROUP = "patient_hup_id"
TIME_RAW = "emu_minute_raw"
SYNC = "synchrony_broadband"

PREDICTORS = [
    "synchrony_broadband",
    "time_since_last_seizure",
    "time_until_next_seizure",
    "emu_minute",
    "ad_ratio",
    "tod_sin",
    "tod_cos",
    "tod_sin2",
    "tod_cos2",
    "spike_rate",
]

COLS_TO_SCALE = [
    "synchrony_broadband",
    "time_since_last_seizure",
    "time_until_next_seizure",
    "emu_minute",
    "ad_ratio",
    "spike_rate",
]

FORMULA = f"{Y} ~ " + " + ".join(PREDICTORS)
FORMULA_NO_SYNC = f"{Y} ~ " + " + ".join(p for p in PREDICTORS if p != SYNC)

LABEL_MAP = {
    "synchrony_broadband":     "Synchrony",
    "time_since_last_seizure": "Time since last seizure",
    "time_until_next_seizure": "Time until next seizure",
    "emu_minute":              "Time since EMU admission",
    "ad_ratio":                "Alpha–delta ratio",
    "tod_sin":                 "Time of day (sin)",
    "tod_cos":                 "Time of day (cos)",
    "tod_sin2":                "Time of day (2nd harm., sin)",
    "tod_cos2":                "Time of day (2nd harm., cos)",
    "spike_rate":              "Spike rate",
}

HIGHLIGHT = "#2b7bba"
HIGHLIGHT_LIGHT = "#a4c8e1"
NEUTRAL = "#34495e"
SOFT_GRAY = "#7f8c8d"


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
def get_patient_hup_ids(directory):
    files = os.listdir(directory)
    return sorted(
        int(f.split("_")[1].split(".")[0])
        for f in files
        if f.startswith("HUP_") and f.endswith(".csv")
    )


def build_all_data(pids):
    dfs = []
    for pid in pids:
        fp = os.path.join(TABLES_PATH, f"HUP_{pid}.csv")
        if not os.path.exists(fp):
            continue
        d = pd.read_csv(fp, low_memory=False)
        d[GROUP] = pid
        keep = [c for c in ([GROUP, "emu_minute"] + PREDICTORS + [Y]) if c in d.columns]
        dfs.append(d[keep])

    all_data = pd.concat(dfs, axis=0, ignore_index=True)
    all_data = all_data.loc[:, ~all_data.columns.duplicated()].copy()
    all_data["emu_minute"] = pd.to_numeric(all_data["emu_minute"], errors="coerce")
    all_data[TIME_RAW] = all_data["emu_minute"].copy()
    needed = [Y, GROUP, "emu_minute", TIME_RAW] + [c for c in PREDICTORS if c in all_data.columns]
    all_data = all_data.dropna(subset=needed).reset_index(drop=True)
    all_data = all_data.sort_values([GROUP, "emu_minute"]).reset_index(drop=True)
    all_data["time_index"] = all_data.groupby(GROUP).cumcount().astype(int)
    scaler = StandardScaler()
    cols_to_scale = [c for c in COLS_TO_SCALE if c in all_data.columns]
    all_data[cols_to_scale] = scaler.fit_transform(all_data[cols_to_scale])
    return all_data


def bin_within_patient(df, bin_minutes):
    d = df.copy()
    d["bin"] = np.floor(d[TIME_RAW] / float(bin_minutes)).astype(int)
    agg_cols = [Y] + PREDICTORS + [GROUP, TIME_RAW]
    out = (
        d[agg_cols + ["bin"]]
        .groupby([GROUP, "bin"], as_index=False)
        .mean(numeric_only=True)
    )
    out = out.sort_values([GROUP, TIME_RAW]).reset_index(drop=True)
    out["time_index"] = out.groupby(GROUP).cumcount().astype(int)
    return out


# -----------------------------------------------------------------------------
# Model fits
# -----------------------------------------------------------------------------
def fit_lme_random_intercept(df, reml=True):
    return smf.mixedlm(FORMULA, df, groups=df[GROUP]).fit(reml=reml)


def fit_lme_random_slope(df, reml=True):
    return smf.mixedlm(
        FORMULA, df, groups=df[GROUP], re_formula=f"~{SYNC}"
    ).fit(reml=reml, method="lbfgs")


def fit_gee_independence_robust(df):
    m = smf.gee(
        formula=FORMULA, groups=df[GROUP], data=df,
        family=Gaussian(), cov_struct=Independence(),
    )
    return m.fit(cov_type="robust")


def fit_gee_unstructured_binned(df, bin_minutes=30, T=20, min_patients=5):
    d0 = bin_within_patient(df, bin_minutes)
    sizes = d0.groupby(GROUP).size()
    eligible = sizes[sizes >= T].index.values
    d1 = d0[d0[GROUP].isin(eligible)].copy()
    if d1[GROUP].nunique() < min_patients:
        raise RuntimeError("Too few patients with required time points")
    d1 = (
        d1.sort_values([GROUP, TIME_RAW])
        .groupby(GROUP, as_index=False, group_keys=False)
        .head(T)
        .reset_index(drop=True)
    )
    d1["time_index"] = d1.groupby(GROUP).cumcount().astype(int)
    m = smf.gee(
        formula=FORMULA, groups=d1[GROUP], data=d1,
        family=Gaussian(), cov_struct=Unstructured(),
        time=d1["time_index"].values,
    )
    res = m.fit(cov_type="robust")
    res._fit_n_patients = int(d1[GROUP].nunique())
    res._fit_n_obs = int(len(d1))
    return res


def lrt(full_res, red_res, label):
    """Likelihood-ratio test between two ML-fit MixedLM results."""
    llf_full = float(full_res.llf)
    llf_red = float(red_res.llf)
    stat = 2.0 * (llf_full - llf_red)
    df_full = int(full_res.df_modelwc)
    df_red = int(red_res.df_modelwc)
    df_diff = df_full - df_red
    p = float(chi2.sf(stat, df_diff)) if df_diff > 0 else float("nan")
    return {
        "label": label,
        "lr_stat": stat,
        "df": df_diff,
        "p": p,
        "llf_full": llf_full,
        "llf_red": llf_red,
    }


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def synchrony_estimate(res):
    beta = float(res.params[SYNC])
    se = float(res.bse[SYNC])
    ci = res.conf_int()
    if SYNC in ci.index:
        ci_low = float(ci.loc[SYNC, 0])
        ci_high = float(ci.loc[SYNC, 1])
    else:
        ci_low = beta - 1.96 * se
        ci_high = beta + 1.96 * se
    p = float(res.pvalues[SYNC])
    return beta, se, ci_low, ci_high, p


def format_pvalue(p):
    if p < 1e-3:
        return f"P = {p:.1e}"
    return f"P = {p:.3f}"


def format_lr_pvalue(p):
    if p < 1e-3:
        # use scientific with one sig digit on mantissa, integer exponent
        mant, exp = f"{p:.1e}".split("e")
        return f"P = {float(mant):.1f}×10$^{{{int(exp)}}}$"
    return f"P = {p:.3f}"


# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------
def plot_figure(res_lme_ri, rows_b, perm, obs_beta_perm, perm_pvalues, lrt_ri, lrt_rs):
    sns.set_theme(style="ticks", context="talk")

    fig = plt.figure(figsize=(20, 15), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.4])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_dict = {"A": ax_a, "B": ax_b, "C": ax_c, "D": ax_d}

    # =========================================================================
    # Panel A: Linear mixed-effects model coefficients (full model)
    # =========================================================================
    fe_params = res_lme_ri.fe_params
    conf_int = res_lme_ri.conf_int(alpha=0.05)
    conf_int.columns = ["ci_lower", "ci_upper"]
    coefs = (
        pd.concat([fe_params, conf_int], axis=1)
        .reset_index()
        .rename(columns={"index": "term", 0: "estimate"})
    )
    coefs = coefs[~coefs["term"].astype(str).str.startswith("Group")].copy()
    coefs = coefs[coefs["term"] != "Intercept"].copy()
    coefs["label"] = coefs["term"].map(LABEL_MAP).fillna(coefs["term"])
    coefs["is_synchrony"] = coefs["term"] == SYNC
    coefs = coefs.sort_values(["is_synchrony", "estimate"]).reset_index(drop=True)

    for i in range(len(coefs)):
        if i % 2 == 0:
            ax_a.axhspan(i - 0.5, i + 0.5, color="#f0f0f0", alpha=0.6, zorder=0)

    err_low_a = coefs["estimate"] - coefs["ci_lower"]
    err_high_a = coefs["ci_upper"] - coefs["estimate"]
    sync_mask = coefs["is_synchrony"].values
    other_mask = ~sync_mask
    other_idx = np.where(other_mask)[0]
    sync_idx = np.where(sync_mask)[0]
    if other_idx.size:
        ax_a.errorbar(
            coefs.loc[other_mask, "estimate"], other_idx,
            xerr=[err_low_a[other_mask], err_high_a[other_mask]],
            fmt="o", color="black", ecolor="black", capsize=3,
            markersize=8, linewidth=1.5, zorder=10,
        )
    if sync_idx.size:
        ax_a.errorbar(
            coefs.loc[sync_mask, "estimate"], sync_idx,
            xerr=[err_low_a[sync_mask], err_high_a[sync_mask]],
            fmt="o", color=HIGHLIGHT, ecolor=HIGHLIGHT, capsize=3,
            markersize=10, linewidth=2.0, zorder=11,
        )
    ax_a.set_yticks(range(len(coefs)))
    ax_a.set_yticklabels(coefs["label"], color="black", fontsize=12)
    for i, row in coefs.iterrows():
        if row["is_synchrony"]:
            ax_a.get_yticklabels()[i].set_fontweight("bold")
            ax_a.get_yticklabels()[i].set_color(HIGHLIGHT)
    ax_a.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax_a.set_xlabel(
        "Fixed-effect estimate (LME, 95% CI)",
        fontsize=14, fontweight="bold", labelpad=10,
    )
    ax_a.set_title(
        "Linear Mixed-Effects Model",
        fontweight="bold", pad=10,
    )
    ax_a.spines["left"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["bottom"].set_linewidth(1.5)
    ax_a.grid(axis="x", linestyle=":", alpha=0.5)
    ax_a.tick_params(axis="y", length=0)

    # =========================================================================
    # Panel B: Convergence forest plot (5 methods)
    # =========================================================================
    df_b = pd.DataFrame(rows_b)
    df_b_plot = df_b.iloc[::-1].reset_index(drop=True)  # top-to-bottom in plot order

    for i in range(len(df_b_plot)):
        if i % 2 == 0:
            ax_b.axhspan(i - 0.5, i + 0.5, color="#f0f0f0", alpha=0.6, zorder=0)

    err_low_b = df_b_plot["beta"] - df_b_plot["ci_low"]
    err_high_b = df_b_plot["ci_high"] - df_b_plot["beta"]
    ax_b.errorbar(
        df_b_plot["beta"], range(len(df_b_plot)),
        xerr=[err_low_b, err_high_b],
        fmt="o", color=HIGHLIGHT, ecolor=HIGHLIGHT, capsize=5,
        markersize=12, linewidth=2.4, zorder=10,
    )
    ax_b.axvline(0, color="black", linestyle="--", linewidth=1.2)

    # combined y-tick labels: method name + dependence assumption + n
    yticklabels_b = []
    for _, row in df_b_plot.iterrows():
        yticklabels_b.append(
            f"{row['label']}\n"
            f"({row['dep_assumption']}, n = {row['n_clusters']} {row['cluster_label']})"
        )
    ax_b.set_yticks(range(len(df_b_plot)))
    ax_b.set_yticklabels(yticklabels_b, fontsize=11)
    ax_b.set_ylim(-0.6, len(df_b_plot) - 0.4)

    ax_b.set_xlabel(
        "Synchrony coefficient (standardized predictor, 95% CI)",
        fontsize=14, fontweight="bold", labelpad=10,
    )
    ax_b.set_title(
        "Synchrony coefficient across longitudinal models",
        fontweight="bold", pad=10,
    )

    # x-limits: only need padding on the right for β/p annotation
    x_min = float(df_b_plot["ci_low"].min())
    x_max = float(df_b_plot["ci_high"].max())
    span = max(0.2, x_max - x_min)
    pad_left = 0.05 * span
    pad_right = 0.40 * span
    ax_b.set_xlim(x_min - pad_left, x_max + pad_right)

    # right-side annotation: β + p-value
    annot_x_right = x_max + 0.05 * span
    for i, row in df_b_plot.iterrows():
        if pd.isna(row["p"]):
            txt = (
                f"β = {row['beta']:.3f}\n"
                f"95% CI [{row['ci_low']:.3f}, {row['ci_high']:.3f}]"
            )
        else:
            txt = (
                f"β = {row['beta']:.3f}\n"
                f"{format_pvalue(row['p'])}"
            )
        ax_b.text(
            annot_x_right, i, txt,
            va="center", ha="left", fontsize=11,
        )

    ax_b.spines["left"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["bottom"].set_linewidth(1.5)
    ax_b.grid(axis="x", linestyle=":", alpha=0.5)
    ax_b.tick_params(axis="x", labelsize=12)
    ax_b.tick_params(axis="y", length=0)

    # =========================================================================
    # Panel C: Circular-shift permutation null
    # =========================================================================
    n_perm = perm_pvalues.n_perm
    n_extreme = perm_pvalues.n_extreme_median_centered_two_sided
    if n_extreme == 0:
        p_emp_str = f"P < {1.0 / n_perm:.3f}"
        n_extreme_label = f"0/{n_perm} permutations"
    else:
        p_emp_str = f"P = {perm_pvalues.empirical_p_median_centered_two_sided:.3f}"
        n_extreme_label = f"{n_extreme}/{n_perm} permutations"

    lo_x = float(min(perm.min(), obs_beta_perm)) - 0.005
    hi_x = float(max(perm.max(), obs_beta_perm)) + 0.005
    bins = np.linspace(lo_x, hi_x, 35)

    ax_c.hist(
        perm, bins=bins, color=SOFT_GRAY, edgecolor="white",
        linewidth=0.6, alpha=0.85, zorder=2,
    )

    # Shade the median-centered two-sided rejection tails.
    lower_cutoff = perm_pvalues.median_centered_lower_cutoff
    upper_cutoff = perm_pvalues.median_centered_upper_cutoff
    if lower_cutoff > lo_x:
        ax_c.axvspan(lo_x, lower_cutoff, color=HIGHLIGHT, alpha=0.10, zorder=1)
    if upper_cutoff < hi_x:
        ax_c.axvspan(upper_cutoff, hi_x, color=HIGHLIGHT, alpha=0.10, zorder=1)

    ax_c.axvline(
        obs_beta_perm, color=HIGHLIGHT, linestyle="-", linewidth=2.8, zorder=5,
        label=f"Observed β = {obs_beta_perm:.3f}",
    )
    ax_c.axvline(
        perm_pvalues.perm_beta_median,
        color=NEUTRAL,
        linestyle="--",
        linewidth=1.6,
        zorder=4,
        label=f"Shifted-null median β = {perm_pvalues.perm_beta_median:.3f}",
    )
    ax_c.set_xlabel(
        "Synchrony coefficient under within-patient circular-shift null",
        fontsize=14, fontweight="bold", labelpad=10,
    )
    ax_c.set_ylabel("Permutation count", fontsize=14, fontweight="bold", labelpad=10)
    ax_c.set_title(
        "Dependence-preserving permutation null",
        fontweight="bold", pad=10,
    )
    ax_c.text(
        0.97, 0.97,
        f"Observed β = {obs_beta_perm:.3f}\n"
        f"{p_emp_str} (median-centered)\n"
        f"{n_extreme_label}\n"
        f"at least as far from null median",
        transform=ax_c.transAxes, fontsize=11.5, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                  alpha=0.94, edgecolor="#cccccc"),
    )
    ax_c.set_xlim(lo_x, hi_x)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    ax_c.spines["bottom"].set_linewidth(1.5)
    ax_c.spines["left"].set_linewidth(1.5)
    ax_c.grid(axis="y", linestyle=":", alpha=0.4)

    # =========================================================================
    # Panel D: Likelihood-ratio tests
    # =========================================================================
    lrt_rows = [
        {
            "label": "Random intercept\n(per patient)",
            "lr_stat": lrt_ri["lr_stat"],
            "p": lrt_ri["p"],
            "llf_full": lrt_ri["llf_full"],
            "llf_red": lrt_ri["llf_red"],
        },
        {
            "label": "Random intercept +\nrandom synchrony slope",
            "lr_stat": lrt_rs["lr_stat"],
            "p": lrt_rs["p"],
            "llf_full": lrt_rs["llf_full"],
            "llf_red": lrt_rs["llf_red"],
        },
    ]
    df_d = pd.DataFrame(lrt_rows)

    bar_positions = np.arange(len(df_d))
    bar_colors = [HIGHLIGHT, HIGHLIGHT_LIGHT]
    bars = ax_d.bar(
        bar_positions, df_d["lr_stat"], color=bar_colors, edgecolor="black",
        linewidth=1.2, width=0.55, zorder=10,
    )
    ax_d.set_xticks(bar_positions)
    ax_d.set_xticklabels(df_d["label"], fontsize=12)
    ax_d.set_yscale("log")
    ax_d.set_ylim(1, max(df_d["lr_stat"]) * 8.0)

    # critical value at chi^2_1 0.05 = 3.841
    crit = 3.841
    ax_d.axhline(
        crit, color="black", linestyle=":", linewidth=1.4, zorder=5,
    )
    ax_d.text(
        -0.42, crit, f"χ²(1) critical, α = 0.05 ({crit:.2f})",
        ha="left", va="center", fontsize=10, color=NEUTRAL,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  alpha=0.92, edgecolor="none"),
        zorder=6,
    )

    # annotate each bar
    for bar, row in zip(bars, lrt_rows):
        height = bar.get_height()
        ax_d.text(
            bar.get_x() + bar.get_width() / 2.0,
            height * 1.18,
            f"LR = {row['lr_stat']:.2f}\n"
            f"{format_lr_pvalue(row['p'])}",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
        )

    ax_d.set_ylabel(
        "Likelihood-ratio statistic\n(2·Δlog-likelihood, log scale)",
        fontsize=13, fontweight="bold", labelpad=10,
    )
    ax_d.set_title(
        "Incremental fit from synchrony fixed effect",
        fontweight="bold", pad=10,
    )
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)
    ax_d.spines["bottom"].set_linewidth(1.5)
    ax_d.spines["left"].set_linewidth(1.5)
    ax_d.grid(axis="y", linestyle=":", alpha=0.4, which="both")

    # Panel labels
    for label, ax in ax_dict.items():
        ax.text(
            -0.18, 1.05, label, transform=ax.transAxes,
            fontsize=24, fontweight="bold", va="bottom", ha="right",
        )

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"\nSaved:\n  {OUT_PNG}\n  {OUT_PDF}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print(f"Tables: {TABLES_PATH}")
    pids = get_patient_hup_ids(TABLES_PATH)
    print(f"Patients found: {len(pids)}")

    df = build_all_data(pids)
    n_patients_full = int(df[GROUP].nunique())
    print(f"Rows after dropna/scale: {len(df)}  Patients: {n_patients_full}")

    # =========================================================================
    # Fit models for Panel A
    # =========================================================================
    print("\n[1/5] Fitting LMM random intercept (REML) ...")
    res_lme_ri = fit_lme_random_intercept(df, reml=True)
    b_ri, _, lo_ri, hi_ri, p_ri = synchrony_estimate(res_lme_ri)
    print(f"  β = {b_ri:.4f}, p = {p_ri:.3g}, CI = [{lo_ri:.3f}, {hi_ri:.3f}]")

    print("\n[2/5] Fitting GEE Independence (robust SE) ...")
    res_gee_ind = fit_gee_independence_robust(df)
    b_gi, _, lo_gi, hi_gi, p_gi = synchrony_estimate(res_gee_ind)
    print(f"  β = {b_gi:.4f}, p = {p_gi:.3g}, CI = [{lo_gi:.3f}, {hi_gi:.3f}]")

    print("\n[3/5] Fitting GEE Unstructured (30-min binned, T=20) ...")
    res_gee_unstruct = fit_gee_unstructured_binned(df, bin_minutes=30, T=20)
    b_gu, _, lo_gu, hi_gu, p_gu = synchrony_estimate(res_gee_unstruct)
    n_pat_unstruct = int(res_gee_unstruct._fit_n_patients)
    print(
        f"  β = {b_gu:.4f}, p = {p_gu:.3g}, CI = [{lo_gu:.3f}, {hi_gu:.3f}], "
        f"patients = {n_pat_unstruct}"
    )

    # =========================================================================
    # Likelihood-ratio tests (ML re-fits)
    # =========================================================================
    print("\n[4/5] Likelihood-ratio tests (ML re-fits) ...")
    print("  RI full vs. reduced ...")
    res_full_ri_ml = smf.mixedlm(FORMULA, df, groups=df[GROUP]).fit(reml=False, method="lbfgs")
    res_red_ri_ml = smf.mixedlm(FORMULA_NO_SYNC, df, groups=df[GROUP]).fit(reml=False, method="lbfgs")
    lrt_ri = lrt(res_full_ri_ml, res_red_ri_ml, "RI")
    print(f"    LR = {lrt_ri['lr_stat']:.3f}, df = {lrt_ri['df']}, p = {lrt_ri['p']:.3g}")

    print("  RI+RS full vs. reduced ...")
    res_full_rs_ml = smf.mixedlm(
        FORMULA, df, groups=df[GROUP], re_formula=f"~{SYNC}"
    ).fit(reml=False, method="lbfgs")
    res_red_rs_ml = smf.mixedlm(
        FORMULA_NO_SYNC, df, groups=df[GROUP], re_formula=f"~{SYNC}"
    ).fit(reml=False, method="lbfgs")
    lrt_rs = lrt(res_full_rs_ml, res_red_rs_ml, "RI+RS")
    print(f"    LR = {lrt_rs['lr_stat']:.3f}, df = {lrt_rs['df']}, p = {lrt_rs['p']:.3g}")

    # =========================================================================
    # Permutation null (loaded from CSV)
    # =========================================================================
    print(f"\n[5/5] Loading permutation betas from {PERM_CSV} ...")
    perm = pd.read_csv(PERM_CSV)["beta_perm"].values
    obs_beta_perm = float(res_gee_ind.params[SYNC])
    perm_pvalues = circular_shift_pvalues(perm, obs_beta_perm)
    print(
        f"  observed β = {obs_beta_perm:.4f}, "
        "median-centered empirical two-sided p = "
        f"{perm_pvalues.empirical_p_median_centered_two_sided:.4f} "
        f"(N_perm = {perm_pvalues.n_perm}, "
        f"n_extreme = {perm_pvalues.n_extreme_median_centered_two_sided})"
    )
    print(
        "  directional lower-tail p = "
        f"{perm_pvalues.empirical_p_lower_tail:.4f} "
        f"({perm_pvalues.n_extreme_lower_tail}/{perm_pvalues.n_perm}); "
        "old absolute-coefficient p = "
        f"{perm_pvalues.empirical_p_abs_coefficient:.4f} "
        f"({perm_pvalues.n_extreme_abs_coefficient}/{perm_pvalues.n_perm})"
    )

    # =========================================================================
    # Build Panel B rows in the same order as the methods narrative: primary
    # mixed-effects model, patient-clustered GEE, then 30-minute binned GEE.
    # =========================================================================
    rows_b = [
        {
            "label": "Primary LMM,\nrandom intercept",
            "dep_assumption": "patient-level random intercept",
            "n_clusters": n_patients_full,
            "cluster_label": "patients",
            "beta": b_ri,
            "ci_low": lo_ri,
            "ci_high": hi_ri,
            "p": p_ri,
        },
        {
            "label": "Patient-clustered GEE,\nrobust sandwich SE",
            "dep_assumption": "patient-level robust",
            "n_clusters": n_patients_full,
            "cluster_label": "patients",
            "beta": b_gi,
            "ci_low": lo_gi,
            "ci_high": hi_gi,
            "p": p_gi,
        },
        {
            "label": "30-min binned GEE,\nunstructured correlation",
            "dep_assumption": "preserves arbitrary\nwithin-patient correlation",
            "n_clusters": n_pat_unstruct,
            "cluster_label": "patients",
            "beta": b_gu,
            "ci_low": lo_gu,
            "ci_high": hi_gu,
            "p": p_gu,
        },
    ]

    plot_figure(
        res_lme_ri=res_lme_ri,
        rows_b=rows_b,
        perm=perm,
        obs_beta_perm=obs_beta_perm,
        perm_pvalues=perm_pvalues,
        lrt_ri=lrt_ri,
        lrt_rs=lrt_rs,
    )


if __name__ == "__main__":
    main()
