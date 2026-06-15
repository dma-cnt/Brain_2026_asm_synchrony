#!/usr/bin/env python3
"""Create revised Figure 3 for Brain submission.

Panel A shows the synchrony coefficient across complementary longitudinal
models. Panel B shows the dependence-preserving within-patient circular-shift
null distribution. The model order in Panel A follows the Results section:
patient-clustered GEE, 30-minute binned GEE, mixed-effects random intercept,
then mixed-effects random intercept plus random synchrony slope.
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
import statsmodels.formula.api as smf
from figure_style import (
    AXIS_LINE_WIDTH,
    BODY_FONT_SIZE,
    CARB_COLOR,
    EDGE_COLOR,
    EXPORT_PAD_INCHES,
    IQR_FILL_COLOR,
    LINE_COLOR,
    PANEL_LABEL_FONT_SIZE,
    ROBUSTNESS_LINE_WIDTH,
    ROBUSTNESS_MARKER_SIZE,
    TICK_FONT_SIZE,
    TITLE_FONT_SIZE,
    apply_brain_style,
    style_axis,
)
from PIL import Image
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler
from statsmodels.genmod.cov_struct import Independence, Unstructured
from statsmodels.genmod.families import Gaussian


warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SOURCE_DIR = REPO_ROOT / "brain_r2"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from circular_shift_pvalues import CircularShiftPValues, circular_shift_pvalues  # noqa: E402


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
    "synchrony_broadband": "Synchrony",
    "time_since_last_seizure": "Time since last seizure",
    "time_until_next_seizure": "Time until next seizure",
    "emu_minute": "Time since EMU admission",
    "ad_ratio": "Alpha-delta ratio",
    "tod_sin": "Time of day (sin)",
    "tod_cos": "Time of day (cos)",
    "tod_sin2": "Time of day (2nd harm., sin)",
    "tod_cos2": "Time of day (2nd harm., cos)",
    "spike_rate": "Spike rate",
}

HIGHLIGHT = CARB_COLOR
HIGHLIGHT_LIGHT = "#8BB9D8"
NEUTRAL = EDGE_COLOR
SOFT_GRAY = IQR_FILL_COLOR


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
        default=SOURCE_DIR / "perm_betas_circular_shift_full_5000perm_seed0.csv",
        help="CSV containing saved circular-shift beta coefficients.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "outputs",
        help="Directory for figure_3.tiff and figure_3_preview.png.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="TIFF resolution in dots per inch.",
    )
    return parser.parse_args()


def find_data_root(explicit: Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    env_path = os.environ.get("CNT_DATA_PATH")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            REPO_ROOT.parent / "cnt-data",
            REPO_ROOT.parent.parent / "cnt-data",
            Path.cwd().parent / "cnt-data",
            Path("/Users/devinma/Documents/cnt-data"),
        ]
    )

    for candidate in candidates:
        table_dir = candidate / "revised_tables_step_4"
        if (table_dir / "HUP_100.csv").exists() or any(table_dir.glob("HUP_*.csv")):
            return candidate

    checked = "\n".join(str(c) for c in candidates)
    raise FileNotFoundError("Could not locate cnt-data/revised_tables_step_4. Checked:\n" + checked)


def get_patient_hup_ids(table_dir: Path) -> list[int]:
    return sorted(
        int(path.stem.split("_")[1])
        for path in table_dir.glob("HUP_*.csv")
        if path.stem.split("_")[1].isdigit()
    )


def build_all_data(table_dir: Path, pids: list[int]) -> pd.DataFrame:
    dfs = []
    for pid in pids:
        fp = table_dir / f"HUP_{pid}.csv"
        if not fp.exists():
            continue
        d = pd.read_csv(fp, low_memory=False)
        d[GROUP] = pid
        keep = [c for c in ([GROUP, "emu_minute"] + PREDICTORS + [Y]) if c in d.columns]
        dfs.append(d[keep])

    if not dfs:
        raise RuntimeError(f"No patient tables found in {table_dir}")

    all_data = pd.concat(dfs, axis=0, ignore_index=True)
    all_data = all_data.loc[:, ~all_data.columns.duplicated()].copy()
    all_data["emu_minute"] = pd.to_numeric(all_data["emu_minute"], errors="coerce")
    all_data[TIME_RAW] = all_data["emu_minute"].copy()
    needed = [Y, GROUP, "emu_minute", TIME_RAW] + [c for c in PREDICTORS if c in all_data.columns]
    all_data = all_data.dropna(subset=needed).reset_index(drop=True)
    all_data = all_data.sort_values([GROUP, "emu_minute"]).reset_index(drop=True)
    all_data["time_index"] = all_data.groupby(GROUP).cumcount().astype(int)
    cols_to_scale = [c for c in COLS_TO_SCALE if c in all_data.columns]
    all_data[cols_to_scale] = StandardScaler().fit_transform(all_data[cols_to_scale])
    return all_data


def bin_within_patient(df: pd.DataFrame, bin_minutes: int) -> pd.DataFrame:
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


def fit_lme_random_intercept(df: pd.DataFrame, reml: bool = True):
    return smf.mixedlm(FORMULA, df, groups=df[GROUP]).fit(reml=reml)


def fit_lme_random_slope(df: pd.DataFrame, reml: bool = True):
    return smf.mixedlm(FORMULA, df, groups=df[GROUP], re_formula=f"~{SYNC}").fit(
        reml=reml,
        method="lbfgs",
    )


def fit_gee_independence_robust(df: pd.DataFrame):
    model = smf.gee(
        formula=FORMULA,
        groups=df[GROUP],
        data=df,
        family=Gaussian(),
        cov_struct=Independence(),
    )
    return model.fit(cov_type="robust")


def fit_gee_unstructured_binned(
    df: pd.DataFrame,
    bin_minutes: int = 30,
    time_points: int = 20,
    min_patients: int = 5,
):
    d0 = bin_within_patient(df, bin_minutes)
    sizes = d0.groupby(GROUP).size()
    eligible = sizes[sizes >= time_points].index.values
    d1 = d0[d0[GROUP].isin(eligible)].copy()
    if d1[GROUP].nunique() < min_patients:
        raise RuntimeError("Too few patients with required time points")

    d1 = (
        d1.sort_values([GROUP, TIME_RAW])
        .groupby(GROUP, as_index=False, group_keys=False)
        .head(time_points)
        .reset_index(drop=True)
    )
    d1["time_index"] = d1.groupby(GROUP).cumcount().astype(int)
    model = smf.gee(
        formula=FORMULA,
        groups=d1[GROUP],
        data=d1,
        family=Gaussian(),
        cov_struct=Unstructured(),
        time=d1["time_index"].values,
    )
    res = model.fit(cov_type="robust")
    res._fit_n_patients = int(d1[GROUP].nunique())
    res._fit_n_obs = int(len(d1))
    return res


def lrt(full_res, red_res, label: str) -> dict[str, float | int | str]:
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


def synchrony_estimate(res) -> tuple[float, float, float, float, float]:
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


def format_pvalue_math(p: float) -> str:
    if not np.isfinite(p):
        return r"$\it{P}$ = NA"
    if p < 1e-3:
        return r"$\it{P}$ < 0.001"
    return rf"$\it{{P}}$ = {p:.3f}"


def format_lr_pvalue_math(p: float) -> str:
    if not np.isfinite(p):
        return r"$\it{P}$ = NA"
    if p < 1e-3:
        return r"$\it{P}$ < 0.001"
    return rf"$\it{{P}}$ = {p:.3f}"


def fixed_effect_coefficients(res_lme_ri) -> pd.DataFrame:
    conf_int = res_lme_ri.conf_int(alpha=0.05)
    conf_int.columns = ["ci_lower", "ci_upper"]
    coefs = (
        pd.concat([res_lme_ri.fe_params, conf_int], axis=1)
        .reset_index()
        .rename(columns={"index": "term", 0: "estimate"})
    )
    coefs = coefs[~coefs["term"].astype(str).str.startswith("Group")].copy()
    coefs = coefs[coefs["term"] != "Intercept"].copy()
    coefs["label"] = coefs["term"].map(LABEL_MAP).fillna(coefs["term"])
    coefs["is_synchrony"] = coefs["term"] == SYNC
    return coefs.sort_values(["is_synchrony", "estimate"]).reset_index(drop=True)


def fit_continuous_results(
    data_root: Path | None = None,
    perm_csv: Path | None = None,
) -> dict[str, object]:
    data_root = find_data_root(data_root)
    table_dir = data_root / "revised_tables_step_4"
    perm_csv = perm_csv or SOURCE_DIR / "perm_betas_circular_shift_full_5000perm_seed0.csv"

    print(f"Tables: {table_dir}")
    pids = get_patient_hup_ids(table_dir)
    print(f"Patients found: {len(pids)}")

    df = build_all_data(table_dir, pids)
    n_patients_full = int(df[GROUP].nunique())
    print(f"Rows after dropna/scale: {len(df)}  Patients: {n_patients_full}")

    print("\n[1/6] Fitting LMM random intercept (REML) ...")
    res_lme_ri = fit_lme_random_intercept(df, reml=True)
    b_ri, _, lo_ri, hi_ri, p_ri = synchrony_estimate(res_lme_ri)
    print(f"  beta = {b_ri:.4f}, p = {p_ri:.3g}, CI = [{lo_ri:.3f}, {hi_ri:.3f}]")

    print("\n[2/6] Fitting GEE Independence (robust SE) ...")
    res_gee_ind = fit_gee_independence_robust(df)
    b_gi, _, lo_gi, hi_gi, p_gi = synchrony_estimate(res_gee_ind)
    print(f"  beta = {b_gi:.4f}, p = {p_gi:.3g}, CI = [{lo_gi:.3f}, {hi_gi:.3f}]")

    print("\n[3/6] Fitting LMM random intercept + random synchrony slope (REML) ...")
    res_lme_rs = fit_lme_random_slope(df, reml=True)
    b_rs, _, lo_rs, hi_rs, p_rs = synchrony_estimate(res_lme_rs)
    print(f"  beta = {b_rs:.4f}, p = {p_rs:.3g}, CI = [{lo_rs:.3f}, {hi_rs:.3f}]")

    print("\n[4/6] Fitting GEE Unstructured (30-min binned, T=20) ...")
    res_gee_unstruct = fit_gee_unstructured_binned(df, bin_minutes=30, time_points=20)
    b_gu, _, lo_gu, hi_gu, p_gu = synchrony_estimate(res_gee_unstruct)
    n_pat_unstruct = int(res_gee_unstruct._fit_n_patients)
    print(
        f"  beta = {b_gu:.4f}, p = {p_gu:.3g}, CI = [{lo_gu:.3f}, {hi_gu:.3f}], "
        f"patients = {n_pat_unstruct}"
    )

    print("\n[5/6] Likelihood-ratio tests (ML re-fits) ...")
    print("  RI full vs. reduced ...")
    res_full_ri_ml = smf.mixedlm(FORMULA, df, groups=df[GROUP]).fit(
        reml=False,
        method="lbfgs",
    )
    res_red_ri_ml = smf.mixedlm(FORMULA_NO_SYNC, df, groups=df[GROUP]).fit(
        reml=False,
        method="lbfgs",
    )
    lrt_ri = lrt(res_full_ri_ml, res_red_ri_ml, "RI")
    print(f"    LR = {lrt_ri['lr_stat']:.3f}, df = {lrt_ri['df']}, p = {lrt_ri['p']:.3g}")

    print("  RI+RS full vs. reduced ...")
    res_full_rs_ml = smf.mixedlm(
        FORMULA,
        df,
        groups=df[GROUP],
        re_formula=f"~{SYNC}",
    ).fit(reml=False, method="lbfgs")
    res_red_rs_ml = smf.mixedlm(
        FORMULA_NO_SYNC,
        df,
        groups=df[GROUP],
        re_formula=f"~{SYNC}",
    ).fit(reml=False, method="lbfgs")
    lrt_rs = lrt(res_full_rs_ml, res_red_rs_ml, "RI+RS")
    print(f"    LR = {lrt_rs['lr_stat']:.3f}, df = {lrt_rs['df']}, p = {lrt_rs['p']:.3g}")

    print(f"\n[6/6] Loading permutation betas from {perm_csv} ...")
    perm = pd.read_csv(perm_csv)["beta_perm"].values
    obs_beta_perm = float(res_gee_ind.params[SYNC])
    perm_pvalues = circular_shift_pvalues(perm, obs_beta_perm)
    print(
        f"  observed beta = {obs_beta_perm:.4f}, "
        "median-centered empirical two-sided p = "
        f"{perm_pvalues.empirical_p_median_centered_two_sided:.4f} "
        f"(N_perm = {perm_pvalues.n_perm}, "
        f"n_extreme = {perm_pvalues.n_extreme_median_centered_two_sided})"
    )

    rows_a = [
        {
            "label": "Patient-clustered GEE,\nrobust sandwich SE",
            "n_patients": n_patients_full,
            "beta": b_gi,
            "ci_low": lo_gi,
            "ci_high": hi_gi,
            "p": p_gi,
        },
        {
            "label": "30-min binned GEE,\nunstructured correlation",
            "n_patients": n_pat_unstruct,
            "beta": b_gu,
            "ci_low": lo_gu,
            "ci_high": hi_gu,
            "p": p_gu,
        },
        {
            "label": "Mixed-effects model,\nrandom intercept",
            "n_patients": n_patients_full,
            "beta": b_ri,
            "ci_low": lo_ri,
            "ci_high": hi_ri,
            "p": p_ri,
        },
        {
            "label": "Mixed-effects model,\nrandom intercept + synchrony slope",
            "n_patients": n_patients_full,
            "beta": b_rs,
            "ci_low": lo_rs,
            "ci_high": hi_rs,
            "p": p_rs,
        },
    ]

    return {
        "df": df,
        "res_lme_ri": res_lme_ri,
        "res_lme_rs": res_lme_rs,
        "res_gee_ind": res_gee_ind,
        "res_gee_unstruct": res_gee_unstruct,
        "rows_a": rows_a,
        "perm": perm,
        "obs_beta_perm": obs_beta_perm,
        "perm_pvalues": perm_pvalues,
        "lrt_ri": lrt_ri,
        "lrt_rs": lrt_rs,
    }


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.16) -> None:
    ax.text(
        x,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_FONT_SIZE,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def draw_coefficient_panel(ax: plt.Axes, rows_a: list[dict[str, object]]) -> None:
    df_plot = pd.DataFrame(rows_a).iloc[::-1].reset_index(drop=True)
    y_positions = np.arange(len(df_plot))

    for i in y_positions:
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#F2F2F2", alpha=0.7, zorder=0)

    err_low = df_plot["beta"] - df_plot["ci_low"]
    err_high = df_plot["ci_high"] - df_plot["beta"]
    ax.errorbar(
        df_plot["beta"],
        y_positions,
        xerr=[err_low, err_high],
        fmt="o",
        color=HIGHLIGHT,
        ecolor=HIGHLIGHT,
        capsize=4,
        markersize=ROBUSTNESS_MARKER_SIZE,
        linewidth=ROBUSTNESS_LINE_WIDTH,
        zorder=10,
    )
    ax.axvline(0, color=LINE_COLOR, linestyle="--", linewidth=AXIS_LINE_WIDTH, zorder=5)

    yticklabels = [
        f"{row['label']}\n(n = {int(row['n_patients'])} patients)"
        for _, row in df_plot.iterrows()
    ]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(yticklabels, fontsize=TICK_FONT_SIZE)
    ax.set_ylim(-0.6, len(df_plot) - 0.4)

    left_limit = float(np.floor((df_plot["ci_low"].min() - 0.04) * 10.0) / 10.0)
    right_limit = 0.2
    ax.set_xlim(left_limit, right_limit)
    ax.set_xlabel("Synchrony coefficient (95% CI)", fontsize=BODY_FONT_SIZE)
    ax.set_title(
        "Synchrony coefficient across longitudinal models",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
        pad=8,
    )

    annot_x = 0.035
    for i, row in df_plot.iterrows():
        ax.text(
            annot_x,
            i,
            rf"$\it{{β}}$ = {row['beta']:.3f}" + "\n" + format_pvalue_math(float(row["p"])),
            va="center",
            ha="left",
            fontsize=TICK_FONT_SIZE,
            linespacing=1.25,
            clip_on=False,
        )

    ax.grid(axis="x", linestyle=":", alpha=0.45)
    ax.tick_params(axis="y", length=0)
    style_axis(ax)
    ax.spines["left"].set_visible(False)
    add_panel_label(ax, "A", x=-0.18)


def draw_permutation_panel(
    ax: plt.Axes,
    perm: np.ndarray,
    obs_beta_perm: float,
    perm_pvalues: CircularShiftPValues,
) -> None:
    n_perm = perm_pvalues.n_perm
    n_extreme = perm_pvalues.n_extreme_median_centered_two_sided
    if n_extreme == 0:
        p_emp_str = rf"$\it{{P}}$ < {1.0 / n_perm:.3f}"
        n_extreme_label = f"0/{n_perm:,} permutations"
    else:
        p_emp_str = format_pvalue_math(perm_pvalues.empirical_p_median_centered_two_sided)
        n_extreme_label = f"{n_extreme:,}/{n_perm:,} permutations"

    lo_x = float(min(perm.min(), obs_beta_perm)) - 0.005
    hi_x = float(max(perm.max(), obs_beta_perm)) + 0.005
    bins = np.linspace(lo_x, hi_x, 35)

    ax.hist(
        perm,
        bins=bins,
        color=SOFT_GRAY,
        edgecolor="white",
        linewidth=0.6,
        alpha=0.9,
        zorder=2,
    )

    lower_cutoff = perm_pvalues.median_centered_lower_cutoff
    upper_cutoff = perm_pvalues.median_centered_upper_cutoff
    if lower_cutoff > lo_x:
        ax.axvspan(lo_x, lower_cutoff, color=HIGHLIGHT, alpha=0.12, zorder=1)
    if upper_cutoff < hi_x:
        ax.axvspan(upper_cutoff, hi_x, color=HIGHLIGHT, alpha=0.12, zorder=1)

    ax.axvline(
        obs_beta_perm,
        color=HIGHLIGHT,
        linestyle="-",
        linewidth=2.2,
        zorder=5,
        label=rf"Observed $\it{{β}}$ = {obs_beta_perm:.3f}",
    )
    ax.axvline(
        perm_pvalues.perm_beta_median,
        color=NEUTRAL,
        linestyle="--",
        linewidth=1.4,
        zorder=4,
        label=rf"Shifted-null median $\it{{β}}$ = {perm_pvalues.perm_beta_median:.3f}",
    )
    ax.annotate(
        rf"Shifted-null median $\it{{β}}$ = {perm_pvalues.perm_beta_median:.3f}",
        xy=(perm_pvalues.perm_beta_median, 0.54),
        xycoords=("data", "axes fraction"),
        xytext=(9, 0),
        textcoords="offset points",
        rotation=90,
        va="center",
        ha="left",
        fontsize=TICK_FONT_SIZE - 1,
        color=NEUTRAL,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.88, edgecolor="none"),
    )
    ax.set_xlabel(
        "Synchrony coefficient under within-patient circular-shift null",
        fontsize=BODY_FONT_SIZE,
    )
    ax.set_ylabel("Permutation count", fontsize=BODY_FONT_SIZE)
    ax.set_title(
        "Dependence-preserving permutation null",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
        pad=8,
    )
    ax.text(
        0.97,
        0.97,
        rf"Observed $\it{{β}}$ = {obs_beta_perm:.3f}" + "\n"
        f"{p_emp_str} (median-centered)\n"
        f"{n_extreme_label}\n"
        "at least as far from null median",
        transform=ax.transAxes,
        fontsize=TICK_FONT_SIZE,
        va="top",
        ha="right",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.94, edgecolor="#CCCCCC"),
    )
    ax.set_xlim(lo_x, hi_x)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    style_axis(ax)
    add_panel_label(ax, "B")


def create_figure(results: dict[str, object]) -> plt.Figure:
    apply_brain_style(mpl)
    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(14.5, 6.2),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.34, 1.0], "wspace": 0.08},
        facecolor="white",
    )
    draw_coefficient_panel(ax_a, results["rows_a"])
    draw_permutation_panel(
        ax_b,
        np.asarray(results["perm"], dtype=float),
        float(results["obs_beta_perm"]),
        results["perm_pvalues"],
    )
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


def save_figure(fig: plt.Figure, output_dir: Path, dpi: int) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tiff_path = output_dir / "figure_3.tiff"
    preview_path = output_dir / "figure_3_preview.png"
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
    results = fit_continuous_results(args.data_root, args.perm_csv)
    fig = create_figure(results)
    tiff_path, preview_path = save_figure(fig, args.output_dir.resolve(), args.dpi)
    plt.close(fig)
    print(f"Wrote {tiff_path}")
    print(f"Wrote {preview_path}")


if __name__ == "__main__":
    main()
