"""Patient-specific synchrony coefficient check for the Brain R2 appeal.

This implements Quy's proposed supplementary analysis:
  1. fit one within-patient model per patient,
  2. extract one synchrony coefficient per patient,
  3. test whether the patient-level coefficients differ from zero.

Run:
    conda run -n cnt python brain_r2/patient_specific_synchrony_check.py
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "cnt-data"))
TABLES_PATH = os.path.join(DATA_PATH, "revised_tables_step_4")
SOZ_PATH = os.path.join(DATA_PATH, "HUP_SOZ.csv")

Y = "med_sum_no_lorazepam_raw"
GROUP = "patient_hup_id"
TIME_RAW = "emu_minute_raw"
SYNC = "synchrony_broadband"

FULL_COVARIATES = [
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

# Reduced model used only when the full within-patient design is unstable.
# It preserves the main clinical/time controls while avoiding the higher-order
# time-of-day terms and spike/ad covariates that can be weakly identified within
# individual patients.
REDUCED_COVARIATES = [
    "time_since_last_seizure",
    "time_until_next_seizure",
    "emu_minute",
    "tod_sin",
    "tod_cos",
]

MINIMAL_COVARIATES = ["emu_minute"]

PREDICTORS = [SYNC] + FULL_COVARIATES
COLS_TO_SCALE = [
    SYNC,
    "time_since_last_seizure",
    "time_until_next_seizure",
    "emu_minute",
    "ad_ratio",
    "spike_rate",
]


def get_patient_hup_ids(directory: str) -> list[int]:
    return sorted(
        int(f.split("_")[1].split(".")[0])
        for f in os.listdir(directory)
        if f.startswith("HUP_") and f.endswith(".csv")
    )


def cohort_patient_ids() -> tuple[list[int], list[int], list[int]]:
    completed = get_patient_hup_ids(TABLES_PATH)

    soz = pd.read_csv(SOZ_PATH)
    soz.columns = [c.replace("\ufeff", "").strip() for c in soz.columns]
    mtle = soz[soz["SOZ"].astype(str).str.lower().eq("mesial temporal")].copy()
    mtle_ids = sorted(
        mtle["pt_id"]
        .astype(str)
        .str.replace("HUP", "", regex=False)
        .astype(int)
        .unique()
        .tolist()
    )

    mtle_completed = sorted([p for p in completed if p in mtle_ids])
    non_mtle_completed = sorted([p for p in completed if p not in mtle_ids])
    return completed, mtle_completed, non_mtle_completed


def build_all_data(pids: list[int]) -> pd.DataFrame:
    dfs = []
    for pid in pids:
        fp = os.path.join(TABLES_PATH, f"HUP_{pid}.csv")
        if not os.path.exists(fp):
            continue
        d = pd.read_csv(fp, low_memory=False)
        d = d.loc[:, ~d.columns.duplicated()].copy()
        d[GROUP] = pid
        keep = list(dict.fromkeys([c for c in ([GROUP, "emu_minute"] + PREDICTORS + [Y]) if c in d.columns]))
        dfs.append(d[keep])

    if not dfs:
        raise RuntimeError("No tables loaded for requested patient IDs.")

    df = pd.concat(dfs, axis=0, ignore_index=True)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    if "emu_minute" not in df.columns:
        raise RuntimeError("Expected 'emu_minute' in tables but it is missing.")

    df["emu_minute"] = pd.to_numeric(df["emu_minute"], errors="coerce")
    df[TIME_RAW] = df["emu_minute"].copy()

    needed = [Y, GROUP, "emu_minute", TIME_RAW] + [c for c in PREDICTORS if c in df.columns]
    df = df.dropna(subset=needed).reset_index(drop=True)
    df = df.sort_values([GROUP, TIME_RAW]).reset_index(drop=True)
    df["time_index"] = df.groupby(GROUP).cumcount().astype(int)

    scaler = StandardScaler()
    cols_to_scale = [c for c in COLS_TO_SCALE if c in df.columns]
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    return df


def bin_within_patient(df: pd.DataFrame, bin_minutes: int | None) -> pd.DataFrame:
    if bin_minutes is None:
        return df.copy()

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


def make_formula(covariates: list[str]) -> str:
    return f"{Y} ~ " + " + ".join([SYNC] + covariates)


def stable_design(model, max_condition: float) -> tuple[bool, str, float, int, int]:
    exog = np.asarray(model.exog, dtype=float)
    rank = int(np.linalg.matrix_rank(exog))
    n_cols = int(exog.shape[1])
    if rank < n_cols:
        return False, f"rank_deficient_{rank}_of_{n_cols}", np.inf, rank, n_cols

    condition = float(np.linalg.cond(exog))
    if not np.isfinite(condition) or condition > max_condition:
        return False, f"ill_conditioned_{condition:.3g}", condition, rank, n_cols

    return True, "ok", condition, rank, n_cols


def fit_ols_with_covariance(model, n: int, args: argparse.Namespace):
    cov_type = args.cov_type.upper()
    if cov_type == "HAC":
        maxlags = min(args.hac_maxlags, max(1, n // 4))
        return model.fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    if cov_type == "NONROBUST":
        return model.fit()
    return model.fit(cov_type=cov_type)


def fit_patient_model(
    g: pd.DataFrame,
    covariates: list[str],
    model_label: str,
    args: argparse.Namespace,
) -> dict[str, object]:
    pid = int(g[GROUP].iloc[0])
    terms = [SYNC] + covariates
    needed = [Y] + terms
    d = g.dropna(subset=needed).copy()

    if len(d) < args.min_n:
        return {"patient_hup_id": pid, "status": "failed", "reason": f"n_lt_{args.min_n}"}
    if len(d) <= len(terms) + 1:
        return {"patient_hup_id": pid, "status": "failed", "reason": "too_few_df"}
    if float(d[Y].std(ddof=0)) <= args.min_sd:
        return {"patient_hup_id": pid, "status": "failed", "reason": "no_outcome_variation"}
    if float(d[SYNC].std(ddof=0)) <= args.min_sd:
        return {"patient_hup_id": pid, "status": "failed", "reason": "no_synchrony_variation"}

    zero_var = [
        term
        for term in terms
        if float(d[term].std(ddof=0)) <= args.min_sd or d[term].nunique(dropna=True) <= 1
    ]
    if zero_var:
        return {
            "patient_hup_id": pid,
            "status": "failed",
            "reason": "zero_variance_terms:" + ",".join(zero_var),
        }

    formula = make_formula(covariates)
    try:
        model = smf.ols(formula, data=d)
        stable, reason, condition, rank, n_cols = stable_design(model, args.max_condition)
        if not stable:
            return {
                "patient_hup_id": pid,
                "status": "failed",
                "reason": reason,
                "condition_number": condition,
                "rank": rank,
                "n_design_cols": n_cols,
            }

        result = fit_ols_with_covariance(model, len(d), args)
        beta = float(result.params[SYNC])
        se = float(result.bse[SYNC])
        p = float(result.pvalues[SYNC])
        if not (np.isfinite(beta) and np.isfinite(se)):
            return {"patient_hup_id": pid, "status": "failed", "reason": "nonfinite_beta_or_se"}

        return {
            "patient_hup_id": pid,
            "status": "success",
            "model": model_label,
            "formula": formula,
            "n": int(len(d)),
            "beta": beta,
            "se": se,
            "p": p,
            "condition_number": condition,
            "rank": rank,
            "n_design_cols": n_cols,
            "r_squared": float(result.rsquared),
        }
    except Exception as exc:
        return {"patient_hup_id": pid, "status": "failed", "reason": repr(exc)}


def fit_patient_with_fallback(g: pd.DataFrame, args: argparse.Namespace) -> dict[str, object]:
    attempts = [
        ("full", FULL_COVARIATES),
        ("reduced", REDUCED_COVARIATES),
        ("minimal", MINIMAL_COVARIATES),
    ]
    failures = []
    for label, covariates in attempts:
        row = fit_patient_model(g, covariates, label, args)
        if row["status"] == "success":
            row["prior_failures"] = "; ".join(failures)
            return row
        failures.append(f"{label}:{row.get('reason', 'failed')}")

    pid = int(g[GROUP].iloc[0])
    return {
        "patient_hup_id": pid,
        "status": "failed",
        "reason": "; ".join(failures),
        "n": int(len(g)),
    }


def signed_rank_p(betas: np.ndarray) -> float:
    if len(betas) == 0 or np.allclose(betas, 0.0):
        return np.nan
    try:
        return float(stats.wilcoxon(betas, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        return np.nan


def summarize_patient_betas(fits: pd.DataFrame, args: argparse.Namespace) -> dict[str, object]:
    d = fits[fits["status"].eq("success")].dropna(subset=["beta"]).copy()
    betas = d["beta"].astype(float).to_numpy()

    n = int(len(betas))
    if n == 0:
        return {"n_patients_modeled": 0}

    nonzero = betas[betas != 0]
    n_negative = int(np.sum(betas < 0))
    n_positive = int(np.sum(betas > 0))
    n_zero = int(np.sum(betas == 0))
    sign_p = np.nan
    if len(nonzero) > 0:
        sign_p = float(
            stats.binomtest(
                int(np.sum(nonzero < 0)),
                int(len(nonzero)),
                0.5,
                alternative="two-sided",
            ).pvalue
        )

    t_p = float(stats.ttest_1samp(betas, 0.0, nan_policy="omit").pvalue) if n > 1 else np.nan

    ivw = d.dropna(subset=["se"]).copy()
    ivw = ivw[np.isfinite(ivw["se"].astype(float)) & (ivw["se"].astype(float) > 0)]
    if len(ivw) >= 2:
        weights = 1.0 / (ivw["se"].astype(float).to_numpy() ** 2)
        ivw_betas = ivw["beta"].astype(float).to_numpy()
        ivw_beta = float(np.sum(weights * ivw_betas) / np.sum(weights))
        ivw_se = float(np.sqrt(1.0 / np.sum(weights)))
        ivw_z = ivw_beta / ivw_se
        ivw_p = float(2.0 * stats.norm.sf(abs(ivw_z)))
        ivw_ci_low = ivw_beta - 1.96 * ivw_se
        ivw_ci_high = ivw_beta + 1.96 * ivw_se
        ivw_top_weight_fraction = float(np.max(weights) / np.sum(weights))
    else:
        ivw_beta = ivw_se = ivw_p = ivw_ci_low = ivw_ci_high = ivw_top_weight_fraction = np.nan

    ivw_reliable = bool(
        np.isfinite(ivw_top_weight_fraction)
        and ivw_top_weight_fraction <= args.ivw_max_top_weight_fraction
    )

    return {
        "cohort": args.cohort,
        "bin_minutes": "none" if args.bin_minutes is None else int(args.bin_minutes),
        "cov_type": args.cov_type,
        "n_patients_total": int(fits[GROUP].nunique() if GROUP in fits.columns else fits["patient_hup_id"].nunique()),
        "n_patients_modeled": n,
        "n_negative": n_negative,
        "n_positive": n_positive,
        "n_zero": n_zero,
        "fraction_negative": float(n_negative / n),
        "median_beta": float(np.median(betas)),
        "q1_beta": float(np.quantile(betas, 0.25)),
        "q3_beta": float(np.quantile(betas, 0.75)),
        "mean_beta": float(np.mean(betas)),
        "sd_beta": float(np.std(betas, ddof=1)) if n > 1 else np.nan,
        "t_test_p": t_p,
        "wilcoxon_signed_rank_p": signed_rank_p(betas),
        "sign_test_p": sign_p,
        "ivw_n": int(len(ivw)),
        "ivw_beta": ivw_beta,
        "ivw_se": ivw_se,
        "ivw_ci_low": ivw_ci_low,
        "ivw_ci_high": ivw_ci_high,
        "ivw_p": ivw_p,
        "ivw_top_weight_fraction": ivw_top_weight_fraction,
        "ivw_reliable_by_top_weight": ivw_reliable,
    }


def fmt_float(value: object) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(x):
        return "nan"
    if x != 0 and abs(x) < 1e-4:
        return f"{x:.3e}"
    return f"{x:.6f}"


def print_summary(summary: dict[str, object], fits: pd.DataFrame) -> None:
    successes = fits[fits["status"].eq("success")].copy()
    failures = fits[~fits["status"].eq("success")].copy()

    print("\nModeling yield")
    print(f"  patients available after dropna: {fits['patient_hup_id'].nunique()}")
    print(f"  patients successfully modeled: {summary['n_patients_modeled']}")
    if len(successes) > 0:
        model_counts = successes["model"].value_counts().to_dict()
        print(f"  model used among successes: {model_counts}")
    print(f"  patients not modeled: {len(failures)}")
    if len(failures) > 0:
        print("  failure reasons:")
        for reason, count in failures["reason"].value_counts().items():
            print(f"    {count}: {reason}")

    print("\nPatient-level synchrony coefficient summary")
    print(
        f"  negative coefficients: {summary['n_negative']}/{summary['n_patients_modeled']} "
        f"({summary['fraction_negative']:.3f})"
    )
    print(
        f"  median beta: {summary['median_beta']:.6f} "
        f"[IQR {summary['q1_beta']:.6f}, {summary['q3_beta']:.6f}]"
    )
    print(f"  mean beta: {summary['mean_beta']:.6f} (SD {summary['sd_beta']:.6f})")
    print(f"  Wilcoxon signed-rank p: {summary['wilcoxon_signed_rank_p']:.6g}")
    print(f"  sign test p: {summary['sign_test_p']:.6g}")
    print(f"  one-sample t-test p: {summary['t_test_p']:.6g}")

    print("\nInverse-variance weighted fixed-effect estimate")
    print(
        f"  beta: {fmt_float(summary['ivw_beta'])} "
        f"SE {fmt_float(summary['ivw_se'])} "
        f"95% CI [{fmt_float(summary['ivw_ci_low'])}, {fmt_float(summary['ivw_ci_high'])}] "
        f"p={summary['ivw_p']:.6g}"
    )
    print(f"  patients with usable SEs: {summary['ivw_n']}")
    print(f"  largest single-patient weight fraction: {summary['ivw_top_weight_fraction']:.3f}")
    if not summary["ivw_reliable_by_top_weight"]:
        print("  reliability flag: do not interpret IVW; weight is too concentrated")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=["full", "mtle", "non_mtle"], default="full")
    parser.add_argument("--bin-minutes", type=int, default=None)
    parser.add_argument("--min-n", type=int, default=20)
    parser.add_argument("--min-sd", type=float, default=1e-12)
    parser.add_argument("--max-condition", type=float, default=1e12)
    parser.add_argument(
        "--cov-type",
        choices=["HC0", "HC1", "HC2", "HC3", "HAC", "nonrobust"],
        default="HC3",
        help="Per-patient covariance estimator used for SEs and IVW weights.",
    )
    parser.add_argument("--hac-maxlags", type=int, default=10)
    parser.add_argument("--ivw-max-top-weight-fraction", type=float, default=0.50)
    parser.add_argument("--out-dir", default=SCRIPT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    completed, mtle_completed, non_mtle_completed = cohort_patient_ids()
    if args.cohort == "full":
        pids = completed
    elif args.cohort == "mtle":
        pids = mtle_completed
    else:
        pids = non_mtle_completed

    df = build_all_data(pids)
    df = bin_within_patient(df, args.bin_minutes)
    os.makedirs(args.out_dir, exist_ok=True)

    print("Patient-specific synchrony coefficient check")
    print(f"  tables: {TABLES_PATH}")
    print(f"  cohort: {args.cohort}")
    print(f"  rows after dropna/binning: {len(df)}")
    print(f"  patients after dropna/binning: {df[GROUP].nunique()}")
    print(f"  bin_minutes: {'none' if args.bin_minutes is None else args.bin_minutes}")
    print(f"  per-patient SE covariance: {args.cov_type}")

    fits = pd.DataFrame([fit_patient_with_fallback(g, args) for _, g in df.groupby(GROUP)])
    summary = summarize_patient_betas(fits, args)
    print_summary(summary, fits)

    suffix = args.cohort
    if args.bin_minutes is not None:
        suffix += f"_binned{args.bin_minutes}min"
    suffix += f"_{args.cov_type.lower()}"

    coeff_path = os.path.join(args.out_dir, f"patient_specific_synchrony_coefficients_{suffix}.csv")
    summary_path = os.path.join(args.out_dir, f"patient_specific_synchrony_summary_{suffix}.csv")
    fits.to_csv(coeff_path, index=False)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print("\nSaved")
    print(f"  coefficients: {coeff_path}")
    print(f"  summary: {summary_path}")


if __name__ == "__main__":
    main()
