from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT.parent / "cnt-data" / "revised_tables_step_4"
OUT_DIR = ROOT / "brain_r2" / "within_patient_slope_audit"

PATIENT = "patient_hup_id"
TIME = "emu_minute"
Y = "med_sum_no_lorazepam_raw"
Y_NORM = "med_sum_no_lorazepam_norm"
X = "synchrony_broadband"

MODEL_SPECS = {
    "unadjusted": [],
    "time_adjusted": [TIME],
    "seizure_time_adjusted": [TIME, "time_since_last_seizure", "time_until_next_seizure"],
    "full_covariate_adjusted": [
        TIME,
        "time_since_last_seizure",
        "time_until_next_seizure",
        "ad_ratio",
        "tod_sin",
        "tod_cos",
        "tod_sin2",
        "tod_cos2",
        "spike_rate",
    ],
}


def load_tables() -> pd.DataFrame:
    frames = []
    for path in sorted(DATA_DIR.glob("HUP_*.csv")):
        pid = int(path.stem.split("_")[1])
        d = pd.read_csv(path, low_memory=False)
        d = d.loc[:, ~d.columns.duplicated()].copy()
        d[PATIENT] = pid
        keep = [PATIENT, TIME, X, Y, Y_NORM] + sorted(
            set(sum(MODEL_SPECS.values(), [])) - {TIME}
        )
        keep = [column for column in keep if column in d.columns]
        frames.append(d[keep])

    if not frames:
        raise RuntimeError(f"No HUP_*.csv files found in {DATA_DIR}")
    return pd.concat(frames, ignore_index=True)


def zscore(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    sd = float(values.std(ddof=0))
    if not math.isfinite(sd) or sd <= 1e-12:
        return pd.Series(np.nan, index=values.index)
    return (values - float(values.mean())) / sd


def residualize(y: np.ndarray, covars: np.ndarray | None) -> tuple[np.ndarray, int]:
    if covars is None or covars.shape[1] == 0:
        return y - np.mean(y), 1

    design = np.column_stack([np.ones(len(y)), covars])
    rank = int(np.linalg.matrix_rank(design))
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    return y - design @ beta, rank


def slope_for_patient(g: pd.DataFrame, model: str, covariates: list[str]) -> dict[str, object]:
    needed = [Y, X] + covariates
    d = g.dropna(subset=needed).copy()
    pid = int(g[PATIENT].iloc[0])

    if len(d) < 20:
        return {PATIENT: pid, "model": model, "status": "failed", "reason": "n_lt_20"}

    z = pd.DataFrame({Y: zscore(d[Y]), X: zscore(d[X])}, index=d.index)
    for covariate in covariates:
        z[covariate] = zscore(d[covariate])

    z = z.dropna()
    if len(z) < 20:
        return {PATIENT: pid, "model": model, "status": "failed", "reason": "low_variation"}

    y = z[Y].to_numpy(dtype=float)
    x = z[X].to_numpy(dtype=float)
    covar_matrix = (
        z[covariates].to_numpy(dtype=float) if covariates else np.empty((len(z), 0))
    )
    y_resid, y_rank = residualize(y, covar_matrix)
    x_resid, x_rank = residualize(x, covar_matrix)

    x_ss = float(np.sum(x_resid**2))
    if x_ss <= 1e-12:
        return {PATIENT: pid, "model": model, "status": "failed", "reason": "no_residual_x"}

    beta = float(np.sum(x_resid * y_resid) / x_ss)
    residuals = y_resid - beta * x_resid
    df = int(len(z) - max(y_rank, x_rank) - 1)
    if df <= 0:
        se = p = ci_low = ci_high = np.nan
    else:
        sigma2 = float(np.sum(residuals**2) / df)
        se = float(np.sqrt(sigma2 / x_ss))
        t_value = beta / se if se > 0 else np.nan
        p = float(2 * stats.t.sf(abs(t_value), df)) if np.isfinite(t_value) else np.nan
        t_crit = float(stats.t.ppf(0.975, df))
        ci_low = beta - t_crit * se
        ci_high = beta + t_crit * se

    return {
        PATIENT: pid,
        "model": model,
        "status": "success",
        "n": int(len(z)),
        "beta": beta,
        "se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p": p,
        "df": df,
        "n_covariates": len(covariates),
        "rank": max(y_rank, x_rank),
        "asm_range_raw": float(d[Y].max() - d[Y].min()),
        "asm_range_norm": float(d[Y_NORM].max() - d[Y_NORM].min()) if Y_NORM in d else np.nan,
    }


def patient_exposure_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, g in df.groupby(PATIENT):
        d = g.dropna(subset=[Y_NORM]).copy()
        far_from_prior = d["time_since_last_seizure"].isna() | (
            d["time_since_last_seizure"] >= 720
        )
        far_from_next = d["time_until_next_seizure"].isna() | (
            d["time_until_next_seizure"] >= 720
        )
        nonperi = d[far_from_prior & far_from_next]
        high = int((nonperi[Y_NORM] >= 0.80).sum())
        low = int((nonperi[Y_NORM] <= 0.50).sum())
        rows.append(
            {
                PATIENT: int(pid),
                "rows_nonperi": int(len(nonperi)),
                "hours_high_ge80_nonperi": high / 6.0,
                "hours_low_le50_nonperi": low / 6.0,
                "exposure_state_eligible": high >= 6 and low >= 6,
                "asm_range_norm_all": float(d[Y_NORM].max() - d[Y_NORM].min())
                if len(d)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize(fits: pd.DataFrame, exposure: pd.DataFrame) -> pd.DataFrame:
    rows = []
    merged = fits.merge(exposure, on=PATIENT, how="left")
    cohorts = {
        "all_modeled": merged["status"].eq("success"),
        "exposure_state_eligible": merged["status"].eq("success")
        & merged["exposure_state_eligible"].fillna(False),
        "asm_range_norm_ge_0.25": merged["status"].eq("success")
        & (merged["asm_range_norm_all"] >= 0.25),
        "asm_range_norm_ge_0.50": merged["status"].eq("success")
        & (merged["asm_range_norm_all"] >= 0.50),
    }

    for model in MODEL_SPECS:
        model_rows = merged[merged["model"].eq(model)].copy()
        for cohort, mask in cohorts.items():
            d = model_rows[mask.loc[model_rows.index]].dropna(subset=["beta"])
            betas = d["beta"].to_numpy(dtype=float)
            n = int(len(betas))
            if n == 0:
                continue
            n_negative = int((betas < 0).sum())
            n_positive = int((betas > 0).sum())
            sign_p = float(stats.binomtest(n_negative, n, 0.5).pvalue)
            wilcoxon_p = (
                float(stats.wilcoxon(betas, zero_method="wilcox").pvalue)
                if n > 0 and not np.allclose(betas, 0)
                else np.nan
            )
            t_res = stats.ttest_1samp(betas, 0.0) if n > 1 else None
            mean_beta = float(np.mean(betas))
            sd_beta = float(np.std(betas, ddof=1)) if n > 1 else np.nan
            se_mean = sd_beta / np.sqrt(n) if n > 1 else np.nan
            ci_low = ci_high = np.nan
            if n > 1:
                t_crit = float(stats.t.ppf(0.975, n - 1))
                ci_low = mean_beta - t_crit * se_mean
                ci_high = mean_beta + t_crit * se_mean
            rows.append(
                {
                    "model": model,
                    "cohort": cohort,
                    "n_patients": n,
                    "n_negative": n_negative,
                    "n_positive": n_positive,
                    "fraction_negative": n_negative / n,
                    "median_beta": float(np.median(betas)),
                    "q1_beta": float(np.quantile(betas, 0.25)),
                    "q3_beta": float(np.quantile(betas, 0.75)),
                    "mean_beta": mean_beta,
                    "mean_ci_low": ci_low,
                    "mean_ci_high": ci_high,
                    "t_test_p": float(t_res.pvalue) if t_res else np.nan,
                    "wilcoxon_p": wilcoxon_p,
                    "sign_test_p": sign_p,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_tables()
    exposure = patient_exposure_summary(df)

    fits = []
    for model, covariates in MODEL_SPECS.items():
        for _, g in df.groupby(PATIENT):
            fits.append(slope_for_patient(g, model, covariates))
    fits_df = pd.DataFrame(fits)
    summary = summarize(fits_df, exposure)

    exposure.to_csv(OUT_DIR / "patient_exposure_summary.csv", index=False)
    fits_df.to_csv(OUT_DIR / "patient_slopes.csv", index=False)
    summary.to_csv(OUT_DIR / "patient_slope_summary.csv", index=False)

    print(summary.to_string(index=False))
    print(f"\nSaved outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
