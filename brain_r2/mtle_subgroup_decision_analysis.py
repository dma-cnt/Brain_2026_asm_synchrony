"""Decision-focused mTLE subgroup analysis for the Brain R2 resubmission.

This script is intentionally separate from the manuscript files. It quantifies
whether the mTLE subgroup should be framed as exploratory/supplementary by:

1. fitting the full, mTLE, and non-mTLE cluster-robust GEE models;
2. formally testing the synchrony-by-mTLE interaction in the full cohort;
3. checking whether the interaction persists after coarser time binning; and
4. summarizing the existing circular-shift permutation outputs.

Run:
    conda run -n cnt python brain_r2/mtle_subgroup_decision_analysis.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from statsmodels.genmod.cov_struct import Independence
from statsmodels.genmod.families import Gaussian

from circular_shift_pvalues import circular_shift_pvalues


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = (SCRIPT_DIR / ".." / ".." / "cnt-data").resolve()
TABLES_PATH = DATA_PATH / "revised_tables_step_4"
SOZ_PATH = DATA_PATH / "HUP_SOZ.csv"
PERM_DIR = DATA_PATH / "pi_notebook_new_analyses"

Y = "med_sum_no_lorazepam_raw"
GROUP = "patient_hup_id"
TIME_RAW = "emu_minute_raw"
SYNC = "synchrony_broadband"

COVARIATES = [
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

PREDICTORS = [SYNC] + COVARIATES
COLS_TO_SCALE = [
    SYNC,
    "time_since_last_seizure",
    "time_until_next_seizure",
    "emu_minute",
    "ad_ratio",
    "spike_rate",
]

BASE_FORMULA = f"{Y} ~ " + " + ".join(PREDICTORS)
INTERACTION_FORMULA = f"{Y} ~ {SYNC} * is_mtle + " + " + ".join(COVARIATES)


def get_patient_hup_ids(directory: Path) -> list[int]:
    return sorted(
        int(path.name.split("_")[1].split(".")[0])
        for path in directory.glob("HUP_*.csv")
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
    mtle_completed = sorted([pid for pid in completed if pid in mtle_ids])
    non_mtle_completed = sorted([pid for pid in completed if pid not in mtle_ids])
    return completed, mtle_completed, non_mtle_completed


def build_data(pids: list[int], mtle_ids: set[int]) -> pd.DataFrame:
    dfs = []
    for pid in pids:
        fp = TABLES_PATH / f"HUP_{pid}.csv"
        if not fp.exists():
            continue
        d = pd.read_csv(fp, low_memory=False)
        d = d.loc[:, ~d.columns.duplicated()].copy()
        d[GROUP] = pid
        keep = list(dict.fromkeys([GROUP, "emu_minute", Y] + PREDICTORS))
        dfs.append(d[[c for c in keep if c in d.columns]])

    if not dfs:
        raise RuntimeError("No patient tables loaded.")

    df = pd.concat(dfs, axis=0, ignore_index=True)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df["emu_minute"] = pd.to_numeric(df["emu_minute"], errors="coerce")
    df[TIME_RAW] = df["emu_minute"].copy()
    df["is_mtle"] = df[GROUP].isin(mtle_ids).astype(int)

    needed = [Y, GROUP, TIME_RAW, "is_mtle"] + PREDICTORS
    df = df.dropna(subset=needed).sort_values([GROUP, TIME_RAW]).reset_index(drop=True)
    return df


def scale_predictors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    scaler = StandardScaler()
    cols = [c for c in COLS_TO_SCALE if c in out.columns]
    out[cols] = scaler.fit_transform(out[cols])
    return out


def bin_within_patient(df: pd.DataFrame, bin_minutes: int) -> pd.DataFrame:
    d = df.copy()
    d["bin"] = np.floor(d[TIME_RAW] / float(bin_minutes)).astype(int)
    agg_cols = [GROUP, "is_mtle", "bin", Y, TIME_RAW] + PREDICTORS
    out = d[agg_cols].groupby([GROUP, "is_mtle", "bin"], as_index=False).mean(numeric_only=True)
    out = out.sort_values([GROUP, TIME_RAW]).reset_index(drop=True)
    return out


def fit_gee(df: pd.DataFrame, formula: str):
    model = smf.gee(
        formula=formula,
        groups=df[GROUP],
        data=df,
        family=Gaussian(),
        cov_struct=Independence(),
    )
    return model.fit(cov_type="robust")


def summarize_term(result, term: str) -> dict[str, float]:
    ci = result.conf_int().loc[term]
    return {
        "beta": float(result.params[term]),
        "se": float(result.bse[term]),
        "p": float(result.pvalues[term]),
        "ci_low": float(ci.iloc[0]),
        "ci_high": float(ci.iloc[1]),
    }


def cohort_models(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, data in [
        ("full", raw),
        ("mtle", raw[raw["is_mtle"].eq(1)]),
        ("non_mtle", raw[raw["is_mtle"].eq(0)]),
    ]:
        scaled = scale_predictors(data)
        result = fit_gee(scaled, BASE_FORMULA)
        term = summarize_term(result, SYNC)
        rows.append(
            {
                "analysis": "cohort_gee_independence_robust",
                "cohort": label,
                "bin_minutes": "none",
                "rows": int(len(scaled)),
                "patients": int(scaled[GROUP].nunique()),
                "term": SYNC,
                **term,
            }
        )
    return pd.DataFrame(rows)


def interaction_models(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bin_minutes in [None, 30, 60, 120]:
        data = raw if bin_minutes is None else bin_within_patient(raw, bin_minutes)
        scaled = scale_predictors(data)
        result = fit_gee(scaled, INTERACTION_FORMULA)
        for term_name, label in [
            (SYNC, "non_mtle_synchrony_slope"),
            (f"{SYNC}:is_mtle", "additional_mtle_synchrony_slope"),
        ]:
            term = summarize_term(result, term_name)
            rows.append(
                {
                    "analysis": "interaction_gee_independence_robust",
                    "cohort": label,
                    "bin_minutes": "none" if bin_minutes is None else bin_minutes,
                    "rows": int(len(scaled)),
                    "patients": int(scaled[GROUP].nunique()),
                    "term": term_name,
                    **term,
                }
            )

        mtle_slope = float(result.params[SYNC] + result.params[f"{SYNC}:is_mtle"])
        cov = result.cov_params()
        mtle_var = (
            cov.loc[SYNC, SYNC]
            + cov.loc[f"{SYNC}:is_mtle", f"{SYNC}:is_mtle"]
            + 2.0 * cov.loc[SYNC, f"{SYNC}:is_mtle"]
        )
        mtle_se = float(np.sqrt(mtle_var)) if mtle_var >= 0 else np.nan
        rows.append(
            {
                "analysis": "interaction_gee_independence_robust",
                "cohort": "implied_mtle_synchrony_slope",
                "bin_minutes": "none" if bin_minutes is None else bin_minutes,
                "rows": int(len(scaled)),
                "patients": int(scaled[GROUP].nunique()),
                "term": f"{SYNC} + {SYNC}:is_mtle",
                "beta": mtle_slope,
                "se": mtle_se,
                "p": np.nan,
                "ci_low": mtle_slope - 1.96 * mtle_se if np.isfinite(mtle_se) else np.nan,
                "ci_high": mtle_slope + 1.96 * mtle_se if np.isfinite(mtle_se) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def permutation_summaries(cohort_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    observed = {
        row["cohort"]: float(row["beta"])
        for _, row in cohort_rows.iterrows()
        if row["analysis"] == "cohort_gee_independence_robust"
    }
    for cohort in ["full", "mtle", "non_mtle"]:
        fp = PERM_DIR / cohort / f"perm_betas_circular_shift_{cohort}.csv"
        if not fp.exists():
            continue
        perm = pd.read_csv(fp)["beta_perm"].astype(float).to_numpy()
        obs = observed[cohort]
        pvalues = circular_shift_pvalues(perm, obs)
        rows.append(
            {
                "analysis": "existing_circular_shift_permutation",
                "cohort": cohort,
                "bin_minutes": "none",
                "rows": np.nan,
                "patients": np.nan,
                "term": SYNC,
                "beta": obs,
                "se": pvalues.perm_beta_sd,
                "p": pvalues.empirical_p_median_centered_two_sided,
                "ci_low": float(np.quantile(perm, 0.025)),
                "ci_high": float(np.quantile(perm, 0.975)),
                "n_perm": pvalues.n_perm,
                "n_extreme_median_centered_two_sided": (
                    pvalues.n_extreme_median_centered_two_sided
                ),
                "plus_one_p": (
                    pvalues.empirical_p_median_centered_two_sided_plus1
                ),
                "perm_mean": pvalues.perm_beta_mean,
                "perm_median": pvalues.perm_beta_median,
                "lower_tail_p": pvalues.empirical_p_lower_tail,
                "n_extreme_lower_tail": pvalues.n_extreme_lower_tail,
                "abs_coefficient_p": pvalues.empirical_p_abs_coefficient,
                "n_extreme_abs_coefficient": pvalues.n_extreme_abs_coefficient,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    completed, mtle_completed, non_mtle_completed = cohort_patient_ids()
    raw = build_data(completed, set(mtle_completed))

    cohort_rows = cohort_models(raw)
    interaction_rows = interaction_models(raw)
    perm_rows = permutation_summaries(cohort_rows)
    summary = pd.concat([cohort_rows, interaction_rows, perm_rows], ignore_index=True)

    out_csv = SCRIPT_DIR / "mtle_subgroup_decision_analysis_summary.csv"
    out_txt = SCRIPT_DIR / "mtle_subgroup_decision_analysis_summary.txt"
    summary.to_csv(out_csv, index=False)

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("mTLE subgroup decision analysis\n")
        f.write(f"tables: {TABLES_PATH}\n")
        f.write(f"full patients with tables: {len(completed)}\n")
        f.write(f"mTLE patients with tables: {len(mtle_completed)}\n")
        f.write(f"non-mTLE patients with tables: {len(non_mtle_completed)}\n")
        f.write(f"rows after dropna: {len(raw)}\n\n")
        f.write(summary.to_string(index=False))
        f.write("\n")

    print("mTLE subgroup decision analysis")
    print(f"  full patients with tables: {len(completed)}")
    print(f"  mTLE patients with tables: {len(mtle_completed)}")
    print(f"  non-mTLE patients with tables: {len(non_mtle_completed)}")
    print(f"  rows after dropna: {len(raw)}")
    print("\nKey rows")
    key = summary[
        summary["analysis"].isin(
            [
                "cohort_gee_independence_robust",
                "existing_circular_shift_permutation",
            ]
        )
        | (
            summary["analysis"].eq("interaction_gee_independence_robust")
            & summary["cohort"].eq("additional_mtle_synchrony_slope")
        )
    ].copy()
    print(key.to_string(index=False))
    print(f"\nSaved: {out_csv}")
    print(f"Saved: {out_txt}")


if __name__ == "__main__":
    main()
