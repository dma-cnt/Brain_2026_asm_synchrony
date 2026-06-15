"""Check mTLE interaction parameterization for the Brain R2 subgroup analysis.

This script diagnoses why separate subgroup coefficients do not numerically
match the interaction coefficient from a single pooled model. It compares:

1. separate subgroup models with subgroup-specific standardization;
2. separate subgroup models with one global standardization;
3. a pooled model with only synchrony-by-mTLE interaction;
4. a pooled model with all covariates interacted by mTLE status.

Run:
    conda run -n cnt python brain_r2/check_mtle_interaction_parameterization.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.preprocessing import StandardScaler
from statsmodels.genmod.cov_struct import Independence
from statsmodels.genmod.families import Gaussian


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = (SCRIPT_DIR / ".." / ".." / "cnt-data").resolve()
TABLES_PATH = DATA_PATH / "revised_tables_step_4"
SOZ_PATH = DATA_PATH / "HUP_SOZ.csv"

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
SYNC_ONLY_INTERACTION_FORMULA = f"{Y} ~ {SYNC} * is_mtle + " + " + ".join(COVARIATES)
ALL_INTERACTIONS_FORMULA = f"{Y} ~ (" + " + ".join(PREDICTORS) + ") * is_mtle"


def get_patient_hup_ids(directory: Path) -> list[int]:
    return sorted(
        int(path.name.split("_")[1].split(".")[0])
        for path in directory.glob("HUP_*.csv")
    )


def mtle_patient_ids() -> set[int]:
    soz = pd.read_csv(SOZ_PATH)
    soz.columns = [c.replace("\ufeff", "").strip() for c in soz.columns]
    mtle = soz[soz["SOZ"].astype(str).str.lower().eq("mesial temporal")].copy()
    return set(
        mtle["pt_id"]
        .astype(str)
        .str.replace("HUP", "", regex=False)
        .astype(int)
        .unique()
        .tolist()
    )


def build_data() -> pd.DataFrame:
    mtle_ids = mtle_patient_ids()
    dfs = []
    for pid in get_patient_hup_ids(TABLES_PATH):
        fp = TABLES_PATH / f"HUP_{pid}.csv"
        d = pd.read_csv(fp, low_memory=False)
        d = d.loc[:, ~d.columns.duplicated()].copy()
        d[GROUP] = pid
        d["is_mtle"] = int(pid in mtle_ids)
        keep = list(dict.fromkeys([GROUP, "is_mtle", "emu_minute", Y] + PREDICTORS))
        dfs.append(d[[c for c in keep if c in d.columns]])

    df = pd.concat(dfs, axis=0, ignore_index=True)
    df["emu_minute"] = pd.to_numeric(df["emu_minute"], errors="coerce")
    df[TIME_RAW] = df["emu_minute"].copy()
    needed = [Y, GROUP, "is_mtle", TIME_RAW] + PREDICTORS
    return df.dropna(subset=needed).sort_values([GROUP, TIME_RAW]).reset_index(drop=True)


def scale_global(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[COLS_TO_SCALE] = StandardScaler().fit_transform(out[COLS_TO_SCALE])
    return out


def scale_within_subset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    parts = []
    for _, g in out.groupby("is_mtle", sort=False):
        h = g.copy()
        h[COLS_TO_SCALE] = StandardScaler().fit_transform(h[COLS_TO_SCALE])
        parts.append(h)
    return pd.concat(parts, axis=0).sort_values([GROUP, TIME_RAW]).reset_index(drop=True)


def fit_gee(df: pd.DataFrame, formula: str):
    model = smf.gee(
        formula=formula,
        groups=df[GROUP],
        data=df,
        family=Gaussian(),
        cov_struct=Independence(),
    )
    return model.fit(cov_type="robust")


def term(result, name: str) -> tuple[float, float, float]:
    return float(result.params[name]), float(result.bse[name]), float(result.pvalues[name])


def wald_p_for_sum(result, terms: list[str]) -> tuple[float, float, float]:
    beta = float(sum(result.params[t] for t in terms))
    cov = result.cov_params()
    var = float(sum(cov.loc[a, b] for a in terms for b in terms))
    se = float(np.sqrt(var)) if var >= 0 else np.nan
    z = beta / se
    p = float(2.0 * stats.norm.sf(abs(z)))
    return beta, se, p


def main() -> None:
    raw = build_data()
    global_scaled = scale_global(raw)
    subset_scaled = scale_within_subset(raw)

    rows = []

    for label, data in [
        ("separate_subset_standardized_non_mtle", subset_scaled[subset_scaled["is_mtle"].eq(0)]),
        ("separate_subset_standardized_mtle", subset_scaled[subset_scaled["is_mtle"].eq(1)]),
        ("separate_global_standardized_non_mtle", global_scaled[global_scaled["is_mtle"].eq(0)]),
        ("separate_global_standardized_mtle", global_scaled[global_scaled["is_mtle"].eq(1)]),
    ]:
        res = fit_gee(data, BASE_FORMULA)
        beta, se, p = term(res, SYNC)
        rows.append(
            {
                "model": label,
                "term": SYNC,
                "patients": int(data[GROUP].nunique()),
                "rows": int(len(data)),
                "beta": beta,
                "se": se,
                "p": p,
            }
        )

    sync_only = fit_gee(global_scaled, SYNC_ONLY_INTERACTION_FORMULA)
    for term_name, label in [
        (SYNC, "pooled_global_sync_only_interaction_non_mtle_slope"),
        (f"{SYNC}:is_mtle", "pooled_global_sync_only_interaction_difference"),
    ]:
        beta, se, p = term(sync_only, term_name)
        rows.append(
            {
                "model": label,
                "term": term_name,
                "patients": int(global_scaled[GROUP].nunique()),
                "rows": int(len(global_scaled)),
                "beta": beta,
                "se": se,
                "p": p,
            }
        )
    beta, se, p = wald_p_for_sum(sync_only, [SYNC, f"{SYNC}:is_mtle"])
    rows.append(
        {
            "model": "pooled_global_sync_only_interaction_implied_mtle_slope",
            "term": f"{SYNC} + {SYNC}:is_mtle",
            "patients": int(global_scaled[GROUP].nunique()),
            "rows": int(len(global_scaled)),
            "beta": beta,
            "se": se,
            "p": p,
        }
    )

    all_int = fit_gee(global_scaled, ALL_INTERACTIONS_FORMULA)
    for term_name, label in [
        (SYNC, "pooled_global_all_covariate_interactions_non_mtle_slope"),
        (f"{SYNC}:is_mtle", "pooled_global_all_covariate_interactions_difference"),
    ]:
        beta, se, p = term(all_int, term_name)
        rows.append(
            {
                "model": label,
                "term": term_name,
                "patients": int(global_scaled[GROUP].nunique()),
                "rows": int(len(global_scaled)),
                "beta": beta,
                "se": se,
                "p": p,
            }
        )
    beta, se, p = wald_p_for_sum(all_int, [SYNC, f"{SYNC}:is_mtle"])
    rows.append(
        {
            "model": "pooled_global_all_covariate_interactions_implied_mtle_slope",
            "term": f"{SYNC} + {SYNC}:is_mtle",
            "patients": int(global_scaled[GROUP].nunique()),
            "rows": int(len(global_scaled)),
            "beta": beta,
            "se": se,
            "p": p,
        }
    )

    out = pd.DataFrame(rows)
    out_csv = SCRIPT_DIR / "check_mtle_interaction_parameterization_summary.csv"
    out_txt = SCRIPT_DIR / "check_mtle_interaction_parameterization_summary.txt"
    out.to_csv(out_csv, index=False)
    out.to_string(out_txt, index=False)

    print(out.to_string(index=False))
    print(f"\nSaved: {out_csv}")
    print(f"Saved: {out_txt}")


if __name__ == "__main__":
    main()
