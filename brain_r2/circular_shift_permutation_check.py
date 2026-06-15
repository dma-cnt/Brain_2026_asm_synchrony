"""Re-run the within-patient circular-shift permutation check.

This mirrors the dependence-preserving permutation analysis from
brain_r2/appeal_analysis.ipynb, but uses a larger default permutation count for
revision checks.

Run:
    conda run -n cnt python brain_r2/circular_shift_permutation_check.py --n-perm 1000
"""

from __future__ import annotations

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from statsmodels.genmod.cov_struct import Independence
from statsmodels.genmod.families import Gaussian

from circular_shift_pvalues import circular_shift_pvalues

warnings.filterwarnings("ignore")


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "cnt-data"))
TABLES_PATH = os.path.join(DATA_PATH, "revised_tables_step_4")
SOZ_PATH = os.path.join(DATA_PATH, "HUP_SOZ.csv")

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

COLS_TO_KEEP = [Y, GROUP, "emu_minute"] + PREDICTORS
COLS_TO_SCALE = [
    "synchrony_broadband",
    "time_since_last_seizure",
    "time_until_next_seizure",
    "emu_minute",
    "ad_ratio",
    "spike_rate",
]

FORMULA = f"{Y} ~ " + " + ".join(PREDICTORS)


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
        d[GROUP] = pid
        keep = [c for c in COLS_TO_KEEP if c in d.columns]
        dfs.append(d[keep])

    if not dfs:
        raise RuntimeError("No tables loaded for requested patient IDs.")

    all_data = pd.concat(dfs, axis=0, ignore_index=True)
    all_data = all_data.loc[:, ~all_data.columns.duplicated()].copy()

    if "emu_minute" not in all_data.columns:
        raise RuntimeError("Expected 'emu_minute' in tables but it is missing.")

    all_data["emu_minute"] = pd.to_numeric(all_data["emu_minute"], errors="coerce")
    all_data[TIME_RAW] = all_data["emu_minute"].copy()

    needed = [Y, GROUP, "emu_minute", TIME_RAW] + [
        c for c in PREDICTORS if c in all_data.columns
    ]
    all_data = all_data.dropna(subset=needed).reset_index(drop=True)
    all_data = all_data.sort_values([GROUP, "emu_minute"]).reset_index(drop=True)
    all_data["time_index"] = all_data.groupby(GROUP).cumcount().astype(int)

    scaler = StandardScaler()
    cols_to_scale = [c for c in COLS_TO_SCALE if c in all_data.columns]
    all_data[cols_to_scale] = scaler.fit_transform(all_data[cols_to_scale])

    return all_data


def fit_gee(df: pd.DataFrame):
    model = smf.gee(
        formula=FORMULA,
        groups=df[GROUP],
        data=df,
        family=Gaussian(),
        cov_struct=Independence(),
    )
    return model.fit(cov_type="robust")


def circular_shift_permutation(
    df: pd.DataFrame,
    n_perm: int,
    seed: int,
    min_shift: int,
    progress_every: int,
) -> tuple[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    obs = fit_gee(df)
    obs_beta = float(obs.params[SYNC])

    base_sync = df[SYNC].to_numpy(copy=True)
    group_indices = [
        np.asarray(idx, dtype=int) for idx in df.groupby(GROUP).groups.values()
    ]
    perm_data = df.copy()
    betas: list[float] = []
    started = time.time()

    for b in range(n_perm):
        sync_perm = base_sync.copy()
        for idx in group_indices:
            if len(idx) <= min_shift:
                continue
            shift = int(rng.integers(min_shift, len(idx)))
            sync_perm[idx] = np.roll(base_sync[idx], shift)

        perm_data[SYNC] = sync_perm
        result = fit_gee(perm_data)
        betas.append(float(result.params[SYNC]))

        done = b + 1
        if progress_every > 0 and (done % progress_every == 0 or done == n_perm):
            elapsed = time.time() - started
            rate = done / elapsed if elapsed > 0 else float("nan")
            print(
                f"  completed {done}/{n_perm} permutation fits "
                f"({rate:.2f} fits/sec)",
                flush=True,
            )

    return obs_beta, np.asarray(betas)


def summarize(
    df: pd.DataFrame,
    obs_beta: float,
    perm_betas: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, object]:
    pvalues = circular_shift_pvalues(perm_betas, obs_beta)

    return {
        "cohort": args.cohort,
        "n_perm_requested": int(args.n_perm),
        "n_perm_used": pvalues.n_perm,
        "seed": int(args.seed),
        "min_shift": int(args.min_shift),
        "rows": int(len(df)),
        "patients": int(df[GROUP].nunique()),
        **pvalues.as_dict(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=["full", "mtle", "non_mtle"], default="full")
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-shift", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--out-dir", default=SCRIPT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_perm < 1000:
        raise ValueError("--n-perm should be at least 1000 for this revision check.")

    completed, mtle_completed, non_mtle_completed = cohort_patient_ids()
    if args.cohort == "full":
        pids = completed
    elif args.cohort == "mtle":
        pids = mtle_completed
    else:
        pids = non_mtle_completed

    df = build_all_data(pids)
    os.makedirs(args.out_dir, exist_ok=True)

    print("Within-patient circular-shift permutation check")
    print(f"  tables: {TABLES_PATH}")
    print(f"  cohort: {args.cohort}")
    print(f"  rows after dropna: {len(df)}")
    print(f"  patients: {df[GROUP].nunique()}")
    print(f"  n_perm: {args.n_perm}")
    print(f"  seed: {args.seed}")
    print(f"  min_shift: {args.min_shift}")

    obs_beta, perm_betas = circular_shift_permutation(
        df=df,
        n_perm=args.n_perm,
        seed=args.seed,
        min_shift=args.min_shift,
        progress_every=args.progress_every,
    )
    summary = summarize(df, obs_beta, perm_betas, args)

    suffix = f"{args.cohort}_{args.n_perm}perm_seed{args.seed}"
    perm_path = os.path.join(args.out_dir, f"perm_betas_circular_shift_{suffix}.csv")
    summary_path = os.path.join(
        args.out_dir, f"circular_shift_permutation_summary_{suffix}.csv"
    )

    pd.DataFrame(
        {
            "perm_index": np.arange(1, len(perm_betas) + 1, dtype=int),
            "beta_perm": perm_betas,
            "median_centered_abs_distance_ge_observed": (
                np.abs(perm_betas - summary["perm_beta_median"])
                >= summary["median_centered_distance_observed"]
            ),
            "lower_tail_beta_perm_le_observed": perm_betas <= obs_beta,
            "abs_beta_perm_ge_abs_observed": np.abs(perm_betas) >= abs(obs_beta),
        }
    ).to_csv(perm_path, index=False)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print("\nResults")
    print(f"  observed beta: {summary['observed_beta']:.6f}")
    print(
        "  permutation beta mean (sd): "
        f"{summary['perm_beta_mean']:.6f} ({summary['perm_beta_sd']:.6f})"
    )
    print(
        "  median-centered empirical two-sided p: "
        f"{summary['empirical_p_median_centered_two_sided']:.6g} "
        f"({summary['n_extreme_median_centered_two_sided']}/{summary['n_perm_used']})"
    )
    print(
        "  lower-tail empirical p: "
        f"{summary['empirical_p_lower_tail']:.6g} "
        f"({summary['n_extreme_lower_tail']}/{summary['n_perm_used']})"
    )
    print(
        "  old absolute-coefficient p (diagnostic only): "
        f"{summary['empirical_p_abs_coefficient']:.6g} "
        f"({summary['n_extreme_abs_coefficient']}/{summary['n_perm_used']})"
    )
    print(
        "  median-centered plus-one Monte Carlo p: "
        f"{summary['empirical_p_median_centered_two_sided_plus1']:.6g}"
    )
    print(f"\nSaved permutation betas: {perm_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
