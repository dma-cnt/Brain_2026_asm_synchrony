#!/usr/bin/env python3
"""Audit continuous longitudinal models using normalized ASM load.

This mirrors the model specifications used by ``figure_generation/figure_3.py``
but changes the outcome from summed raw non-lorazepam ASM load to
patient-normalized non-lorazepam ASM load. It is intended to check the
manuscript Methods/results normalization mismatch without changing existing
figure code.

Run:
    conda run -n cnt python brain_r2/continuous_normalized_load_models.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from circular_shift_pvalues import circular_shift_pvalues

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE_3_PATH = REPO_ROOT / "figure_generation" / "figure_3.py"
NORMALIZED_OUTCOME = "med_sum_no_lorazepam_norm"


def load_figure_3_module():
    """Load figure_generation/figure_3.py without colliding with brain_r2/figure_3.py."""
    figure_dir = str(FIGURE_3_PATH.parent)
    if figure_dir not in sys.path:
        sys.path.insert(0, figure_dir)
    spec = importlib.util.spec_from_file_location(
        "revised_figure_3_for_normalized_audit",
        FIGURE_3_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {FIGURE_3_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_outcome(f3, outcome: str) -> None:
    """Patch the imported Figure 3 module to use a different outcome."""
    f3.Y = outcome
    f3.FORMULA = f"{outcome} ~ " + " + ".join(f3.PREDICTORS)
    f3.FORMULA_NO_SYNC = (
        f"{outcome} ~ " + " + ".join(p for p in f3.PREDICTORS if p != f3.SYNC)
    )


def extract_fixed_effects(
    f3,
    model_name: str,
    n_patients: int,
    n_observations: int,
    result,
) -> list[dict[str, object]]:
    ci = result.conf_int(alpha=0.05)
    rows: list[dict[str, object]] = []
    for term in f3.PREDICTORS:
        if term not in result.params.index:
            continue
        rows.append(
            {
                "model": model_name,
                "n_patients": n_patients,
                "n_observations": n_observations,
                "term": term,
                "covariate": f3.LABEL_MAP.get(term, term),
                "estimate": float(result.params[term]),
                "se": float(result.bse[term]),
                "ci_low": float(ci.loc[term, 0]),
                "ci_high": float(ci.loc[term, 1]),
                "p": float(result.pvalues[term]),
            }
        )
    return rows


def format_pvalue(p_value: float) -> str:
    if p_value < 0.001:
        return "<0.001"
    return f"{p_value:.3f}"


def print_synchrony_summary(coefs: pd.DataFrame) -> None:
    sync = coefs[coefs["term"] == "synchrony_broadband"].copy()
    print("\nSynchrony coefficient summary")
    print(
        sync[
            [
                "model",
                "n_patients",
                "n_observations",
                "estimate",
                "se",
                "ci_low",
                "ci_high",
                "p",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.6g}")
    )


def print_full_covariate_table(coefs: pd.DataFrame) -> None:
    print("\nFull fixed-effect table")
    print("model\tn_patients\tn_observations\tcovariate\testimate\tse\tci_low\tci_high\tp")
    for row in coefs.itertuples(index=False):
        print(
            f"{row.model}\t"
            f"{row.n_patients}\t"
            f"{row.n_observations}\t"
            f"{row.covariate}\t"
            f"{row.estimate:.6g}\t"
            f"{row.se:.6g}\t"
            f"{row.ci_low:.6g}\t"
            f"{row.ci_high:.6g}\t"
            f"{format_pvalue(row.p)}"
        )


def fit_lrt_models(f3, df: pd.DataFrame) -> tuple[dict[str, object], dict[str, object]]:
    full_ri_ml = f3.smf.mixedlm(f3.FORMULA, df, groups=df[f3.GROUP]).fit(
        reml=False,
        method="lbfgs",
    )
    red_ri_ml = f3.smf.mixedlm(f3.FORMULA_NO_SYNC, df, groups=df[f3.GROUP]).fit(
        reml=False,
        method="lbfgs",
    )
    lrt_ri = f3.lrt(full_ri_ml, red_ri_ml, "Random intercept")

    full_rs_ml = f3.smf.mixedlm(
        f3.FORMULA,
        df,
        groups=df[f3.GROUP],
        re_formula=f"~{f3.SYNC}",
    ).fit(reml=False, method="lbfgs")
    red_rs_ml = f3.smf.mixedlm(
        f3.FORMULA_NO_SYNC,
        df,
        groups=df[f3.GROUP],
        re_formula=f"~{f3.SYNC}",
    ).fit(reml=False, method="lbfgs")
    lrt_rs = f3.lrt(full_rs_ml, red_rs_ml, "Random intercept + random synchrony slope")
    return lrt_ri, lrt_rs


def circular_shift_permutation(
    f3,
    df: pd.DataFrame,
    observed_beta: float,
    n_perm: int,
    seed: int,
    min_shift: int,
    progress_every: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    base_sync = df[f3.SYNC].to_numpy(copy=True)
    group_indices = [
        np.asarray(index, dtype=int) for index in df.groupby(f3.GROUP).groups.values()
    ]
    perm_data = df.copy()
    betas = np.empty(n_perm, dtype=float)
    started = time.time()

    for permutation_index in range(n_perm):
        sync_perm = base_sync.copy()
        for index in group_indices:
            if len(index) <= min_shift:
                continue
            shift = int(rng.integers(min_shift, len(index)))
            sync_perm[index] = np.roll(base_sync[index], shift)

        perm_data[f3.SYNC] = sync_perm
        result = f3.fit_gee_independence_robust(perm_data)
        betas[permutation_index] = float(result.params[f3.SYNC])

        done = permutation_index + 1
        if progress_every > 0 and (done % progress_every == 0 or done == n_perm):
            elapsed = time.time() - started
            rate = done / elapsed if elapsed > 0 else float("nan")
            print(f"  completed {done}/{n_perm} permutation fits ({rate:.2f} fits/sec)")

    pvalues = circular_shift_pvalues(betas, observed_beta)
    return {
        "observed_beta": observed_beta,
        "n_perm": int(pvalues.n_perm),
        "perm_beta_mean": float(pvalues.perm_beta_mean),
        "perm_beta_sd": float(pvalues.perm_beta_sd),
        "perm_beta_median": float(pvalues.perm_beta_median),
        "n_extreme_median_centered_two_sided": int(
            pvalues.n_extreme_median_centered_two_sided
        ),
        "empirical_p_median_centered_two_sided": float(
            pvalues.empirical_p_median_centered_two_sided
        ),
        "empirical_p_median_centered_two_sided_plus1": float(
            pvalues.empirical_p_median_centered_two_sided_plus1
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Path to cnt-data. If omitted, the Figure 3 search paths are used.",
    )
    parser.add_argument(
        "--outcome",
        default=NORMALIZED_OUTCOME,
        help="Outcome column to use. Default: med_sum_no_lorazepam_norm.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Optional path to write the full fixed-effect table as CSV.",
    )
    parser.add_argument(
        "--skip-lrt",
        action="store_true",
        help="Skip likelihood-ratio tests for faster coefficient-only checks.",
    )
    parser.add_argument(
        "--n-perm",
        type=int,
        default=0,
        help="Optional circular-shift permutations for the normalized GEE. Default: 0.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-shift", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    f3 = load_figure_3_module()
    configure_outcome(f3, args.outcome)

    data_root = f3.find_data_root(args.data_root)
    table_dir = data_root / "revised_tables_step_4"
    patient_ids = f3.get_patient_hup_ids(table_dir)
    df = f3.build_all_data(table_dir, patient_ids)
    n_patients = int(df[f3.GROUP].nunique())
    n_observations = int(len(df))

    print("Continuous normalized ASM-load model audit")
    print(f"  outcome: {args.outcome}")
    print(f"  tables: {table_dir}")
    print(f"  processed tables found: {len(patient_ids)}")
    print(f"  complete-case patients: {n_patients}")
    print(f"  complete-case 10-min observations: {n_observations}")

    rows: list[dict[str, object]] = []

    print("\nFitting patient-clustered GEE...")
    gee = f3.fit_gee_independence_robust(df)
    rows.extend(
        extract_fixed_effects(
            f3,
            "Patient-clustered GEE (robust sandwich SE)",
            n_patients,
            n_observations,
            gee,
        )
    )

    print("Fitting 30-min binned GEE...")
    binned_gee = f3.fit_gee_unstructured_binned(df, bin_minutes=30, time_points=20)
    rows.extend(
        extract_fixed_effects(
            f3,
            "30-min binned GEE (unstructured; first 20 bins)",
            int(binned_gee._fit_n_patients),
            int(binned_gee._fit_n_obs),
            binned_gee,
        )
    )

    print("Fitting linear mixed-effects random-intercept model...")
    lme_ri = f3.fit_lme_random_intercept(df, reml=True)
    rows.extend(
        extract_fixed_effects(
            f3,
            "Linear mixed-effects model (random intercept)",
            n_patients,
            n_observations,
            lme_ri,
        )
    )

    print("Fitting linear mixed-effects random-intercept + random-slope model...")
    lme_rs = f3.fit_lme_random_slope(df, reml=True)
    rows.extend(
        extract_fixed_effects(
            f3,
            "Linear mixed-effects model (random intercept + random synchrony slope)",
            n_patients,
            n_observations,
            lme_rs,
        )
    )

    coefs = pd.DataFrame(rows)
    print_synchrony_summary(coefs)

    if not args.skip_lrt:
        print("\nFitting mixed-effects likelihood-ratio tests...")
        lrt_ri, lrt_rs = fit_lrt_models(f3, df)
        print("\nLikelihood-ratio tests for adding fixed synchrony effect")
        for row in (lrt_ri, lrt_rs):
            print(
                f"  {row['label']}: LR = {row['lr_stat']:.6g}, "
                f"df = {row['df']}, P = {format_pvalue(float(row['p']))}"
            )

    if args.n_perm > 0:
        print("\nRunning normalized-outcome circular-shift permutation check...")
        permutation_summary = circular_shift_permutation(
            f3=f3,
            df=df,
            observed_beta=float(gee.params[f3.SYNC]),
            n_perm=args.n_perm,
            seed=args.seed,
            min_shift=args.min_shift,
            progress_every=args.progress_every,
        )
        print("\nCircular-shift permutation summary")
        print(f"  observed beta: {permutation_summary['observed_beta']:.6g}")
        print(
            "  permutation beta median: "
            f"{permutation_summary['perm_beta_median']:.6g}"
        )
        print(
            "  median-centered empirical two-sided P: "
            f"{permutation_summary['empirical_p_median_centered_two_sided']:.6g} "
            f"({permutation_summary['n_extreme_median_centered_two_sided']}/"
            f"{permutation_summary['n_perm']})"
        )
        print(
            "  plus-one Monte Carlo P: "
            f"{permutation_summary['empirical_p_median_centered_two_sided_plus1']:.6g}"
        )

    print_full_covariate_table(coefs)

    if args.csv_out is not None:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        coefs.to_csv(args.csv_out, index=False)
        print(f"\nSaved CSV: {args.csv_out}")


if __name__ == "__main__":
    main()
