"""Check circular-shift P-value sensitivity to using fewer permutations.

This script does not refit the GEE models. It reads the saved 5000 circular-shift
permutation coefficients, then recomputes the same P-value definitions using
only the first 1000, first 2000 and all 5000 permutation draws. It also estimates
how variable the result would be if 1000 or 2000 permutations were sampled from
the completed 5000-permutation run.

Run:
    conda run -n cnt python brain_r2/circular_shift_permutation_count_sensitivity.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from circular_shift_pvalues import circular_shift_pvalues


SCRIPT_DIR = Path(__file__).resolve().parent
PERM_CSV = SCRIPT_DIR / "perm_betas_circular_shift_full_5000perm_seed0.csv"
SUMMARY_CSV = SCRIPT_DIR / "circular_shift_permutation_summary_full_5000perm_seed0.csv"
OUT_PREFIX_CSV = SCRIPT_DIR / "circular_shift_permutation_count_prefix_sensitivity.csv"
OUT_RESAMPLE_CSV = SCRIPT_DIR / "circular_shift_permutation_count_resample_sensitivity.csv"

PREFIX_COUNTS = [1000, 2000, 5000]
RESAMPLE_COUNTS = [1000, 2000]
N_RESAMPLES = 1000
RNG_SEED = 20260524


def observed_beta() -> float:
    summary = pd.read_csv(SUMMARY_CSV)
    if len(summary) != 1:
        raise ValueError(f"Expected one row in {SUMMARY_CSV}, found {len(summary)}")
    return float(summary.loc[0, "observed_beta"])


def compact_row(label: str, pvalues) -> dict[str, float | int | str]:
    return {
        "label": label,
        "n_perm": pvalues.n_perm,
        "observed_beta": pvalues.observed_beta,
        "perm_beta_median": pvalues.perm_beta_median,
        "perm_beta_sd": pvalues.perm_beta_sd,
        "n_extreme_median_centered_two_sided": pvalues.n_extreme_median_centered_two_sided,
        "p_median_centered_two_sided": pvalues.empirical_p_median_centered_two_sided,
        "p_median_centered_two_sided_plus1": (
            pvalues.empirical_p_median_centered_two_sided_plus1
        ),
        "n_extreme_lower_tail": pvalues.n_extreme_lower_tail,
        "p_lower_tail": pvalues.empirical_p_lower_tail,
        "p_lower_tail_plus1": pvalues.empirical_p_lower_tail_plus1,
        "n_extreme_abs_coefficient": pvalues.n_extreme_abs_coefficient,
        "p_abs_coefficient": pvalues.empirical_p_abs_coefficient,
        "p_abs_coefficient_plus1": pvalues.empirical_p_abs_coefficient_plus1,
        "min_possible_positive_p": pvalues.min_possible_positive_p,
    }


def prefix_sensitivity(perm_betas: np.ndarray, obs_beta: float) -> pd.DataFrame:
    rows = []
    for n_perm in PREFIX_COUNTS:
        if n_perm > len(perm_betas):
            continue
        pvalues = circular_shift_pvalues(perm_betas[:n_perm], obs_beta)
        rows.append(compact_row(f"first_{n_perm}", pvalues))
    return pd.DataFrame(rows)


def resample_sensitivity(perm_betas: np.ndarray, obs_beta: float) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for n_perm in RESAMPLE_COUNTS:
        median_centered_ps = []
        lower_tail_ps = []
        abs_coefficient_ps = []
        for _ in range(N_RESAMPLES):
            subset = rng.choice(perm_betas, size=n_perm, replace=False)
            pvalues = circular_shift_pvalues(subset, obs_beta)
            median_centered_ps.append(pvalues.empirical_p_median_centered_two_sided)
            lower_tail_ps.append(pvalues.empirical_p_lower_tail)
            abs_coefficient_ps.append(pvalues.empirical_p_abs_coefficient)

        for name, values in [
            ("median_centered_two_sided", median_centered_ps),
            ("lower_tail", lower_tail_ps),
            ("abs_coefficient", abs_coefficient_ps),
        ]:
            arr = np.asarray(values, dtype=float)
            rows.append(
                {
                    "n_perm": n_perm,
                    "p_definition": name,
                    "n_resamples": N_RESAMPLES,
                    "p_min": float(np.min(arr)),
                    "p_q025": float(np.quantile(arr, 0.025)),
                    "p_median": float(np.median(arr)),
                    "p_q975": float(np.quantile(arr, 0.975)),
                    "p_max": float(np.max(arr)),
                    "fraction_below_0_05": float(np.mean(arr < 0.05)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    perm_betas = pd.read_csv(PERM_CSV)["beta_perm"].to_numpy(dtype=float)
    obs_beta = observed_beta()

    prefix = prefix_sensitivity(perm_betas, obs_beta)
    resampled = resample_sensitivity(perm_betas, obs_beta)

    prefix.to_csv(OUT_PREFIX_CSV, index=False)
    resampled.to_csv(OUT_RESAMPLE_CSV, index=False)

    print("Circular-shift permutation count sensitivity")
    print(f"  source permutations: {PERM_CSV}")
    print(f"  observed beta: {obs_beta:.6f}")
    print("\nPrefix results")
    print(
        prefix[
            [
                "label",
                "n_perm",
                "n_extreme_median_centered_two_sided",
                "p_median_centered_two_sided",
                "p_median_centered_two_sided_plus1",
                "n_extreme_lower_tail",
                "p_lower_tail",
                "p_lower_tail_plus1",
            ]
        ].to_string(index=False)
    )
    print("\nResampled stability")
    print(resampled.to_string(index=False))
    print("\nSaved")
    print(f"  {OUT_PREFIX_CSV}")
    print(f"  {OUT_RESAMPLE_CSV}")


if __name__ == "__main__":
    main()
