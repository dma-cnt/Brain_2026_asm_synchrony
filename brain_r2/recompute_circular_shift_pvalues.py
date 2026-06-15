"""Recompute circular-shift permutation P-values from saved beta draws.

This does not refit any GEE models. It reads the saved permutation coefficients
and observed beta from the corresponding summary file, then writes a compact
P-value summary with median-centered two-sided and lower-tail definitions.

Run:
    conda run -n cnt python brain_r2/recompute_circular_shift_pvalues.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from circular_shift_pvalues import circular_shift_pvalues


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PERM_CSV = SCRIPT_DIR / "perm_betas_circular_shift_full_5000perm_seed0.csv"
DEFAULT_SUMMARY_CSV = SCRIPT_DIR / "circular_shift_permutation_summary_full_5000perm_seed0.csv"
DEFAULT_OUT_CSV = SCRIPT_DIR / "circular_shift_permutation_pvalues_full_5000perm_seed0.csv"
DEFAULT_UPDATED_SUMMARY_CSV = (
    SCRIPT_DIR / "circular_shift_permutation_summary_full_5000perm_seed0.csv"
)
LEGACY_ABS_VALUE_COLUMNS = [
    "n_extreme_two_sided",
    "empirical_p_two_sided",
    "empirical_p_two_sided_plus1",
]


def read_observed_beta(summary_csv: Path) -> float:
    summary = pd.read_csv(summary_csv)
    if len(summary) != 1:
        raise ValueError(f"Expected one summary row in {summary_csv}, found {len(summary)}.")
    if "observed_beta" not in summary.columns:
        raise ValueError(f"Missing observed_beta column in {summary_csv}.")
    return float(summary.loc[0, "observed_beta"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--perm-csv", type=Path, default=DEFAULT_PERM_CSV)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument(
        "--updated-summary-csv",
        type=Path,
        default=DEFAULT_UPDATED_SUMMARY_CSV,
        help="Summary CSV to update with the recomputed P-value columns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    perm = pd.read_csv(args.perm_csv)["beta_perm"].astype(float)
    observed_beta = read_observed_beta(args.summary_csv)
    pvalues = circular_shift_pvalues(perm.to_numpy(), observed_beta)

    pvalue_row = pvalues.as_dict()
    pd.DataFrame([pvalue_row]).to_csv(args.out_csv, index=False)

    summary = pd.read_csv(args.summary_csv)
    updated = pd.concat(
        [
            summary.drop(
                columns=[
                    c
                    for c in list(pvalue_row) + LEGACY_ABS_VALUE_COLUMNS
                    if c in summary.columns and c != "observed_beta"
                ],
                errors="ignore",
            ),
            pd.DataFrame([{k: v for k, v in pvalue_row.items() if k != "observed_beta"}]),
        ],
        axis=1,
    )
    updated.to_csv(args.updated_summary_csv, index=False)

    print("Circular-shift permutation P-values")
    print(f"  permutation betas: {args.perm_csv}")
    print(f"  observed beta: {pvalues.observed_beta:.6f}")
    print(f"  shifted-null median: {pvalues.perm_beta_median:.6f}")
    print(
        "  median-centered two-sided p: "
        f"{pvalues.empirical_p_median_centered_two_sided:.6g} "
        f"({pvalues.n_extreme_median_centered_two_sided}/{pvalues.n_perm})"
    )
    print(
        "  directional lower-tail p: "
        f"{pvalues.empirical_p_lower_tail:.6g} "
        f"({pvalues.n_extreme_lower_tail}/{pvalues.n_perm})"
    )
    print(
        "  old absolute-coefficient p (diagnostic only): "
        f"{pvalues.empirical_p_abs_coefficient:.6g} "
        f"({pvalues.n_extreme_abs_coefficient}/{pvalues.n_perm})"
    )
    print(f"  saved P-value summary: {args.out_csv}")
    print(f"  updated summary: {args.updated_summary_csv}")


if __name__ == "__main__":
    main()
