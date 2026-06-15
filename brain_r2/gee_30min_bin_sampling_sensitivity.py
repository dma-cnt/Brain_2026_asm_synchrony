#!/usr/bin/env python3
"""Sensitivity checks for selecting 20 complete 30-minute bins per patient.

The manuscript's 30-minute binned GEE uses a balanced panel so an
unstructured working correlation matrix can be estimated. The original
implementation retained the first 20 valid 30-minute bins for each eligible
patient. This script compares that deterministic choice with bins selected to
span the admission and with repeated random within-patient samples.

The script is read-only with respect to project data and writes no output
files. Run from the repository root with:

    /opt/miniconda3/envs/cnt/bin/python brain_r2/gee_30min_bin_sampling_sensitivity.py
"""

from __future__ import annotations

import argparse
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Unstructured
from statsmodels.genmod.families import Gaussian


warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]

Y = "med_sum_no_lorazepam_raw"
GROUP = "patient_hup_id"
TIME = "emu_minute"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Path to cnt-data. If omitted, common local paths are searched.",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=20,
        help="Number of 30-minute bins to retain per patient.",
    )
    parser.add_argument(
        "--n-random",
        type=int,
        default=1000,
        help="Number of repeated random within-patient bin samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260525,
        help="Random seed for repeated random sampling.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N random repetitions; set 0 to disable.",
    )
    return parser.parse_args()


def find_data_root(explicit: Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
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
        if any(table_dir.glob("HUP_*.csv")):
            return candidate

    checked = "\n".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Could not locate cnt-data/revised_tables_step_4. Checked:\n" + checked
    )


def standardize_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        values = out[column].astype(float)
        sd = float(values.std(ddof=0))
        if sd == 0 or not np.isfinite(sd):
            raise ValueError(f"Cannot standardize {column}: invalid standard deviation.")
        out[column] = (values - float(values.mean())) / sd
    return out


def load_complete_10min_data(table_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    needed = list(dict.fromkeys([GROUP, TIME, TIME_RAW, Y] + PREDICTORS))
    read_cols = set([TIME, Y] + PREDICTORS)

    for path in sorted(table_dir.glob("HUP_*.csv")):
        try:
            hup_id = int(path.stem.split("_")[1])
        except (IndexError, ValueError):
            continue

        data = pd.read_csv(path, usecols=lambda c: c in read_cols, low_memory=False)
        data[GROUP] = hup_id
        data[TIME] = pd.to_numeric(data[TIME], errors="coerce")
        data[TIME_RAW] = data[TIME].copy()
        frames.append(data[[c for c in needed if c in data.columns]])

    if not frames:
        raise RuntimeError(f"No HUP_*.csv files found in {table_dir}")

    all_data = pd.concat(frames, axis=0, ignore_index=True)
    all_data = all_data.dropna(subset=needed).reset_index(drop=True)
    all_data = all_data.sort_values([GROUP, TIME_RAW]).reset_index(drop=True)
    all_data = standardize_columns(all_data, COLS_TO_SCALE)
    return all_data


def bin_30_minutes(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["bin"] = np.floor(data[TIME_RAW] / 30.0).astype(int)
    agg_cols = [Y] + PREDICTORS + [GROUP, TIME_RAW]
    out = (
        data[agg_cols + ["bin"]]
        .groupby([GROUP, "bin"], as_index=False)
        .mean(numeric_only=True)
        .sort_values([GROUP, TIME_RAW])
        .reset_index(drop=True)
    )
    return out


def even_positions(n_available: int, n_select: int) -> np.ndarray:
    """Return deterministic row positions spanning first to last available bin."""
    positions = np.rint(np.linspace(0, n_available - 1, n_select)).astype(int)
    if len(np.unique(positions)) == n_select:
        return positions

    # This should rarely be needed when n_available >= n_select, but keep the
    # selection deterministic if rounding produces duplicate positions.
    used: set[int] = set()
    repaired: list[int] = []
    for position in positions:
        candidates = sorted(
            range(n_available),
            key=lambda x: (abs(x - int(position)), x),
        )
        for candidate in candidates:
            if candidate not in used:
                used.add(candidate)
                repaired.append(candidate)
                break
    return np.asarray(sorted(repaired), dtype=int)


def select_panel(
    binned: pd.DataFrame,
    eligible_patients: np.ndarray,
    n_bins: int,
    mode: str,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []

    for patient_id in eligible_patients:
        patient = binned[binned[GROUP] == patient_id].sort_values(TIME_RAW).reset_index(drop=True)
        n_available = len(patient)

        if mode == "first":
            positions = np.arange(n_bins)
        elif mode == "even":
            positions = even_positions(n_available, n_bins)
        elif mode == "random":
            if rng is None:
                raise ValueError("rng is required for random sampling.")
            positions = np.sort(rng.choice(n_available, size=n_bins, replace=False))
        else:
            raise ValueError(f"Unknown mode: {mode}")

        selected.append(patient.iloc[positions].copy())

    panel = pd.concat(selected, axis=0, ignore_index=True)
    panel = panel.sort_values([GROUP, TIME_RAW]).reset_index(drop=True)
    panel["time_index"] = panel.groupby(GROUP).cumcount().astype(int)
    return panel


def fit_unstructured_gee(panel: pd.DataFrame) -> dict[str, float]:
    result = smf.gee(
        formula=FORMULA,
        groups=panel[GROUP],
        data=panel,
        family=Gaussian(),
        cov_struct=Unstructured(),
        time=panel["time_index"].values,
    ).fit(cov_type="robust")
    ci = result.conf_int().loc[SYNC]
    return {
        "beta": float(result.params[SYNC]),
        "se": float(result.bse[SYNC]),
        "ci_low": float(ci.iloc[0]),
        "ci_high": float(ci.iloc[1]),
        "p": float(result.pvalues[SYNC]),
    }


def print_single_result(label: str, result: dict[str, float]) -> None:
    print(
        f"{label}: beta={result['beta']:.6f}, SE={result['se']:.6f}, "
        f"95% CI=({result['ci_low']:.6f}, {result['ci_high']:.6f}), "
        f"P={result['p']:.6g}"
    )


def summarize_random(results: list[dict[str, float]]) -> None:
    beta = np.asarray([row["beta"] for row in results], dtype=float)
    p_values = np.asarray([row["p"] for row in results], dtype=float)
    ci_low = np.asarray([row["ci_low"] for row in results], dtype=float)
    ci_high = np.asarray([row["ci_high"] for row in results], dtype=float)

    print("Random 20-bin sampling summary:")
    print(f"  repetitions fitted: {len(beta)}")
    print(f"  median beta: {np.median(beta):.6f}")
    print(f"  beta IQR: ({np.quantile(beta, 0.25):.6f}, {np.quantile(beta, 0.75):.6f})")
    print(f"  beta 2.5%-97.5% range: ({np.quantile(beta, 0.025):.6f}, {np.quantile(beta, 0.975):.6f})")
    print(f"  proportion beta < 0: {np.mean(beta < 0):.3f}")
    print(f"  proportion P < 0.05: {np.mean(p_values < 0.05):.3f}")
    print(f"  proportion 95% CI entirely < 0: {np.mean(ci_high < 0):.3f}")
    print(f"  proportion 95% CI crossing 0: {np.mean((ci_low <= 0) & (ci_high >= 0)):.3f}")


def main() -> None:
    args = parse_args()
    data_root = find_data_root(args.data_root)
    table_dir = data_root / "revised_tables_step_4"

    print(f"Data root: {data_root}")
    data_10min = load_complete_10min_data(table_dir)
    print(
        f"Complete-case 10-minute rows: {len(data_10min)}; "
        f"patients: {data_10min[GROUP].nunique()}"
    )

    binned = bin_30_minutes(data_10min)
    counts = binned.groupby(GROUP).size()
    eligible = counts[counts >= args.n_bins].index.to_numpy()
    print(
        f"Eligible patients with at least {args.n_bins} valid 30-minute bins: "
        f"{len(eligible)}"
    )
    print(f"Balanced panel rows per fit: {len(eligible) * args.n_bins}")

    first_panel = select_panel(binned, eligible, args.n_bins, "first")
    first_result = fit_unstructured_gee(first_panel)
    print_single_result("First 20 chronological bins", first_result)

    even_panel = select_panel(binned, eligible, args.n_bins, "even")
    even_result = fit_unstructured_gee(even_panel)
    print_single_result("Evenly spaced 20 bins", even_result)

    rng = np.random.default_rng(args.seed)
    random_results: list[dict[str, float]] = []
    failures = 0
    for repetition in range(1, args.n_random + 1):
        try:
            random_panel = select_panel(binned, eligible, args.n_bins, "random", rng)
            random_results.append(fit_unstructured_gee(random_panel))
        except Exception as exc:  # pragma: no cover - diagnostic path
            failures += 1
            print(f"Random repetition {repetition} failed: {exc}")

        if args.progress_every and repetition % args.progress_every == 0:
            print(f"Completed {repetition}/{args.n_random} random repetitions")

    if failures:
        print(f"Random repetitions failed: {failures}")
    if random_results:
        summarize_random(random_results)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="asm_synchrony_gee_") as _tmp:
        main()
