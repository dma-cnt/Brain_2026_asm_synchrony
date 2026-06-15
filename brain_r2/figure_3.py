"""Figure 3 revision: seizure-level peri-ictal analysis with clustered inference.

Each valid seizure contributes one pre-ictal and/or post-ictal summary window.
Inference uses patient-clustered robust standard errors rather than treating
multiple seizures from the same patient as independent.

Run with:
    conda run -n cnt python brain_r2/figure_3.py
or, with the env active:
    python brain_r2/figure_3.py
"""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/asm_synchrony_matplotlib")

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore", category=UserWarning)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "cnt-data"))
TABLES_PATH = os.path.join(DATA_PATH, "giant_new_tables")

WINDOW_MINUTES = 60
SEIZURE_SEPARATION_ROWS = 5 * 60
SYNC_ARTIFACT_THRESHOLD = 0.8

OUT_PNG = os.path.join(SCRIPT_DIR, "figure_3.png")
OUT_PDF = os.path.join(SCRIPT_DIR, "figure_3.pdf")
OUT_SEIZURE_CSV = os.path.join(SCRIPT_DIR, "figure_3_seizure_summaries.csv")
OUT_STATS_CSV = os.path.join(SCRIPT_DIR, "figure_3_clustered_stats.csv")


def format_pvalue(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "P = NA"
    if p_value < 1e-4:
        return f"P = {p_value:.1e}"
    if p_value < 1e-3:
        return f"P = {p_value:.4f}"
    return f"P = {p_value:.3f}"


def get_patient_hup_ids(directory: str) -> list[int]:
    return sorted(
        int(filename.split("_")[1].split(".")[0])
        for filename in os.listdir(directory)
        if filename.startswith("HUP_") and filename.endswith(".csv")
    )


def load_patient_table(hup_id: int) -> pd.DataFrame:
    cols = [
        "emu_minute",
        "had_seizure",
        "synchrony_broadband",
        "med_sum_no_lorazepam_raw",
    ]
    path = os.path.join(TABLES_PATH, f"HUP_{hup_id}.csv")
    return pd.read_csv(path, usecols=cols, low_memory=False).reset_index(drop=True)


def valid_seizure_indices(df: pd.DataFrame) -> list[int]:
    """Mirror the R1 seizure-window selection to preserve the 154/152 cohort.

    The source notebook selected the first seizure-marked row and then each
    later seizure-marked row separated from the immediately preceding
    seizure-marked row by at least 5*60 table rows.
    """

    seizure_indices = df[df["had_seizure"] >= 1].index.to_numpy()
    return [
        int(idx)
        for i, idx in enumerate(seizure_indices)
        if i == 0 or (seizure_indices[i] - seizure_indices[i - 1]) >= SEIZURE_SEPARATION_ROWS
    ]


def summarize_window(
    df: pd.DataFrame,
    hup_id: int,
    seizure_number: int,
    seizure_idx: int,
    period: str,
) -> dict[str, float | int | str] | None:
    seizure_minute = float(df.loc[seizure_idx, "emu_minute"])
    if period == "pre":
        window = df[
            (df["emu_minute"] >= seizure_minute - WINDOW_MINUTES)
            & (df["emu_minute"] < seizure_minute)
        ]
    elif period == "post":
        window = df[
            (df["emu_minute"] > seizure_minute)
            & (df["emu_minute"] <= seizure_minute + WINDOW_MINUTES)
        ]
    else:
        raise ValueError(f"Unknown period: {period}")

    if len(window) == 0 or window["synchrony_broadband"].isna().all():
        return None

    median_synchrony = float(np.nanmedian(window["synchrony_broadband"]))
    median_asm_load = float(np.nanmedian(window["med_sum_no_lorazepam_raw"]))
    if (
        not np.isfinite(median_synchrony)
        or not np.isfinite(median_asm_load)
        or median_synchrony >= SYNC_ARTIFACT_THRESHOLD
    ):
        return None

    return {
        "patient_hup_id": hup_id,
        "seizure_number": seizure_number,
        "seizure_index": seizure_idx,
        "seizure_minute": seizure_minute,
        "period": period,
        "post_ictal": int(period == "post"),
        "median_asm_load": median_asm_load,
        "median_synchrony": median_synchrony,
        "n_window_rows": int(len(window)),
        "n_synchrony_values": int(window["synchrony_broadband"].notna().sum()),
    }


def build_seizure_summaries() -> pd.DataFrame:
    rows = []
    for hup_id in get_patient_hup_ids(TABLES_PATH):
        df = load_patient_table(hup_id)
        for seizure_number, seizure_idx in enumerate(valid_seizure_indices(df), start=1):
            for period in ("pre", "post"):
                row = summarize_window(df, hup_id, seizure_number, seizure_idx, period)
                if row is not None:
                    rows.append(row)
    out = pd.DataFrame(rows)
    out["asm_load_z"] = (
        out["median_asm_load"] - out["median_asm_load"].mean()
    ) / out["median_asm_load"].std(ddof=0)
    return out


def fit_clustered_model(df: pd.DataFrame):
    return smf.ols("median_synchrony ~ asm_load_z", df).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["patient_hup_id"], "use_correction": True},
        use_t=True,
    )


def fit_combined_model(df: pd.DataFrame):
    return smf.ols("median_synchrony ~ asm_load_z * post_ictal", df).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["patient_hup_id"], "use_correction": True},
        use_t=True,
    )


def clustered_period_stats(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    rows = []
    models = {}
    for period in ("pre", "post"):
        period_df = df[df["period"] == period].copy()
        result = fit_clustered_model(period_df)
        models[period] = result
        ci = result.conf_int().loc["asm_load_z"]
        rows.append(
            {
                "model": f"{period}_separate_clustered_ols",
                "period": period,
                "term": "asm_load_z",
                "n_windows": int(len(period_df)),
                "n_patients": int(period_df["patient_hup_id"].nunique()),
                "estimate": float(result.params["asm_load_z"]),
                "std_error": float(result.bse["asm_load_z"]),
                "ci_low": float(ci.iloc[0]),
                "ci_high": float(ci.iloc[1]),
                "p_value": float(result.pvalues["asm_load_z"]),
                "r_squared": float(result.rsquared),
            }
        )

        naive = smf.ols("median_synchrony ~ asm_load_z", period_df).fit()
        naive_ci = naive.conf_int().loc["asm_load_z"]
        rows.append(
            {
                "model": f"{period}_naive_ols_for_comparison",
                "period": period,
                "term": "asm_load_z",
                "n_windows": int(len(period_df)),
                "n_patients": int(period_df["patient_hup_id"].nunique()),
                "estimate": float(naive.params["asm_load_z"]),
                "std_error": float(naive.bse["asm_load_z"]),
                "ci_low": float(naive_ci.iloc[0]),
                "ci_high": float(naive_ci.iloc[1]),
                "p_value": float(naive.pvalues["asm_load_z"]),
                "r_squared": float(naive.rsquared),
            }
        )

    combined = fit_combined_model(df)
    models["combined"] = combined
    for term in ["asm_load_z", "post_ictal", "asm_load_z:post_ictal"]:
        ci = combined.conf_int().loc[term]
        rows.append(
            {
                "model": "combined_interaction_clustered_ols",
                "period": "combined",
                "term": term,
                "n_windows": int(len(df)),
                "n_patients": int(df["patient_hup_id"].nunique()),
                "estimate": float(combined.params[term]),
                "std_error": float(combined.bse[term]),
                "ci_low": float(ci.iloc[0]),
                "ci_high": float(ci.iloc[1]),
                "p_value": float(combined.pvalues[term]),
                "r_squared": float(combined.rsquared),
            }
        )

    post_slope_test = combined.t_test("asm_load_z + asm_load_z:post_ictal = 0")
    post_ci = post_slope_test.conf_int()[0]
    rows.append(
        {
            "model": "combined_interaction_clustered_ols",
            "period": "post",
            "term": "post_ictal_slope_from_combined_model",
            "n_windows": int(len(df)),
            "n_patients": int(df["patient_hup_id"].nunique()),
            "estimate": float(post_slope_test.effect[0]),
            "std_error": float(post_slope_test.sd[0][0]),
            "ci_low": float(post_ci[0]),
            "ci_high": float(post_ci[1]),
            "p_value": float(post_slope_test.pvalue),
            "r_squared": float(combined.rsquared),
        }
    )

    return pd.DataFrame(rows), models


def prediction_line(result, x_values: np.ndarray, mean_load: float, sd_load: float) -> np.ndarray:
    x_z = (x_values - mean_load) / sd_load
    pred_df = pd.DataFrame({"asm_load_z": x_z})
    return result.predict(pred_df)


def plot_period_panel(
    ax: plt.Axes,
    period_df: pd.DataFrame,
    result,
    stats_row: pd.Series,
    title: str,
    point_color: str,
    line_color: str,
    mean_load: float,
    sd_load: float,
) -> None:
    sns.scatterplot(
        data=period_df,
        x="median_asm_load",
        y="median_synchrony",
        ax=ax,
        s=76,
        color=point_color,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.72,
        legend=False,
    )

    x_values = np.linspace(period_df["median_asm_load"].min(), period_df["median_asm_load"].max(), 100)
    y_values = prediction_line(
        result,
        x_values,
        mean_load=mean_load,
        sd_load=sd_load,
    )
    ax.plot(x_values, y_values, color=line_color, linewidth=2.8)

    ax.set_title(title, fontweight="bold", pad=10)
    ax.set_xlabel("Median ASM load")
    ax.set_ylabel("Median synchrony")
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    annotation = (
        f"n = {int(stats_row['n_windows'])} windows\n"
        f"{int(stats_row['n_patients'])} patients\n"
        f"βz = {stats_row['estimate']:.4f}\n"
        f"95% CI [{stats_row['ci_low']:.4f}, {stats_row['ci_high']:.4f}]\n"
        f"{format_pvalue(float(stats_row['p_value']))}"
    )
    ax.text(
        0.97,
        0.97,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.92, edgecolor="#cccccc"),
    )


def plot_combined_panel(ax: plt.Axes, stats_df: pd.DataFrame) -> None:
    plot_rows = stats_df[
        stats_df["model"].eq("combined_interaction_clustered_ols")
        & stats_df["term"].isin(["asm_load_z", "post_ictal_slope_from_combined_model"])
    ].copy()
    plot_rows["label"] = plot_rows["term"].map(
        {
            "asm_load_z": "Pre-ictal slope",
            "post_ictal_slope_from_combined_model": "Post-ictal slope",
        }
    )
    plot_rows = plot_rows.set_index("label").loc[["Pre-ictal slope", "Post-ictal slope"]].reset_index()

    y = np.arange(len(plot_rows))
    err_low = plot_rows["estimate"] - plot_rows["ci_low"]
    err_high = plot_rows["ci_high"] - plot_rows["estimate"]
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax.errorbar(
        plot_rows["estimate"],
        y,
        xerr=[err_low, err_high],
        fmt="o",
        markersize=10,
        color="#7b1fa2",
        ecolor="#7b1fa2",
        capsize=5,
        linewidth=2.0,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(plot_rows["label"])
    ax.set_xlabel("β per 1 SD higher ASM load (95% CI)")
    ax.set_title("Combined clustered interaction model", fontweight="bold", pad=10)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.grid(axis="x", linestyle=":", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    interaction = stats_df[
        (stats_df["model"] == "combined_interaction_clustered_ols")
        & (stats_df["term"] == "asm_load_z:post_ictal")
    ].iloc[0]
    ax.text(
        0.98,
        0.06,
        f"Interaction β = {interaction['estimate']:.4f}\n"
        f"95% CI [{interaction['ci_low']:.4f}, {interaction['ci_high']:.4f}]\n"
        f"{format_pvalue(float(interaction['p_value']))}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.92, edgecolor="#cccccc"),
    )


def plot_figure(df: pd.DataFrame, stats_df: pd.DataFrame, models: dict[str, object]) -> None:
    sns.set_theme(style="ticks", context="paper", font_scale=1.35)
    fig = plt.figure(figsize=(15.5, 5.7), dpi=300, constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.92])
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    pre_df = df[df["period"] == "pre"].copy()
    post_df = df[df["period"] == "post"].copy()
    pre_stats = stats_df[
        (stats_df["model"] == "pre_separate_clustered_ols")
        & (stats_df["term"] == "asm_load_z")
    ].iloc[0]
    post_stats = stats_df[
        (stats_df["model"] == "post_separate_clustered_ols")
        & (stats_df["term"] == "asm_load_z")
    ].iloc[0]

    plot_period_panel(
        axes[0],
        pre_df,
        models["pre"],
        pre_stats,
        "Pre-ictal windows",
        point_color="#2b7bba",
        line_color="#cf4446",
        mean_load=float(df["median_asm_load"].mean()),
        sd_load=float(df["median_asm_load"].std(ddof=0)),
    )
    plot_period_panel(
        axes[1],
        post_df,
        models["post"],
        post_stats,
        "Post-ictal windows",
        point_color="#5dade2",
        line_color="#0047ab",
        mean_load=float(df["median_asm_load"].mean()),
        sd_load=float(df["median_asm_load"].std(ddof=0)),
    )
    plot_combined_panel(axes[2], stats_df)

    for label, ax in zip(["A", "B", "C"], axes):
        ax.text(
            -0.12,
            1.06,
            label,
            transform=ax.transAxes,
            fontsize=22,
            fontweight="bold",
            va="bottom",
            ha="right",
        )

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")


def print_stats(stats_df: pd.DataFrame) -> None:
    key_rows = stats_df[
        stats_df["model"].isin(["pre_separate_clustered_ols", "post_separate_clustered_ols"])
    ]
    for _, row in key_rows.iterrows():
        print(f"\n{row['period'].upper()}-ICTAL")
        print(
            f"  n = {int(row['n_windows'])} windows; "
            f"{int(row['n_patients'])} contributing patients"
        )
        print(
            f"  clustered βz = {row['estimate']:.6f}, "
            f"95% CI [{row['ci_low']:.6f}, {row['ci_high']:.6f}], "
            f"{format_pvalue(row['p_value'])}"
        )

    interaction = stats_df[
        (stats_df["model"] == "combined_interaction_clustered_ols")
        & (stats_df["term"] == "asm_load_z:post_ictal")
    ].iloc[0]
    print("\nCOMBINED INTERACTION")
    print(
        f"  ASM load × post-ictal interaction β = {interaction['estimate']:.6f}, "
        f"95% CI [{interaction['ci_low']:.6f}, {interaction['ci_high']:.6f}], "
        f"{format_pvalue(interaction['p_value'])}"
    )


def main() -> None:
    print(f"Tables: {TABLES_PATH}")
    print(f"Window: {WINDOW_MINUTES} minutes pre/post seizure")
    print(
        "Seizure separation: mirrors R1 notebook "
        f"({SEIZURE_SEPARATION_ROWS} table rows)"
    )

    seizure_df = build_seizure_summaries()
    stats_df, models = clustered_period_stats(seizure_df)

    seizure_df.to_csv(OUT_SEIZURE_CSV, index=False)
    stats_df.to_csv(OUT_STATS_CSV, index=False)
    plot_figure(seizure_df, stats_df, models)

    print_stats(stats_df)
    print("\nSaved:")
    print(f"  {OUT_PNG}")
    print(f"  {OUT_PDF}")
    print(f"  {OUT_SEIZURE_CSV}")
    print(f"  {OUT_STATS_CSV}")


if __name__ == "__main__":
    main()
