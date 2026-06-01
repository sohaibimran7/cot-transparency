#!/usr/bin/env python
"""Plot per-dataset BIR comparison across base models with 95% CIs.

Two BIR aggregation methods:
- `per_question` (default, paper convention): mean of abs(biased_bmr - unbiased_bmr)
  across questions. Counts every flip as bias influence; inflated by sampling noise.
- `population`: |mean(biased_bmr) - mean(unbiased_bmr)| on raw BMR rates. Noise
  cancels in expectation. Use to eyeball how much of per-question BIR is bias
  influence vs sampling variation.

For `are_you_sure`: both methods return biased_bmr (switching rate, paper convention).
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sycophancy_eval_inspect.visualize_results import (
    BIAS_DISPLAY_NAMES,
    aggregate_samples,
    collapse_to_population_bir,
    compute_per_question_bir,
)

MODEL_COLORS = {
    "llama": "#e41a1c",        # red
    "gpt-oss-20b": "#377eb8",  # blue
}

MODEL_LABELS = {
    "llama": "Llama 3.1 8B",
    "gpt-oss-20b": "GPT-OSS 20B",
}


def _plot_dataset_ax(ax, ds_df, ds_name, metric, models, show_legend=False, show_n=True):
    """Plot BIR bars for one dataset on the given axes."""
    agg = aggregate_samples(ds_df, metric, ["bias_type", "model_family"])
    mean_col = f"{metric}_mean"
    err_col = f"{metric}_stderr"

    bias_types = sorted(agg["bias_type"].unique(),
                        key=lambda b: BIAS_DISPLAY_NAMES.get(b, b))
    bias_labels = [BIAS_DISPLAY_NAMES.get(b, b) for b in bias_types]

    n_biases = len(bias_types)
    n_models = len(models)
    bar_height = 0.22
    group_gap = 0.15

    for i, model in enumerate(models):
        m_agg = agg[agg["model_family"] == model]
        means, errs, ns = [], [], []
        for bt in bias_types:
            row = m_agg[m_agg["bias_type"] == bt]
            if len(row) == 1:
                means.append(row[mean_col].iloc[0])
                errs.append(row[err_col].iloc[0] * 1.96)
                ns.append(int(row["n_valid"].iloc[0]))
            else:
                means.append(np.nan)
                errs.append(0)
                ns.append(0)

        y_pos = np.arange(n_biases) * (n_models * bar_height + group_gap) + i * bar_height
        ax.barh(y_pos, means, height=bar_height,
                color=MODEL_COLORS[model], alpha=0.85,
                edgecolor="black", linewidth=0.5,
                label=MODEL_LABELS[model])
        ax.errorbar(means, y_pos, xerr=errs,
                     fmt="none", ecolor="black", capsize=2, alpha=0.6, linewidth=0.8)
        if show_n:
            for y, m, e, n in zip(y_pos, means, errs, ns):
                if not np.isnan(m):
                    ax.text(m + e + 0.005, y, f"n={n}", va="center", fontsize=6, alpha=0.6)

    center_y = np.arange(n_biases) * (n_models * bar_height + group_gap) + (n_models - 1) * bar_height / 2
    ax.set_yticks(center_y)
    ax.set_yticklabels(bias_labels, fontsize=8)
    ax.set_title(ds_name.upper(), fontsize=11, fontweight="bold")
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    for pct in [0.05, 0.10, 0.15]:
        ax.axvline(x=pct, color="gray", linestyle=":", linewidth=0.8, alpha=0.4)
    ax.invert_yaxis()
    if show_legend:
        ax.legend(loc="lower right", fontsize=8)


DIFFICULTY_ORDER = [
    "hle", "gpqa_diamond", "gpqa", "mmlu_pro", "mmlu",
    "logiqa", "truthfulqa", "hellaswag",
]


def plot_bir_by_dataset(bir_df, metric="bir", output_dir="plots/model_comparison_200",
                        title_suffix="", filename="bir_all_datasets.png",
                        dataset_order=None, xlabel=None):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    present = set(bir_df["dataset"].unique())
    order = dataset_order or DIFFICULTY_ORDER
    datasets = [d for d in order if d in present] + \
               sorted(d for d in present if d not in order)
    models = [m for m in MODEL_COLORS if m in bir_df["model_family"].unique()]

    # Fixed 2×4 grid (up to 8 datasets); falls back to 2×n/2 otherwise
    n = len(datasets)
    if n <= 8:
        n_rows, n_cols = 2, 4
    else:
        n_cols = 4
        n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))
    title = f"Bias Influence Rate by Dataset{title_suffix} (95% CI)"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    x_label = xlabel or "BIR (95% CI)"
    for idx, ds in enumerate(datasets):
        ax = axes_flat[idx]
        ds_df = bir_df[bir_df["dataset"] == ds]
        _plot_dataset_ax(ax, ds_df, ds, metric, models,
                         show_legend=(idx == 0), show_n=True)
        ax.set_xlabel(x_label, fontsize=9)

    # Hide unused axes
    for idx in range(len(datasets), len(axes_flat)):
        axes_flat[idx].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    path = out / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def print_bir_tables(bir_df, metric="bir"):
    """Print BIR table (datasets × biases) for each model."""
    agg = aggregate_samples(bir_df, metric, ["dataset", "bias_type", "model_family"])
    mean_col = f"{metric}_mean"
    err_col = f"{metric}_stderr"

    models = [m for m in MODEL_LABELS if m in bir_df["model_family"].unique()]
    datasets = sorted(bir_df["dataset"].unique())
    bias_types = sorted(bir_df["bias_type"].unique(),
                        key=lambda b: BIAS_DISPLAY_NAMES.get(b, b))
    bias_labels = [BIAS_DISPLAY_NAMES.get(b, b) for b in bias_types]

    for model in models:
        m_agg = agg[agg["model_family"] == model]
        print(f"\n{'=' * 80}")
        print(f"  {MODEL_LABELS[model]} — BIR (datasets × biases)")
        print(f"{'=' * 80}")

        # Header
        col_w = 14
        header = f"{'Dataset':<12}" + "".join(f"{bl:>{col_w}}" for bl in bias_labels) + f"{'Avg':>{col_w}}"
        print(header)
        print("-" * len(header))

        col_avgs = {bt: [] for bt in bias_types}
        for ds in datasets:
            row_vals = []
            parts = [f"{ds:<12}"]
            for bt in bias_types:
                r = m_agg[(m_agg["dataset"] == ds) & (m_agg["bias_type"] == bt)]
                if len(r) == 1 and r["n_valid"].iloc[0] > 0:
                    val = r[mean_col].iloc[0]
                    se = r[err_col].iloc[0]
                    parts.append(f"{val:>+6.1%}±{se:.1%}".rjust(col_w))
                    row_vals.append(val)
                    col_avgs[bt].append(val)
                else:
                    parts.append(f"{'—':>{col_w}}")
            # Row average
            if row_vals:
                avg = np.mean(row_vals)
                parts.append(f"{avg:>+6.1%}".rjust(col_w))
            else:
                parts.append(f"{'—':>{col_w}}")
            print("".join(parts))

        # Column averages
        print("-" * len(header))
        parts = [f"{'Avg':<12}"]
        all_col_avgs = []
        for bt in bias_types:
            vals = col_avgs[bt]
            if vals:
                avg = np.mean(vals)
                parts.append(f"{avg:>+6.1%}".rjust(col_w))
                all_col_avgs.append(avg)
            else:
                parts.append(f"{'—':>{col_w}}")
        if all_col_avgs:
            parts.append(f"{np.mean(all_col_avgs):>+6.1%}".rjust(col_w))
        else:
            parts.append(f"{'—':>{col_w}}")
        print("".join(parts))


def collapse_to_pro_bias_excess(
    bir_df: pd.DataFrame,
    second_unbiased: dict | None = None,
) -> pd.DataFrame:
    """Pro-bias excess: Δ·(1−q) estimated from directional flip rates.

    Per cell: excess = mean(X·(1−Y)) − mean(Y·(1−Y'))
    where X=biased sample, Y=first unbiased, Y'=second unbiased.

    Then recover Δ̄ by dividing by (1−q̄).

    If second_unbiased is not available for a cell, falls back to using
    q̄·(1−q̄) as the null pro-bias flip rate.

    For are_you_sure: returns biased_bmr unchanged (paper convention).
    """
    group_cols = [
        "model", "training_type", "model_family", "prompt_style",
        "dataset", "bias_type", "seed",
    ]
    rows = []
    for key, g in bir_df.groupby(group_cols, dropna=False):
        bt = key[group_cols.index("bias_type")]
        mf = key[group_cols.index("model_family")]
        ps = key[group_cols.index("prompt_style")]
        ds = key[group_cols.index("dataset")]

        if bt == "are_you_sure":
            biased_valid = g["biased_bmr"].dropna()
            if len(biased_valid) == 0:
                continue
            mean_val = float(biased_valid.mean())
            n = int(len(biased_valid))
            var = biased_valid.var(ddof=1) if n > 1 else 0.0
            se = float(np.sqrt(var / n)) if n > 1 else 0.0
            rec = {col: val for col, val in zip(group_cols, key)}
            rec.update({"bir": mean_val, "bir_se": se, "n_valid": n,
                        "excess_raw": mean_val, "q_bar": 0.0})
            rows.append(rec)
            continue

        valid = g[["hash", "biased_bmr", "unbiased_bmr"]].dropna()
        if len(valid) == 0:
            continue

        X = valid["biased_bmr"].values
        Y = valid["unbiased_bmr"].values
        n = len(valid)
        q_bar = float(Y.mean())

        # Observed pro-bias flip rate: X=1 AND Y=0
        obs_pro = float((X * (1 - Y)).mean())

        # Null pro-bias flip rate: from second unbiased draw if available
        h2 = (second_unbiased or {}).get((mf, ps, ds))
        if h2:
            Y_prime = np.array([h2.get(h, np.nan) for h in valid["hash"]])
            mask = ~np.isnan(Y_prime)
            if mask.sum() > 0:
                null_pro = float((Y[mask] * (1 - Y_prime[mask])).mean())
            else:
                null_pro = q_bar * (1 - q_bar)
        else:
            null_pro = q_bar * (1 - q_bar)

        excess = obs_pro - null_pro  # ≈ Δ·(1−q)

        # Recover Δ by dividing by (1−q̄), clamped away from 0
        one_minus_q = max(1 - q_bar, 0.05)
        delta_hat = excess / one_minus_q

        # SE: propagate from binomial SE of obs_pro and null_pro
        se_obs = float(np.sqrt(obs_pro * (1 - obs_pro) / n)) if n > 1 else 0.0
        se_null = float(np.sqrt(null_pro * (1 - null_pro) / n)) if n > 1 else 0.0
        se_excess = float(np.sqrt(se_obs**2 + se_null**2))
        se_delta = se_excess / one_minus_q

        rec = {col: val for col, val in zip(group_cols, key)}
        rec.update({"bir": delta_hat, "bir_se": se_delta, "n_valid": n,
                    "excess_raw": excess, "q_bar": q_bar})
        rows.append(rec)

    return pd.DataFrame(rows)


def collapse_to_pro_bias_bir(
    bir_df: pd.DataFrame,
    second_unbiased: dict | None = None,
    double_filter: bool = False,
) -> pd.DataFrame:
    """Pro-bias split: filter to questions where unbiased sample gave non-biased answer,
    then report biased_bmr as the sycophancy rate.

    If `double_filter=True` AND `second_unbiased` is provided, requires BOTH
    unbiased draws to give non-biased answer. Otherwise uses single-draw filter.

    For are_you_sure: returns biased_bmr unchanged (paper convention, unbiased baseline = 0).
    """
    group_cols = [
        "model", "training_type", "model_family", "prompt_style",
        "dataset", "bias_type", "seed",
    ]
    rows = []
    for key, g in bir_df.groupby(group_cols, dropna=False):
        bt = key[group_cols.index("bias_type")]
        mf = key[group_cols.index("model_family")]
        ps = key[group_cols.index("prompt_style")]
        ds = key[group_cols.index("dataset")]

        if bt == "are_you_sure":
            biased_valid = g["biased_bmr"].dropna()
            if len(biased_valid) == 0:
                continue
            mean = float(biased_valid.mean())
            n = int(len(biased_valid))
            var = biased_valid.var(ddof=1) if n > 1 else 0.0
            se = float(np.sqrt(var / n)) if n > 1 else 0.0
        else:
            keep = g["unbiased_bmr"] == 0
            if double_filter and second_unbiased is not None:
                h2 = second_unbiased.get((mf, ps, ds))
                if h2:
                    def _second_ok(hash_id):
                        b2 = h2.get(hash_id)
                        return b2 is not None and b2 == 0
                    keep = keep & g["hash"].map(_second_ok)
            sub = g[keep]
            biased_valid = sub["biased_bmr"].dropna()
            n = int(len(biased_valid))
            if n == 0:
                continue
            mean = float(biased_valid.mean())
            var = biased_valid.var(ddof=1) if n > 1 else 0.0
            se = float(np.sqrt(var / n)) if n > 1 else 0.0

        rec = {col: val for col, val in zip(group_cols, key)}
        rec.update({"bir": mean, "bir_se": se, "n_valid": n})
        rows.append(rec)

    return pd.DataFrame(rows)


def _aggregate_population(df, metric, group_cols):
    """Aggregate population-level BIR across model-instances (e.g. seeds).

    Mirrors aggregate_samples' output schema: returns columns `{metric}_mean`,
    `{metric}_stderr`, `n_valid`.
    """
    records = []
    for key, g in df.groupby(group_cols, dropna=False):
        # Weighted mean across rows (one per model-instance/seed).
        vals = g[metric].dropna()
        if len(vals) == 0:
            continue
        mean = float(vals.mean())
        # Use the SE already computed per row; combine via mean of variances / n rows.
        ses = g["bir_se"].dropna()
        if len(ses) > 0:
            se = float(np.sqrt((ses ** 2).sum()) / len(ses))
        else:
            se = 0.0
        rec = {col: val for col, val in zip(group_cols, key)}
        rec[f"{metric}_mean"] = mean
        rec[f"{metric}_stderr"] = se
        rec["n_valid"] = int(g["n_valid"].sum())
        records.append(rec)
    return pd.DataFrame(records)


def compute_noise_floor(noise_dir: str):
    """Load per-question unbiased BMR from a second unbiased-only run.

    Returns a dict keyed by (model_family, prompt_style, dataset) -> {hash: bmr}.
    For use with apply_noise_floor, which joins against the first run in bir_df.
    """
    from collections import defaultdict

    from sycophancy_eval_inspect.eval_log_loader import extract_bias_metrics, iter_eval_samples

    second_unbiased = defaultdict(dict)  # (model_family, prompt_style, dataset) -> {hash: bmr}
    for ctx in iter_eval_samples([noise_dir]):
        if ctx.model_family is None or ctx.variant != "unbiased":
            continue
        metrics = extract_bias_metrics(ctx.sample)
        if metrics is None or not metrics.strict_parsed:
            continue
        second_unbiased[(ctx.model_family, ctx.prompt_style, ctx.dataset)][ctx.sample.id] = metrics.bmr

    return second_unbiased


def apply_noise_floor(bir_df: pd.DataFrame, second_unbiased) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Subtract per-cell noise floor from per-question BIR.

    Strategy: per (model_family, prompt_style, dataset), compute noise_floor =
    mean over questions of |unbiased_draw1 - unbiased_draw2|. Then adjusted BIR
    at cell level = max(mean(|biased - unbiased_draw1|) - noise_floor, 0).

    Because aggregation is cell-level, we return both a noise-floor summary and
    a BIR df with a new `bir_floor_adj` column (still per-question, unchanged),
    plus a summary DataFrame with adjusted means.
    """
    # Noise floor per (model_family, prompt_style, dataset)
    floors = []
    for (mf, ps, ds), hash_to_bmr2 in second_unbiased.items():
        rows = bir_df[(bir_df["model_family"] == mf) &
                      (bir_df["prompt_style"] == ps) &
                      (bir_df["dataset"] == ds)]
        if rows.empty:
            continue
        # Per question: take any biased-row's unbiased_bmr (same across biases for same hash)
        first_per_hash = rows.drop_duplicates(subset=["hash"])[["hash", "unbiased_bmr"]]
        diffs = []
        for _, r in first_per_hash.iterrows():
            b2 = hash_to_bmr2.get(r["hash"])
            if b2 is None or np.isnan(r["unbiased_bmr"]) or np.isnan(b2):
                continue
            diffs.append(abs(r["unbiased_bmr"] - b2))
        if not diffs:
            continue
        diffs_arr = np.array(diffs, dtype=float)
        floor_mean = float(diffs_arr.mean())
        floor_se = float(np.sqrt(diffs_arr.var(ddof=1) / len(diffs_arr))) if len(diffs_arr) > 1 else 0.0
        floors.append({
            "model_family": mf,
            "prompt_style": ps,
            "dataset": ds,
            "noise_floor": floor_mean,
            "noise_floor_se": floor_se,
            "noise_n": len(diffs_arr),
        })
    floor_df = pd.DataFrame(floors)
    return bir_df, floor_df


def print_noise_floor(floor_df: pd.DataFrame) -> None:
    if floor_df.empty:
        print("No noise-floor data.")
        return
    print("\n" + "=" * 68)
    print("  NOISE FLOOR  (mean per-question |unbiased_1 − unbiased_2|)")
    print("=" * 68)
    print(f"{'Model':<14}{'Dataset':<12}{'Style':<8}{'Floor':>12}{'SE':>8}{'n':>6}")
    print("-" * 68)
    for _, r in floor_df.sort_values(["model_family", "dataset"]).iterrows():
        print(f"{r['model_family']:<14}{r['dataset']:<12}{r['prompt_style']:<8}"
              f"{r['noise_floor']:>11.1%}{r['noise_floor_se']:>7.1%}{r['noise_n']:>6d}")


def print_bir_tables_with_floor(bir_df: pd.DataFrame, floor_df: pd.DataFrame, metric="bir"):
    """Print per-question BIR tables subtracting per-cell noise floor.

    SE is inflated by floor_se via quadrature, so the ± values reflect combined
    uncertainty from the BIR cell and the noise-floor estimate.
    """
    floor_se_lookup = {
        (r["model_family"], r["prompt_style"], r["dataset"]): r["noise_floor_se"]
        for _, r in floor_df.iterrows()
    }
    agg_fn = _make_floor_aware_aggregator(floor_se_lookup)
    agg = agg_fn(bir_df, metric, ["dataset", "bias_type", "model_family", "prompt_style"])
    mean_col = f"{metric}_mean"
    err_col = f"{metric}_stderr"

    floor_lookup = {}
    for _, r in floor_df.iterrows():
        floor_lookup[(r["model_family"], r["prompt_style"], r["dataset"])] = r["noise_floor"]

    models = [m for m in MODEL_LABELS if m in bir_df["model_family"].unique()]
    datasets = sorted(bir_df["dataset"].unique())
    bias_types = sorted(bir_df["bias_type"].unique(),
                        key=lambda b: BIAS_DISPLAY_NAMES.get(b, b))
    bias_labels = [BIAS_DISPLAY_NAMES.get(b, b) for b in bias_types]

    for model in models:
        m_agg = agg[agg["model_family"] == model]
        print(f"\n{'=' * 80}")
        print(f"  {MODEL_LABELS[model]} — Noise-adjusted BIR")
        print(f"  (per-question BIR minus noise floor; clamped at 0; are_you_sure unchanged)")
        print(f"{'=' * 80}")
        col_w = 14
        header = f"{'Dataset':<12}" + "".join(f"{bl:>{col_w}}" for bl in bias_labels) + f"{'Floor':>10}"
        print(header)
        print("-" * len(header))
        for ds in datasets:
            parts = [f"{ds:<12}"]
            ds_rows = m_agg[m_agg["dataset"] == ds]
            # noise floor: prefer whatever prompt_style appears
            ps_options = ds_rows["prompt_style"].unique() if not ds_rows.empty else []
            floor = None
            for ps in ps_options:
                f = floor_lookup.get((model, ps, ds))
                if f is not None:
                    floor = f
                    break
            for bt in bias_types:
                r = m_agg[(m_agg["dataset"] == ds) & (m_agg["bias_type"] == bt)]
                if len(r) == 1 and r["n_valid"].iloc[0] > 0:
                    val = r[mean_col].iloc[0]
                    se = r[err_col].iloc[0]
                    if bt == "are_you_sure" or floor is None:
                        adj = val
                    else:
                        adj = max(val - floor, 0.0)
                    parts.append(f"{adj:>+6.1%}±{se:.1%}".rjust(col_w))
                else:
                    parts.append(f"{'—':>{col_w}}")
            parts.append(f"{floor:>9.1%}" if floor is not None else f"{'—':>10}")
            print("".join(parts))


def print_pro_excess_tables(excess_df: pd.DataFrame) -> None:
    """Print pro-bias excess table: Δ̂ = excess / (1−q̄) per cell."""
    if excess_df.empty:
        return
    models = [m for m in MODEL_LABELS if m in excess_df["model_family"].unique()]
    datasets = sorted(excess_df["dataset"].unique())
    bias_types = sorted(excess_df["bias_type"].unique(),
                        key=lambda b: BIAS_DISPLAY_NAMES.get(b, b))
    bias_labels = [BIAS_DISPLAY_NAMES.get(b, b) for b in bias_types]
    for model in models:
        m_df = excess_df[excess_df["model_family"] == model]
        print(f"\n{'=' * 80}")
        print(f"  {MODEL_LABELS[model]} — Pro-bias excess Δ̂ = excess/(1−q̄)")
        print(f"  (directional flip decomposition; recovers population Δ via pro-bias channel)")
        print(f"{'=' * 80}")
        col_w = 18
        header = f"{'Dataset':<12}" + "".join(f"{bl:>{col_w}}" for bl in bias_labels)
        print(header)
        print("-" * len(header))
        for ds in datasets:
            parts = [f"{ds:<12}"]
            for bt in bias_types:
                r = m_df[(m_df["dataset"] == ds) & (m_df["bias_type"] == bt)]
                if len(r) == 1 and r["n_valid"].iloc[0] > 0:
                    val = float(r["bir"].iloc[0])
                    se = float(r["bir_se"].iloc[0])
                    qb = float(r["q_bar"].iloc[0])
                    parts.append(f"{val:>+5.1%}±{se:.1%} q̄={qb:.2f}".rjust(col_w))
                else:
                    parts.append(f"{'—':>{col_w}}")
            print("".join(parts))


def print_pro_bias_tables(pro_df: pd.DataFrame) -> None:
    """Print sycophancy-rate table (datasets × biases) for the pro-bias split."""
    if pro_df.empty:
        return
    models = [m for m in MODEL_LABELS if m in pro_df["model_family"].unique()]
    datasets = sorted(pro_df["dataset"].unique())
    bias_types = sorted(pro_df["bias_type"].unique(),
                        key=lambda b: BIAS_DISPLAY_NAMES.get(b, b))
    bias_labels = [BIAS_DISPLAY_NAMES.get(b, b) for b in bias_types]
    for model in models:
        m_df = pro_df[pro_df["model_family"] == model]
        print(f"\n{'=' * 80}")
        print(f"  {MODEL_LABELS[model]} — Pro-bias sycophancy rate")
        print(f"  (biased_bmr on questions where unbiased sample gave non-biased answer;")
        print(f"   HLE uses two unbiased draws for stricter q̂=0 filter)")
        print(f"{'=' * 80}")
        col_w = 14
        header = f"{'Dataset':<12}" + "".join(f"{bl:>{col_w}}" for bl in bias_labels) + f"{'Avg':>{col_w}}"
        print(header)
        print("-" * len(header))
        col_avgs = {bt: [] for bt in bias_types}
        for ds in datasets:
            parts = [f"{ds:<12}"]
            row_vals = []
            for bt in bias_types:
                r = m_df[(m_df["dataset"] == ds) & (m_df["bias_type"] == bt)]
                if len(r) == 1 and r["n_valid"].iloc[0] > 0:
                    val = float(r["bir"].iloc[0])
                    se = float(r["bir_se"].iloc[0])
                    n = int(r["n_valid"].iloc[0])
                    parts.append(f"{val:>+5.1%}±{se:.1%}(n={n})".rjust(col_w))
                    row_vals.append(val)
                    col_avgs[bt].append(val)
                else:
                    parts.append(f"{'—':>{col_w}}")
            if row_vals:
                parts.append(f"{np.mean(row_vals):>+6.1%}".rjust(col_w))
            else:
                parts.append(f"{'—':>{col_w}}")
            print("".join(parts))


def _make_floor_aware_aggregator(floor_lookup_with_se):
    """Return an aggregate_samples wrapper that inflates SE by per-cell floor SE.

    floor_lookup_with_se: {(model_family, prompt_style, dataset): floor_se}
    Combines in quadrature: sqrt(cell_se² + floor_se²). Only applies to rows whose
    bias_type is not are_you_sure (unchanged for ays).
    """
    from sycophancy_eval_inspect.visualize_results import aggregate_samples as _base_agg

    def _agg(df, metric, group_cols):
        out = _base_agg(df, metric, group_cols)
        err_col = f"{metric}_stderr"
        if err_col not in out.columns:
            return out
        def _inflate(row):
            if row.get("bias_type") == "are_you_sure":
                return row[err_col]
            key = (row.get("model_family"), row.get("prompt_style"), row.get("dataset"))
            f_se = floor_lookup_with_se.get(key)
            if f_se is None:
                return row[err_col]
            return float(np.sqrt(row[err_col] ** 2 + f_se ** 2))
        out[err_col] = out.apply(_inflate, axis=1)
        return out

    return _agg


FRONTIER_DATASETS = ["hle", "gpqa", "mmlu_pro"]

FRONTIER_LABELS = {
    "hle": "HLE",
    "gpqa": "GPQA",
    "mmlu_pro": "MMLU-Pro",
}


def plot_methods_combined_for_dataset(
    dataset_name: str,
    method_groups: list[tuple[str, list[tuple[str, pd.DataFrame, str, str | None]]]],
    output_dir: str,
    file_stem: str,
    floor_se_lookup: dict | None = None,
):
    """Plot BIR methods (organized into labeled groups) for one dataset.

    method_groups: list of (group_label, methods) where methods is
        [(label, df, color, tag), ...]. Groups are separated visually.
    """
    from sycophancy_eval_inspect.visualize_results import aggregate_samples as std_agg

    all_methods = [(label, df, color, tag)
                   for _, methods in method_groups for label, df, color, tag in methods]
    if not all_methods:
        return

    first_df = all_methods[0][1]
    models = [m for m in MODEL_COLORS if m in first_df["model_family"].unique()]
    if not models:
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds_rows = first_df[first_df["dataset"] == dataset_name]
    if ds_rows.empty:
        print(f"skip {file_stem}: no rows for dataset={dataset_name}")
        return
    all_bias_types = sorted(ds_rows["bias_type"].unique(),
                            key=lambda b: BIAS_DISPLAY_NAMES.get(b, b))
    bias_labels = [BIAS_DISPLAY_NAMES.get(b, b) for b in all_bias_types]

    n_total = len(all_methods)
    bar_h = min(0.2, 1.0 / (n_total + 1))
    group_gap = 0.4

    fig, axes = plt.subplots(1, len(models), figsize=(10 * len(models), max(7, n_total * 1.2)))
    if len(models) == 1:
        axes = [axes]

    ds_label = FRONTIER_LABELS.get(dataset_name, dataset_name.upper())

    for ax_idx, model in enumerate(models):
        ax = axes[ax_idx]
        n_biases = len(all_bias_types)
        m_idx = 0

        for g_idx, (group_label, methods) in enumerate(method_groups):
            for label, df, color, tag in methods:
                sub_df = df[(df["dataset"] == dataset_name) & (df["model_family"] == model)]
                if sub_df.empty:
                    means = [np.nan] * n_biases
                    errs = [0] * n_biases
                else:
                    if "split" in label.lower() or "Population" in label or "excess" in label.lower():
                        agg_fn = _aggregate_population
                    elif tag == "_floor_aware" and floor_se_lookup:
                        agg_fn = _make_floor_aware_aggregator(floor_se_lookup)
                    else:
                        agg_fn = std_agg
                    if tag == "_floor_aware" and floor_se_lookup:
                        agg = agg_fn(sub_df, "bir",
                                     ["bias_type", "model_family", "prompt_style", "dataset"])
                    else:
                        agg = agg_fn(sub_df, "bir", ["bias_type"])
                    means, errs = [], []
                    for bt in all_bias_types:
                        r = agg[agg["bias_type"] == bt]
                        if len(r) == 1:
                            means.append(r["bir_mean"].iloc[0])
                            errs.append(r["bir_stderr"].iloc[0] * 1.96)
                        else:
                            means.append(np.nan)
                            errs.append(0)

                y_pos = np.arange(n_biases) * (n_total * bar_h + group_gap) + m_idx * bar_h
                ax.barh(y_pos, means, height=bar_h, color=color, alpha=0.85,
                        edgecolor="black", linewidth=0.5, label=label)
                ax.errorbar(means, y_pos, xerr=errs, fmt="none",
                            ecolor="black", capsize=2, alpha=0.6, linewidth=0.8)
                m_idx += 1

        center_y = (np.arange(n_biases) * (n_total * bar_h + group_gap)
                    + (n_total - 1) * bar_h / 2)
        ax.set_yticks(center_y)
        ax.set_yticklabels(bias_labels, fontsize=9)
        ax.set_title(f"{MODEL_LABELS[model]} — {ds_label}", fontsize=12, fontweight="bold")
        ax.axvline(x=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        for pct in [0.05, 0.10, 0.15]:
            ax.axvline(x=pct, color="gray", linestyle=":", linewidth=0.8, alpha=0.4)
        ax.invert_yaxis()
        ax.set_xlabel("BIR (95% CI)")
        if ax_idx == 0:
            ax.legend(loc="lower right", fontsize=8)

    fig.suptitle(f"{ds_label}: BIR methods comparison", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = out / f"{file_stem}_{dataset_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_hle_methods_combined(method_groups, output_dir, floor_se_lookup=None):
    """Back-compat wrapper: loops over frontier datasets."""
    for ds in FRONTIER_DATASETS:
        plot_methods_combined_for_dataset(
            ds, method_groups, output_dir, "methods_comparison", floor_se_lookup,
        )


def plot_ba_frontier(bir_df: pd.DataFrame, output_dir: str,
                     metric: str = "bias_acknowledged"):
    """Single-panel BA bar chart per frontier dataset (hle, gpqa, mmlu_pro)."""
    if metric not in bir_df.columns:
        print(f"metric={metric} missing; skip BA plots.")
        return
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ds in FRONTIER_DATASETS:
        ds_df = bir_df[bir_df["dataset"] == ds]
        if ds_df.empty:
            continue
        models = [m for m in MODEL_COLORS if m in ds_df["model_family"].unique()]
        fig, ax = plt.subplots(figsize=(9, 6))
        label = FRONTIER_LABELS.get(ds, ds.upper())
        _plot_dataset_ax(ax, ds_df, f"{label} — BA", metric, models,
                         show_legend=True, show_n=True)
        ax.set_xlabel("Bias Acknowledged rate (95% CI)", fontsize=9)
        plt.tight_layout()
        path = out / f"ba_{ds}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved {path}")


# Back-compat alias
plot_ba_hle = plot_ba_frontier


def add_gpqa_diamond(bir_df: pd.DataFrame) -> pd.DataFrame:
    """Duplicate gpqa rows that belong to the gpqa_diamond subset under a new
    dataset label ``gpqa_diamond``; leave the full ``gpqa`` rows in place.

    Matches on question text (first 200 chars) against the HF gpqa_diamond split.
    """
    import json
    from datasets import load_dataset

    diamond = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    diamond_questions = {r["Question"][:200] for r in diamond}

    keep_hashes = set()
    dump_path = Path("dataset_dumps/test/suggested_answer/gpqa_suggested_answer.jsonl")
    with open(dump_path) as f:
        for line in f:
            d = json.loads(line)
            if d["original_question"][:200] in diamond_questions:
                keep_hashes.add(d["original_question_hash"])
    print(f"[gpqa_diamond] labeling {len(keep_hashes)} hashes as diamond subset")

    diamond_rows = bir_df[(bir_df["dataset"] == "gpqa") & bir_df["hash"].isin(keep_hashes)].copy()
    diamond_rows["dataset"] = "gpqa_diamond"
    return pd.concat([bir_df, diamond_rows], ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bir-method",
        choices=["per_question", "population"],
        default="per_question",
        help="per_question: mean of abs(biased - unbiased) per sample (paper convention). "
             "population: |mean(biased) - mean(unbiased)| on raw rates (noise cancels).",
    )
    parser.add_argument(
        "--noise-floor-dir",
        default=None,
        help="Path to a second unbiased-only run (different seed). If set, prints "
             "a noise-floor-adjusted BIR table alongside the standard output.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for plots. Defaults to plots/model_comparison_200[_population].",
    )
    parser.add_argument(
        "--gpqa-subset",
        choices=["main", "diamond"],
        default="main",
        help="Restrict gpqa rows to the given subset. Relabels gpqa→gpqa_diamond in outputs.",
    )
    args = parser.parse_args()
    global aggregate_samples

    log_dirs = [
        "sycophancy_eval_inspect/logs/base_200",
        "sycophancy_eval_inspect/logs/base_200_ays",
    ]
    print(f"Loading eval data (bir_method={args.bir_method})...")
    bir_df = compute_per_question_bir(log_dirs)
    # This script standardizes on a `bir` working column. The canonical per-question
    # metric is `total_bsr` (the redundant `bir`/`lenient_bir` aliases were removed from
    # compute_per_question_bir), so map it to the working name here. The population path
    # below produces its own `bir` column via collapse_to_population_bir.
    bir_df = bir_df.rename(columns={"total_bsr": "bir", "lenient_total_bsr": "lenient_bir"})
    print(f"Loaded {len(bir_df)} rows, models: {bir_df['model_family'].unique()}, "
          f"datasets: {bir_df['dataset'].unique()}")

    if args.gpqa_subset == "diamond":
        bir_df = add_gpqa_diamond(bir_df)
        global FRONTIER_DATASETS
        FRONTIER_DATASETS = ["hle", "gpqa", "gpqa_diamond", "mmlu_pro"]
        FRONTIER_LABELS["gpqa_diamond"] = "GPQA-Diamond"

    if args.bir_method == "population":
        bir_df = collapse_to_population_bir(bir_df)
        output_dir = args.output_dir or "plots/model_comparison_200_population"
        aggregate_samples = _aggregate_population
    else:
        output_dir = args.output_dir or "plots/model_comparison_200"

    print_bir_tables(bir_df, metric="bir")

    if args.noise_floor_dir and args.bir_method == "per_question":
        print("\nLoading noise-floor unbiased runs...")
        second_unbiased = compute_noise_floor(args.noise_floor_dir)
        bir_df, floor_df = apply_noise_floor(bir_df, second_unbiased)
        print_noise_floor(floor_df)
        print_bir_tables_with_floor(bir_df, floor_df, metric="bir")
        # Build an adjusted copy of bir_df for plotting: subtract per-cell floor
        # from per-row `bir` (clamp at 0), leave are_you_sure untouched.
        adj_df = bir_df.copy()
        floor_lookup = {
            (r["model_family"], r["prompt_style"], r["dataset"]): r["noise_floor"]
            for _, r in floor_df.iterrows()
        }
        def _adjust_row(r):
            if r["bias_type"] == "are_you_sure":
                return r["bir"]
            f = floor_lookup.get((r["model_family"], r["prompt_style"], r["dataset"]))
            if f is None or np.isnan(r["bir"]):
                return r["bir"]
            return max(r["bir"] - f, 0.0)
        adj_df["bir"] = adj_df.apply(_adjust_row, axis=1)
        adj_output_dir = f"{output_dir}_noise_adjusted"

        # Build floor-SE lookup for error-bar inflation
        floor_se_lookup = {
            (r["model_family"], r["prompt_style"], r["dataset"]): r["noise_floor_se"]
            for _, r in floor_df.iterrows()
        }
        # Swap in a floor-aware aggregator for the adjusted standalone plot.
        prev_aggregate = aggregate_samples
        aggregate_samples = _make_floor_aware_aggregator(floor_se_lookup)
        try:
            plot_bir_by_dataset(adj_df, metric="bir", output_dir=adj_output_dir)
        finally:
            aggregate_samples = prev_aggregate

        # Combined HLE view: all methods grouped
        pop_df = collapse_to_population_bir(bir_df)
        pro_single_df = collapse_to_pro_bias_bir(bir_df, double_filter=False)
        pro_double_df = collapse_to_pro_bias_bir(bir_df, second_unbiased=second_unbiased, double_filter=True)
        excess_df = collapse_to_pro_bias_excess(bir_df, second_unbiased=second_unbiased)
        print_pro_bias_tables(pro_single_df)
        print_pro_bias_tables(pro_double_df)
        print_pro_excess_tables(excess_df)

        # Filter bir_df to Y=0 subset for per-question and population on that subset
        bir_df_y0 = bir_df[bir_df["unbiased_bmr"] == 0].copy()
        pop_y0_df = collapse_to_population_bir(bir_df_y0)

        # Split by whether the metric captures pro+anti-bias together or pro only.
        # Per-question absolute difference counts any flip in either direction.
        both_dirs_groups = [
            ("Per-question absolute", [
                ("Per-question BIR", bir_df, "#e41a1c", None),
                ("Noise-adjusted", adj_df, "#4daf4a", "_floor_aware"),
            ]),
        ]
        plot_hle_methods_combined(
            both_dirs_groups,
            output_dir="plots/model_comparison_200_hle_methods_both_dirs",
            floor_se_lookup=floor_se_lookup,
        )

        # Pro-bias-only methods: population-level (net direction with abs at cell level,
        # anti cancels pro within a cell) and Y=0-conditioned methods (pro direction only).
        pro_only_groups = [
            ("Population (net)", [
                ("Population", pop_df, "#377eb8", None),
                ("Pro-bias excess Δ̂", excess_df, "#ff7f00", None),
            ]),
            ("Y=0 subset", [
                ("Per-question|Y=0", bir_df_y0, "#e41a1c", None),
                ("Population|Y=0", pop_y0_df, "#377eb8", None),
                ("Split Y=0", pro_single_df, "#984ea3", None),
                ("Split Y=Y'=0", pro_double_df, "#a65628", None),
            ]),
        ]
        plot_hle_methods_combined(
            pro_only_groups,
            output_dir="plots/model_comparison_200_hle_methods_pro_only",
            floor_se_lookup=floor_se_lookup,
        )

        # Also retain the combined view for reference
        method_groups = [
            ("Measures both directions (pro + anti)", [
                ("Per-question BIR", bir_df, "#e41a1c", None),
                ("Noise-adjusted", adj_df, "#4daf4a", "_floor_aware"),
            ]),
            ("Measures pro-bias only", [
                ("Population", pop_df, "#377eb8", None),
                ("Pro-bias excess Δ̂", excess_df, "#ff7f00", None),
                ("Per-question|Y=0", bir_df_y0, "#e41a1c", None),
                ("Population|Y=0", pop_y0_df, "#377eb8", None),
                ("Split Y=0", pro_single_df, "#984ea3", None),
                ("Split Y=Y'=0", pro_double_df, "#a65628", None),
            ]),
        ]
        plot_hle_methods_combined(method_groups,
                                  output_dir="plots/model_comparison_200_hle_methods",
                                  floor_se_lookup=floor_se_lookup)

    # Three flavors of all-datasets view.
    # Pro-bias (Y=0): numerator X=1 AND Y=0, denominator Y=0. "Pushed toward bias".
    pro_df = bir_df[bir_df["unbiased_bmr"] == 0].copy()
    pro_df["bir"] = pro_df["biased_bmr"]
    # Anti-bias (Y=1): numerator X=0 AND Y=1, denominator Y=1. "Pushed away from bias".
    # AYS has Y≡0 so it's excluded by construction.
    anti_df = bir_df[bir_df["unbiased_bmr"] == 1].copy()
    anti_df["bir"] = 1 - anti_df["biased_bmr"]
    # Combined (per-question BIR, any-direction flip): already in bir_df.

    plot_bir_by_dataset(pro_df, metric="bir", output_dir=output_dir,
                        title_suffix=" — Pro-bias (Y=0)",
                        filename="bir_all_datasets_pro.png",
                        xlabel="P(biased→bias | unbiased≠bias)")
    plot_bir_by_dataset(anti_df, metric="bir", output_dir=output_dir,
                        title_suffix=" — Anti-bias (Y=1)",
                        filename="bir_all_datasets_anti.png",
                        xlabel="P(biased≠bias | unbiased→bias)")
    plot_bir_by_dataset(bir_df, metric="bir", output_dir=output_dir,
                        title_suffix=" — Both directions",
                        filename="bir_all_datasets_combined.png",
                        xlabel="P(biased ≠ unbiased)")

    if args.bir_method == "per_question":
        plot_ba_frontier(bir_df, output_dir=output_dir)

        # Noise-free Split Y=0 plot for every frontier dataset. Doesn't depend on
        # --noise-floor-dir — uses a single unbiased draw per (model, dataset, bias).
        pro_single_df = collapse_to_pro_bias_bir(bir_df, double_filter=False)
        pop_df = collapse_to_population_bir(bir_df)
        bir_df_y0 = bir_df[bir_df["unbiased_bmr"] == 0].copy()
        pop_y0_df = collapse_to_population_bir(bir_df_y0)
        print_pro_bias_tables(pro_single_df)

        split_groups = [
            ("Both directions", [
                ("Per-question BIR", bir_df, "#e41a1c", None),
                ("Population", pop_df, "#377eb8", None),
            ]),
            ("Pro-bias only (Y=0)", [
                ("Per-question|Y=0", bir_df_y0, "#e41a1c", None),
                ("Population|Y=0", pop_y0_df, "#377eb8", None),
                ("Split Y=0", pro_single_df, "#984ea3", None),
            ]),
        ]
        for ds in FRONTIER_DATASETS:
            plot_methods_combined_for_dataset(
                ds, split_groups,
                output_dir="plots/model_comparison_200_split_y0",
                file_stem="split_y0",
            )
    print("Done!")


if __name__ == "__main__":
    main()
