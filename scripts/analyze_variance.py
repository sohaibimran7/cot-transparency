#!/usr/bin/env python
"""
Variance diagnostics for BIR estimates across repeated evals.

Compares same-checkpoint evaluations (A vs B) to determine whether
observed BIR differences are within expected noise or systematic.

Analyses:
  1. Split-k: divide each model's samples into k groups → within-model variance
  2. Cross-model subsets: same question split across base/A/B
  3. Paired bootstrap: p-value for mean(BIR_A - BIR_B)
  4. Variance decomposition: between-question vs between-run components
  5. Agreement matrix: per-question BIR concordance heatmap

Usage:
    python scripts/analyze_variance.py \
        --log-dir artifacts/eval_suites/cot_300samples/eval_logs \
        --output-dir artifacts/eval_suites/variance_diagnostics/plots
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sycophancy_eval_inspect.visualize_results import (
    TRAINING_TYPE_NAMES,
    compute_per_question_bir,
    filter_to_common_questions,
)

# ── Configuration ────────────────────────────────────────────────────────────

MODELS = ["base", "rlct_s50_a", "rlct_s50_b"]
MODEL_COLORS = {"base": "#1f77b4", "rlct_s50_a": "#ff7f0e", "rlct_s50_b": "#2ca02c"}
MODEL_LABELS = {"base": "Base", "rlct_s50_a": "RLCT A", "rlct_s50_b": "RLCT B"}

BIAS_DISPLAY = {
    "distractor_argument": "Argument",
    "distractor_fact": "Fact",
    "spurious_few_shot_squares": "Squares",
    "suggested_answer": "Sugg. Answer",
    "wrong_few_shot": "Wrong FS",
}


def _bt_label(bt: str) -> str:
    return BIAS_DISPLAY.get(bt, bt.replace("_", " ").title())


# ── Data loading ─────────────────────────────────────────────────────────────


def load_data(log_dir: str, metric: str = "lenient_bir") -> pd.DataFrame:
    """Load per-question BIR, filter to common questions and target models."""
    bir_df = compute_per_question_bir(log_dirs=[log_dir])
    bir_df = bir_df[bir_df["training_type"].isin(MODELS)]
    bir_df = filter_to_common_questions(bir_df, metric_col=metric)
    return bir_df


# ── 1. Split-k within model ─────────────────────────────────────────────────


def _stratified_split(df: pd.DataFrame, k: int, rng) -> list[set]:
    """Split hashes into k groups, stratified by dataset."""
    groups = [[] for _ in range(k)]
    for ds in df["dataset"].unique():
        ds_hashes = df[df["dataset"] == ds]["hash"].unique().copy()
        rng.shuffle(ds_hashes)
        for i, chunk in enumerate(np.array_split(ds_hashes, k)):
            groups[i].extend(chunk)
    return [set(g) for g in groups]


def plot_split_k(
    bir_df: pd.DataFrame,
    metric: str,
    k_values: list[int],
    output_dir: Path,
    seed: int = 42,
    stratify: bool = False,
    ci_multiplier: float = 1.0,
):
    """For each model, split its questions into k random groups.
    Plot BIR +/- (ci_multiplier * SE) per group.

    stratify=True balances dataset composition across groups.
    ci_multiplier=2.0 gives approximate 95% CI."""
    rng = np.random.default_rng(seed)
    bias_types = sorted(bir_df["bias_type"].unique())
    ci_label = f"\u00b1{ci_multiplier:.0f}" if ci_multiplier == int(ci_multiplier) else f"\u00b1{ci_multiplier}"
    suffix = ""
    if stratify:
        suffix += "_stratified"
    if ci_multiplier != 1.0:
        suffix += f"_{ci_multiplier:.0f}se"

    for k in k_values:
        fig, axes = plt.subplots(
            1, len(bias_types), figsize=(4.5 * len(bias_types), 5), squeeze=False
        )
        n_per_group = bir_df.groupby(["bias_type", "training_type"])["hash"].nunique().min() // k
        strat_txt = ", stratified by dataset" if stratify else ""
        fig.suptitle(
            f"Split-{k}: {k} random subsets (n\u2248{n_per_group}){strat_txt}, "
            f"error bars = {ci_label} SEM",
            fontsize=12,
        )

        for ax_idx, bt in enumerate(bias_types):
            ax = axes[0, ax_idx]
            bt_df = bir_df[bir_df["bias_type"] == bt]

            pos = 0
            tick_positions = []
            tick_labels = []
            model_centers = []

            for tt in MODELS:
                tt_df = bt_df[bt_df["training_type"] == tt]

                if stratify:
                    groups = _stratified_split(tt_df, k, rng)
                else:
                    hashes = tt_df["hash"].unique().copy()
                    rng.shuffle(hashes)
                    groups = [set(g) for g in np.array_split(hashes, k)]

                start_pos = pos
                for g_idx, g_hashes in enumerate(groups):
                    vals = (
                        tt_df[tt_df["hash"].isin(g_hashes)][metric]
                        .dropna()
                        .values
                    )
                    mean = vals.mean() * 100 if len(vals) else 0
                    se = (
                        (vals.std(ddof=1) / np.sqrt(len(vals))) * 100
                        if len(vals) > 1
                        else 0
                    )

                    ax.bar(
                        pos,
                        mean,
                        yerr=se * ci_multiplier,
                        capsize=3,
                        width=0.7,
                        color=MODEL_COLORS[tt],
                        alpha=0.7,
                        edgecolor="black",
                        linewidth=0.5,
                    )
                    tick_positions.append(pos)
                    tick_labels.append(f"#{g_idx + 1}")
                    pos += 1

                model_centers.append((start_pos + pos - 1) / 2)
                pos += 0.5  # gap between models

            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, fontsize=8)
            ax.set_ylabel("BIR %")
            ax.set_title(_bt_label(bt), fontsize=11)
            ax.axhline(y=0, color="grey", linestyle="--", linewidth=0.5)

            # Model labels below
            ylo = ax.get_ylim()[0]
            for i, tt in enumerate(MODELS):
                ax.text(
                    model_centers[i],
                    ylo - 0.08 * (ax.get_ylim()[1] - ylo),
                    MODEL_LABELS[tt],
                    ha="center",
                    fontsize=9,
                    fontweight="bold",
                    color=MODEL_COLORS[tt],
                    clip_on=False,
                )

        plt.tight_layout(rect=[0, 0.04, 1, 0.95])
        path = output_dir / f"split_{k}{suffix}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


# ── 2. Cross-model subsets ───────────────────────────────────────────────────


def plot_cross_model_subsets(
    bir_df: pd.DataFrame,
    metric: str,
    k: int,
    output_dir: Path,
    seed: int = 42,
):
    """Split questions into k groups (SAME split for all models).
    For each group, plot base/A/B side by side on the same questions."""
    rng = np.random.default_rng(seed)
    bias_types = sorted(bir_df["bias_type"].unique())

    fig, axes = plt.subplots(
        1, len(bias_types), figsize=(4.5 * len(bias_types), 5), squeeze=False
    )
    fig.suptitle(
        f"Cross-model: Same {k} question subsets, base/A/B side by side", fontsize=13
    )

    for ax_idx, bt in enumerate(bias_types):
        ax = axes[0, ax_idx]
        bt_df = bir_df[bir_df["bias_type"] == bt]

        # Common hashes present in all models
        hash_counts = bt_df.groupby("hash")["training_type"].nunique()
        common_hashes = hash_counts[hash_counts == len(MODELS)].index.values.copy()
        rng.shuffle(common_hashes)
        groups = np.array_split(common_hashes, k)

        width = 0.25
        x = np.arange(k)

        for m_idx, tt in enumerate(MODELS):
            means, ses = [], []
            for g_hashes in groups:
                vals = (
                    bt_df[
                        (bt_df["training_type"] == tt)
                        & (bt_df["hash"].isin(set(g_hashes)))
                    ][metric]
                    .dropna()
                    .values
                )
                means.append(vals.mean() * 100 if len(vals) else 0)
                ses.append(
                    (vals.std(ddof=1) / np.sqrt(len(vals))) * 100
                    if len(vals) > 1
                    else 0
                )

            ax.bar(
                x + m_idx * width,
                means,
                width,
                yerr=ses,
                capsize=3,
                color=MODEL_COLORS[tt],
                alpha=0.7,
                label=MODEL_LABELS[tt],
                edgecolor="black",
                linewidth=0.5,
            )

        ax.set_xticks(x + width)
        ax.set_xticklabels([f"Group {i + 1}" for i in range(k)], fontsize=9)
        ax.set_ylabel("BIR %")
        ax.set_title(_bt_label(bt), fontsize=11)
        ax.axhline(y=0, color="grey", linestyle="--", linewidth=0.5)
        if ax_idx == len(bias_types) - 1:
            ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = output_dir / f"cross_model_k{k}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 3. Paired bootstrap test ────────────────────────────────────────────────


def paired_bootstrap_test(
    bir_df: pd.DataFrame,
    metric: str,
    output_dir: Path,
    n_bootstrap: int = 5000,
    seed: int = 42,
):
    """Bootstrap CI for mean(BIR_A - BIR_B) using paired questions.
    Tests whether the A-B difference is statistically significant."""
    rng = np.random.default_rng(seed)
    bias_types = sorted(bir_df["bias_type"].unique())

    results = []
    fig, axes = plt.subplots(
        1, len(bias_types), figsize=(4.5 * len(bias_types), 4.5), squeeze=False
    )
    fig.suptitle(
        "Paired Bootstrap: Distribution of Mean(BIR_A \u2212 BIR_B)", fontsize=13
    )

    for ax_idx, bt in enumerate(bias_types):
        ax = axes[0, ax_idx]
        bt_df = bir_df[bir_df["bias_type"] == bt]

        a = bt_df[bt_df["training_type"] == "rlct_s50_a"].set_index("hash")[metric]
        b = bt_df[bt_df["training_type"] == "rlct_s50_b"].set_index("hash")[metric]
        common = a.index.intersection(b.index)

        diff = (a.loc[common] - b.loc[common]).dropna().values
        if len(diff) == 0:
            continue

        observed = diff.mean() * 100
        boot = (
            np.array(
                [
                    rng.choice(diff, len(diff), replace=True).mean()
                    for _ in range(n_bootstrap)
                ]
            )
            * 100
        )

        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
        p_val = 2 * min((boot >= 0).mean(), (boot <= 0).mean())

        results.append(
            {
                "bias_type": _bt_label(bt),
                "n": len(diff),
                "mean_diff%": round(observed, 2),
                "95%CI_lo": round(ci_lo, 2),
                "95%CI_hi": round(ci_hi, 2),
                "p_value": round(p_val, 4),
                "sig": (
                    "***"
                    if p_val < 0.001
                    else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                ),
            }
        )

        ax.hist(boot, bins=50, alpha=0.7, color="#5599cc", edgecolor="black", linewidth=0.3)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="H\u2080: no diff")
        ax.axvline(
            observed, color="orange", linewidth=1.5, label=f"Obs: {observed:+.1f}%"
        )
        ax.axvspan(ci_lo, ci_hi, alpha=0.15, color="grey", label="95% CI")
        ax.set_title(f"{_bt_label(bt)}\np={p_val:.3f}", fontsize=10)
        ax.set_xlabel("BIR_A \u2212 BIR_B (%)")
        ax.legend(fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = output_dir / "paired_bootstrap_ab.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    df = pd.DataFrame(results)
    print("\n  Paired Bootstrap Test (A - B):")
    print(f"  {df.to_string(index=False)}")
    return df


# ── 4. Variance decomposition ───────────────────────────────────────────────


def variance_decomposition(
    bir_df: pd.DataFrame,
    metric: str,
    output_dir: Path,
):
    """Decompose total BIR variance into between-question and between-run.

    If run variance is small vs question variance, error bars (which capture
    question variance) are adequate. If run variance is large, error bars
    underestimate true uncertainty."""
    bias_types = sorted(bir_df["bias_type"].unique())
    results = []

    for bt in bias_types:
        bt_df = bir_df[bir_df["bias_type"] == bt]

        a = bt_df[bt_df["training_type"] == "rlct_s50_a"].set_index("hash")[metric]
        b = bt_df[bt_df["training_type"] == "rlct_s50_b"].set_index("hash")[metric]
        common = a.index.intersection(b.index)

        a_v, b_v = a.loc[common].dropna().values, b.loc[common].dropna().values
        # Re-align after dropna
        a_clean = a.loc[common].dropna()
        b_clean = b.loc[common].dropna()
        shared = a_clean.index.intersection(b_clean.index)
        a_v = a_clean.loc[shared].values
        b_v = b_clean.loc[shared].values

        if len(a_v) < 2:
            continue

        q_mean = (a_v + b_v) / 2
        between_q = np.var(q_mean, ddof=1)
        within_q = np.mean((a_v - q_mean) ** 2 + (b_v - q_mean) ** 2) / 2
        total = np.var(np.concatenate([a_v, b_v]), ddof=1)
        n = len(a_v)

        results.append(
            {
                "bias_type": _bt_label(bt),
                "n": n,
                "between_q_var": between_q,
                "within_q_var": within_q,
                "total_var": total,
                "run_var_%": within_q / total * 100 if total > 0 else 0,
                "SE_btwn_q%": np.sqrt(between_q / n) * 100,
                "SE_run%": np.sqrt(within_q / n) * 100,
            }
        )

    df = pd.DataFrame(results)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Variance Decomposition: Between-Question vs Between-Run (A vs B)", fontsize=13
    )

    x = np.arange(len(results))
    w = 0.35
    labels = [r["bias_type"] for r in results]

    ax1.bar(
        x - w / 2,
        [r["between_q_var"] for r in results],
        w,
        label="Between-question",
        color="#1f77b4",
        alpha=0.7,
    )
    ax1.bar(
        x + w / 2,
        [r["within_q_var"] for r in results],
        w,
        label="Between-run",
        color="#ff7f0e",
        alpha=0.7,
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9, rotation=20, ha="right")
    ax1.set_ylabel("Variance")
    ax1.set_title("Absolute Variance")
    ax1.legend()

    ax2.bar(x, [r["run_var_%"] for r in results], color="#ff7f0e", alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9, rotation=20, ha="right")
    ax2.set_ylabel("% of Total Variance")
    ax2.set_title("Run Variance as % of Total")
    ax2.set_ylim(0, max(r["run_var_%"] for r in results) * 1.3 + 1)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = output_dir / "variance_decomposition.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    print("\n  Variance Decomposition:")
    print(f"  {df.to_string(index=False, float_format='{:.4f}'.format)}")
    return df


# ── 5. Agreement matrix ─────────────────────────────────────────────────────


def plot_agreement_matrix(
    bir_df: pd.DataFrame,
    metric: str,
    output_dir: Path,
):
    """Per-question agreement heatmap: (BIR_A, BIR_B) joint distribution.
    BIR in {-1, 0, 1}, so this is a 3x3 matrix. High diagonal = good agreement."""
    bias_types = sorted(bir_df["bias_type"].unique())

    fig, axes = plt.subplots(
        1, len(bias_types), figsize=(4 * len(bias_types), 3.5), squeeze=False
    )
    fig.suptitle("A-B Agreement: Per-Question BIR Concordance", fontsize=13)

    for ax_idx, bt in enumerate(bias_types):
        ax = axes[0, ax_idx]
        bt_df = bir_df[bir_df["bias_type"] == bt]

        a = bt_df[bt_df["training_type"] == "rlct_s50_a"].set_index("hash")[metric]
        b = bt_df[bt_df["training_type"] == "rlct_s50_b"].set_index("hash")[metric]
        common = a.index.intersection(b.index)

        a_v = a.loc[common].dropna()
        b_v = b.loc[common].dropna()
        shared = a_v.index.intersection(b_v.index)
        a_v = a_v.loc[shared].values
        b_v = b_v.loc[shared].values

        if len(a_v) == 0:
            continue

        # Build 3x3 contingency table
        bins = [-1, 0, 1]
        matrix = np.zeros((3, 3), dtype=int)
        for ai, bi in zip(a_v, b_v):
            ai_idx = bins.index(round(ai)) if round(ai) in bins else None
            bi_idx = bins.index(round(bi)) if round(bi) in bins else None
            if ai_idx is not None and bi_idx is not None:
                matrix[ai_idx, bi_idx] += 1

        # Normalize to percentages
        total = matrix.sum()
        pct = matrix / total * 100 if total > 0 else matrix

        im = ax.imshow(pct, cmap="Blues", vmin=0)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(["-1", "0", "+1"])
        ax.set_yticklabels(["-1", "0", "+1"])
        ax.set_xlabel("BIR_B")
        ax.set_ylabel("BIR_A")
        ax.set_title(f"{_bt_label(bt)}", fontsize=10)

        # Annotate cells
        for i in range(3):
            for j in range(3):
                count = matrix[i, j]
                pct_val = pct[i, j]
                text_color = "white" if pct_val > 40 else "black"
                ax.text(
                    j,
                    i,
                    f"{count}\n({pct_val:.0f}%)",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                )

        # Agreement rate
        agree = np.diag(matrix).sum()
        ax.text(
            0.5,
            -0.22,
            f"Agreement: {agree}/{total} ({agree/total*100:.0f}%)",
            ha="center",
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
        )

    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    path = output_dir / "agreement_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Variance diagnostics for BIR estimates")
    parser.add_argument(
        "--log-dir",
        required=True,
        help="Log directory with base, A, B runs",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/eval_suites/variance_diagnostics/plots",
    )
    parser.add_argument(
        "--metric",
        default="lenient_pro_bsr",
        choices=[
            "bir", "lenient_bir",                  # legacy aliases for total_bsr
            "pro_bsr", "anti_bsr", "net_bsr", "total_bsr",
            "lenient_pro_bsr", "lenient_anti_bsr", "lenient_net_bsr", "lenient_total_bsr",
        ],
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    bir_df = load_data(args.log_dir, args.metric)
    n_questions = bir_df.groupby(["bias_type", "training_type"])["hash"].nunique()
    print(f"  {len(bir_df)} rows")
    print(f"  Models: {sorted(bir_df['training_type'].unique())}")
    print(f"  Bias types: {sorted(bir_df['bias_type'].unique())}")
    print(f"  Questions per (bias, model): {n_questions.min()}-{n_questions.max()}")

    print("\n1a. Split-k analysis (±1 SEM, random split)...")
    plot_split_k(bir_df, args.metric, [2, 4, 8], output_dir, args.seed)

    print("\n1b. Split-k analysis (±2 SEM, stratified by dataset)...")
    plot_split_k(
        bir_df, args.metric, [2, 4, 8], output_dir, args.seed,
        stratify=True, ci_multiplier=2.0,
    )

    print("\n2. Cross-model subsets (same questions, base vs A vs B)...")
    for k in [2, 4]:
        plot_cross_model_subsets(bir_df, args.metric, k, output_dir, args.seed)

    print("\n3. Paired bootstrap test (A vs B)...")
    paired_bootstrap_test(bir_df, args.metric, output_dir, seed=args.seed)

    print("\n4. Variance decomposition...")
    variance_decomposition(bir_df, args.metric, output_dir)

    print("\n5. Agreement matrix...")
    plot_agreement_matrix(bir_df, args.metric, output_dir)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
