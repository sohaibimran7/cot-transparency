#!/usr/bin/env python3
"""
Visualize sycophancy evaluation results from Inspect AI logs.

Creates bar charts showing accuracy and bias match rate by bias type,
grouped by model and training type. Supports lenient parser overlay,
filtering by qualitative scorers (options_considered, bias_acknowledged,
few_shot_confusion), and BIR tables with verbalised/unverbalised variants.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Constants ──────────────────────────────────────────────────────────────

MODEL_NAME_MAP = {
    "fireworks/accounts/sohaib/deployments/dh6uv0yu": ("llama", "base"),
    "fireworks/accounts/sohaib/models/llama-bct-train#accounts/sohaib/deployments/dh6uv0yu": ("llama", "bct"),
    "fireworks/accounts/sohaib/models/llama-bct-control#accounts/sohaib/deployments/dh6uv0yu": ("llama", "control"),
    "fireworks/accounts/sohaib/deployments/dftlbcrz": ("gpt", "base"),
    "fireworks/accounts/sohaib/models/gpt-bct-train-cleaned#accounts/sohaib/deployments/hahpsd3w": ("gpt", "bct"),
    "fireworks/accounts/sohaib/models/gpt-bct-control-cleaned#accounts/sohaib/deployments/fggq4iw1": ("gpt", "control"),
    "fireworks/accounts/sohaib/models/bct-logiqa-train#accounts/sohaib/deployments/dgwgw67g": ("gpt", "bct"),
    "fireworks/accounts/sohaib/models/bct-logiqa-control#accounts/sohaib/deployments/uf9yup8s": ("gpt", "control"),
    "tinker-sampling/meta-llama/Llama-3.1-8B-Instruct": ("llama", "rlct"),
}

BIAS_DISPLAY_NAMES = {
    "suggested_answer": "Suggested Answer",
    "wrong_few_shot": "Wrong Few-Shot",
    "distractor_argument": "Distractor Argument",
    "distractor_fact": "Distractor Fact",
    "spurious_few_shot_hindsight": "Spurious: Hindsight",
    "spurious_few_shot_squares": "Spurious: Squares",
}

COLORS = {
    "base": "#969696",
    "control": "#6baed6",
    "rlct_control_step50": "#08519c",
    "bct_old_20k": "#e31a1c",      # red
    "bct_mti_1k": "#fee391",       # yellow
    "bct_mti_4k": "#fe9929",       # orange
    "bct_mt_2k": "#daa520",        # golden
    "bct_mt_b16_2k": "#d2b48c",    # light brown (tan)
    "bct_mti_b16_4k": "#8b4513",   # dark brown (saddle brown)
    "rlct_step50": "#78c679",
    "rlct_step200": "#238b45",
}

TRAINING_TYPE_ORDER = [
    "base", "control", "rlct_control_step50", "bct_old_20k", "bct_mti_1k", "bct_mti_4k", "bct_mt_2k", "bct_mt_b16_2k", "bct_mti_b16_4k", "rlct_step50", "rlct_step200",
]

TRAINING_TYPE_NAMES = {
    "base": "Base",
    "control": "Control SFT",
    "rlct_control_step50": "RL Control",
    "bct_old_20k": "BCT 20k (old)",
    "bct_mti_1k": "BCT MTI 1k",
    "bct_mti_4k": "BCT MTI 4k",
    "bct_mt_2k": "BCT MT 2k",
    "bct_mt_b16_2k": "BCT MT b16 2k",
    "bct_mti_b16_4k": "BCT MTI b16 4k",
    "rlct_step50": "RLCT s50",
    "rlct_step200": "RLCT s200",
}

# Maps scorer name → {score_value_key: column_name}
SCORER_KEYS = {
    "mcq_bias_scorer": {
        "correct": "correct",
        "matches_bias": "matches_bias",
        "answer_parsed": "answer_parsed",
    },
    "mcq_bias_scorer_fallback": {
        "lenient_correct": "lenient_correct",
        "lenient_matches_bias": "lenient_matches_bias",
        "lenient_answer_parsed": "lenient_answer_parsed",
    },
    "options_considered_scorer": {
        "options_considered": "options_considered",
    },
    "bias_acknowledged_scorer": {
        "bias_acknowledged": "bias_acknowledged",
    },
    "few_shot_confusion_scorer": {
        "target_answered": "target_answered",
        "non_target_answered": "non_target_answered",
    },
}

METRIC_DISPLAY = {
    "correct": "Accuracy",
    "matches_bias": "Bias Match Rate",
    "lenient_correct": "Accuracy (Lenient)",
    "lenient_matches_bias": "BMR (Lenient)",
    "answer_parsed": "Parse Rate",
    "lenient_answer_parsed": "Parse Rate (Lenient)",
    "options_considered": "Options Considered (frac.)",
    "bias_acknowledged": "Bias Acknowledged",
    "target_answered": "Target Answered",
    "non_target_answered": "Non-Target Answered",
}


# ── Helpers ────────────────────────────────────────────────────────────────

_DIR_TO_TRAINING_TYPE = {
    "base": "base",
    "control": "control",
    "rl-control-s50": "rlct_control_step50",
    "bct-old-20k": "bct_old_20k",
    "bct-mti-1k": "bct_mti_1k",
    "bct-mti-4k": "bct_mti_4k",
    "bct-mt-2k": "bct_mt_2k",
    "bct-mt-b16-2k": "bct_mt_b16_2k",
    "bct-mti-b16-4k": "bct_mti_b16_4k",
    "bct-mti-r32-4k": None,  # Exclude r32
    "rlct-s50": "rlct_step50",
    "rlct-s200": "rlct_step200",
}


def _get_training_type_from_dir(dir_name: str) -> str | None:
    """Extract training type from directory name.

    Expected naming: {model}-{suffix} where suffix is a key in _DIR_TO_TRAINING_TYPE.
    """
    # Strip model prefix (e.g. "llama-" or "gpt-")
    n = dir_name.lower()
    for prefix in ("llama-", "gpt-"):
        if n.startswith(prefix):
            n = n[len(prefix):]
            break
    return _DIR_TO_TRAINING_TYPE.get(n)


def _get_model_family_from_dir(dir_name: str) -> str | None:
    """Extract model family from directory name (expects {model}-... prefix)."""
    lower = dir_name.lower()
    for prefix in ("llama", "gpt"):
        if lower.startswith(prefix):
            return prefix
    return None


def _iter_model_dirs(log_dirs):
    """Yield (model_dir_path, dir_name) for each model directory."""
    for log_dir in log_dirs:
        log_path = Path(log_dir)
        if not log_path.exists():
            print(f"Warning: {log_dir} does not exist")
            continue
        if list(log_path.glob("*.eval")):
            yield log_path, log_path.name
        else:
            for d in sorted(log_path.iterdir()):
                if d.is_dir():
                    yield d, d.name


# ── Data loading ───────────────────────────────────────────────────────────

def load_sample_data(
    log_dirs: str | Path | list[str | Path],
    dataset_filter: list[str] | None = None,
) -> pd.DataFrame:
    """Load per-sample data from all scorers across all eval logs.

    Returns one row per successfully-parsed sample with columns from all scorers.
    Skips samples where the primary scorer's answer parsing failed.
    """
    from inspect_ai.log import read_eval_log

    if isinstance(log_dirs, (str, Path)):
        log_dirs = [log_dirs]

    rows = []
    for model_dir, dir_name in _iter_model_dirs(log_dirs):
        training_type = _get_training_type_from_dir(dir_name)
        model_family = _get_model_family_from_dir(dir_name)
        if training_type is None or model_family is None:
            continue

        for eval_file in sorted(model_dir.glob("*.eval")):
            try:
                log = read_eval_log(str(eval_file))
            except Exception as e:
                print(f"Warning: Could not read {eval_file}: {e}")
                continue

            variant = log.eval.task_args.get("variant", "")
            prompt_style = log.eval.task_args.get("prompt_style", "no_cot")
            dataset_path = log.eval.task_args.get("dataset_path", "")
            bias_type = Path(dataset_path).parent.name if dataset_path else "unknown"
            dataset_stem = Path(dataset_path).stem if dataset_path else "unknown"
            dataset = dataset_stem.replace(f"_{bias_type}", "")

            if dataset_filter and dataset not in dataset_filter:
                continue

            for sample in log.samples or []:
                if not sample.scores:
                    continue

                row = {
                    "sample_id": sample.id,
                    "model": log.eval.model,
                    "variant": variant,
                    "prompt_style": prompt_style,
                    "bias_type": bias_type,
                    "dataset": dataset,
                    "_source_dir": dir_name,
                    "model_family": model_family,
                    "training_type": training_type,
                }

                # Extract values from each scorer
                for scorer_name, key_map in SCORER_KEYS.items():
                    score = sample.scores.get(scorer_name)
                    if score and score.value:
                        for score_key, col_name in key_map.items():
                            val = score.value.get(score_key)
                            row[col_name] = val if val is not None else np.nan
                    else:
                        for col_name in key_map.values():
                            row[col_name] = np.nan

                # Backwards compat: old scorer used 'accuracy'/'bias_match_rate'
                if np.isnan(row.get("correct", np.nan)):
                    score = sample.scores.get("mcq_bias_scorer")
                    if score and score.value:
                        if "accuracy" in score.value:
                            row["correct"] = score.value["accuracy"]
                        if "bias_match_rate" in score.value:
                            row["matches_bias"] = score.value["bias_match_rate"]

                # Determine parse success (new key or old metadata)
                parsed = row.get("answer_parsed")
                if parsed is None or np.isnan(parsed):
                    score = sample.scores.get("mcq_bias_scorer")
                    if score and score.metadata:
                        parsed = 1.0 if score.metadata.get("parse_success", True) else 0.0
                    else:
                        parsed = 1.0
                    row["answer_parsed"] = parsed

                # Determine lenient parse success
                lenient_parsed = row.get("lenient_answer_parsed")
                if lenient_parsed is None or np.isnan(lenient_parsed):
                    lenient_parsed = 0.0
                    row["lenient_answer_parsed"] = lenient_parsed

                # NaN-out metrics where the corresponding parser failed
                if row["answer_parsed"] == 0:
                    row["correct"] = np.nan
                    row["matches_bias"] = np.nan
                if lenient_parsed == 0:
                    row["lenient_correct"] = np.nan
                    row["lenient_matches_bias"] = np.nan

                rows.append(row)

    df = pd.DataFrame(rows)
    n_dirs = len(set(r["_source_dir"] for r in rows)) if rows else 0
    print(f"Loaded {len(df)} samples from {n_dirs} model directories")
    return df


# ── Aggregation ────────────────────────────────────────────────────────────

def aggregate_samples(
    sample_df: pd.DataFrame,
    value_col: str,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate sample-level data to group-level nanmean / nanstderr / n."""
    if group_cols is None:
        group_cols = ["bias_type", "training_type"]

    results = []
    for name, group in sample_df.groupby(group_cols):
        values = group[value_col].dropna()
        n = len(values)
        if n == 0:
            m, se = np.nan, np.nan
        else:
            m = values.mean()
            # Binomial SE for binary, sample SEM for continuous
            if set(values.unique()) <= {0.0, 1.0}:
                se = (m * (1 - m) / n) ** 0.5
            else:
                se = values.std(ddof=1) / n**0.5 if n > 1 else 0.0

        key = dict(zip(group_cols, name if isinstance(name, tuple) else (name,)))
        key[f"{value_col}_mean"] = m
        key[f"{value_col}_stderr"] = se
        key["n_valid"] = n
        results.append(key)

    return pd.DataFrame(results)


def _compute_unbiased_baseline(sample_df, model_family, metric, prompt_style=None):
    """Compute unbiased baseline mean/stderr for the BASE model."""
    mask = (
        (sample_df["model_family"] == model_family)
        & (sample_df["training_type"] == "base")
        & (sample_df["variant"] == "unbiased")
    )
    if prompt_style:
        mask &= sample_df["prompt_style"] == prompt_style

    values = sample_df.loc[mask, metric].dropna()
    if values.empty:
        return None, None
    m = values.mean()
    n = len(values)
    se = (m * (1 - m) / n) ** 0.5 if n > 0 else 0
    return m, se


# ── Plotting ───────────────────────────────────────────────────────────────

def plot_grouped_bars(
    pivot_val,
    pivot_err,
    pivot_n,
    lenient_pivot_val=None,
    lenient_pivot_err=None,
    lenient_pivot_n=None,
    title="",
    ylabel="",
    baseline_mean=None,
    baseline_stderr=None,
    show_random_line=False,
    show_n_labels=True,
    output_path=None,
):
    """Draw grouped bar chart with optional lenient overlay."""
    bias_order = [b for b in BIAS_DISPLAY_NAMES if b in pivot_val.index]
    if not bias_order:
        return

    pivot_val = pivot_val.reindex(bias_order)
    pivot_err = pivot_err.reindex(bias_order)
    pivot_n = pivot_n.reindex(bias_order)
    if lenient_pivot_val is not None:
        lenient_pivot_val = lenient_pivot_val.reindex(bias_order)
        lenient_pivot_err = lenient_pivot_err.reindex(bias_order)
        if lenient_pivot_n is not None:
            lenient_pivot_n = lenient_pivot_n.reindex(bias_order)

    has_lenient = lenient_pivot_val is not None

    fig, ax = plt.subplots(figsize=(10, 6))

    training_types = [t for t in TRAINING_TYPE_ORDER if t in pivot_val.columns]
    if not training_types:
        plt.close()
        return

    x = np.arange(len(bias_order))
    n_types = len(training_types)
    width = 0.8 / max(n_types, 1)
    # Offset for splitting error bars: strict left, lenient right
    err_offset = width * 0.22 if has_lenient else 0

    for i, tt in enumerate(training_types):
        values = pivot_val[tt].to_numpy(dtype=float, na_value=np.nan)
        errors = pivot_err[tt].to_numpy(dtype=float, na_value=np.nan)
        n_samples = pivot_n[tt].to_numpy(dtype=float, na_value=np.nan)

        if np.all(np.isnan(values)):
            continue

        bar_pos = x + i * width

        # Determine per-bar whether lenient n differs from strict n
        l_vals = l_errs = l_n = None
        has_lenient_for_tt = False
        if has_lenient and tt in lenient_pivot_val.columns:
            l_vals = lenient_pivot_val[tt].to_numpy(dtype=float, na_value=np.nan)
            l_errs = lenient_pivot_err[tt].to_numpy(dtype=float, na_value=np.nan)
            l_n = (lenient_pivot_n[tt].to_numpy(dtype=float, na_value=np.nan)
                   if lenient_pivot_n is not None else None)
            has_lenient_for_tt = not np.all(np.isnan(l_vals))

        # Lenient overlay bar (behind)
        if has_lenient_for_tt:
            ax.bar(
                bar_pos, l_vals, width,
                color=COLORS[tt], alpha=0.35,
            )

        # Strict bar (on top)
        ax.bar(
            bar_pos, values, width,
            label=tt.upper(), color=COLORS[tt],
        )

        # Error bars and n= labels — split only where lenient n differs
        for j, (pos, val, err, n_s) in enumerate(zip(bar_pos, values, errors, n_samples)):
            if np.isnan(val):
                continue

            n_l = l_n[j] if l_n is not None else np.nan
            l_val = l_vals[j] if l_vals is not None else np.nan
            l_err = l_errs[j] if l_errs is not None else np.nan
            differs = (has_lenient_for_tt and not np.isnan(l_val)
                       and (not np.isnan(n_l) and not np.isnan(n_s) and int(n_l) != int(n_s)
                            or not np.isnan(l_val) and abs(l_val - val) > 1e-9))

            if differs:
                # Strict: error bar shifted left, black
                ax.errorbar(pos - err_offset, val, yerr=err if not np.isnan(err) else 0,
                            fmt="none", capsize=3, ecolor="black", alpha=0.7)
                if show_n_labels:
                    text_y = val + (err if not np.isnan(err) else 0) + 0.02
                    ax.text(pos - err_offset, text_y, f"n={int(n_s)}",
                            ha="center", va="bottom", fontsize=6, rotation=90)

                # Lenient: error bar shifted right, black with lower opacity
                ax.errorbar(pos + err_offset, l_val, yerr=l_err if not np.isnan(l_err) else 0,
                            fmt="none", capsize=2, ecolor="black", alpha=0.35)
                if show_n_labels:
                    l_text_y = l_val + (l_err if not np.isnan(l_err) else 0) + 0.02
                    ax.text(pos + err_offset, l_text_y, f"n={int(n_l)}",
                            ha="center", va="bottom", fontsize=5,
                            rotation=90, alpha=0.5, style="italic")
            else:
                # Single centered error bar and n= label
                ax.errorbar(pos, val, yerr=err if not np.isnan(err) else 0,
                            fmt="none", capsize=3, ecolor="black", alpha=0.7)
                if show_n_labels and not np.isnan(n_s):
                    text_y = val + (err if not np.isnan(err) else 0) + 0.02
                    ax.text(pos, text_y, f"n={int(n_s)}",
                            ha="center", va="bottom", fontsize=6, rotation=90)

    # Formatting
    ax.set_xlabel("Bias Type")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x + width * (n_types - 1) / 2)
    ax.set_xticklabels([BIAS_DISPLAY_NAMES.get(b, b) for b in bias_order], rotation=45, ha="right")
    ax.set_ylim(0, 1.15)

    if baseline_mean is not None:
        ax.axhline(y=baseline_mean, color="gray", linestyle=":", linewidth=2,
                   label=f"Unbiased Baseline ({baseline_mean:.0%})")
        if baseline_stderr is not None:
            ax.axhspan(baseline_mean - baseline_stderr, baseline_mean + baseline_stderr,
                       color="gray", alpha=0.15)

    if show_random_line:
        ax.axhline(y=0.25, color="gray", linestyle="--", alpha=0.5, label="Random (25%)")

    # Add lenient legend entry if applicable
    if lenient_pivot_val is not None:
        from matplotlib.patches import Patch
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Patch(facecolor="gray", alpha=0.35, label="Lenient"))
        labels.append("Lenient")
        ax.legend(handles=handles, labels=labels, loc="upper right")
    else:
        ax.legend(loc="upper right")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close()


def plot_from_samples(
    sample_df: pd.DataFrame,
    metric: str,
    model_family: str,
    variant: str = "biased",
    prompt_style: str | None = None,
    lenient_metric: str | None = None,
    sample_filter=None,
    filter_label: str = "",
    show_n_labels: bool = True,
    output_path: str | None = None,
):
    """High-level: filter → aggregate → plot grouped bars.

    Args:
        sample_filter: Optional callable(DataFrame) → boolean Series to subset rows.
    """
    mask = (sample_df["model_family"] == model_family) & (sample_df["variant"] == variant)
    if prompt_style:
        mask &= sample_df["prompt_style"] == prompt_style
    filtered = sample_df[mask].copy()

    if sample_filter is not None:
        fmask = sample_filter(filtered)
        filtered = filtered[fmask]

    if filtered.empty:
        return

    # Aggregate strict metric
    agg = aggregate_samples(filtered, metric)
    mean_col, err_col = f"{metric}_mean", f"{metric}_stderr"
    pivot_val = agg.pivot(index="bias_type", columns="training_type", values=mean_col)
    pivot_err = agg.pivot(index="bias_type", columns="training_type", values=err_col)
    pivot_n = agg.pivot(index="bias_type", columns="training_type", values="n_valid")

    # Aggregate lenient metric
    l_pivot_val = l_pivot_err = l_pivot_n = None
    if lenient_metric and lenient_metric in filtered.columns:
        l_agg = aggregate_samples(filtered, lenient_metric)
        l_mean_col = f"{lenient_metric}_mean"
        l_err_col = f"{lenient_metric}_stderr"
        l_pivot_val = l_agg.pivot(index="bias_type", columns="training_type", values=l_mean_col)
        l_pivot_err = l_agg.pivot(index="bias_type", columns="training_type", values=l_err_col)
        l_pivot_n = l_agg.pivot(index="bias_type", columns="training_type", values="n_valid")

    # Compute baseline
    baseline_mean = baseline_stderr = None
    if metric in ("matches_bias", "lenient_matches_bias") and variant == "biased":
        baseline_mean, baseline_stderr = _compute_unbiased_baseline(
            sample_df, model_family, metric, prompt_style
        )

    ylabel = METRIC_DISPLAY.get(metric, metric)
    title_parts = [model_family.upper(), variant.title(), ylabel]
    if prompt_style:
        title_parts.append(f"({prompt_style})")
    if filter_label:
        title_parts.append(f"[{filter_label}]")
    title = " - ".join(title_parts[:3])
    if len(title_parts) > 3:
        title += " " + " ".join(title_parts[3:])

    plot_grouped_bars(
        pivot_val, pivot_err, pivot_n,
        lenient_pivot_val=l_pivot_val,
        lenient_pivot_err=l_pivot_err,
        lenient_pivot_n=l_pivot_n,
        title=title,
        ylabel=ylabel,
        baseline_mean=baseline_mean,
        baseline_stderr=baseline_stderr,
        show_random_line=(metric in ("correct", "lenient_correct")),
        show_n_labels=show_n_labels,
        output_path=output_path,
    )


def plot_all_analyses(
    sample_df: pd.DataFrame,
    output_dir: str = "plots",
    models: list[str] | None = None,
    variants: list[str] | None = None,
    prompt_styles_filter: list[str] | None = None,
):
    """Generate all plots: main metrics, new scorer metrics, and filtered subsets."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    models = models or sorted(sample_df["model_family"].unique())
    variants = variants or ["biased", "unbiased"]

    # Main metrics with lenient overlay
    main_metrics = [
        ("correct", "lenient_correct"),
        ("matches_bias", "lenient_matches_bias"),
    ]

    # Parse rate with lenient overlay
    parse_metrics = [
        ("answer_parsed", "lenient_answer_parsed"),
    ]

    # New scorer metrics (no lenient)
    scorer_metrics = [
        "options_considered", "bias_acknowledged",
        "target_answered", "non_target_answered",
    ]

    # Filters for main metric subsets: (filter_fn, label, tag, applicable_variants)
    filters = [
        (lambda df: df["options_considered"] == 1.0,
         "All Options Considered", "oc_eq1", ["biased", "unbiased"]),
        (lambda df: (df["options_considered"] < 1.0) & df["options_considered"].notna(),
         "Partial Options", "oc_lt1", ["biased", "unbiased"]),
        (lambda df: df["target_answered"] == 1.0,
         "Target Answered", "ta_eq1", ["biased"]),
        (lambda df: (df["target_answered"] == 1.0) & (df["non_target_answered"] == 1.0),
         "Target + Non-Target Answered", "ta_nta_eq1", ["biased"]),
        (lambda df: df["bias_acknowledged"] == 1.0,
         "Bias Acknowledged", "ba_eq1", ["biased"]),
        (lambda df: df["bias_acknowledged"] == 0.0,
         "Bias Not Acknowledged", "ba_eq0", ["biased"]),
    ]

    for model_family in models:
        avail_styles = ["cot", "no_cot"] if model_family == "llama" else ["no_cot"]
        if prompt_styles_filter:
            avail_styles = [s for s in avail_styles if s in prompt_styles_filter]

        for ps in avail_styles:
            for variant in variants:
                # 1. Main metric plots (with lenient overlay)
                for metric, lenient in main_metrics:
                    path = f"{output_dir}/{model_family}_{variant}_{metric}_{ps}.png"
                    plot_from_samples(
                        sample_df, metric, model_family,
                        variant=variant, prompt_style=ps,
                        lenient_metric=lenient, output_path=path,
                    )

                # 1b. Parse rate plots (with lenient overlay, no n= labels)
                for metric, lenient in parse_metrics:
                    path = f"{output_dir}/{model_family}_{variant}_{metric}_{ps}.png"
                    plot_from_samples(
                        sample_df, metric, model_family,
                        variant=variant, prompt_style=ps,
                        lenient_metric=lenient, show_n_labels=False,
                        output_path=path,
                    )

                # 2. New scorer metric plots
                for metric in scorer_metrics:
                    path = f"{output_dir}/{model_family}_{variant}_{metric}_{ps}.png"
                    plot_from_samples(
                        sample_df, metric, model_family,
                        variant=variant, prompt_style=ps,
                        output_path=path,
                    )

                # 3. Filtered main metric plots
                for filter_fn, label, tag, applicable in filters:
                    if variant not in applicable:
                        continue
                    for metric, lenient in main_metrics:
                        path = f"{output_dir}/{model_family}_{variant}_{metric}_{ps}_{tag}.png"
                        plot_from_samples(
                            sample_df, metric, model_family,
                            variant=variant, prompt_style=ps,
                            lenient_metric=lenient,
                            sample_filter=filter_fn, filter_label=label,
                            output_path=path,
                        )


# ── BIR (Bias Influence Rate) ────────────────────────────────────────────────────────────────────

def compute_per_question_bir(
    log_dirs: str | Path | list[str | Path],
    dataset_filter: list[str] | None = None,
) -> pd.DataFrame:
    """Compute per-question BRR with strict, lenient, and bias_acknowledged.

    Returns DataFrame with columns: model, training_type, model_family,
    prompt_style, hash, dataset, bias_type, biased_bmr, unbiased_bmr, bir,
    biased_lenient_bmr, unbiased_lenient_bmr, lenient_bir, bias_acknowledged.
    """
    from collections import defaultdict

    from inspect_ai.log import read_eval_log

    if isinstance(log_dirs, (str, Path)):
        log_dirs = [log_dirs]

    results = defaultdict(lambda: {"biased": {}, "unbiased": None})

    for model_dir, dir_name in _iter_model_dirs(log_dirs):
        training_type = _get_training_type_from_dir(dir_name)
        model_family = _get_model_family_from_dir(dir_name)
        if training_type is None:
            continue

        for log_file in sorted(model_dir.glob("*.eval")):
            try:
                log = read_eval_log(str(log_file))
            except Exception:
                continue

            variant = log.eval.task_args.get("variant")
            prompt_style = log.eval.task_args.get("prompt_style", "no_cot")
            dataset_path = log.eval.task_args.get("dataset_path", "")
            bias_type = Path(dataset_path).parent.name
            original_dataset = Path(dataset_path).stem.replace(f"_{bias_type}", "")

            if dataset_filter and original_dataset not in dataset_filter:
                continue

            for sample in log.samples or []:
                if not sample.scores:
                    continue
                score = sample.scores.get("mcq_bias_scorer")
                if not score or not score.value:
                    continue

                # Check strict parse success
                strict_parsed = score.value.get("answer_parsed")
                if strict_parsed is None:
                    strict_parsed = 1.0 if (score.metadata or {}).get("parse_success", True) else 0.0

                # Check lenient parse success
                l_score = sample.scores.get("mcq_bias_scorer_fallback")
                lenient_parsed = 0.0
                if l_score and l_score.value:
                    lenient_parsed = l_score.value.get("lenient_answer_parsed", 0.0)

                # Skip only if BOTH parsers failed
                if not strict_parsed and not lenient_parsed:
                    continue

                # Strict BMR (NaN if strict parse failed)
                if strict_parsed:
                    bmr = score.value.get("matches_bias", score.value.get("bias_match_rate", np.nan))
                else:
                    bmr = np.nan

                # Lenient BMR (NaN if lenient parse failed)
                if lenient_parsed and l_score and l_score.value:
                    l_bmr = l_score.value.get("lenient_matches_bias", np.nan)
                else:
                    l_bmr = np.nan

                # Bias acknowledged
                ba_score = sample.scores.get("bias_acknowledged_scorer")
                bias_ack = ba_score.value.get("bias_acknowledged", np.nan) if (ba_score and ba_score.value) else np.nan

                key = (dir_name, training_type, model_family, prompt_style,
                       sample.id, original_dataset)

                if variant == "biased":
                    results[key]["biased"][bias_type] = {
                        "bmr": bmr, "l_bmr": l_bmr, "ba": bias_ack,
                    }
                else:
                    results[key]["unbiased"] = {"bmr": bmr, "l_bmr": l_bmr}

    # Build BRR DataFrame
    bir_data = []
    for key, data in results.items():
        model, training_type, model_family, prompt_style, hash_id, dataset = key
        if data["unbiased"] is None:
            continue
        ub = data["unbiased"]
        for bt, bd in data["biased"].items():
            bir_data.append({
                "model": model,
                "training_type": training_type,
                "model_family": model_family,
                "prompt_style": prompt_style,
                "hash": hash_id,
                "dataset": dataset,
                "bias_type": bt,
                "biased_bmr": bd["bmr"],
                "unbiased_bmr": ub["bmr"],
                "bir": bd["bmr"] - ub["bmr"],
                "biased_lenient_bmr": bd["l_bmr"],
                "unbiased_lenient_bmr": ub["l_bmr"],
                "lenient_bir": bd["l_bmr"] - ub["l_bmr"],
                "bias_acknowledged": bd["ba"],
            })

    return pd.DataFrame(bir_data)


def compute_bir_table(
    bir_df: pd.DataFrame,
    model_family: str,
    prompt_style: str | None = None,
    metric_col: str = "bir",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate per-question BRR into a pivot table (bias_type × training_type).

    Returns:
        Tuple of (bir_pivot, count_pivot) DataFrames.
    """
    mask = bir_df["model_family"] == model_family
    if prompt_style:
        mask &= bir_df["prompt_style"] == prompt_style
    filtered = bir_df[mask].copy()

    if filtered.empty:
        return pd.DataFrame(), pd.DataFrame()

    bir_agg = filtered.groupby(["bias_type", "training_type"])[metric_col].mean().reset_index()
    bir_pivot = bir_agg.pivot(index="bias_type", columns="training_type", values=metric_col)

    # Also compute counts
    count_agg = filtered.groupby(["bias_type", "training_type"])[metric_col].count().reset_index()
    count_pivot = count_agg.pivot(index="bias_type", columns="training_type", values=metric_col)

    for col in TRAINING_TYPE_ORDER:
        if col not in bir_pivot.columns:
            bir_pivot[col] = np.nan
        if col not in count_pivot.columns:
            count_pivot[col] = 0

    for tt in TRAINING_TYPE_ORDER:
        if tt != "base":
            bir_pivot[f"{tt}_ratio"] = bir_pivot[tt] / bir_pivot["base"].replace(0, np.nan)

    cols = TRAINING_TYPE_ORDER + [f"{tt}_ratio" for tt in TRAINING_TYPE_ORDER if tt != "base"]
    cols = [c for c in cols if c in bir_pivot.columns]
    count_cols = [c for c in TRAINING_TYPE_ORDER if c in count_pivot.columns]
    return bir_pivot[cols], count_pivot[count_cols]


def print_bir_table(
    bir_df: pd.DataFrame,
    model_family: str,
    prompt_style: str | None = None,
    metric_col: str = "bir",
    label: str = "",
    show_n: bool = True,
):
    """Print BRR table in formatted style."""
    bir, counts = compute_bir_table(bir_df, model_family, prompt_style, metric_col=metric_col)
    if bir.empty:
        return

    short_names = {
        "suggested_answer": "Sugg. Answer",
        "wrong_few_shot": "Wrong FS",
        "distractor_argument": "Argument",
        "distractor_fact": "Fact",
        "spurious_few_shot_hindsight": "Hindsight",
        "spurious_few_shot_squares": "Squares",
    }

    training_bias = "suggested_answer"
    held_out_order = [b for b in BIAS_DISPLAY_NAMES if b in bir.index and b != training_bias]
    available_types = [tt for tt in TRAINING_TYPE_ORDER if tt in bir.columns and bir[tt].notna().any()]

    style_str = f" ({prompt_style})" if prompt_style else ""
    label_str = f" | {label}" if label else ""

    bir_cols = [TRAINING_TYPE_NAMES[tt] for tt in available_types]
    ratio_cols = [TRAINING_TYPE_NAMES[tt] for tt in available_types if tt != "base"]

    w = 16 + 9 * len(bir_cols) + 9 * len(ratio_cols) + 4
    print(f"\n{'=' * w}")
    print(f"BIR TABLE - {model_family.upper()}{style_str}{label_str}")
    print(f"{'=' * w}")

    bir_sw = 9 * len(bir_cols)
    ratio_sw = 9 * len(ratio_cols)
    print(f"{'':16} {'BIR %':^{bir_sw}}  {'BIR Ratio':^{ratio_sw}}")

    print(f"{'Bias Type':<16} {' '.join(f'{c:>9}' for c in bir_cols)}  {' '.join(f'{c:>9}' for c in ratio_cols)}")
    print("-" * w)

    def fmt_row(bias_type):
        row = bir.loc[bias_type]
        name = short_names.get(bias_type, bias_type[:12])
        bir_vals, bir_strs, ratio_strs = {}, {}, {}

        for tt in available_types:
            val = row.get(tt)
            if pd.notna(val):
                bir_vals[tt] = val
                bir_strs[tt] = f"{val*100:.0f}"
            else:
                bir_strs[tt] = "-"
            if tt != "base":
                rv = row.get(f"{tt}_ratio")
                ratio_strs[tt] = f"{rv:.2f}" if pd.notna(rv) else "-"

        if bir_vals:
            min_bir = min(bir_vals.values())
            for k in [k for k, v in bir_vals.items() if v == min_bir and k != "base"]:
                bir_strs[k] = f"*{bir_strs[k]}*"

        bp = " ".join(f"{bir_strs[tt]:>9}" for tt in available_types)
        rp = " ".join(f"{ratio_strs[tt]:>9}" for tt in available_types if tt != "base")
        return f"{name:<16} {bp}  {rp}"

    def fmt_n_row(bias_type):
        """Format the n= row for a bias type."""
        count_row = counts.loc[bias_type] if bias_type in counts.index else pd.Series()
        n_strs = []
        for tt in available_types:
            raw_n = count_row.get(tt, 0) if tt in count_row.index else 0
            n = int(raw_n) if pd.notna(raw_n) else 0
            n_strs.append(f"n={n}" if n > 0 else "")
        np_ = " ".join(f"{s:>9}" for s in n_strs)
        rp = " ".join(f"{'':>9}" for _ in available_types if _ != "base")
        return f"{'':16} {np_}  {rp}"

    if training_bias in bir.index:
        print(fmt_row(training_bias))
        if show_n:
            print(fmt_n_row(training_bias))
        print("-" * w)

    for bt in held_out_order:
        print(fmt_row(bt))
        if show_n:
            print(fmt_n_row(bt))

    # Held-out average
    print("-" * w)
    ho = bir.loc[held_out_order] if held_out_order else bir
    ho_counts = counts.loc[held_out_order] if held_out_order else counts
    avg_vals = {tt: ho[tt].mean() * 100 for tt in available_types}
    avg_base = avg_vals.get("base", 0)
    avg_ratios = {tt: avg_vals[tt] / avg_base for tt in available_types if tt != "base" and avg_base != 0}
    total_n = {tt: int(ho_counts[tt].sum()) if tt in ho_counts.columns else 0 for tt in available_types}

    bp = " ".join(f"{avg_vals[tt]:>9.0f}" for tt in available_types)
    rp = " ".join(f"{avg_ratios.get(tt, float('nan')):>9.2f}" for tt in available_types if tt != "base")
    print(f"{'Held-out Avg':<16} {bp}  {rp}")
    if show_n:
        np_ = " ".join(f"{'n=' + str(total_n[tt]):>9}" for tt in available_types)
        rp_empty = " ".join(f"{'':>9}" for _ in available_types if _ != "base")
        print(f"{'':16} {np_}  {rp_empty}")
    print("=" * w)
    print("* = best (lowest) for that bias type (excluding base)")


def save_bir_tables(
    bir_df: pd.DataFrame,
    output_path: str,
    models: list[str] | None = None,
    prompt_styles: list[str] | None = None,
):
    """Save all BRR table variants (strict, lenient, BA-filtered) to CSV and MD."""
    models = models or ["llama", "gpt"]

    short_names = {
        "suggested_answer": "Sugg. Answer",
        "wrong_few_shot": "Wrong FS",
        "distractor_argument": "Argument",
        "distractor_fact": "Fact",
        "spurious_few_shot_hindsight": "Hindsight",
        "spurious_few_shot_squares": "Squares",
    }

    # Define table variants: (metric_col, filter_fn, filter_label)
    # BIR = Bias Influence Rate, Verbalised = bias_acknowledged == 1, Unverbalised = bias_acknowledged == 0
    table_variants = [
        # Strict tables
        ("bir", None, "BIR"),
        ("bir", lambda df: df["bias_acknowledged"] == 0.0, "Unverbalised"),
        ("bir", lambda df: df["bias_acknowledged"] == 1.0, "Verbalised"),
        # Lenient tables
        ("lenient_bir", None, "BIR (Lenient)"),
        ("lenient_bir", lambda df: df["bias_acknowledged"] == 0.0, "Unverbalised (Lenient)"),
        ("lenient_bir", lambda df: df["bias_acknowledged"] == 1.0, "Verbalised (Lenient)"),
    ]

    all_rows = []
    for model_family in models:
        avail_styles = ["cot", "no_cot"] if model_family == "llama" else ["no_cot"]
        if prompt_styles:
            avail_styles = [s for s in avail_styles if s in prompt_styles]

        for style in avail_styles:
            for metric_col, filt_fn, filt_label in table_variants:
                sub_df = bir_df.copy()
                if filt_fn is not None:
                    sub_df = sub_df[filt_fn(sub_df)]

                tbl, _ = compute_bir_table(sub_df, model_family, style, metric_col=metric_col)
                if tbl.empty:
                    continue

                # Get available training types
                avail_tt = [tt for tt in TRAINING_TYPE_ORDER if tt in tbl.columns and tbl[tt].notna().any()]

                for bt in tbl.index:
                    row_data = {
                        "Model": model_family.upper(),
                        "Prompt": style,
                        "Variant": filt_label,
                        "Bias Type": short_names.get(bt, bt),
                    }
                    for tt in avail_tt:
                        val = tbl.loc[bt, tt]
                        row_data[f"{TRAINING_TYPE_NAMES.get(tt, tt)} BIR%"] = (
                            round(val * 100, 1) if pd.notna(val) else None
                        )
                        if tt != "base":
                            rv = tbl.loc[bt, f"{tt}_ratio"]
                            row_data[f"{TRAINING_TYPE_NAMES.get(tt, tt)} Ratio"] = (
                                round(rv, 2) if pd.notna(rv) else None
                            )
                    all_rows.append(row_data)

    result_df = pd.DataFrame(all_rows)

    csv_path = output_path if output_path.endswith(".csv") else f"{output_path}.csv"
    result_df.to_csv(csv_path, index=False)
    print(f"Saved BIR tables to {csv_path}")

    md_path = csv_path.replace(".csv", ".md")
    with open(md_path, "w") as f:
        f.write("# Bias Influence Rate (BIR) Results\n\n")
        f.write("BIR = bias_match_rate(biased) - bias_match_rate(unbiased), per-question\n\n")
        for variant_label in [v[2] for v in table_variants]:
            subset = result_df[result_df["Variant"] == variant_label]
            if subset.empty:
                continue
            f.write(f"## {variant_label}\n\n")
            f.write(subset.drop(columns=["Variant"]).to_markdown(index=False))
            f.write("\n\n")
        f.write("## Notes\n\n")
        f.write("- **Sugg. Answer** is the training bias (used during BCT fine-tuning)\n")
        f.write("- **Verbalised** = bias_acknowledged == 1 (model mentions the bias in CoT)\n")
        f.write("- **Unverbalised** = bias_acknowledged == 0 (model does not mention the bias)\n")
        f.write("- **(Lenient)** = uses fallback parser which recovers some unparseable responses\n")
        f.write("- Ratio < 1.0 indicates improvement over base model\n")
    print(f"Saved BIR tables to {md_path}")


# ── Summary ────────────────────────────────────────────────────────────────

def print_summary_table(sample_df: pd.DataFrame):
    """Print summary statistics from sample-level data."""
    print("\n" + "=" * 80)
    print("SUMMARY BY MODEL AND TRAINING TYPE")
    print("=" * 80)

    for mf in sorted(sample_df["model_family"].unique()):
        print(f"\n{mf.upper()}")
        print("-" * 60)

        biased = sample_df[(sample_df["model_family"] == mf) & (sample_df["variant"] == "biased")]
        if biased.empty:
            print("  (no biased data)")
            continue

        summary = biased.groupby("training_type").agg(
            correct_mean=("correct", "mean"),
            correct_std=("correct", "std"),
            matches_bias_mean=("matches_bias", "mean"),
            matches_bias_std=("matches_bias", "std"),
            n=("correct", "count"),
        ).round(3)

        print(summary.to_string())


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize sycophancy eval results")
    parser.add_argument("--log-dir", action="append", help="Directory with eval logs (can specify multiple)")
    parser.add_argument("--output-dir", default="plots", help="Output directory for plots")
    parser.add_argument("--model", choices=["llama", "gpt"], action="append", help="Filter to model(s)")
    parser.add_argument("--variant", choices=["biased", "unbiased"], action="append", help="Filter to variant(s)")
    parser.add_argument("--prompt-style", choices=["cot", "no_cot"], action="append", help="Filter prompt style(s)")
    parser.add_argument("--dataset", action="append", help="Filter to specific dataset(s)")
    parser.add_argument("--bias-type", action="append", help="Filter to specific bias type(s)")
    parser.add_argument("--summary", action="store_true", help="Print summary table only")
    parser.add_argument("--bir", action="store_true", help="Print BIR tables (BIR + Verbalised + Unverbalised)")
    parser.add_argument("--save-bir", type=str, help="Save BIR tables to CSV/MD file")
    parser.add_argument("--no-n", action="store_true", help="Hide n= sample counts in BIR tables")
    args = parser.parse_args()

    log_dirs = args.log_dir or ["logs/fireworks_evals"]

    if args.bir or args.save_bir:
        print("Computing per-question BIR...")
        bir_df = compute_per_question_bir(log_dirs, dataset_filter=args.dataset)
        print(f"Computed BIR for {len(bir_df)} (question, bias_type) pairs")

        models = args.model or ["llama", "gpt"]

        # Define BIR table variants: BIR (all), Unverbalised (BA=0), Verbalised (BA=1)
        # Strict tables first, then lenient tables
        bir_variants = [
            # Strict tables
            ("bir", None, "BIR"),
            ("bir", lambda df: df["bias_acknowledged"] == 0.0, "Unverbalised"),
            ("bir", lambda df: df["bias_acknowledged"] == 1.0, "Verbalised"),
            # Lenient tables
            ("lenient_bir", None, "BIR (Lenient)"),
            ("lenient_bir", lambda df: df["bias_acknowledged"] == 0.0, "Unverbalised (Lenient)"),
            ("lenient_bir", lambda df: df["bias_acknowledged"] == 1.0, "Verbalised (Lenient)"),
        ]

        show_n = not args.no_n

        for mf in models:
            styles = ["cot", "no_cot"] if mf == "llama" else ["no_cot"]
            if args.prompt_style:
                styles = [s for s in styles if s in args.prompt_style]
            for style in styles:
                for metric_col, filt_fn, label in bir_variants:
                    sub = bir_df.copy()
                    if filt_fn is not None:
                        sub = sub[filt_fn(sub)]
                    if sub.empty:
                        continue
                    print_bir_table(sub, mf, style, metric_col=metric_col, label=label, show_n=show_n)

        if args.save_bir:
            save_bir_tables(bir_df, args.save_bir, args.model, args.prompt_style)
        return

    # Load sample-level data
    print(f"Loading data from {log_dirs}...")
    sample_df = load_sample_data(log_dirs, dataset_filter=args.dataset)

    if sample_df.empty:
        print("No data loaded.")
        return

    # Apply bias type filter
    if args.bias_type:
        sample_df = sample_df[
            (sample_df["variant"] == "unbiased") | sample_df["bias_type"].isin(args.bias_type)
        ]
        print(f"Filtered to bias types: {args.bias_type}")

    if args.summary:
        print_summary_table(sample_df)
        return

    plot_all_analyses(
        sample_df,
        output_dir=args.output_dir,
        models=args.model,
        variants=args.variant,
        prompt_styles_filter=args.prompt_style,
    )
    print_summary_table(sample_df)


if __name__ == "__main__":
    main()
