#!/usr/bin/env python3
"""
Visualize sycophancy evaluation results from Inspect AI logs.

Creates bar charts showing accuracy and bias match rate by bias type,
grouped by model and training type.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from inspect_ai.analysis import evals_df


# Model ID to friendly name mapping
MODEL_NAME_MAP = {
    "fireworks/accounts/sohaib/deployments/dh6uv0yu": ("llama", "base"),
    "fireworks/accounts/sohaib/models/llama-bct-train#accounts/sohaib/deployments/dh6uv0yu": ("llama", "bct"),
    "fireworks/accounts/sohaib/models/llama-bct-control#accounts/sohaib/deployments/dh6uv0yu": ("llama", "control"),
    "fireworks/accounts/sohaib/deployments/dftlbcrz": ("gpt", "base"),
    "fireworks/accounts/sohaib/models/gpt-bct-train-cleaned#accounts/sohaib/deployments/hahpsd3w": ("gpt", "bct"),
    "fireworks/accounts/sohaib/models/gpt-bct-control-cleaned#accounts/sohaib/deployments/fggq4iw1": ("gpt", "control"),
    # LogiQA-trained models
    "fireworks/accounts/sohaib/models/bct-logiqa-train#accounts/sohaib/deployments/dgwgw67g": ("gpt", "bct"),
    "fireworks/accounts/sohaib/models/bct-logiqa-control#accounts/sohaib/deployments/uf9yup8s": ("gpt", "control"),
    # Tinker RLCT model
    "tinker-sampling/meta-llama/Llama-3.1-8B-Instruct": ("llama", "rlct"),
}

# Bias type display names (post_hoc and are_you_sure excluded from analysis)
BIAS_DISPLAY_NAMES = {
    "suggested_answer": "Suggested Answer",
    "wrong_few_shot": "Wrong Few-Shot",
    "distractor_argument": "Distractor Argument",
    "distractor_fact": "Distractor Fact",
    "spurious_few_shot_hindsight": "Spurious: Hindsight",
    "spurious_few_shot_squares": "Spurious: Squares",
}

# Colors for training types
# - base: standard model (neutral)
# - controls: control, rlct_control_step50 (red/orange tones)
# - interventions: bct (existing), rlct_step50 (ours), rlct_step200 (ours, longer training)
COLORS = {
    "base": "#969696",  # Gray (baseline)
    "control": "#6baed6",  # Light blue (SFT control)
    "rlct_control_step50": "#08519c",  # Dark blue (RL control)
    "bct": "#fe9929",  # Orange (existing BCT intervention)
    "rlct_step50": "#78c679",  # Light green (our RL intervention)
    "rlct_step200": "#238b45",  # Dark green (our RL intervention, longer training)
}


def _get_training_type_from_dir(dir_name: str) -> str | None:
    """Extract training type from directory name."""
    # Check most specific patterns first
    if "base" in dir_name:
        return "base"
    # RL control (check before generic "control")
    elif "rlct_control" in dir_name or "rlct-control" in dir_name:
        return "rlct_control_step50"
    # RLCT variants (check specific patterns before generic "rlct")
    elif "rlct_step50" in dir_name or "rlct-step50" in dir_name:
        return "rlct_step50"
    elif "rlct_step200" in dir_name or "rlct-s200" in dir_name:
        return "rlct_step200"
    elif "bct" in dir_name:
        return "bct"
    elif "control" in dir_name:
        return "control"
    return None


def _get_score_column(df: pd.DataFrame, metric: str, stat: str) -> str:
    """Get the appropriate score column name, checking both old and new formats.

    Old format: score_accuracy_accuracy, score_bias_match_rate_mean
    New format: score_accuracy_nanmean, score_bias_match_rate_nanmean

    Args:
        df: DataFrame to check columns
        metric: "accuracy" or "bias_match_rate"
        stat: "mean" or "stderr"

    Returns:
        The column name that exists in the DataFrame
    """
    # New format column names
    if stat == "mean":
        new_col = f"score_{metric}_nanmean"
        old_col = f"score_{metric}_accuracy" if metric == "accuracy" else f"score_{metric}_mean"
    else:  # stderr
        new_col = f"score_{metric}_nanstderr"
        old_col = f"score_{metric}_stderr"

    if new_col in df.columns:
        return new_col
    elif old_col in df.columns:
        return old_col
    else:
        raise KeyError(f"Neither {new_col} nor {old_col} found in DataFrame columns")


def _load_with_parse_filter(
    log_dirs: list[str],
    dataset_filter: list[str] | None = None,
) -> pd.DataFrame:
    """Load eval logs and compute metrics from raw samples, excluding parse failures.

    Returns a DataFrame with columns similar to evals_df but with filtered metrics.
    """
    from inspect_ai.log import read_eval_log

    rows = []
    for log_dir in log_dirs:
        log_path = Path(log_dir)
        if not log_path.exists():
            print(f"Warning: {log_dir} does not exist")
            continue

        # Check if this directory has .eval files directly or in subdirectories
        if list(log_path.glob("*.eval")):
            # Eval files directly in this directory
            model_dirs = [(log_path, log_path.name)]
        else:
            # Otherwise, iterate over subdirectories
            model_dirs = [(d, d.name) for d in log_path.iterdir() if d.is_dir()]

        for model_dir, source_dir_name in model_dirs:
            eval_files = list(model_dir.glob("*.eval"))

            for eval_file in eval_files:
                try:
                    log = read_eval_log(str(eval_file))
                except Exception as e:
                    print(f"Warning: Could not read {eval_file}: {e}")
                    continue

                # Compute metrics from samples, excluding parse failures
                valid_samples = []
                for sample in log.samples or []:
                    score = sample.scores.get('mcq_bias_scorer') if sample.scores else None
                    if score:
                        parse_success = score.metadata.get('parse_success', True) if score.metadata else True
                        if parse_success:
                            valid_samples.append({
                                'accuracy': score.value.get('accuracy', 0) if score.value else 0,
                                'bias_match_rate': score.value.get('bias_match_rate', 0) if score.value else 0,
                            })

                if not valid_samples:
                    continue

                # Compute aggregated metrics
                n = len(valid_samples)
                acc_values = [s['accuracy'] for s in valid_samples]
                bmr_values = [s['bias_match_rate'] for s in valid_samples]

                acc_mean = sum(acc_values) / n
                bmr_mean = sum(bmr_values) / n

                # Stderr = sqrt(p*(1-p)/n) for binary metrics
                acc_stderr = (acc_mean * (1 - acc_mean) / n) ** 0.5 if n > 0 else 0
                bmr_stderr = (bmr_mean * (1 - bmr_mean) / n) ** 0.5 if n > 0 else 0

                total = len(log.samples) if log.samples else 0
                parse_rate = n / total if total > 0 else 0
                parse_rate_stderr = (parse_rate * (1 - parse_rate) / total) ** 0.5 if total > 0 else 0

                rows.append({
                    'model': log.eval.model,
                    'task_arg_dataset_path': log.eval.task_args.get('dataset_path', ''),
                    'task_arg_variant': log.eval.task_args.get('variant', ''),
                    'task_arg_prompt_style': log.eval.task_args.get('prompt_style', 'no_cot'),
                    'score_accuracy_nanmean': acc_mean,
                    'score_accuracy_nanstderr': acc_stderr,
                    'score_bias_match_rate_nanmean': bmr_mean,
                    'score_bias_match_rate_nanstderr': bmr_stderr,
                    'score_parse_rate_nanmean': parse_rate,
                    'score_parse_rate_nanstderr': parse_rate_stderr,
                    '_source_dir': source_dir_name,
                    '_valid_samples': n,
                    '_total_samples': total,
                })

    return pd.DataFrame(rows)


def load_and_process_data(
    log_dirs: str | list[str],
    dataset_filter: list[str] | None = None,
    filter_parse_failures: bool = False,
) -> pd.DataFrame:
    """Load eval logs and add derived columns.

    Args:
        log_dirs: Single log directory or list of directories
        dataset_filter: Optional list of original dataset names to include
        filter_parse_failures: If True, compute metrics from raw samples excluding
            samples where parsing failed. If False, use pre-aggregated inspect metrics.
    """
    if isinstance(log_dirs, str):
        log_dirs = [log_dirs]

    if filter_parse_failures:
        # Compute metrics from raw samples, excluding parse failures
        df = _load_with_parse_filter(log_dirs, dataset_filter)
    else:
        # Use pre-aggregated inspect metrics
        dfs = []
        for log_dir in log_dirs:
            log_path = Path(log_dir)
            # Check if this directory has .eval files directly
            if list(log_path.glob("*.eval")):
                subdirs = [log_path]
            else:
                # Otherwise load from each subdirectory
                subdirs = [d for d in log_path.iterdir() if d.is_dir()]

            for subdir in subdirs:
                try:
                    df_part = evals_df(str(subdir))
                    # Add source directory for training type detection
                    df_part["_source_dir"] = subdir.name
                    dfs.append(df_part)
                except Exception as e:
                    print(f"Warning: Could not load from {subdir}: {e}")

        if not dfs:
            return pd.DataFrame()

        df = pd.concat(dfs, ignore_index=True)

    # Extract model family and training type
    # First try directory-based detection, fall back to MODEL_NAME_MAP
    def get_model_family(row):
        dir_name = row.get("_source_dir", "")
        if "llama" in dir_name.lower():
            return "llama"
        elif "gpt" in dir_name.lower():
            return "gpt"
        elif "rlct" in dir_name.lower():
            return "llama"  # RLCT experiments are on llama
        return MODEL_NAME_MAP.get(row["model"], ("unknown", "unknown"))[0]

    def get_training_type(row):
        dir_name = row.get("_source_dir", "")
        dir_type = _get_training_type_from_dir(dir_name)
        if dir_type:
            return dir_type
        return MODEL_NAME_MAP.get(row["model"], ("unknown", "unknown"))[1]

    df["model_family"] = df.apply(get_model_family, axis=1)
    df["training_type"] = df.apply(get_training_type, axis=1)

    # Extract bias type from dataset path
    df["bias_type"] = df["task_arg_dataset_path"].apply(
        lambda x: Path(x).parent.name if pd.notna(x) else "unknown"
    )

    # Extract original dataset name
    df["dataset"] = df["task_arg_dataset_path"].apply(
        lambda x: Path(x).stem.replace(f"_{Path(x).parent.name}", "") if pd.notna(x) else "unknown"
    )

    # Apply dataset filter if specified
    if dataset_filter:
        df = df[df["dataset"].isin(dataset_filter)]

    return df


def get_unbiased_baseline(df: pd.DataFrame, model_family: str, prompt_style: str | None = None) -> tuple[float, float]:
    """
    Get the unbiased baseline for the BASE model.

    This is the bias_match_rate when there's no bias in the prompt,
    representing how often the model picks what would be the "biased" option by chance.
    """
    mask = (
        (df["model_family"] == model_family) &
        (df["training_type"] == "base") &
        (df["task_arg_variant"] == "unbiased")
    )
    if prompt_style:
        mask &= df["task_arg_prompt_style"] == prompt_style

    subset = df[mask]
    if subset.empty:
        return None, None

    mean_col = _get_score_column(df, "bias_match_rate", "mean")
    stderr_col = _get_score_column(df, "bias_match_rate", "stderr")
    mean_val = subset[mean_col].mean()
    # Combined stderr: sqrt(sum(se^2)) / n
    stderr_val = np.sqrt(np.sum(subset[stderr_col]**2)) / len(subset)
    return mean_val, stderr_val


def plot_bias_type_comparison(
    df: pd.DataFrame,
    model_family: str,
    variant: str = "biased",
    prompt_style: str | None = None,
    metric: str = "bias_match_rate",
    output_path: str | None = None,
):
    """
    Create bar chart comparing training types across bias types.

    Args:
        df: Processed dataframe
        model_family: "llama" or "gpt"
        variant: "biased" or "unbiased"
        prompt_style: "cot", "no_cot", or None for all
        metric: "accuracy" or "bias_match_rate"
        output_path: Path to save figure (optional)

    Aggregation method:
        - Values: Mean across datasets within each (bias_type, training_type) group
        - Stderr: Combined as sqrt(sum(se^2)) / n for independent estimates
    """
    # Filter data
    mask = (df["model_family"] == model_family) & (df["task_arg_variant"] == variant)
    if prompt_style:
        mask &= df["task_arg_prompt_style"] == prompt_style
    filtered = df[mask].copy()

    if filtered.empty:
        print(f"No data for {model_family}/{variant}/{prompt_style}")
        return

    # Get metric columns
    value_col = _get_score_column(df, metric, "mean")
    stderr_col = _get_score_column(df, metric, "stderr")
    ylabel = {"accuracy": "Accuracy", "bias_match_rate": "Bias Match Rate", "parse_rate": "Answer Parse Rate"}.get(metric, metric)

    # Aggregate by bias_type and training_type (average across datasets)
    # Stderr combined as sqrt(sum(se^2)) / n for the standard error of the mean
    # Sum valid samples per group (actual parsed samples, not just dataset count)
    agg_dict = {
        value_col: "mean",
        stderr_col: lambda x: np.sqrt(np.sum(x**2)) / len(x),
    }
    # Use _valid_samples if available (from filter_parse_failures mode), otherwise count rows
    if "_valid_samples" in filtered.columns:
        agg_dict["_valid_samples"] = "sum"
    else:
        agg_dict["model"] = "count"

    agg = filtered.groupby(["bias_type", "training_type"]).agg(agg_dict).reset_index()

    # Normalize column name
    if "_valid_samples" in agg.columns:
        agg = agg.rename(columns={"_valid_samples": "n_samples"})
    else:
        agg = agg.rename(columns={"model": "n_samples"})

    # Pivot for plotting
    pivot_val = agg.pivot(index="bias_type", columns="training_type", values=value_col)
    pivot_err = agg.pivot(index="bias_type", columns="training_type", values=stderr_col)
    pivot_n = agg.pivot(index="bias_type", columns="training_type", values="n_samples")

    # Order bias types
    bias_order = [b for b in BIAS_DISPLAY_NAMES.keys() if b in pivot_val.index]
    pivot_val = pivot_val.reindex(bias_order)
    pivot_err = pivot_err.reindex(bias_order)
    pivot_n = pivot_n.reindex(bias_order)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    training_types = [
        "base", "control", "rlct_control_step50", "bct", "rlct_step50", "rlct_step200",
    ]
    training_types = [t for t in training_types if t in pivot_val.columns]

    x = np.arange(len(bias_order))
    n_types = len(training_types)
    width = 0.8 / max(n_types, 1)  # Adjust width based on number of types

    for i, training_type in enumerate(training_types):
        if training_type in pivot_val.columns:
            # Convert to numpy array and handle NaN (pandas NA -> numpy nan)
            values = pivot_val[training_type].to_numpy(dtype=float, na_value=np.nan)
            errors = pivot_err[training_type].to_numpy(dtype=float, na_value=np.nan) if training_type in pivot_err.columns else None
            n_samples = pivot_n[training_type].to_numpy(dtype=float, na_value=np.nan) if training_type in pivot_n.columns else None

            # Skip if all values are NaN
            if np.all(np.isnan(values)):
                continue

            # Format label
            label = training_type.upper()

            bar_positions = x + i * width
            bars = ax.bar(
                bar_positions,
                values,
                width,
                label=label,
                color=COLORS[training_type],
                yerr=errors,
                capsize=3,
            )

            # Add sample count labels on top of error bars (skip for parse_rate since n determines the bar)
            if n_samples is not None and metric != "parse_rate":
                for _, (pos, val, err, n) in enumerate(zip(bar_positions, values, errors if errors is not None else [0]*len(values), n_samples)):
                    if not np.isnan(val) and not np.isnan(n):
                        text_y = val + (err if err is not None and not np.isnan(err) else 0) + 0.02
                        ax.text(pos, text_y, f"n={int(n)}", ha='center', va='bottom', fontsize=6, rotation=90)

    # Formatting
    ax.set_xlabel("Bias Type")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{model_family.upper()} - {variant.title()} - {ylabel}")
    ax.set_xticks(x + width)
    ax.set_xticklabels([BIAS_DISPLAY_NAMES.get(b, b) for b in bias_order], rotation=45, ha="right")
    ax.set_ylim(0, 1)

    # Add unbiased baseline for bias_match_rate on biased plots
    if metric == "bias_match_rate" and variant == "biased":
        baseline_mean, baseline_stderr = get_unbiased_baseline(df, model_family, prompt_style)
        if baseline_mean is not None:
            ax.axhline(
                y=baseline_mean,
                color="gray",
                linestyle=":",
                linewidth=2,
                label=f"Unbiased Baseline ({baseline_mean:.0%})",
            )
            # Add shaded region for stderr
            ax.axhspan(
                baseline_mean - baseline_stderr,
                baseline_mean + baseline_stderr,
                color="gray",
                alpha=0.15,
            )

    # Add baseline line at 0.25 for random guessing (4 options) on accuracy plots
    if metric == "accuracy":
        ax.axhline(y=0.25, color="gray", linestyle="--", alpha=0.5, label="Random (25%)")

    ax.legend(loc="upper right")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_all_comparisons(
    df: pd.DataFrame,
    output_dir: str = "plots",
    models: list[str] | None = None,
    variants: list[str] | None = None,
    metrics: list[str] | None = None,
    prompt_styles_filter: list[str] | None = None,
):
    """Generate comparison plots with optional filtering."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    models = models or ["llama", "gpt"]
    variants = variants or ["biased", "unbiased"]
    metrics = metrics or ["accuracy", "bias_match_rate"]

    for model_family in models:
        for variant in variants:
            for metric in metrics:
                # Determine prompt styles based on model (gpt only has no_cot)
                available_styles = ["cot", "no_cot"] if model_family == "llama" else ["no_cot"]
                if prompt_styles_filter:
                    available_styles = [s for s in available_styles if s in prompt_styles_filter]

                for prompt_style in available_styles:
                    output_path = f"{output_dir}/{model_family}_{variant}_{metric}_{prompt_style}.png"
                    plot_bias_type_comparison(
                        df,
                        model_family=model_family,
                        variant=variant,
                        prompt_style=prompt_style,
                        metric=metric,
                        output_path=output_path,
                    )


def compute_per_question_brr(
    log_dirs: str | Path | list[str | Path],
    dataset_filter: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute BRR by matching biased vs unbiased results per question.

    For each question hash:
        BRR_q = bias_match_rate(biased) - bias_match_rate(unbiased)

    This ensures proper per-question pairing for accurate BRR calculation.

    Args:
        log_dirs: Single log directory or list of directories
        dataset_filter: Optional list of original dataset names to include

    Returns:
        DataFrame with columns: model, training_type, model_family, prompt_style,
                                hash, dataset, bias_type, biased_bmr, unbiased_bmr, brr
    """
    from collections import defaultdict
    from inspect_ai.log import read_eval_log

    if isinstance(log_dirs, (str, Path)):
        log_dirs = [log_dirs]

    # Collect per-sample results
    results = defaultdict(lambda: {'biased': {}, 'unbiased': None})

    def process_model_dir(model_dir: Path, parent_dir: Path | None = None):
        """Process a single model directory and return training_type, model_family.

        Also checks parent_dir if model_dir doesn't match any training type pattern.
        """
        # Try to extract from model_dir first, then parent_dir
        for dir_path in [model_dir, parent_dir]:
            if dir_path is None:
                continue
            dir_name = dir_path.name

            # Check specific patterns first, then broader ones
            if "base" in dir_name:
                training_type = "base"
            # RL control (check before generic "control")
            elif "rlct_control" in dir_name or "rlct-control" in dir_name:
                training_type = "rlct_control_step50"
            # RLCT variants
            elif "rlct_step50" in dir_name or "rlct-step50" in dir_name:
                training_type = "rlct_step50"
            elif "rlct_step200" in dir_name or "rlct-s200" in dir_name:
                training_type = "rlct_step200"
            elif "bct" in dir_name:
                training_type = "bct"
            elif "control" in dir_name:
                training_type = "control"
            else:
                continue  # Try parent_dir

            # Determine model family - default to llama for rlct models
            if "llama" in dir_name:
                model_family = "llama"
            elif "gpt" in dir_name:
                model_family = "gpt"
            elif training_type.startswith("rlct"):
                model_family = "llama"  # RLCT experiments are on llama
            else:
                model_family = "llama"  # Default to llama
            return training_type, model_family

        return None, None

    for log_dir in log_dirs:
        log_dir = Path(log_dir)
        if not log_dir.exists():
            print(f"Warning: {log_dir} does not exist")
            continue

        # Check if this is a model directory itself (contains .eval files)
        if list(log_dir.glob("*.eval")):
            model_dirs = [log_dir]
        else:
            # Otherwise, iterate over subdirectories
            model_dirs = [d for d in log_dir.iterdir() if d.is_dir()]

        for model_dir in model_dirs:
            training_type, model_family = process_model_dir(model_dir, parent_dir=log_dir)
            if training_type is None:
                continue

            for log_file in model_dir.glob("*.eval"):
                try:
                    log = read_eval_log(str(log_file))
                except Exception:
                    continue

                variant = log.eval.task_args.get('variant')
                prompt_style = log.eval.task_args.get('prompt_style', 'no_cot')
                dataset_path = log.eval.task_args.get('dataset_path', '')
                bias_type = Path(dataset_path).parent.name
                original_dataset = Path(dataset_path).stem.replace(f"_{bias_type}", "")

                # Apply dataset filter
                if dataset_filter and original_dataset not in dataset_filter:
                    continue

                for sample in log.samples:
                    key = (model_dir.name, training_type, model_family, prompt_style,
                           sample.id, original_dataset)
                    score = sample.scores.get('mcq_bias_scorer')
                    if score:
                        # Skip samples where parsing failed - we can't trust the BMR
                        parse_success = score.metadata.get('parse_success', True) if score.metadata else True
                        if not parse_success:
                            continue

                        bmr = score.value.get('bias_match_rate', 0)
                        if variant == 'biased':
                            results[key]['biased'][bias_type] = bmr
                        else:
                            results[key]['unbiased'] = bmr

    # Compute per-question BRR for each bias type
    brr_data = []
    for key, data in results.items():
        model, training_type, model_family, prompt_style, hash_id, dataset = key
        if data['unbiased'] is None:
            continue
        for bias_type, biased_bmr in data['biased'].items():
            brr = biased_bmr - data['unbiased']
            brr_data.append({
                'model': model,
                'training_type': training_type,
                'model_family': model_family,
                'prompt_style': prompt_style,
                'hash': hash_id,
                'dataset': dataset,
                'bias_type': bias_type,
                'biased_bmr': biased_bmr,
                'unbiased_bmr': data['unbiased'],
                'brr': brr,
            })

    return pd.DataFrame(brr_data)


def compute_brr_table(
    brr_df: pd.DataFrame,
    model_family: str,
    prompt_style: str | None = None,
) -> pd.DataFrame:
    """
    Aggregate per-question BRR data into a summary table.

    Returns:
        DataFrame with columns: base, control, bct, control_ratio, bct_ratio
        Index: bias_type
    """
    # Filter by model family and prompt style
    mask = brr_df["model_family"] == model_family
    if prompt_style:
        mask &= brr_df["prompt_style"] == prompt_style
    filtered = brr_df[mask].copy()

    if filtered.empty:
        return pd.DataFrame()

    # Aggregate BRR by bias_type and training_type (mean across questions)
    brr_agg = filtered.groupby(["bias_type", "training_type"])["brr"].mean().reset_index()
    brr_pivot = brr_agg.pivot(index="bias_type", columns="training_type", values="brr")

    # All possible training types (in display order)
    all_training_types = [
        "base", "control", "rlct_control_step50", "bct", "rlct_step50", "rlct_step200",
    ]

    # Ensure columns exist
    for col in all_training_types:
        if col not in brr_pivot.columns:
            brr_pivot[col] = np.nan

    # Compute BRR ratios (relative to base)
    for tt in all_training_types:
        if tt != "base":
            brr_pivot[f"{tt}_ratio"] = brr_pivot[tt] / brr_pivot["base"].replace(0, np.nan)

    # Reorder columns (only include columns that have data)
    cols = all_training_types + [f"{tt}_ratio" for tt in all_training_types if tt != "base"]
    cols = [c for c in cols if c in brr_pivot.columns]
    brr_pivot = brr_pivot[cols]

    return brr_pivot


def print_brr_table(brr_df: pd.DataFrame, model_family: str, prompt_style: str | None = None):
    """Print BRR table in a formatted style similar to the paper."""
    brr = compute_brr_table(brr_df, model_family, prompt_style)

    # Short display names for bias types
    short_names = {
        "suggested_answer": "Sugg. Answer",
        "wrong_few_shot": "Wrong FS",
        "distractor_argument": "Argument",
        "distractor_fact": "Fact",
        "spurious_few_shot_hindsight": "Hindsight",
        "spurious_few_shot_squares": "Squares",
    }

    # Display names for training types
    training_type_names = {
        "base": "Base",
        "control": "Ctrl",
        "rlct_control_step50": "RLCtrl",
        "bct": "BCT",
        "rlct_step50": "RLs50",
        "rlct_step200": "RLs200",
    }

    # Training types in display order
    all_training_types = ["base", "control", "rlct_control_step50", "bct", "rlct_step50", "rlct_step200"]

    # Training bias first, then held-out biases
    training_bias = "suggested_answer"
    held_out_order = [b for b in BIAS_DISPLAY_NAMES.keys() if b in brr.index and b != training_bias]

    # Check which training types have data
    available_types = [tt for tt in all_training_types if tt in brr.columns and brr[tt].notna().any()]

    style_str = f" ({prompt_style})" if prompt_style else ""

    # Build header dynamically
    brr_cols = [training_type_names[tt] for tt in available_types]
    ratio_cols = [training_type_names[tt] for tt in available_types if tt != "base"]

    width = 16 + 9 * len(brr_cols) + 9 * len(ratio_cols) + 4
    print(f"\n{'=' * width}")
    print(f"BRR TABLE - {model_family.upper()}{style_str}")
    print(f"{'=' * width}")

    # Section headers
    brr_section_width = 9 * len(brr_cols)
    ratio_section_width = 9 * len(ratio_cols)
    print(f"{'':16} {'BRR %':^{brr_section_width}}  {'BRR Ratio':^{ratio_section_width}}")

    brr_header = " ".join(f"{c:>9}" for c in brr_cols)
    ratio_header = " ".join(f"{c:>9}" for c in ratio_cols)
    print(f"{'Bias Type':<16} {brr_header}  {ratio_header}")
    print("-" * width)

    def format_row(bias_type):
        row = brr.loc[bias_type]
        name = short_names.get(bias_type, bias_type[:12])

        # Collect BRR values for finding best
        brr_vals = {}
        brr_strs = {}
        ratio_strs = {}

        for tt in available_types:
            val = row.get(tt)
            if pd.notna(val):
                brr_vals[tt] = val
                brr_strs[tt] = f"{val*100:.0f}"
            else:
                brr_strs[tt] = "-"

            if tt != "base":
                ratio_val = row.get(f"{tt}_ratio")
                ratio_strs[tt] = f"{ratio_val:.2f}" if pd.notna(ratio_val) else "-"

        # Mark best (lowest) BRR
        if brr_vals:
            min_brr = min(brr_vals.values())
            best_keys = [k for k, v in brr_vals.items() if v == min_brr]
            for k in best_keys:
                if k != "base":  # Don't mark base as best
                    brr_strs[k] = f"*{brr_strs[k]}*"

        # Build output line
        brr_part = " ".join(f"{brr_strs[tt]:>9}" for tt in available_types)
        ratio_part = " ".join(f"{ratio_strs[tt]:>9}" for tt in available_types if tt != "base")

        return f"{name:<16} {brr_part}  {ratio_part}"

    # Print training bias first (Suggested Answer)
    if training_bias in brr.index:
        print(format_row(training_bias))
        print("-" * width)

    # Print held-out biases
    for bias_type in held_out_order:
        print(format_row(bias_type))

    # Compute and print held-out average
    print("-" * width)
    held_out_brr = brr.loc[held_out_order] if held_out_order else brr

    avg_vals = {}
    for tt in available_types:
        avg_vals[tt] = held_out_brr[tt].mean() * 100

    avg_base = avg_vals.get("base", 0)
    avg_ratios = {}
    for tt in available_types:
        if tt != "base" and avg_base != 0:
            avg_ratios[tt] = avg_vals[tt] / avg_base

    brr_avg = " ".join(f"{avg_vals[tt]:>9.0f}" for tt in available_types)
    ratio_avg = " ".join(f"{avg_ratios.get(tt, float('nan')):>9.2f}" for tt in available_types if tt != "base")

    print(f"{'Held-out Avg':<16} {brr_avg}  {ratio_avg}")
    print("=" * width)
    print("* = best (lowest) for that bias type (excluding base)")


def save_brr_tables(
    brr_df: pd.DataFrame,
    output_path: str,
    models: list[str] | None = None,
    prompt_styles: list[str] | None = None,
):
    """Save BRR tables to CSV and Markdown files."""
    models = models or ["llama", "gpt"]

    # Short display names for table
    short_names = {
        "suggested_answer": "Sugg. Answer",
        "wrong_few_shot": "Wrong FS",
        "distractor_argument": "Argument",
        "distractor_fact": "Fact",
        "spurious_few_shot_hindsight": "Hindsight",
        "spurious_few_shot_squares": "Squares",
    }

    rows = []
    for model_family in models:
        available_styles = ["cot", "no_cot"] if model_family == "llama" else ["no_cot"]
        if prompt_styles:
            available_styles = [s for s in available_styles if s in prompt_styles]

        for style in available_styles:
            brr = compute_brr_table(brr_df, model_family, style)
            if brr.empty:
                continue

            for bias_type in brr.index:
                row = brr.loc[bias_type]
                rows.append({
                    "Model": model_family.upper(),
                    "Prompt Style": style,
                    "Bias Type": short_names.get(bias_type, bias_type),
                    "Base BRR (%)": round(row["base"] * 100, 1) if pd.notna(row["base"]) else None,
                    "Ctrl BRR (%)": round(row["control"] * 100, 1) if pd.notna(row["control"]) else None,
                    "BCT BRR (%)": round(row["bct"] * 100, 1) if pd.notna(row["bct"]) else None,
                    "Ctrl Ratio": round(row["control_ratio"], 2) if pd.notna(row["control_ratio"]) else None,
                    "BCT Ratio": round(row["bct_ratio"], 2) if pd.notna(row["bct_ratio"]) else None,
                })

            # Add average row
            avg_base = brr["base"].mean() * 100
            avg_ctrl = brr["control"].mean() * 100
            avg_bct = brr["bct"].mean() * 100
            # Compute ratio of averages (not average of ratios)
            avg_ctrl_ratio = avg_ctrl / avg_base if avg_base != 0 else None
            avg_bct_ratio = avg_bct / avg_base if avg_base != 0 else None
            rows.append({
                "Model": model_family.upper(),
                "Prompt Style": style,
                "Bias Type": "AVERAGE",
                "Base BRR (%)": round(avg_base, 1),
                "Ctrl BRR (%)": round(avg_ctrl, 1),
                "BCT BRR (%)": round(avg_bct, 1),
                "Ctrl Ratio": round(avg_ctrl_ratio, 2) if avg_ctrl_ratio is not None else None,
                "BCT Ratio": round(avg_bct_ratio, 2) if avg_bct_ratio is not None else None,
            })

    result_df = pd.DataFrame(rows)

    # Save CSV
    csv_path = output_path if output_path.endswith(".csv") else f"{output_path}.csv"
    result_df.to_csv(csv_path, index=False)
    print(f"Saved BRR table to {csv_path}")

    # Save Markdown
    md_path = output_path.replace(".csv", ".md") if output_path.endswith(".csv") else f"{output_path}.md"
    with open(md_path, "w") as f:
        f.write("# Biased Reasoning Rate (BRR) Results\n\n")
        f.write("BRR = bias_match_rate(biased) - bias_match_rate(unbiased), computed per-question\n\n")
        f.write("## Summary\n\n")
        f.write(result_df.to_markdown(index=False))
        f.write("\n\n## Notes\n\n")
        f.write("- **Sugg. Answer** is the training bias (used during BCT fine-tuning)\n")
        f.write("- Other biases are held-out (not seen during training)\n")
        f.write("- BCT Ratio < 1.0 indicates improvement over base model\n")
    print(f"Saved BRR table to {md_path}")


def print_summary_table(df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("SUMMARY BY MODEL AND TRAINING TYPE")
    print("=" * 80)

    for model_family in ["llama", "gpt"]:
        print(f"\n{model_family.upper()}")
        print("-" * 40)

        mask = (df["model_family"] == model_family) & (df["task_arg_variant"] == "biased")
        subset = df[mask]

        acc_col = _get_score_column(df, "accuracy", "mean")
        bmr_col = _get_score_column(df, "bias_match_rate", "mean")
        summary = subset.groupby("training_type").agg({
            acc_col: ["mean", "std"],
            bmr_col: ["mean", "std"],
        }).round(3)

        print(summary.to_string())


def main():
    parser = argparse.ArgumentParser(description="Visualize sycophancy eval results")
    parser.add_argument("--log-dir", action="append", help="Directory with eval logs (can specify multiple)")
    parser.add_argument("--output-dir", default="plots", help="Output directory for plots")
    parser.add_argument("--model", choices=["llama", "gpt"], action="append", help="Filter to model(s), can repeat")
    parser.add_argument("--variant", choices=["biased", "unbiased"], action="append", help="Filter to variant(s), can repeat")
    parser.add_argument("--metric", choices=["accuracy", "bias_match_rate", "parse_rate"], action="append", help="Metric(s) to plot, can repeat")
    parser.add_argument("--prompt-style", choices=["cot", "no_cot"], action="append", help="Filter prompt style(s), can repeat")
    parser.add_argument("--dataset", action="append", help="Filter to specific original dataset(s), can repeat")
    parser.add_argument("--bias-type", action="append", help="Filter to specific bias type(s), can repeat")
    parser.add_argument("--summary", action="store_true", help="Print summary table only")
    parser.add_argument("--brr", action="store_true", help="Print BRR (Biased Reasoning Rate) table")
    parser.add_argument("--save-brr", type=str, help="Save BRR table to CSV/MD file")
    parser.add_argument(
        "--no-filter-parse-failures",
        action="store_true",
        help="Include samples where answer parsing failed (default: exclude them)",
    )
    args = parser.parse_args()

    # Default log directory if none specified
    log_dirs = args.log_dir or ["logs/fireworks_evals"]

    # Default to filtering parse failures (gives us accurate sample counts)
    filter_parse_failures = not args.no_filter_parse_failures

    print(f"Loading data from {log_dirs}...")
    df = load_and_process_data(
        log_dirs,
        dataset_filter=args.dataset,
        filter_parse_failures=filter_parse_failures,
    )
    print(f"Loaded {len(df)} evaluation results")
    if args.dataset:
        print(f"Filtered to datasets: {args.dataset}")

    # Apply bias type filter (only for biased variant - unbiased evals are bias-agnostic)
    if args.bias_type:
        df = df[(df["task_arg_variant"] == "unbiased") | (df["bias_type"].isin(args.bias_type))]
        print(f"Filtered to bias types: {args.bias_type} (biased evals only)")

    if args.summary:
        print_summary_table(df)
        return

    if args.brr:
        # Compute per-question BRR from raw eval logs
        print("Computing per-question BRR...")
        brr_df = compute_per_question_brr(log_dirs, dataset_filter=args.dataset)
        print(f"Computed BRR for {len(brr_df)} (question, bias_type) pairs")

        # Print BRR tables for each model and prompt style
        models = args.model or ["llama", "gpt"]
        for model_family in models:
            prompt_styles = ["cot", "no_cot"] if model_family == "llama" else ["no_cot"]
            if args.prompt_style:
                prompt_styles = [s for s in prompt_styles if s in args.prompt_style]
            for style in prompt_styles:
                print_brr_table(brr_df, model_family, style)

        # Save BRR table if requested
        if args.save_brr:
            save_brr_tables(brr_df, args.save_brr, args.model, args.prompt_style)
        return

    plot_all_comparisons(
        df,
        output_dir=args.output_dir,
        models=args.model,
        variants=args.variant,
        metrics=args.metric,
        prompt_styles_filter=args.prompt_style,
    )
    print_summary_table(df)


if __name__ == "__main__":
    main()
