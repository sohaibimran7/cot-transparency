"""Variance-across plots: per-(bias_type, training_type) means with SE
computed across groups defined by `split_by` (e.g. seed, dataset).

Adapted from legacy `plot_variance_across` in `_archive/visualize_results.py`.
The two-stage aggregation (per-group mean → cross-group mean+SE) is expressed
on the long-form MetricFrame produced by `aggregate_metric`, so the same
renderer (`bar_plot`) can be used.

`n=` labels here count groups (not total samples).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .plot import bar_plot
from .recipes import METRIC_LABELS
from .theme import PUBLICATION_THEME


def _aggregate_variance(
    wide: pd.DataFrame,
    metric: str,
    split_by: str,
    model_family: str,
    *,
    variant: str | None = None,
    prompt_style: str | None = None,
) -> pd.DataFrame:
    """Two-stage aggregation: per-group means, then cross-group mean+SE.

    Returns a long-form frame with columns expected by `bar_plot`:
    (model_family, prompt_style, training_type, bias_type, metric, value, stderr, n).
    `n` counts the number of valid groups (same as legacy `n_valid`).
    """
    if metric not in wide.columns or split_by not in wide.columns:
        return pd.DataFrame()

    sub = wide[wide["model_family"] == model_family]
    if variant is not None and "variant" in sub.columns:
        sub = sub[sub["variant"] == variant]
    if prompt_style is not None and "prompt_style" in sub.columns:
        sub = sub[sub["prompt_style"] == prompt_style]
    sub = sub.dropna(subset=[split_by])
    if sub.empty or sub[split_by].nunique() < 2:
        return pd.DataFrame()

    # Stage 1: per-group means
    group_means = (
        sub.groupby([split_by, "bias_type", "training_type"])[metric]
        .mean()
        .reset_index()
        .rename(columns={metric: "_group_mean"})
    )

    # Stage 2: cross-group mean + SE
    rows = []
    for (bt, tt), grp in group_means.groupby(["bias_type", "training_type"]):
        means = grp["_group_mean"].dropna()
        k = len(means)
        if k == 0:
            continue
        rows.append(
            {
                "model_family": model_family,
                "prompt_style": prompt_style,
                "training_type": tt,
                "bias_type": bt,
                "metric": metric,
                "value": float(means.mean()),
                "stderr": float(means.std(ddof=1) / k**0.5) if k > 1 else 0.0,
                "n": int(k),
            }
        )
    return pd.DataFrame(rows)


def _styles_for(model_family: str, prompt_styles: list[str] | None) -> list[str]:
    if prompt_styles:
        return list(prompt_styles)
    return ["cot", "no_cot"] if model_family == "llama" else ["no_cot"]


def render_variance_across(
    wide: pd.DataFrame,
    *,
    split_by: str,
    metrics: tuple[str, ...],
    output_dir: str | Path,
    models: list[str] | None = None,
    prompt_styles: list[str] | None = None,
    variants: list[str] | None = None,
    training_biases: frozenset[str] | None = None,
    n_labels: bool = True,
) -> int:
    """Render one variance figure per (model, prompt_style[, variant], metric).

    `wide` may be either the per-question BIR frame (no `variant` column;
    `variants` is ignored) or the sample-level frame (has `variant`; iterates).

    Returns the number of figures written.
    """
    output_dir = Path(output_dir)

    if split_by not in wide.columns:
        print(f"  Warning: column '{split_by}' not in data, skipping variance plots")
        return 0

    has_variant = "variant" in wide.columns
    iter_variants = variants if has_variant else [None]
    if has_variant and not iter_variants:
        iter_variants = ["biased", "unbiased"]

    if not models:
        models = sorted(wide["model_family"].dropna().unique().tolist())

    n_groups = wide[split_by].nunique()
    if n_groups < 2:
        print(f"Only {n_groups} group(s) for '{split_by}', skipping variance plots")
        return 0
    print(f"\nGenerating variance plots split by '{split_by}' ({n_groups} groups)...")

    n_written = 0
    for mf in models:
        for ps in _styles_for(mf, prompt_styles):
            for variant in iter_variants:
                for metric in metrics:
                    agg = _aggregate_variance(
                        wide,
                        metric,
                        split_by,
                        mf,
                        variant=variant,
                        prompt_style=ps,
                    )
                    if agg.empty:
                        continue
                    ylabel = METRIC_LABELS.get(metric, metric)
                    title_suffix = f" (Variance across {split_by})"
                    if variant is not None:
                        out_name = f"{mf}_{variant}_{metric}_{ps}_var_{split_by}.png"
                        ylabel_full = f"{ylabel} - {variant.title()}{title_suffix}"
                    else:
                        out_name = f"{mf}_{metric}_{ps}_var_{split_by}.png"
                        ylabel_full = f"{ylabel}{title_suffix}"
                    bar_plot(
                        agg,
                        metric=metric,
                        model_family=mf,
                        prompt_style=ps,
                        theme=PUBLICATION_THEME,
                        output_path=output_dir / out_name,
                        ylabel=ylabel_full,
                        n_labels=n_labels,
                    )
                    n_written += 1
    return n_written
