"""Long-form MetricFrame: the canonical input shape for all transforms and plots.

Per-question DataFrame columns:
    model_family, training_type, prompt_style, bias_type, dataset, hash, seed,
    metric, value

Aggregated DataFrame columns:
    model_family, training_type, prompt_style, bias_type, metric,
    value, stderr, n
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Type alias for clarity. We don't subclass DataFrame to keep things boring.
MetricFrame = pd.DataFrame


# Wide → long melt for the per-question BIR/BSR/BA frame.
# These metric columns are emitted by visualize_results.compute_per_question_bir.
PER_QUESTION_METRIC_COLUMNS = (
    "pro_bsr",
    "anti_bsr",
    "net_bsr",
    "total_bsr",
    "bir",
    "lenient_pro_bsr",
    "lenient_anti_bsr",
    "lenient_net_bsr",
    "lenient_total_bsr",
    "lenient_bir",
    "biased_bmr",
    "unbiased_bmr",
    "biased_lenient_bmr",
    "unbiased_lenient_bmr",
    "bias_acknowledged",
)

ID_COLUMNS = (
    "model",
    "training_type",
    "model_family",
    "prompt_style",
    "hash",
    "dataset",
    "bias_type",
    "seed",
)


def melt_per_question(
    wide: pd.DataFrame,
    metric_columns: tuple[str, ...] = PER_QUESTION_METRIC_COLUMNS,
) -> MetricFrame:
    """Convert wide per-question frame (one column per metric) to long form.

    Returns one row per (sample × metric). Drops rows where the metric is NaN.
    """
    metrics = [m for m in metric_columns if m in wide.columns]
    id_cols = [c for c in ID_COLUMNS if c in wide.columns]
    long = wide.melt(id_vars=id_cols, value_vars=metrics, var_name="metric", value_name="value")
    return long


def aggregate_metric(
    long: MetricFrame,
    group_cols: tuple[str, ...] = (
        "model_family",
        "training_type",
        "prompt_style",
        "bias_type",
        "metric",
    ),
) -> MetricFrame:
    """Aggregate long-form per-question frame to (mean, stderr, n) per group.

    Binomial SE for binary-valued metrics; sample SEM for continuous.
    """
    rows = []
    for keys, grp in long.groupby(list(group_cols), dropna=False):
        vals = grp["value"].dropna().to_numpy()
        n = len(vals)
        if n == 0:
            mean, se = np.nan, np.nan
        else:
            mean = float(vals.mean())
            uniq = set(np.unique(vals).tolist())
            if uniq <= {0.0, 1.0}:
                se = float((mean * (1.0 - mean) / n) ** 0.5)
            else:
                se = float(vals.std(ddof=1) / n**0.5) if n > 1 else 0.0
        rec = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        rec.update(value=mean, stderr=se, n=n)
        rows.append(rec)
    return pd.DataFrame(rows)


def filter_metric(
    long: MetricFrame,
    *,
    metric: str | None = None,
    model_family: str | None = None,
    prompt_style: str | None = None,
    variant_predicate=None,
) -> MetricFrame:
    """Slice a long-form frame by common selectors. Returns a copy."""
    out = long
    if metric is not None:
        out = out[out["metric"] == metric]
    if model_family is not None:
        out = out[out["model_family"] == model_family]
    if prompt_style is not None and "prompt_style" in out.columns:
        out = out[out["prompt_style"] == prompt_style]
    if variant_predicate is not None:
        out = out[variant_predicate(out)]
    return out.copy()
