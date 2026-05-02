"""Aggregations and group-level reductions ported from legacy visualize_results.py.

Three pure DataFrame helpers:
  - aggregate_samples: per-(group_cols) nanmean/SEM/n
  - aggregate_training_types: remap training_type to its registry aggregate group
  - filter_to_common_questions: restrict per-question metric frame to questions
    answered (non-NaN) by every training_type within each
    (bias_type, dataset, model_family, prompt_style) group.

All registry-derived constants (training_type → aggregate group, training type
ordering) come from `viz.registry.REGISTRY` rather than legacy module globals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .registry import REGISTRY


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


def _aggregate_map() -> dict[str, str]:
    """Build a {training_type → aggregate_group} map from REGISTRY entries.

    Mirrors legacy `_AGGREGATE_MAP`, but sourced from each TrainingTypeInfo's
    `aggregate_group` attribute (set in `experiments.toml`). Entries without
    an aggregate_group are left as-is by `aggregate_training_types`.
    """
    return {
        key: info.aggregate_group
        for key, info in REGISTRY.training_types.items()
        if info.aggregate_group
    }


def aggregate_training_types(df: pd.DataFrame) -> pd.DataFrame:
    """Remap training_type column to aggregate groups."""
    df = df.copy()
    agg_map = _aggregate_map()
    df["training_type"] = df["training_type"].map(
        lambda t: agg_map.get(t, t)
    )
    return df


def filter_to_common_questions(
    bir_df: pd.DataFrame,
    metric_col: str = "bir",
) -> pd.DataFrame:
    """Filter BIR DataFrame to only questions answered by ALL training types.

    For each (bias_type, dataset, model_family, prompt_style) group, finds the
    intersection of question hashes that have non-NaN metric values across every
    training_type present. This ensures all models are compared on exactly the
    same questions, eliminating selection bias from differential parse failures.

    Returns a filtered copy of bir_df.
    """
    group_cols = ["bias_type", "dataset", "model_family", "prompt_style"]
    filtered_parts = []

    for group_key, group_df in bir_df.groupby(group_cols):
        training_types = group_df["training_type"].unique()

        # For each training type, get set of hashes with non-NaN metric
        hash_sets = []
        for tt in training_types:
            tt_df = group_df[group_df["training_type"] == tt]
            valid = tt_df[tt_df[metric_col].notna()]["hash"].unique()
            hash_sets.append(set(valid))

        # Intersection: only keep hashes present (with valid metric) in ALL models
        if hash_sets:
            common_hashes = set.intersection(*hash_sets)
        else:
            common_hashes = set()

        # Filter this group to common hashes
        part = group_df[group_df["hash"].isin(common_hashes)]
        filtered_parts.append(part)

    if not filtered_parts:
        return bir_df.iloc[:0]  # empty with same columns

    result = pd.concat(filtered_parts, ignore_index=True)
    return result
