"""Compatibility facade for the modular visualization pipeline.

The plotting implementation lives in :mod:`sycophancy_eval_inspect.viz`, but
this module remains the stable analysis entry point. In particular,
``compute_per_question_bir`` and ``collapse_to_population_bir`` are the
canonical BIR functions used by downstream paper scripts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .viz import BIAS_DISPLAY_NAMES
from .viz.aggregations import (
    aggregate_samples,
    aggregate_training_types,
    filter_to_common_questions,
)
from .viz.cli import main
from .viz.loaders import compute_per_question_bsr as _compute_per_question_bsr
from .viz.loaders import load_samples
from .viz.tables import print_bir_table, save_bir_tables


def load_sample_data(
    log_dirs: str | Path | list[str | Path],
    dataset_filter: list[str] | None = None,
    *,
    dedup: str = "last",
) -> pd.DataFrame:
    """Load normalized eval samples, optionally filtering by dataset."""
    samples = load_samples(log_dirs, dedup=dedup)
    if dataset_filter and not samples.empty:
        samples = samples[samples["dataset"].isin(dataset_filter)].copy()
    return samples


def compute_per_question_bir(
    log_dirs: str | Path | list[str | Path],
    dataset_filter: list[str] | None = None,
    *,
    dedup: str = "last",
) -> pd.DataFrame:
    """Compute canonical matched per-question bias-switch metrics."""
    samples = load_sample_data(log_dirs, dataset_filter, dedup=dedup)
    return _compute_per_question_bsr(samples)


# Backward-compatible spelling used by some callers.
compute_per_question_bsr = compute_per_question_bir


def collapse_to_population_bir(bir_df: pd.DataFrame, signed: bool = False) -> pd.DataFrame:
    """Collapse matched per-question rows into population-level BIR.

    ``signed=False`` returns the absolute matched BMR gap; ``signed=True``
    returns the signed net gap. For ``are_you_sure``, both modes use the
    biased switching rate by convention.
    """
    if bir_df.empty:
        return pd.DataFrame()

    group_cols = [
        "model",
        "training_type",
        "model_family",
        "prompt_style",
        "dataset",
        "bias_type",
        "seed",
    ]
    rows = []
    for key, group in bir_df.groupby(group_cols, dropna=False):
        bias_type = key[group_cols.index("bias_type")]
        if bias_type == "are_you_sure":
            biased = group["biased_bmr"].dropna()
            if biased.empty:
                continue
            gap = float(biased.mean())
            n = len(biased)
            variance = biased.var(ddof=1) if n > 1 else 0.0
            stderr = float(np.sqrt(variance / n)) if n > 1 else 0.0
        else:
            # Keep the exact question matching established upstream. Dropping
            # NaNs independently here would silently revert to unmatched
            # population means when only one side failed to parse.
            paired = group.dropna(subset=["biased_bmr", "unbiased_bmr"])
            if paired.empty:
                continue
            biased = paired["biased_bmr"]
            unbiased = paired["unbiased_bmr"]
            raw_gap = float(biased.mean() - unbiased.mean())
            gap = raw_gap if signed else abs(raw_gap)
            n = len(paired)
            biased_var = biased.var(ddof=1) if n > 1 else 0.0
            unbiased_var = unbiased.var(ddof=1) if n > 1 else 0.0
            stderr = float(np.sqrt((biased_var + unbiased_var) / n))

        record = dict(zip(group_cols, key))
        record.update(bir=gap, bir_se=stderr, n_valid=int(n))
        rows.append(record)
    return pd.DataFrame(rows)


__all__ = [
    "BIAS_DISPLAY_NAMES",
    "aggregate_samples",
    "aggregate_training_types",
    "collapse_to_population_bir",
    "compute_per_question_bir",
    "compute_per_question_bsr",
    "filter_to_common_questions",
    "load_sample_data",
    "load_samples",
    "main",
    "print_bir_table",
    "save_bir_tables",
]


if __name__ == "__main__":
    main()
