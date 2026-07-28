"""Render the slim main-body figure set: pro_bsr + bias_verbalised
(all + towards split), no controls, no n-labels, only DA-trained methods,
faceted by model.

Each plot ends up as 1 figure with 2 panels (gpt-oss-20b on top, llama on
bottom), Base/BCT/RLCT clusters per bias_type. Dropping the dawfs training
types removes the second training-bias-set panel; faceting by model adds
the cross-model split that the per-model pipeline can't.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sycophancy_eval_inspect.viz.aggregations import aggregate_training_types
from sycophancy_eval_inspect.viz.frame import (
    aggregate_metric,
    melt_per_question,
)
from sycophancy_eval_inspect.viz.loaders import (
    compute_per_question_bsr,
    load_samples,
)
from sycophancy_eval_inspect.viz.plot import bar_plot
from sycophancy_eval_inspect.viz.recipes import BIR_METRIC_COLUMNS, METRIC_LABELS
from sycophancy_eval_inspect.viz.registry import REGISTRY
from sycophancy_eval_inspect.viz.theme import PUBLICATION_THEME
from sycophancy_eval_inspect.viz.transforms import facet_by_column

MODEL_DISPLAY = {
    "gpt-oss-20b": "OpenAI GPT OSS 20B",
    "llama": "Meta Llama 3.1 8B Instruct",
}
MODEL_ORDER = ["gpt-oss-20b", "llama"]


def _da_only_training_types() -> list[str]:
    """Base plus DA-only trained methods (drop DA+WFS and other phenomena)."""
    da_biases = frozenset({"distractor_argument"})
    return [
        key
        for key, info in REGISTRY.training_types.items()
        if key == "base" or (info.data_scale == "da" and info.training_biases == da_biases and not info.is_control)
    ]


def _aggregate_metric_long(wide: pd.DataFrame, *, metric: str, predicate=None) -> pd.DataFrame:
    if wide.empty:
        return pd.DataFrame()
    if predicate is not None:
        wide = wide[predicate(wide)]
        if wide.empty:
            return pd.DataFrame()
    long = melt_per_question(wide, metric_columns=BIR_METRIC_COLUMNS)
    long = long[long["metric"] == metric]
    return aggregate_metric(long, group_cols=("model_family", "training_type", "prompt_style", "bias_type", "metric"))


def render(*, log_dirs: list[str], output_dir: Path, dedup: str = "last") -> int:
    samples = load_samples(log_dirs, dedup=dedup)
    bir = compute_per_question_bsr(samples)
    bir = aggregate_training_types(bir)

    keep_tts = set(_da_only_training_types())
    bir = bir[bir["training_type"].isin(keep_tts)]
    bir = bir[bir["model_family"].isin(MODEL_ORDER)]

    output_dir.mkdir(parents=True, exist_ok=True)
    # (filename_stem, metric, predicate) — predicate=None ⇒ no row filter.
    # The third entry uses bias_acknowledged with net_bsr>0 (towards-bias only):
    # this gives "BV conditional on the model switching toward the biased
    # answer", a main-body figure separate from the unconditional BV plot.
    figs = [
        ("main_pro_bsr", "pro_bsr", None),
        ("main_bias_verbalised", "bias_acknowledged", None),
        ("main_bias_verbalised_towards", "bias_acknowledged", lambda df: df["net_bsr"] > 0),
    ]
    n_written = 0
    for stem, metric, predicate in figs:
        agg = _aggregate_metric_long(bir, metric=metric, predicate=predicate)
        if agg.empty:
            continue
        ylabel = METRIC_LABELS.get(metric, metric)
        # No prompt_style filter: gpt-oss-20b only emits no_cot logs and llama
        # only emits cot logs, so each (model_family, bias_type, training_type)
        # cell is unique within its panel even though prompt_style differs
        # across panels.
        bar_plot(
            agg,
            metric=metric,
            model_family=None,
            prompt_style=None,
            theme=PUBLICATION_THEME,
            # Highlight the DA-trained bias (distractor_argument_g4) on every
            # panel so it sits on the left with the cream background, matching
            # the training-bias-faceted layout. The model-axis split inherits
            # this annotation per panel.
            faceter=facet_by_column(
                "model_family",
                label=lambda v: MODEL_DISPLAY.get(str(v), str(v)),
                annotate=lambda v: {"highlighted_biases": frozenset({"distractor_argument"})},
                sort_key=lambda v: MODEL_ORDER.index(str(v)) if str(v) in MODEL_ORDER else len(MODEL_ORDER),
            ),
            ylabel=ylabel,
            ylim=(0, 1.05) if metric == "bias_acknowledged" else None,
            random_line=None,
            zero_line=False,
            show_controls=False,
            n_labels=False,
            auto_held_out=True,
            output_path=output_dir / f"{stem}.png",
        )
        n_written += 1
    return n_written


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", action="append", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dedup", choices=("last", "mean", "none"), default="last")
    args = p.parse_args()
    n = render(
        log_dirs=args.log_dir,
        output_dir=Path(args.output_dir),
        dedup=args.dedup,
    )
    print(f"Wrote {n} figures to {args.output_dir}")


if __name__ == "__main__":
    main()
