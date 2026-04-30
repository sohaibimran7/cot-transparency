"""Recipe list driving the publication plot suite.

A `Recipe` is a small declarative spec: which wide data source, which metric,
optional predicate (to filter rows), optional ratio transform, output naming,
and a few plot-knob overrides (ylim, baseline lines).

The dispatcher (`render_all_publication_plots`) iterates models × prompt_styles
× recipes, applies each recipe end-to-end, and writes the figures + n.csv files.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import pandas as pd

from .frame import melt_per_question, aggregate_metric, ID_COLUMNS
from .plot import bar_plot
from .registry import REGISTRY, model_info
from .theme import PUBLICATION_THEME, Theme
from .transforms import (
    panels_for_training_types,
    order_biases_by_panels,
    pct_change_vs_baseline,
)


# ── Per-source metric columns ─────────────────────────────────────────────
SAMPLE_METRIC_COLUMNS = (
    "correct", "matches_bias", "lenient_correct", "lenient_matches_bias",
    "answer_parsed", "lenient_answer_parsed",
    "options_considered", "bias_acknowledged",
)

BIR_METRIC_COLUMNS = (
    "pro_bsr", "anti_bsr", "net_bsr", "total_bsr", "bir",
    "lenient_pro_bsr", "lenient_anti_bsr", "lenient_net_bsr", "lenient_total_bsr",
    "lenient_bir",
    "biased_bmr", "unbiased_bmr",
    "biased_lenient_bmr", "unbiased_lenient_bmr",
    "bias_acknowledged",
)

# Matches METRIC_DISPLAY in legacy module — single source of truth for ylabels.
METRIC_LABELS = {
    "correct": "Accuracy",
    "matches_bias": "Bias Match Rate",
    "lenient_correct": "Accuracy (Lenient)",
    "lenient_matches_bias": "BMR (Lenient)",
    "answer_parsed": "Parse Rate",
    "lenient_answer_parsed": "Parse Rate (Lenient)",
    "options_considered": "Options Considered (frac.)",
    "bias_acknowledged": "Bias Acknowledged",
    "bir": "Total-BSR",
    "pro_bsr": "Pro-BSR",
    "anti_bsr": "Anti-BSR",
    "net_bsr": "Net-BSR",
    "total_bsr": "Total-BSR",
}


@dataclass(frozen=True)
class Recipe:
    """One plot to render.

    `template` placeholders: {mf}, {ps}. Suffix `.png` is added automatically.
    Examples:
        "{mf}_pro_bsr_{ps}"                  → llama_pro_bsr_cot.png
        "{mf}_pro_bsr_ratio_{ps}"            → llama_pro_bsr_ratio_cot.png
        "{mf}_pro_bsr_{ps}_verbalised"       → llama_pro_bsr_cot_verbalised.png
        "{mf}_biased_correct_{ps}_oc_eq1"    → llama_biased_correct_cot_oc_eq1.png
    """
    source: Literal["samples", "bir"]
    metric: str
    template: str
    predicate: Callable[[pd.DataFrame], pd.Series] | None = None
    variant: str | None = None              # samples only: "biased" | "unbiased"
    ratio: bool = False
    show_baseline: bool = False             # matches_bias plots → unbiased baseline
    show_random_line: bool = False          # accuracy → 0.25 line
    ylim: tuple[float, float] | None = None  # default: auto


def _default_recipes() -> list[Recipe]:
    """Mirrors plot_all_analyses + plot_all_bir_ba in the legacy module.

    Filename layout: filter tag goes AFTER `_{ps}` (matching legacy).
    """
    rs: list[Recipe] = []

    # ── A. Sample-based main metrics (correct, matches_bias) ──────────────
    for variant in ("biased", "unbiased"):
        for metric in ("correct", "matches_bias"):
            rs.append(Recipe(
                source="samples", metric=metric, variant=variant,
                template=f"{{mf}}_{variant}_{metric}_{{ps}}",
                show_baseline=(metric == "matches_bias" and variant == "biased"),
                show_random_line=(metric == "correct"),
                ylim=(0, 1.05),
            ))
        # Parse rate
        rs.append(Recipe(
            source="samples", metric="answer_parsed", variant=variant,
            template=f"{{mf}}_{variant}_answer_parsed_{{ps}}",
            ylim=(0, 1.05),
        ))
        # Scorer metrics
        for metric in ("options_considered", "bias_acknowledged"):
            rs.append(Recipe(
                source="samples", metric=metric, variant=variant,
                template=f"{{mf}}_{variant}_{metric}_{{ps}}",
                ylim=(0, 1.05),
            ))

    # ── B. Sample-based filtered subsets ──────────────────────────────────
    sample_filters = [
        ("oc_eq1",
         lambda df: df["options_considered"] == 1.0,
         ("biased", "unbiased")),
        ("oc_lt1",
         lambda df: (df["options_considered"] < 1.0)
                    & df["options_considered"].notna(),
         ("biased", "unbiased")),
        ("ba_eq1",
         lambda df: df["bias_acknowledged"] == 1.0,
         ("biased",)),
        ("ba_eq0",
         lambda df: df["bias_acknowledged"] == 0.0,
         ("biased",)),
    ]
    for tag, pred, applicable in sample_filters:
        for variant in applicable:
            for metric in ("correct", "matches_bias"):
                rs.append(Recipe(
                    source="samples", metric=metric, variant=variant,
                    predicate=pred,
                    template=f"{{mf}}_{variant}_{metric}_{{ps}}_{tag}",
                    show_baseline=(metric == "matches_bias" and variant == "biased"),
                    show_random_line=(metric == "correct"),
                    ylim=(0, 1.05),
                ))

    # ── C. BIR-based BSR variants ─────────────────────────────────────────
    bsr_variants = ("pro_bsr", "anti_bsr", "net_bsr", "total_bsr")
    ba_splits = [
        ("verbalised", lambda df: df["bias_acknowledged"] == 1.0),
        ("unverbalised", lambda df: df["bias_acknowledged"] == 0.0),
    ]
    for variant in bsr_variants:
        rs.append(Recipe(source="bir", metric=variant,
                         template=f"{{mf}}_{variant}_{{ps}}"))
        rs.append(Recipe(source="bir", metric=variant, ratio=True,
                         template=f"{{mf}}_{variant}_ratio_{{ps}}"))
        for tag, pred in ba_splits:
            rs.append(Recipe(source="bir", metric=variant, predicate=pred,
                             template=f"{{mf}}_{variant}_{{ps}}_{tag}"))
            rs.append(Recipe(source="bir", metric=variant, predicate=pred,
                             ratio=True,
                             template=f"{{mf}}_{variant}_ratio_{{ps}}_{tag}"))

    # ── D. BA (bias_acknowledged) plots ───────────────────────────────────
    rs.append(Recipe(source="bir", metric="bias_acknowledged",
                     template="{mf}_ba_{ps}"))
    rs.append(Recipe(source="bir", metric="bias_acknowledged", ratio=True,
                     template="{mf}_ba_ratio_{ps}"))
    bsr_dir_splits = [
        ("toward", lambda df: df["net_bsr"] > 0),
        ("away", lambda df: df["net_bsr"] < 0),
        ("unchanged", lambda df: df["net_bsr"] == 0),
    ]
    for tag, pred in bsr_dir_splits:
        rs.append(Recipe(source="bir", metric="bias_acknowledged", predicate=pred,
                         template=f"{{mf}}_ba_{{ps}}_{tag}"))
        rs.append(Recipe(source="bir", metric="bias_acknowledged", predicate=pred,
                         ratio=True,
                         template=f"{{mf}}_ba_ratio_{{ps}}_{tag}"))

    return rs


def _aggregate_for_recipe(wide: pd.DataFrame, recipe: Recipe,
                          model_family: str, prompt_style: str | None,
                          ) -> tuple[pd.DataFrame, str]:
    """Apply variant filter, predicate, optional ratio. Return (agg_long, metric).

    Returns the long-form aggregated frame and the metric name to plot with
    (which equals recipe.metric in all cases).
    """
    sub = wide[wide["model_family"] == model_family]
    if prompt_style is not None and "prompt_style" in sub.columns:
        sub = sub[sub["prompt_style"] == prompt_style]
    if recipe.variant is not None and "variant" in sub.columns:
        sub = sub[sub["variant"] == recipe.variant]
    if recipe.predicate is not None:
        sub = sub[recipe.predicate(sub)]

    if sub.empty:
        return pd.DataFrame(), recipe.metric

    metric_cols = (SAMPLE_METRIC_COLUMNS if recipe.source == "samples"
                   else BIR_METRIC_COLUMNS)
    long = melt_per_question(sub, metric_columns=metric_cols)
    long = long[long["metric"] == recipe.metric]
    agg = aggregate_metric(long, group_cols=("model_family", "training_type",
                                             "prompt_style", "bias_type", "metric"))
    if recipe.ratio:
        agg = pct_change_vs_baseline(agg, baseline_training_type="base")
        if not agg.empty and "metric" not in agg.columns:
            agg["metric"] = recipe.metric
    return agg, recipe.metric


def _unbiased_baseline(samples_wide: pd.DataFrame, *,
                       model_family: str, metric: str,
                       prompt_style: str | None) -> tuple[float | None, float | None]:
    """Compute (mean, sem) of the unbiased base model for one metric."""
    sel = ((samples_wide["model_family"] == model_family)
           & (samples_wide["training_type"] == "base")
           & (samples_wide["variant"] == "unbiased"))
    if prompt_style is not None and "prompt_style" in samples_wide.columns:
        sel = sel & (samples_wide["prompt_style"] == prompt_style)
    sub = samples_wide.loc[sel, metric].dropna()
    if sub.empty:
        return None, None
    n = len(sub)
    m = float(sub.mean())
    se = float((m * (1 - m) / n) ** 0.5) if n > 0 else 0.0
    return m, se


def render_all_publication_plots(
    *,
    samples_wide: pd.DataFrame,
    bir_wide: pd.DataFrame,
    output_dir: Path | str,
    models: list[str] | None = None,
    prompt_styles: list[str] | None = None,
    show_controls: bool = True,
    theme: Theme = PUBLICATION_THEME,
    recipes: list[Recipe] | None = None,
) -> int:
    """Run every recipe across models × prompt_styles. Returns # of plots written."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    recipes = recipes or _default_recipes()

    # Pick model families. If not specified, take everything with data.
    available = sorted(set(samples_wide["model_family"].unique()) |
                       set(bir_wide["model_family"].unique()))
    families = models or available

    n_written = 0
    for mf in families:
        info = model_info(mf)
        # Display key for filename: legacy uses "llama" and "gpt-oss-20b"
        # (registry's model key already matches that).
        styles = prompt_styles or list(info.prompt_styles)
        for ps in styles:
            for recipe in recipes:
                source_wide = samples_wide if recipe.source == "samples" else bir_wide
                if source_wide.empty:
                    continue
                agg, metric = _aggregate_for_recipe(source_wide, recipe, mf, ps)
                if agg.empty:
                    continue
                # Baseline (unbiased base) for matches_bias plots
                baseline_hline = baseline_band = None
                if recipe.show_baseline and not recipe.ratio:
                    bm, bse = _unbiased_baseline(
                        samples_wide, model_family=mf,
                        metric=recipe.metric, prompt_style=ps)
                    if bm is not None:
                        baseline_hline = bm
                        if bse is not None:
                            baseline_band = (bm - bse, bm + bse)
                ylabel = METRIC_LABELS.get(metric, metric)
                if recipe.ratio:
                    ylabel = f"% Δ {METRIC_LABELS.get(metric, metric)}"
                yformatter = ((lambda v, _: f"{v:+.0f}%" if v != 0 else "0")
                              if recipe.ratio else None)
                output = output_dir / (recipe.template.format(mf=mf, ps=ps) + ".png")
                bar_plot(
                    agg,
                    metric=metric,
                    model_family=mf,
                    prompt_style=ps,
                    theme=theme,
                    ylabel=ylabel,
                    ylim=recipe.ylim if not recipe.ratio else None,
                    yformatter=yformatter,
                    baseline_hline=baseline_hline,
                    baseline_band=baseline_band,
                    random_line=0.25 if recipe.show_random_line else None,
                    zero_line=recipe.ratio,
                    show_controls=show_controls,
                    output_path=output,
                    # Legacy plot_metric_ratio_publication didn't emit n.csv;
                    # only the grouped-bars publication function did.
                    write_n_csv=not recipe.ratio,
                )
                n_written += 1
    return n_written
