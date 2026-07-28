"""Recipe list driving the publication plot suite.

A `Recipe` is a small declarative spec: which wide data source, which metric,
optional predicate (to filter rows), optional ratio transform, output naming,
and a few plot-knob overrides (ylim, baseline lines).

The dispatcher (`render_all_publication_plots`) iterates over training-bias
sets × prompt_styles × recipes; each output figure renders one panel per
model_family.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import pandas as pd

from .frame import aggregate_metric, melt_per_question
from .plot import bar_plot
from .registry import training_type_info
from .theme import PUBLICATION_THEME, Theme
from .transforms import (
    facet_by_column,
    pct_change_vs_baseline,
)

# Short codes for the most common training_biases sets, used in filenames.
_BIAS_SET_SHORT = {
    frozenset({"distractor_argument"}): "da",
    frozenset({"distractor_argument", "wrong_few_shot"}): "dawfs",
    frozenset({"suggested_answer"}): "sa",
    frozenset({"suggested_answer", "wrong_few_shot"}): "sawfs",
}


def _bias_set_code(bs: frozenset[str]) -> str:
    if bs in _BIAS_SET_SHORT:
        return _BIAS_SET_SHORT[bs]
    return "_".join(sorted(bs))


# Display names + ordering for the per-panel model facet. Order picked so
# gpt-oss panels render above llama panels, matching render_main_paper_figures.
MODEL_DISPLAY = {
    "gpt-oss-20b": "OpenAI GPT OSS 20B",
    "gpt-oss-120b": "gpt-oss-120b",
    "gpt": "GPT",
    "llama": "Meta Llama 3.1 8B Instruct",
    "qwen3": "Qwen3",
    "qwen3-8b": "Qwen3-8B",
    "qwen3-30b-a3b": "Qwen3-30B-A3B",
}
_MODEL_ORDER = ("gpt-oss-120b", "gpt-oss-20b", "gpt", "qwen3-30b-a3b", "qwen3-8b", "qwen3", "llama")


# ── Per-source metric columns ─────────────────────────────────────────────
SAMPLE_METRIC_COLUMNS = (
    "correct",
    "matches_bias",
    "lenient_correct",
    "lenient_matches_bias",
    "answer_parsed",
    "lenient_answer_parsed",
    "options_considered",
    "bias_acknowledged",
)

BIR_METRIC_COLUMNS = (
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

# Matches METRIC_DISPLAY in legacy module — single source of truth for ylabels.
METRIC_LABELS = {
    "correct": "Accuracy",
    "matches_bias": "Bias Match Rate",
    "lenient_correct": "Accuracy (Lenient)",
    "lenient_matches_bias": "BMR (Lenient)",
    "answer_parsed": "Parse Rate",
    "lenient_answer_parsed": "Parse Rate (Lenient)",
    "options_considered": "Options Considered (frac.)",
    "bias_acknowledged": "Bias verbalisation rate",
    "bir": "Total bias switch rate",
    "pro_bsr": "Towards-bias switch rate",
    "anti_bsr": "Away-from-bias switch rate",
    "net_bsr": "Net towards-bias switch rate",
    "total_bsr": "Total bias switch rate",
}


@dataclass(frozen=True)
class Recipe:
    """One plot to render.

    `template` placeholders: {bs}. Suffix `.png` is added automatically. `bs`
    is a short code for the training-bias set (e.g. "da", "dawfs"). Each
    figure includes all prompt styles in the data, faceted by model_family,
    so the filename does not encode prompt style. Examples:
        "{bs}_pro_bsr"                  → da_pro_bsr.png
        "{bs}_pro_bsr_ratio"            → da_pro_bsr_ratio.png
        "{bs}_pro_bsr_verbalised"       → da_pro_bsr_verbalised.png
        "{bs}_biased_correct_oc_eq1"    → da_biased_correct_oc_eq1.png
    """

    source: Literal["samples", "bir"]
    metric: str
    template: str
    predicate: Callable[[pd.DataFrame], pd.Series] | None = None
    variant: str | None = None  # samples only: "biased" | "unbiased"
    ratio: bool = False
    show_baseline: bool = False  # matches_bias plots → unbiased baseline
    show_random_line: bool = False  # accuracy → 0.25 line
    ylim: tuple[float, float] | None = None  # default: auto
    n_labels: bool = True  # draw "n=NNN" above each bar


def _default_recipes(*, include_splits: bool = True) -> list[Recipe]:
    """Mirrors plot_all_analyses + plot_all_bir_ba in the legacy module.

    Each recipe yields one figure faceted by model_family; prompt_style is
    not split into separate files (panels naturally separate it via model).
    """
    rs: list[Recipe] = []

    # ── A. Sample-based main metrics (correct, matches_bias) ──────────────
    # Only `biased` recipes are bias-faceted: the unbiased prompt is
    # bias-agnostic, and the eval pipeline labels every unbiased row with one
    # spurious bias_type, so per-bias_type unbiased plots can't be rendered
    # meaningfully through this recipe path. Render unbiased BMR via the
    # standalone `scripts/render_unbiased_bmr.py` (no bias x-axis).
    for metric in ("correct", "matches_bias"):
        rs.append(
            Recipe(
                source="samples",
                metric=metric,
                variant="biased",
                template=f"{{bs}}_biased_{metric}",
                show_baseline=(metric == "matches_bias"),
                show_random_line=(metric == "correct"),
                ylim=(0, 1.05),
            )
        )
    # Parse rate
    rs.append(
        Recipe(
            source="samples",
            metric="answer_parsed",
            variant="biased",
            template="{bs}_biased_answer_parsed",
            ylim=(0, 1.05),
        )
    )
    # Scorer metrics. `bias_acknowledged` files are named "biased_bv_*" to
    # match the BIR-source "bv_*" naming after the rename.
    rs.append(
        Recipe(
            source="samples",
            metric="options_considered",
            variant="biased",
            template="{bs}_biased_options_considered",
            ylim=(0, 1.05),
        )
    )
    rs.append(
        Recipe(
            source="samples",
            metric="bias_acknowledged",
            variant="biased",
            template="{bs}_biased_bv",
            ylim=(0, 1.05),
        )
    )

    # ── B. Sample-based filtered subsets ──────────────────────────────────
    sample_filters = [
        ("oc_eq1", lambda df: df["options_considered"] == 1.0, ("biased",)),
        ("oc_lt1", lambda df: (df["options_considered"] < 1.0) & df["options_considered"].notna(), ("biased",)),
        ("ba_eq1", lambda df: df["bias_acknowledged"] == 1.0, ("biased",)),
        ("ba_eq0", lambda df: df["bias_acknowledged"] == 0.0, ("biased",)),
    ]
    for tag, pred, applicable in sample_filters:
        for variant in applicable:
            for metric in ("correct", "matches_bias"):
                rs.append(
                    Recipe(
                        source="samples",
                        metric=metric,
                        variant=variant,
                        predicate=pred,
                        template=f"{{bs}}_{variant}_{metric}_{tag}",
                        show_baseline=(metric == "matches_bias" and variant == "biased"),
                        show_random_line=(metric == "correct"),
                        ylim=(0, 1.05),
                    )
                )

    # ── C. BIR-based BSR variants ─────────────────────────────────────────
    bsr_variants = ("pro_bsr", "anti_bsr", "net_bsr", "total_bsr")
    ba_splits = [
        ("verbalised", lambda df: df["bias_acknowledged"] == 1.0),
        ("unverbalised", lambda df: df["bias_acknowledged"] == 0.0),
    ]
    for variant in bsr_variants:
        rs.append(Recipe(source="bir", metric=variant, template=f"{{bs}}_{variant}"))
        rs.append(Recipe(source="bir", metric=variant, ratio=True, template=f"{{bs}}_{variant}_ratio"))
        if include_splits:
            for tag, pred in ba_splits:
                rs.append(Recipe(source="bir", metric=variant, predicate=pred, template=f"{{bs}}_{variant}_{tag}"))
                rs.append(
                    Recipe(
                        source="bir",
                        metric=variant,
                        predicate=pred,
                        ratio=True,
                        template=f"{{bs}}_{variant}_ratio_{tag}",
                    )
                )

    # ── D. BV (bias_verbalised) plots ─────────────────────────────────────
    rs.append(Recipe(source="bir", metric="bias_acknowledged", template="{bs}_bv"))
    rs.append(Recipe(source="bir", metric="bias_acknowledged", ratio=True, template="{bs}_bv_ratio"))
    bsr_dir_splits = [
        ("toward", lambda df: df["net_bsr"] > 0),
        ("away", lambda df: df["net_bsr"] < 0),
        ("unchanged", lambda df: df["net_bsr"] == 0),
        # `changed` = toward ∪ away: BV conditional on the model actually
        # switching its answer (in either direction) under the biased prompt.
        ("changed", lambda df: df["net_bsr"] != 0),
    ]
    if include_splits:
        for tag, pred in bsr_dir_splits:
            rs.append(Recipe(source="bir", metric="bias_acknowledged", predicate=pred, template=f"{{bs}}_bv_{tag}"))
            rs.append(
                Recipe(
                    source="bir",
                    metric="bias_acknowledged",
                    predicate=pred,
                    ratio=True,
                    template=f"{{bs}}_bv_ratio_{tag}",
                )
            )

    return rs


def _aggregate_for_recipe(
    wide: pd.DataFrame,
    recipe: Recipe,
    training_biases_set: frozenset[str] | None,
    prompt_style: str | None,
) -> tuple[pd.DataFrame, str]:
    """Apply variant filter, predicate, optional ratio. Return (agg_long, metric).

    Restricts ``wide`` to the rows whose training_type belongs to
    ``training_biases_set`` (along with shared types such as ``base`` and any
    associated controls) so the resulting figure represents one training-bias
    set faceted by model.
    """
    sub = wide
    if training_biases_set is not None:
        keep_tts = set()
        for tt in sub["training_type"].unique():
            info = training_type_info(tt)
            if info.training_biases == training_biases_set:
                keep_tts.add(tt)
            elif info.control_for is not None:
                base_tt = info.control_for
                base_info = training_type_info(base_tt)
                if base_info.training_biases == training_biases_set:
                    keep_tts.add(tt)
            elif not info.training_biases:
                # Shared types with no training_biases (e.g. "base") are
                # included in every panel.
                keep_tts.add(tt)
        sub = sub[sub["training_type"].isin(keep_tts)]
    if prompt_style is not None and "prompt_style" in sub.columns:
        sub = sub[sub["prompt_style"] == prompt_style]
    if recipe.variant is not None and "variant" in sub.columns:
        sub = sub[sub["variant"] == recipe.variant]
    if recipe.predicate is not None:
        sub = sub[recipe.predicate(sub)]

    if sub.empty:
        return pd.DataFrame(), recipe.metric

    metric_cols = SAMPLE_METRIC_COLUMNS if recipe.source == "samples" else BIR_METRIC_COLUMNS
    long = melt_per_question(sub, metric_columns=metric_cols)
    long = long[long["metric"] == recipe.metric]
    agg = aggregate_metric(long, group_cols=("model_family", "training_type", "prompt_style", "bias_type", "metric"))
    if recipe.ratio:
        # Each (model_family, prompt_style, bias_type) ratio is computed against
        # that group's own base row, since groupby keys include both columns.
        agg = pct_change_vs_baseline(agg, baseline_training_type="base")
        if not agg.empty and "metric" not in agg.columns:
            agg["metric"] = recipe.metric
    return agg, recipe.metric


def _unbiased_baseline(
    samples_wide: pd.DataFrame, *, model_family: str, metric: str, prompt_style: str | None
) -> tuple[float | None, float | None]:
    """Compute (mean, sem) of the unbiased base model for one metric."""
    sel = (
        (samples_wide["model_family"] == model_family)
        & (samples_wide["training_type"] == "base")
        & (samples_wide["variant"] == "unbiased")
    )
    if prompt_style is not None and "prompt_style" in samples_wide.columns:
        sel = sel & (samples_wide["prompt_style"] == prompt_style)
    sub = samples_wide.loc[sel, metric].dropna()
    if sub.empty:
        return None, None
    n = len(sub)
    m = float(sub.mean())
    se = float((m * (1 - m) / n) ** 0.5) if n > 0 else 0.0
    return m, se


def _highlighted_for_set(bs: frozenset[str]) -> Callable[[object], dict]:
    """Faceter `annotate` callback: every model panel highlights the same
    training-bias set so trained biases get the cream band on the left."""

    def annotate(_value: object) -> dict:
        return {"highlighted_biases": bs}

    return annotate


def _model_sort_key(value: object) -> int:
    s = str(value)
    if s in _MODEL_ORDER:
        return _MODEL_ORDER.index(s)
    return len(_MODEL_ORDER)


def _model_label(value: object) -> str:
    s = str(value)
    return MODEL_DISPLAY.get(s, s)


def _training_bias_sets_in(wide: pd.DataFrame) -> list[frozenset[str]]:
    """All unique training_biases sets that appear in the data."""
    if wide.empty or "training_type" not in wide.columns:
        return []
    sets: set[frozenset[str]] = set()
    for tt in wide["training_type"].unique():
        info = training_type_info(tt)
        if info.training_biases:
            sets.add(info.training_biases)
    # Sort by (size, sorted-tuple) for determinism
    return sorted(sets, key=lambda s: (len(s), tuple(sorted(s))))


def _baseline_for_panels(
    samples_wide: pd.DataFrame,
    recipe: Recipe,
    training_biases_set: frozenset[str] | None,
    prompt_style: str | None,
) -> tuple[float | None, float | None]:
    """Single (mean, sem) for the cream baseline band on `matches_bias` plots.

    With model panels stacked, we only draw one horizontal line per figure; we
    pool unbiased base across all models for that figure. Per-panel baselines
    would clutter the layout and aren't visually distinguishable.
    """
    if not (recipe.show_baseline and not recipe.ratio):
        return None, None
    sel = (samples_wide["training_type"] == "base") & (samples_wide["variant"] == "unbiased")
    if prompt_style is not None and "prompt_style" in samples_wide.columns:
        sel = sel & (samples_wide["prompt_style"] == prompt_style)
    sub = samples_wide.loc[sel, recipe.metric].dropna()
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
    show_n_labels: bool = True,
    include_splits: bool = True,
    theme: Theme = PUBLICATION_THEME,
    recipes: list[Recipe] | None = None,
) -> int:
    """Render every recipe across (training_biases_set × prompt_style).

    Each output figure has one panel per ``model_family``. ``models`` and
    ``prompt_styles`` filter the input frames; absent values are simply
    omitted. Returns # of plots written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if recipes is None:
        recipes = _default_recipes(include_splits=include_splits)

    if models:
        if not samples_wide.empty:
            samples_wide = samples_wide[samples_wide["model_family"].isin(models)]
        if not bir_wide.empty:
            bir_wide = bir_wide[bir_wide["model_family"].isin(models)]
    if prompt_styles:
        if not samples_wide.empty and "prompt_style" in samples_wide.columns:
            samples_wide = samples_wide[samples_wide["prompt_style"].isin(prompt_styles)]
        if not bir_wide.empty and "prompt_style" in bir_wide.columns:
            bir_wide = bir_wide[bir_wide["prompt_style"].isin(prompt_styles)]

    bias_sets = sorted(
        set(_training_bias_sets_in(samples_wide)) | set(_training_bias_sets_in(bir_wide)),
        key=lambda s: (len(s), tuple(sorted(s))),
    )
    if not bias_sets:
        return 0

    n_written = 0
    for bs in bias_sets:
        bs_code = _bias_set_code(bs)
        faceter = facet_by_column(
            "model_family",
            label=_model_label,
            annotate=_highlighted_for_set(bs),
            sort_key=_model_sort_key,
        )

        for recipe in recipes:
            source_wide = samples_wide if recipe.source == "samples" else bir_wide
            if source_wide.empty:
                continue
            # prompt_style=None: each model_family panel naturally restricts
            # to that model's emitted prompt_styles, so no per-figure split.
            agg, metric = _aggregate_for_recipe(source_wide, recipe, bs, None)
            if agg.empty:
                continue
            bm, bse = _baseline_for_panels(samples_wide, recipe, bs, None)
            baseline_hline = bm
            baseline_band = (bm - bse, bm + bse) if (bm is not None and bse) else None
            ylabel = METRIC_LABELS.get(metric, metric)
            if recipe.ratio:
                ylabel = f"% Δ {METRIC_LABELS.get(metric, metric)}"
            yformatter = (lambda v, _: f"{v:+.0f}%" if v != 0 else "0") if recipe.ratio else None
            output = output_dir / (recipe.template.format(bs=bs_code) + ".png")
            bar_plot(
                agg,
                metric=metric,
                model_family=None,
                prompt_style=None,
                theme=theme,
                faceter=faceter,
                ylabel=ylabel,
                ylim=recipe.ylim if not recipe.ratio else None,
                yformatter=yformatter,
                baseline_hline=baseline_hline,
                baseline_band=baseline_band,
                random_line=0.25 if recipe.show_random_line else None,
                zero_line=recipe.ratio,
                show_controls=show_controls,
                output_path=output,
                n_labels=recipe.n_labels and show_n_labels,
            )
            n_written += 1
    return n_written
