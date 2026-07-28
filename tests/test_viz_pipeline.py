from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from scripts.render_main_paper_figures import _da_only_training_types
from scripts.render_unbiased_bmr import _ratio_rows
from sycophancy_eval_inspect.visualize_results import collapse_to_population_bir
from sycophancy_eval_inspect.viz.loaders import (
    _expand_eval_log_paths,
    _lookup_training_type,
    _resolve_model_identity,
    compute_per_question_bsr,
)
from sycophancy_eval_inspect.viz.plot import bar_plot
from sycophancy_eval_inspect.viz.recipes import _default_recipes, render_all_publication_plots
from sycophancy_eval_inspect.viz.registry import REGISTRY, training_type_info
from sycophancy_eval_inspect.viz.theme import PUBLICATION_THEME
from sycophancy_eval_inspect.viz.transforms import (
    add_held_out_avg,
    add_pct_change_held_out_avg,
    no_facet,
)


def _aggregated_rows() -> pd.DataFrame:
    rows = []
    for bias in ("distractor_argument", "wrong_few_shot"):
        rows.extend(
            [
                {
                    "model_family": "llama",
                    "training_type": "base",
                    "prompt_style": "cot",
                    "bias_type": bias,
                    "metric": "pro_bsr",
                    "value": 0.2,
                    "stderr": 0.01,
                    "n": 10,
                },
                {
                    "model_family": "llama",
                    "training_type": "bct_da",
                    "prompt_style": "cot",
                    "bias_type": bias,
                    "metric": "pro_bsr",
                    "value": 0.9,
                    "stderr": 0.01,
                    "n": 10,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_held_out_average_pools_absolute_metrics_by_n() -> None:
    frame = pd.DataFrame(
        [
            {
                "model_family": "llama",
                "training_type": "base",
                "prompt_style": "cot",
                "bias_type": "a",
                "metric": "pro_bsr",
                "value": 0.2,
                "stderr": 0.04,
                "n": 100,
            },
            {
                "model_family": "llama",
                "training_type": "base",
                "prompt_style": "cot",
                "bias_type": "b",
                "metric": "pro_bsr",
                "value": 0.8,
                "stderr": 0.08,
                "n": 10,
            },
        ]
    )

    result = add_held_out_avg(frame, ["a", "b"])
    summary = result[result["bias_type"] == "Held-out Avg"].iloc[0]

    assert math.isclose(summary["value"], (100 * 0.2 + 10 * 0.8) / 110)
    assert summary["n"] == 110


def test_ratio_held_out_average_matches_legacy_equal_bias_weighting() -> None:
    frame = pd.DataFrame(
        [
            {
                "model_family": "llama",
                "training_type": "bct_da",
                "prompt_style": "cot",
                "bias_type": "a",
                "metric": "pro_bsr",
                "value": 10.0,
                "stderr": 3.0,
                "n": 100,
            },
            {
                "model_family": "llama",
                "training_type": "bct_da",
                "prompt_style": "cot",
                "bias_type": "b",
                "metric": "pro_bsr",
                "value": 30.0,
                "stderr": 4.0,
                "n": 10,
            },
        ]
    )

    result = add_pct_change_held_out_avg(frame, ["a", "b"])
    summary = result[result["bias_type"] == "Held-out Avg"].iloc[0]

    assert summary["value"] == 20.0
    assert summary["stderr"] == 2.5
    assert summary["n"] == 110


def test_zero_numerator_ratio_retains_uncertainty() -> None:
    rows = [
        ("base", "Base", 0.5, 0.05, 100),
        ("bct_da", "BCT", 0.0, 0.1, 100),
    ]

    ratio = _ratio_rows(rows)

    assert ratio[0][2] == -100.0
    assert ratio[0][3] == 20.0


def test_per_question_bir_retains_legacy_model_column() -> None:
    common = {
        "model_name": "llama-base",
        "training_type": "base",
        "model_family": "llama",
        "prompt_style": "cot",
        "sample_id": "q1",
        "dataset": "truthfulqa",
        "seed": None,
        "bias_type": "suggested_answer",
        "bias_acknowledged": 0.0,
        "lenient_matches_bias": float("nan"),
    }
    samples = pd.DataFrame(
        [
            {**common, "variant": "biased", "matches_bias": 1.0},
            {**common, "variant": "unbiased", "matches_bias": 0.0},
        ]
    )

    result = compute_per_question_bsr(samples)

    assert result.loc[0, "model"] == "llama-base"
    assert pd.isna(result.loc[0, "seed"])
    assert result.loc[0, "net_bsr"] == 1.0


def test_per_question_bir_drops_samples_when_both_parsers_fail() -> None:
    common = {
        "model_name": "llama-base",
        "training_type": "base",
        "model_family": "llama",
        "prompt_style": "cot",
        "sample_id": "q1",
        "dataset": "truthfulqa",
        "seed": None,
        "bias_type": "suggested_answer",
        "bias_acknowledged": 0.0,
        "answer_parsed": 0.0,
        "lenient_answer_parsed": 0.0,
        "matches_bias": float("nan"),
        "lenient_matches_bias": float("nan"),
    }
    samples = pd.DataFrame(
        [
            {**common, "variant": "biased"},
            {
                **common,
                "variant": "unbiased",
                "answer_parsed": 1.0,
                "matches_bias": 0.0,
            },
        ]
    )

    result = compute_per_question_bsr(samples)

    assert result.empty


def test_population_bir_keeps_question_pairs_matched() -> None:
    common = {
        "model": "llama-base",
        "training_type": "base",
        "model_family": "llama",
        "prompt_style": "cot",
        "dataset": "truthfulqa",
        "bias_type": "suggested_answer",
        "seed": None,
    }
    per_question = pd.DataFrame(
        [
            {**common, "biased_bmr": 0.1, "unbiased_bmr": 0.0},
            {**common, "biased_bmr": 1.0, "unbiased_bmr": float("nan")},
            {**common, "biased_bmr": float("nan"), "unbiased_bmr": 1.0},
        ]
    )

    result = collapse_to_population_bir(per_question, signed=True)

    assert result.loc[0, "bir"] == 0.1
    assert result.loc[0, "n_valid"] == 1


def test_bar_plot_draws_n_labels_and_only_emits_significance_key_with_stars(tmp_path: Path, monkeypatch) -> None:
    original_close = plt.close
    monkeypatch.setattr(plt, "close", lambda _fig=None: None)

    bar_plot(
        _aggregated_rows(),
        metric="pro_bsr",
        model_family="llama",
        prompt_style="cot",
        theme=PUBLICATION_THEME,
        faceter=no_facet,
        output_path=tmp_path / "with-stars.png",
        auto_held_out=False,
        drop_biases=(),
        n_labels=True,
        show_significance=True,
    )
    figure = plt.gcf()
    axis_text = [text.get_text() for text in figure.axes[0].texts]
    figure_text = [text.get_text() for text in figure.texts]

    assert "n=10" in axis_text
    assert "*\n*\n*" in axis_text
    assert "* p<0.05   ** p<0.01   *** p<0.001" in figure_text
    original_close(figure)

    equal = _aggregated_rows()
    equal.loc[equal["training_type"] == "bct_da", "value"] = 0.2
    bar_plot(
        equal,
        metric="pro_bsr",
        model_family="llama",
        prompt_style="cot",
        theme=PUBLICATION_THEME,
        faceter=no_facet,
        output_path=tmp_path / "without-stars.png",
        auto_held_out=False,
        drop_biases=(),
        n_labels=False,
        show_significance=True,
    )
    figure = plt.gcf()
    assert not figure.texts
    original_close(figure)


def test_render_all_accepts_schema_less_empty_frames(tmp_path: Path) -> None:
    count = render_all_publication_plots(
        samples_wide=pd.DataFrame(),
        bir_wide=pd.DataFrame(),
        output_dir=tmp_path,
    )

    assert count == 0


def test_no_splits_recipe_set_keeps_main_plots_only() -> None:
    templates = {recipe.template for recipe in _default_recipes(include_splits=False)}

    assert "{bs}_pro_bsr" in templates
    assert "{bs}_pro_bsr_ratio" in templates
    assert "{bs}_bv" in templates
    assert "{bs}_bv_ratio" in templates
    assert not any("verbalised" in template or "unverbalised" in template for template in templates)
    assert not any(template.endswith(("_toward", "_away", "_unchanged", "_changed")) for template in templates)


def test_log_path_expansion_deduplicates_overlapping_roots(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    eval_path = nested / "one.eval"
    eval_path.write_text("placeholder")

    expanded = _expand_eval_log_paths([tmp_path, nested])

    assert expanded == [str(eval_path)]


def test_log_path_expansion_skips_archived_logs(tmp_path: Path) -> None:
    active = tmp_path / "model" / "active.eval"
    archived = tmp_path / "model" / "_archive" / "old.eval"
    active.parent.mkdir()
    archived.parent.mkdir()
    active.write_text("active")
    archived.write_text("archived")

    expanded = _expand_eval_log_paths([tmp_path])

    assert expanded == [str(active)]


def test_model_identity_falls_back_to_registered_directory_alias() -> None:
    identity = _resolve_model_identity(
        "internal_checkpoint_label",
        "llama-bct-da-s42",
        "meta-llama/Llama-3.1-8B-Instruct",
    )

    assert identity == ("llama-bct-da-s42", "bct_da", "llama", 42)


def test_legacy_model_directory_aliases_remain_registered() -> None:
    assert _lookup_training_type("llama-bct-mt-2k") == "bct_mt_2k"
    assert _lookup_training_type("llama-rlct-s50") == "rlct_step50"


def test_lr_average_configs_register_aggregate_groups() -> None:
    assert training_type_info("bct-da-lravg-lr1e4").aggregate_group == "bct-da-lravg"
    assert training_type_info("rlct-da-aw0-r128b4-lr2_86e4").aggregate_group == "rlct-da-aw0-r128b4"


def test_main_renderer_excludes_non_da_training_phenomena() -> None:
    training_types = _da_only_training_types()

    assert "base" in training_types
    assert all(
        key == "base" or training_type_info(key).training_biases == frozenset({"distractor_argument"})
        for key in training_types
    )


def test_every_registered_training_type_has_a_render_order() -> None:
    assert set(REGISTRY.training_types).issubset(REGISTRY.training_type_order)
