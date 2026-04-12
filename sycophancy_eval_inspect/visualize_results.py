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
import json
import re
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

DEFAULT_TRAINING_BIASES = frozenset({"suggested_answer"})


def _split_biases(bias_list: list[str], training_biases: set[str] | None = None) -> tuple[list[str], list[str]]:
    """Split bias list into (trained_on, held_out), preserving order within each group."""
    tb = training_biases if training_biases is not None else DEFAULT_TRAINING_BIASES
    trained = [b for b in bias_list if b in tb]
    held_out = [b for b in bias_list if b not in tb]
    return trained, held_out


# ── Visual encoding system ──────────────────────────────────────────────
# Color hue  = method family (grey=Base, blue-grey=Control, blue=BCT, green=RLCT, purple=VFT)
# Lightness  = training scale (lighter=less data/steps, darker=more)
# Hatch      = training variant (none=MT, '//'=MTI/instruct, '/'=DA, 'x'=DA+WFS)

def _style(color, hatch="", edgecolor="black"):
    return {"color": color, "hatch": hatch, "edgecolor": edgecolor}

STYLE_MAP = {
    # ── Base ──
    "base":                  _style("#969696"),
    # ── Controls (grey with faint treatment hue tint) ──
    "control":               _style("#b0b0b8"),         # grey, faint blue tint (BCT control)
    "bct_control_mt_2k":     _style("#c0c0c8"),         # lighter grey, faint blue tint
    "rlct_control_step50":   _style("#b0b5b0"),         # grey, faint green tint
    "rl_control_step50":     _style("#b0b5b0"),         # grey, faint green tint (alias)
    "rlct_control_step100":  _style("#a0a5a0"),         # darker grey, faint green tint
    # ── BCT family (blues, lighter→less data, darker→more) ──
    "bct_mti_1k":            _style("#c6dbef", "//"),   # very light blue, instruct
    "bct_mt_2k":             _style("#9ecae1"),         # light blue
    "bct_mt_b16_2k":         _style("#6baed6"),         # medium blue
    "bct_mti_4k":            _style("#4292c6", "//"),   # medium-dark blue, instruct
    "bct_mti_b16_4k":        _style("#2171b5", "//"),   # dark blue, instruct
    "bct_old_20k":           _style("#08519c"),         # very dark blue
    "bct_da_4k":             _style("#4292c6", "/"),    # medium blue, DA
    "bct_da_wfs_6k":         _style("#2171b5", "x"),   # dark blue, DA+WFS
    # ── RLCT family (greens, lighter→fewer steps, darker→more) ──
    "rlct_step50":           _style("#a1d99b"),         # light green (50 steps)
    "rlct_step50_v2":        _style("#8ed18e"),         # slight variation
    "rlct_step25_r3":        _style("#bae4b3"),         # very light (25 steps)
    "rlct_step50_v3":        _style("#82c982"),         # variation
    "rlct_step50_v4":        _style("#76c176"),         # variation
    "rlct_step50_v5":        _style("#6ab96a"),         # variation
    "rlct_step50_v6":        _style("#5eb15e"),         # variation
    "rlct_step50_v7":        _style("#52a952"),         # variation
    "rlct_s50_noempty":      _style("#78c679"),         # medium-light green
    "rlct_sa_g1_r1":         _style("#c7e9c0"),         # very light (small ablation)
    "rlct_sa_g16_r16":       _style("#a1d99b"),         # light
    "rlct_sa_g16_t16_r128":  _style("#74c476"),         # medium
    "rlct_sa_g16_r128":      _style("#74c476"),         # medium
    "rlct_s50_re_eval":      _style("#31a354"),         # medium-dark green (re-eval)
    "rlct_s50_a":            _style("#a1d99b"),         # light green (run A)
    "rlct_s50_b":            _style("#41ab5d"),         # medium-dark green (run B)
    "rlct_step100":          _style("#41ab5d"),         # medium-dark green (100 steps)
    "rlct_step200":          _style("#238b45"),         # dark green (200 steps)
    "rlct_da":               _style("#41ab5d", "/"),    # medium-dark green, DA
    "rlct_da_wfs":           _style("#238b45", "x"),    # dark green, DA+WFS
    # ── Anchor ablation (darker greens) ──
    "rlct_a0_r1":            _style("#006d2c"),         # very dark green (α=0, run 1)
    "rlct_a0_r2":            _style("#00441b"),         # darkest green (α=0, run 2)
    "rlct_a0_ctrl":          _style("#b0b5b0"),         # grey-green (α=0 control)
    "rlct_a05_r1":           _style("#238b45"),         # dark green (α=0.5, run 1)
    "rlct_a05_r2":           _style("#1a7a3a"),         # dark green (α=0.5, run 2)
    "rlct_a05_ctrl":         _style("#a0a8a0"),         # grey-green (α=0.5 control)
    "rlct_a1_ctrl":          _style("#909890"),         # grey-green (α=1.0 control)
    # ── VFT (purple) ──
    "vft_mt_1675":           _style("#9e9ac8"),         # purple
    # ── Aggregate groups ──
    "control_agg":           _style("#b0b0b0"),         # grey (aggregate control)
    "bct_agg":               _style("#3182bd"),         # solid blue (aggregate BCT)
    "rlct_agg":              _style("#31a354"),         # solid green (aggregate RLCT)
    "rlct_a0_agg":           _style("#006d2c"),         # very dark green (α=0 avg)
    "rlct_a05_agg":          _style("#238b45"),         # dark green (α=0.5 avg)
}

# Backward-compat alias
COLORS = {k: v["color"] for k, v in STYLE_MAP.items()}

# Maps training type → set of bias types used during training.
# Used to compute per-model held-out averages (exclude training biases).
TRAINING_BIAS_TYPES = {
    "base": set(),
    "control": set(),
    "bct_control_mt_2k": set(),
    "rlct_control_step50": set(),
    "rl_control_step50": set(),
    "rlct_control_step100": set(),
    "bct_old_20k": {"suggested_answer"},
    "bct_mti_1k": {"suggested_answer"},
    "bct_mti_4k": {"suggested_answer"},
    "bct_mt_2k": {"suggested_answer"},
    "bct_mt_b16_2k": {"suggested_answer"},
    "bct_mti_b16_4k": {"suggested_answer"},
    "bct_da_4k": {"distractor_argument"},
    "bct_da_wfs_6k": {"distractor_argument", "wrong_few_shot"},
    "vft_mt_1675": {"suggested_answer"},
    "rlct_step50": {"suggested_answer"},
    "rlct_step100": {"suggested_answer"},
    "rlct_step200": {"suggested_answer"},
    "rlct_step50_v2": {"suggested_answer"},
    "rlct_step25_r3": {"suggested_answer"},
    "rlct_step50_v3": {"suggested_answer"},
    "rlct_step50_v4": {"suggested_answer"},
    "rlct_step50_v5": {"suggested_answer"},
    "rlct_step50_v6": {"suggested_answer"},
    "rlct_step50_v7": {"suggested_answer"},
    "rlct_s50_noempty": {"suggested_answer"},
    "rlct_sa_g1_r1": {"suggested_answer"},
    "rlct_sa_g16_r16": {"suggested_answer"},
    "rlct_sa_g16_t16_r128": {"suggested_answer"},
    "rlct_sa_g16_r128": {"suggested_answer"},
    "rlct_s50_re_eval": {"suggested_answer"},
    "rlct_s50_a": {"suggested_answer"},
    "rlct_s50_b": {"suggested_answer"},
    "rlct_da": {"distractor_argument"},
    "rlct_da_wfs": {"distractor_argument", "wrong_few_shot"},
    # Anchor ablation
    "rlct_a0_r1": {"suggested_answer"},
    "rlct_a0_r2": {"suggested_answer"},
    "rlct_a0_ctrl": set(),
    "rlct_a05_r1": {"suggested_answer"},
    "rlct_a05_r2": {"suggested_answer"},
    "rlct_a05_ctrl": set(),
    "rlct_a1_ctrl": set(),
    # Aggregate groups (union of training biases)
    "control_agg": set(),
    "bct_agg": {"suggested_answer"},
    "rlct_agg": {"suggested_answer"},
    "rlct_a0_agg": {"suggested_answer"},
    "rlct_a05_agg": {"suggested_answer"},
}

TRAINING_TYPE_ORDER = [
    "base", "vft_mt_1675", "bct_control_mt_2k", "control", "rlct_control_step50", "rl_control_step50", "rlct_control_step100", "bct_old_20k", "bct_mti_1k", "bct_mti_4k", "bct_mt_2k", "bct_mt_b16_2k", "bct_mti_b16_4k", "rlct_step50", "rlct_step50_v2", "rlct_step25_r3", "rlct_step50_v3", "rlct_step50_v4", "rlct_step50_v5", "rlct_step50_v6", "rlct_step50_v7", "rlct_s50_noempty", "rlct_sa_g1_r1", "rlct_sa_g16_r16", "rlct_sa_g16_t16_r128", "rlct_sa_g16_r128", "rlct_s50_re_eval", "rlct_s50_a", "rlct_s50_b", "rlct_step100", "rlct_step200",
    "bct_da_4k", "bct_da_wfs_6k", "rlct_da", "rlct_da_wfs",
    # Anchor ablation
    "rlct_a0_r1", "rlct_a0_r2", "rlct_a0_ctrl",
    "rlct_a05_r1", "rlct_a05_r2", "rlct_a05_ctrl",
    "rlct_a1_ctrl",
    # Aggregate groups
    "control_agg", "bct_agg", "rlct_agg", "rlct_a0_agg", "rlct_a05_agg",
]

TRAINING_TYPE_NAMES = {
    "base": "Base",
    "control": "Control SFT",
    "rlct_control_step50": "RL Control s50",
    "rlct_control_step100": "RL Control s100",
    "bct_old_20k": "BCT 20k (old)",
    "bct_mti_1k": "BCT MTI 1k",
    "bct_mti_4k": "BCT MTI 4k",
    "bct_mt_2k": "BCT MT 2k",
    "bct_mt_b16_2k": "BCT MT b16 2k",
    "bct_mti_b16_4k": "BCT MTI b16 4k",
    "bct_control_mt_2k": "BCT Control MT 2k",
    "rl_control_step50": "RL Control s50",
    "rlct_step50": "RLCT s50",
    "rlct_step100": "RLCT s100",
    "rlct_step200": "RLCT s200",
    "rlct_step50_v2": "RLCT s50 v2",
    "rlct_step25_r3": "RLCT s25 r3",
    "rlct_step50_v3": "RLCT s50 v3 (skip empty)",
    "rlct_step50_v4": "RLCT s50 v4 (original)",
    "rlct_step50_v5": "RLCT s50 v5 (auto LR)",
    "rlct_step50_v6": "RLCT s50 v6 (auto LR+pipe r1)",
    "rlct_step50_v7": "RLCT s50 v7 (auto LR+pipe r3)",
    "rlct_s50_noempty": "RLCT s50 noempty",
    "rlct_sa_g1_r1": "RLCT SA g1 r1",
    "rlct_sa_g16_r16": "RLCT SA g16 r16",
    "rlct_sa_g16_t16_r128": "RLCT SA g16 t16 r128",
    "rlct_sa_g16_r128": "RLCT SA g16 r128",
    "rlct_s50_re_eval": "RLCT s50 re-eval",
    "rlct_s50_a": "RLCT s50 A",
    "rlct_s50_b": "RLCT s50 B",
    "vft_mt_1675": "VFT MT 1675",
    "bct_da_4k": "BCT DA 4k",
    "bct_da_wfs_6k": "BCT DA+WFS 6k",
    "rlct_da": "RLCT DA",
    "rlct_da_wfs": "RLCT DA+WFS",
    # Anchor ablation
    "rlct_a0_r1": "RLCT α=0 r1",
    "rlct_a0_r2": "RLCT α=0 r2",
    "rlct_a0_ctrl": "RL Ctrl α=0",
    "rlct_a05_r1": "RLCT α=0.5 r1",
    "rlct_a05_r2": "RLCT α=0.5 r2",
    "rlct_a05_ctrl": "RL Ctrl α=0.5",
    "rlct_a1_ctrl": "RL Ctrl α=1",
    # Aggregate groups
    "control_agg": "Control (avg)",
    "bct_agg": "BCT (avg)",
    "rlct_agg": "RLCT (avg)",
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
    "strict_bias_acknowledged": "Bias Acknowledged (strict)",
    "bir": "Bias Influence Rate (BIR)",
    "lenient_bir": "BIR (Lenient)",
    "target_answered": "Target Answered",
    "non_target_answered": "Non-Target Answered",
}


# ── Aggregate group mappings ───────────────────────────────────────────────
# Maps individual training types → aggregate group key.
# Types not listed here are kept as-is (base, controls).
_AGGREGATE_MAP = {
    # BCT family → bct_agg
    "bct_old_20k": "bct_agg",
    "bct_mti_1k": "bct_agg",
    "bct_mti_4k": "bct_agg",
    "bct_mt_2k": "bct_agg",
    "bct_mt_b16_2k": "bct_agg",
    "bct_mti_b16_4k": "bct_agg",
    "bct_da_4k": "bct_agg",
    "bct_da_wfs_6k": "bct_agg",
    # RLCT family → rlct_agg
    "rlct_step50": "rlct_agg",
    "rlct_step100": "rlct_agg",
    "rlct_step200": "rlct_agg",
    "rlct_step50_v2": "rlct_agg",
    "rlct_step25_r3": "rlct_agg",
    "rlct_step50_v3": "rlct_agg",
    "rlct_step50_v4": "rlct_agg",
    "rlct_step50_v5": "rlct_agg",
    "rlct_step50_v6": "rlct_agg",
    "rlct_step50_v7": "rlct_agg",
    "rlct_s50_noempty": "rlct_agg",
    "rlct_sa_g1_r1": "rlct_agg",
    "rlct_sa_g16_r16": "rlct_agg",
    "rlct_sa_g16_t16_r128": "rlct_agg",
    "rlct_sa_g16_r128": "rlct_agg",
    "rlct_s50_re_eval": "rlct_agg",
    "rlct_s50_a": "rlct_agg",
    "rlct_s50_b": "rlct_agg",
    "rlct_da": "rlct_agg",
    "rlct_da_wfs": "rlct_agg",
    # Anchor ablation RLCT → separate per-α groups
    "rlct_a0_r1": "rlct_a0_agg",
    "rlct_a0_r2": "rlct_a0_agg",
    "rlct_a05_r1": "rlct_a05_agg",
    "rlct_a05_r2": "rlct_a05_agg",
    # Controls → control_agg (merge SFT/RL controls)
    "control": "control_agg",
    "bct_control_mt_2k": "control_agg",
    "rlct_control_step50": "control_agg",
    "rl_control_step50": "control_agg",
    "rlct_control_step100": "control_agg",
    "rlct_a0_ctrl": "control_agg",
    "rlct_a05_ctrl": "control_agg",
    "rlct_a1_ctrl": "control_agg",
}

# Order & display for aggregate mode
_AGGREGATE_ORDER = ["base", "control_agg", "bct_agg", "rlct_agg", "rlct_a0_agg", "rlct_a05_agg"]
_AGGREGATE_NAMES = {
    "base": "Base",
    "control_agg": "Control (avg)",
    "bct_agg": "BCT (avg)",
    "rlct_agg": "RLCT (avg)",
    "rlct_a0_agg": "RLCT α=0 (avg)",
    "rlct_a05_agg": "RLCT α=0.5 (avg)",
}


def aggregate_training_types(df: "pd.DataFrame") -> "pd.DataFrame":
    """Remap training_type column to aggregate groups."""
    df = df.copy()
    df["training_type"] = df["training_type"].map(
        lambda t: _AGGREGATE_MAP.get(t, t)
    )
    return df


# ── Helpers ────────────────────────────────────────────────────────────────

_DIR_TO_TRAINING_TYPE = {
    "base": "base",
    "control": "control",
    "bct-control-mt-2k": "bct_control_mt_2k",
    "rl-control-s50": "rlct_control_step50",
    "bct-old-20k": "bct_old_20k",
    "bct-mti-1k": "bct_mti_1k",
    "bct-mti-4k": "bct_mti_4k",
    "bct-mt-2k": "bct_mt_2k",
    "bct-mt-b16-2k": "bct_mt_b16_2k",
    "bct-mti-b16-4k": "bct_mti_b16_4k",
    "bct-mti-bs16-4k": "bct_mti_b16_4k",
    "bct-mti-r32-4k": None,  # Exclude r32
    "bct-control-mt-2k": "bct_control_mt_2k",
    "rl-control-step50": "rl_control_step50",
    "control-mti-4k": "control",
    "bct-mti-4k-100samples": "bct_mti_4k",
    "control-mti-4k-100samples": "control",
    "rlct-s50": "rlct_step50",
    "rlct-s50-100samples": "rlct_step50",
    "rlct-s100": "rlct_step100",
    "rlct-s100-100samples": "rlct_step100",
    "rlct-s200": "rlct_step200",
    "rlct-s50-v2": "rlct_step50_v2",
    "rlct-s25-r3": "rlct_step25_r3",
    "rlct-s50-v3": "rlct_step50_v3",
    "rlct-s50-v4": "rlct_step50_v4",
    "rlct-s50-noempty": "rlct_s50_noempty",
    "rlct-sa-g1-r1": "rlct_sa_g1_r1",
    "rlct-sa-g16-r16": "rlct_sa_g16_r16",
    "rlct-sa-g16-t16-r128": "rlct_sa_g16_t16_r128",
    "rlct-sa-g16-r128": "rlct_sa_g16_r128",
    "rlct-s50-v5": "rlct_step50_v5",
    "rlct-s50-v6": "rlct_step50_v6",
    "rlct-s50-v7": "rlct_step50_v7",
    "rl-control-s50-100samples": "rlct_control_step50",
    "rl-control-s100-100samples": "rlct_control_step100",
    "vft-mt-1675": "vft_mt_1675",
    "bct-da-4k": "bct_da_4k",
    "bct-da-wfs-6k": "bct_da_wfs_6k",
    "rlct-s50-re-eval": "rlct_s50_re_eval",
    "rlct-s50-a": "rlct_s50_a",
    "rlct-s50-b": "rlct_s50_b",
    "rlct-da": "rlct_da",
    "rlct-da-wfs": "rlct_da_wfs",
    # Anchor ablation runs
    "rlct-a0-r1": "rlct_a0_r1",
    "rlct-a0-r2": "rlct_a0_r2",
    "rlct-a0-ctrl": "rlct_a0_ctrl",
    "rlct-a05-r1": "rlct_a05_r1",
    "rlct-a05-r2": "rlct_a05_r2",
    "rlct-a05-ctrl": "rlct_a05_ctrl",
    "rlct-a1-ctrl": "rlct_a1_ctrl",
}


# ── Model registry (extends hardcoded dicts with external entries) ─────────
_REGISTRY_PATH = Path(__file__).parent / "model_registry.json"
_MODEL_PREFIXES = ["llama-", "gpt-oss-120b-", "gpt-oss-20b-", "gpt-"]


def _load_model_registry() -> None:
    """Load model_registry.json and merge entries into module-level dicts."""
    global _MODEL_PREFIXES
    if not _REGISTRY_PATH.exists():
        return
    with open(_REGISTRY_PATH) as f:
        registry = json.load(f)

    # Extend model prefixes (deduplicated, order preserved)
    for p in registry.get("model_prefixes", []):
        if p not in _MODEL_PREFIXES:
            _MODEL_PREFIXES.append(p)

    for suffix, entry in registry.get("models", {}).items():
        tt = entry.get("training_type", suffix.replace("-", "_"))
        # _DIR_TO_TRAINING_TYPE
        if suffix not in _DIR_TO_TRAINING_TYPE:
            _DIR_TO_TRAINING_TYPE[suffix] = tt
        # STYLE_MAP
        if tt not in STYLE_MAP:
            STYLE_MAP[tt] = _style(
                entry.get("color", "#888888"),
                entry.get("hatch", ""),
                entry.get("edgecolor", "black"),
            )
            COLORS[tt] = STYLE_MAP[tt]["color"]
        # TRAINING_TYPE_NAMES
        if tt not in TRAINING_TYPE_NAMES:
            TRAINING_TYPE_NAMES[tt] = entry.get("display_name", tt)
        # TRAINING_BIAS_TYPES
        if tt not in TRAINING_BIAS_TYPES:
            TRAINING_BIAS_TYPES[tt] = set(entry.get("training_biases", []))
        # TRAINING_TYPE_ORDER — append if missing
        if tt not in TRAINING_TYPE_ORDER:
            TRAINING_TYPE_ORDER.append(tt)


_load_model_registry()


_SEED_SUFFIX_RE = re.compile(r"-s(\d+)$")


def _strip_seed_suffix(name: str) -> tuple[str, int | None]:
    """Strip -s{N} seed suffix from a dir name component. Returns (stripped, seed_or_None)."""
    m = _SEED_SUFFIX_RE.search(name)
    if m:
        return name[:m.start()], int(m.group(1))
    return name, None


def _strip_model_prefix(dir_name: str) -> str:
    """Strip model prefix from directory name, returning the suffix."""
    n = dir_name.lower()
    for prefix in sorted(_MODEL_PREFIXES, key=len, reverse=True):
        if n.startswith(prefix):
            return n[len(prefix):]
    return n


def _get_training_type_from_dir(dir_name: str) -> str | None:
    """Extract training type from directory name.

    Expected naming: {model}-{suffix} where suffix is a key in _DIR_TO_TRAINING_TYPE.
    Prefixes are loaded from model_registry.json so new models work automatically.
    Handles seed suffixes (-s{N}) by stripping them before lookup, and
    control+seed combos like bct-sa-s42-ctrl.
    """
    n = _strip_model_prefix(dir_name)

    # 1. Try original name first (handles e.g. rlct-s50 = "step 50")
    result = _DIR_TO_TRAINING_TYPE.get(n)
    if result is not None:
        return result

    # 2. Try with seed suffix stripped (e.g. bct-sa-s42 -> bct-sa)
    n_no_seed, _ = _strip_seed_suffix(n)
    if n_no_seed != n:
        result = _DIR_TO_TRAINING_TYPE.get(n_no_seed)
        if result is not None:
            return result

    # 3. Handle control+seed (e.g. bct-sa-s42-ctrl -> bct-sa-ctrl)
    if n.endswith("-ctrl"):
        n_no_ctrl = n[:-5]
        n_no_ctrl_no_seed, _ = _strip_seed_suffix(n_no_ctrl)
        if n_no_ctrl_no_seed != n_no_ctrl:
            return _DIR_TO_TRAINING_TYPE.get(n_no_ctrl_no_seed + "-ctrl")

    return None


def _get_seed_from_dir(dir_name: str) -> int | None:
    """Extract seed from directory name's -s{N} suffix, if present.

    Returns None if the full suffix (with -s{N}) is already a known training type
    (collision guard: e.g. rlct-s50 means 'step 50', not 'seed 50').
    """
    n = _strip_model_prefix(dir_name)

    # If the full name is already registered, it's not a seed variant
    if n in _DIR_TO_TRAINING_TYPE:
        return None

    # Check for seed directly (e.g. bct-sa-s42)
    _, seed = _strip_seed_suffix(n)
    if seed is not None:
        return seed

    # Check for seed before -ctrl (e.g. bct-sa-s42-ctrl)
    if n.endswith("-ctrl"):
        _, seed = _strip_seed_suffix(n[:-5])
        return seed

    return None


def _get_model_family_from_dir(dir_name: str) -> str | None:
    """Extract model family from directory name.

    Derives family from registered prefixes (e.g. 'llama-' -> 'llama').
    """
    lower = dir_name.lower()
    for prefix in sorted(_MODEL_PREFIXES, key=len, reverse=True):
        if lower.startswith(prefix):
            return prefix.rstrip("-")
    return None


def _iter_model_dirs(log_dirs):
    """Yield (model_dir_path, dir_name) for each model directory."""
    for log_dir in log_dirs:
        log_path = Path(log_dir)
        if not log_path.exists():
            print(f"Warning: {log_dir} does not exist")
            continue
        if any(log_path.glob("*.eval")):
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
        seed = _get_seed_from_dir(dir_name)

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
                    "seed": seed,
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
    if not df.empty:
        # Strict bias_acknowledged: NaN out when model answered a non-target (few-shot) question
        # For few-shot bias types (wrong_few_shot, spurious_few_shot_*), this filters out
        # samples where the model got confused by the few-shot examples.
        df["strict_bias_acknowledged"] = df["bias_acknowledged"].copy()
        few_shot_mask = df["bias_type"].str.contains("few_shot", na=False)
        nta_mask = df["non_target_answered"] == 1.0
        df.loc[few_shot_mask & nta_mask, "strict_bias_acknowledged"] = np.nan
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
    ylim=None,
    output_path=None,
    training_biases: set[str] | None = None,
):
    """Draw grouped bar chart with optional lenient overlay.

    Args:
        ylim: Tuple (ymin, ymax) for y-axis limits. If None, auto-scales
              from data with padding. Pass (0, 1.15) for standard 0-1 metrics.
    """
    all_biases = [b for b in BIAS_DISPLAY_NAMES if b in pivot_val.index]
    if not all_biases:
        return
    trained_biases, held_out_biases = _split_biases(all_biases, training_biases)
    bias_order = trained_biases + held_out_biases
    n_trained = len(trained_biases)

    # Add average rows to pivots
    tb_set = training_biases if training_biases is not None else DEFAULT_TRAINING_BIASES
    _trained_in_data = [b for b in bias_order if b in tb_set]
    _held_out_in_data = [b for b in bias_order if b not in tb_set]

    avg_keys = []
    for label, subset in [("Training\nAvg", _trained_in_data), ("Held-out\nAvg", _held_out_in_data)]:
        if subset and any(b in pivot_val.index for b in subset):
            present = [b for b in subset if b in pivot_val.index]
            pivot_val.loc[label] = pivot_val.loc[present].mean()
            # SE of average: sqrt(sum(SE_i^2)) / k
            k = len(present)
            pivot_err.loc[label] = (pivot_err.loc[present] ** 2).sum() ** 0.5 / k
            pivot_n.loc[label] = pivot_n.loc[present].sum()
            if lenient_pivot_val is not None:
                l_present = [b for b in present if b in lenient_pivot_val.index]
                if l_present:
                    lenient_pivot_val.loc[label] = lenient_pivot_val.loc[l_present].mean()
                    lenient_pivot_err.loc[label] = (lenient_pivot_err.loc[l_present] ** 2).sum() ** 0.5 / len(l_present)
                    if lenient_pivot_n is not None:
                        lenient_pivot_n.loc[label] = lenient_pivot_n.loc[l_present].sum()
            avg_keys.append(label)

    display_order = bias_order + avg_keys

    pivot_val = pivot_val.reindex(display_order)
    pivot_err = pivot_err.reindex(display_order)
    pivot_n = pivot_n.reindex(display_order)
    if lenient_pivot_val is not None:
        lenient_pivot_val = lenient_pivot_val.reindex(display_order)
        lenient_pivot_err = lenient_pivot_err.reindex(display_order)
        if lenient_pivot_n is not None:
            lenient_pivot_n = lenient_pivot_n.reindex(display_order)

    has_lenient = lenient_pivot_val is not None

    fig, ax = plt.subplots(figsize=(max(10, 2 * len(display_order)), 6))

    training_types = [t for t in TRAINING_TYPE_ORDER if t in pivot_val.columns]
    if not training_types:
        plt.close()
        return

    # X positions with gaps for separators
    x_positions = []
    gap = 0.0
    for i in range(len(display_order)):
        if i == n_trained and n_trained > 0 and n_trained < len(bias_order):
            gap += 0.5  # gap between trained and held-out biases
        if i == len(bias_order) and avg_keys:
            gap += 0.5  # gap before averages
        x_positions.append(float(i) + gap)
    x = np.array(x_positions)
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

        style = STYLE_MAP.get(tt, _style("#999999"))

        # Lenient overlay bar (behind)
        if has_lenient_for_tt:
            ax.bar(
                bar_pos, l_vals, width,
                color=style["color"], alpha=0.35,
            )

        # Strict bar (on top)
        ax.bar(
            bar_pos, values, width,
            label=TRAINING_TYPE_NAMES.get(tt, tt),
            color=style["color"],
            hatch=style["hatch"],
            edgecolor=style["edgecolor"],
            linewidth=0.8 if style["hatch"] else 0.5,
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
                # Strict: error bar shifted left, black (±2 SEM ≈ 95% CI)
                ax.errorbar(pos - err_offset, val, yerr=2*err if not np.isnan(err) else 0,
                            fmt="none", capsize=3, ecolor="black", alpha=0.7)
                if show_n_labels:
                    text_y = val + (2*err if not np.isnan(err) else 0) + 0.02
                    ax.text(pos - err_offset, text_y, f"n={int(n_s)}",
                            ha="center", va="bottom", fontsize=6, rotation=90)

                # Lenient: error bar shifted right, black with lower opacity (±2 SEM ≈ 95% CI)
                ax.errorbar(pos + err_offset, l_val, yerr=2*l_err if not np.isnan(l_err) else 0,
                            fmt="none", capsize=2, ecolor="black", alpha=0.35)
                if show_n_labels:
                    l_text_y = l_val + (2*l_err if not np.isnan(l_err) else 0) + 0.02
                    ax.text(pos + err_offset, l_text_y, f"n={int(n_l)}",
                            ha="center", va="bottom", fontsize=5,
                            rotation=90, alpha=0.5, style="italic")
            else:
                # Single centered error bar and n= label (±2 SEM ≈ 95% CI)
                ax.errorbar(pos, val, yerr=2*err if not np.isnan(err) else 0,
                            fmt="none", capsize=3, ecolor="black", alpha=0.7)
                if show_n_labels and not np.isnan(n_s):
                    text_y = val + (2*err if not np.isnan(err) else 0) + 0.02
                    ax.text(pos, text_y, f"n={int(n_s)}",
                            ha="center", va="bottom", fontsize=6, rotation=90)

    # Separator lines
    if n_trained > 0 and held_out_biases:
        sep_x = (x[n_trained - 1] + x[n_trained]) / 2 + width * (n_types - 1) / 2
        ax.axvline(x=sep_x, color="gray", linestyle=":", linewidth=1.2, alpha=0.6)
    if avg_keys:
        sep_x2 = (x[len(bias_order) - 1] + x[len(bias_order)]) / 2 + width * (n_types - 1) / 2
        ax.axvline(x=sep_x2, color="gray", linestyle=":", linewidth=1.2, alpha=0.6)

    # Formatting
    ax.set_xlabel("Bias Type")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x + width * (n_types - 1) / 2)
    display_labels = [BIAS_DISPLAY_NAMES.get(b, b) for b in display_order]
    ax.set_xticklabels(display_labels, rotation=45, ha="right")
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        # Auto-scale: gather all plotted values to find range
        all_vals = pivot_val.to_numpy(dtype=float).flatten()
        if lenient_pivot_val is not None:
            all_vals = np.concatenate([all_vals, lenient_pivot_val.to_numpy(dtype=float).flatten()])
        all_vals = all_vals[~np.isnan(all_vals)]
        if len(all_vals) > 0:
            ymin = min(0, np.min(all_vals) - 0.05)
            ymax = max(1.15, np.max(all_vals) + 0.15)
            ax.set_ylim(ymin, ymax)
        else:
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
        ax.legend(handles=handles, labels=labels, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=7)
    else:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=7)

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
    training_biases: set[str] | None = None,
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
        ylim=(0, 1.15),
        output_path=output_path,
        training_biases=training_biases,
    )


# ── BIR / BA plotting ────────────────────────────────────────────────────


def plot_metric_bars(
    df: pd.DataFrame,
    metric_col: str,
    model_family: str,
    prompt_style: str | None = None,
    lenient_metric_col: str | None = None,
    sample_filter=None,
    filter_label: str = "",
    show_n_labels: bool = True,
    output_path: str | None = None,
    training_biases: set[str] | None = None,
):
    """Plot grouped bars for a metric from per-question data (e.g. BIR or BA).

    Works on any DataFrame with columns: model_family, prompt_style,
    bias_type, training_type, and the metric column(s).
    """
    mask = df["model_family"] == model_family
    if prompt_style:
        mask &= df["prompt_style"] == prompt_style
    filtered = df[mask].copy()

    if sample_filter is not None:
        filtered = filtered[sample_filter(filtered)]

    if filtered.empty:
        return

    # Aggregate strict metric
    agg = aggregate_samples(filtered, metric_col, ["bias_type", "training_type"])
    mean_col = f"{metric_col}_mean"
    err_col = f"{metric_col}_stderr"
    pivot_val = agg.pivot(index="bias_type", columns="training_type", values=mean_col)
    pivot_err = agg.pivot(index="bias_type", columns="training_type", values=err_col)
    pivot_n = agg.pivot(index="bias_type", columns="training_type", values="n_valid")

    # Aggregate lenient metric
    l_pivot_val = l_pivot_err = l_pivot_n = None
    if lenient_metric_col and lenient_metric_col in filtered.columns:
        l_agg = aggregate_samples(filtered, lenient_metric_col, ["bias_type", "training_type"])
        l_pivot_val = l_agg.pivot(
            index="bias_type", columns="training_type",
            values=f"{lenient_metric_col}_mean",
        )
        l_pivot_err = l_agg.pivot(
            index="bias_type", columns="training_type",
            values=f"{lenient_metric_col}_stderr",
        )
        l_pivot_n = l_agg.pivot(
            index="bias_type", columns="training_type", values="n_valid",
        )

    ylabel = METRIC_DISPLAY.get(metric_col, metric_col)
    title_parts = [model_family.upper(), ylabel]
    if prompt_style:
        title_parts.append(f"({prompt_style})")
    if filter_label:
        title_parts.append(f"[{filter_label}]")
    title = " - ".join(title_parts[:2])
    if len(title_parts) > 2:
        title += " " + " ".join(title_parts[2:])

    plot_grouped_bars(
        pivot_val, pivot_err, pivot_n,
        lenient_pivot_val=l_pivot_val,
        lenient_pivot_err=l_pivot_err,
        lenient_pivot_n=l_pivot_n,
        title=title,
        ylabel=ylabel,
        show_n_labels=show_n_labels,
        output_path=output_path,
        training_biases=training_biases,
    )


def plot_metric_ratio(
    df: pd.DataFrame,
    metric_col: str,
    model_family: str,
    baseline_type: str = "base",
    prompt_style: str | None = None,
    sample_filter=None,
    filter_label: str = "",
    output_path: str | None = None,
    training_biases: set[str] | None = None,
):
    """Plot diverging horizontal bars showing % change from baseline.

    Each row is a bias type (+ averages), with grouped horizontal bars
    for each training type. Bars extend left (improvement) or right (degradation)
    from the 0% center line. Training and held-out averages are computed based
    on the training_biases parameter.
    """
    mask = df["model_family"] == model_family
    if prompt_style:
        mask &= df["prompt_style"] == prompt_style
    filtered = df[mask].copy()

    if sample_filter is not None:
        filtered = filtered[sample_filter(filtered)]

    if filtered.empty:
        return

    agg = aggregate_samples(filtered, metric_col, ["bias_type", "training_type"])
    mean_col = f"{metric_col}_mean"
    err_col = f"{metric_col}_stderr"
    pivot_mean = agg.pivot(index="bias_type", columns="training_type", values=mean_col)
    pivot_err = agg.pivot(index="bias_type", columns="training_type", values=err_col)

    if baseline_type not in pivot_mean.columns:
        return

    # Compute % change from baseline with propagated error bars.
    # For ratio R = A/B: SE(R) = R * sqrt((SE_A/A)^2 + (SE_B/B)^2)
    # For % change P = (R-1)*100: SE(P) = SE(R) * 100
    baseline_vals = pivot_mean[baseline_type].replace(0, np.nan)
    baseline_errs = pivot_err[baseline_type] if baseline_type in pivot_err.columns else pd.Series(dtype=float)
    training_types = [t for t in TRAINING_TYPE_ORDER
                      if t in pivot_mean.columns and t != baseline_type]
    if not training_types:
        return

    all_biases = [b for b in BIAS_DISPLAY_NAMES if b in pivot_mean.index]
    if not all_biases:
        return
    trained_biases_list, held_out_biases_list = _split_biases(all_biases, training_biases)
    bias_order = trained_biases_list + held_out_biases_list
    n_trained = len(trained_biases_list)

    # Compute per-bias % changes and propagated errors
    pct_changes = {}   # tt -> {bias_type: pct_change}
    pct_errors = {}    # tt -> {bias_type: pct_stderr}
    for tt in training_types:
        changes = {}
        errors = {}
        for bt in bias_order:
            base_val = baseline_vals.get(bt, np.nan)
            base_err = baseline_errs.get(bt, np.nan) if bt in baseline_errs.index else np.nan
            tt_val = pivot_mean.loc[bt, tt] if bt in pivot_mean.index else np.nan
            tt_err = pivot_err.loc[bt, tt] if bt in pivot_err.index else np.nan
            if pd.notna(base_val) and pd.notna(tt_val) and base_val != 0:
                ratio = tt_val / base_val
                changes[bt] = (ratio - 1.0) * 100
                # Propagate SE: SE(R) = R * sqrt((SE_A/A)^2 + (SE_B/B)^2)
                rel_a = (tt_err / tt_val) ** 2 if (pd.notna(tt_err) and tt_val != 0) else 0
                rel_b = (base_err / base_val) ** 2 if (pd.notna(base_err) and base_val != 0) else 0
                errors[bt] = abs(ratio) * (rel_a + rel_b) ** 0.5 * 100
            else:
                changes[bt] = np.nan
                errors[bt] = np.nan
        pct_changes[tt] = changes
        pct_errors[tt] = errors

    # Compute per-training-type training and held-out averages and SE
    tb_set = training_biases if training_biases is not None else DEFAULT_TRAINING_BIASES
    _trained_in_data = [b for b in bias_order if b in tb_set]
    _held_out_in_data = [b for b in bias_order if b not in tb_set]

    training_avgs = {}
    training_errs = {}
    held_out_avgs = {}
    held_out_errs = {}
    for tt in training_types:
        # Training average
        tr_vals = [pct_changes[tt][bt] for bt in _trained_in_data
                   if not np.isnan(pct_changes[tt].get(bt, np.nan))]
        tr_errs_list = [pct_errors[tt][bt] for bt in _trained_in_data
                        if not np.isnan(pct_errors[tt].get(bt, np.nan))]
        training_avgs[tt] = np.mean(tr_vals) if tr_vals else np.nan
        training_errs[tt] = (sum(e ** 2 for e in tr_errs_list) ** 0.5 / len(tr_errs_list)) if tr_errs_list else np.nan

        # Held-out average
        ho_vals = [pct_changes[tt][bt] for bt in _held_out_in_data
                   if not np.isnan(pct_changes[tt].get(bt, np.nan))]
        ho_errs_list = [pct_errors[tt][bt] for bt in _held_out_in_data
                        if not np.isnan(pct_errors[tt].get(bt, np.nan))]
        held_out_avgs[tt] = np.mean(ho_vals) if ho_vals else np.nan
        held_out_errs[tt] = (sum(e ** 2 for e in ho_errs_list) ** 0.5 / len(ho_errs_list)) if ho_errs_list else np.nan

    # Row labels: bias types + averages
    avg_row_labels = []
    if _trained_in_data:
        avg_row_labels.append("Training Avg")
    if _held_out_in_data:
        avg_row_labels.append("Held-out Avg")
    row_labels = [BIAS_DISPLAY_NAMES.get(b, b) for b in bias_order] + avg_row_labels
    n_rows = len(row_labels)
    n_types = len(training_types)
    bar_height = 0.8 / max(n_types, 1)

    fig_height = max(5, n_rows * (0.15 * n_types + 0.3) + 2)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    y = np.arange(n_rows)

    # Dynamic x-axis range using IQR method (robust to outliers).
    # Values outside the whiskers are clipped and marked with triangles.
    all_pct_vals = []
    for tt in training_types:
        for bt in bias_order:
            v = pct_changes[tt].get(bt, np.nan)
            if not np.isnan(v):
                all_pct_vals.append(v)
        if not np.isnan(held_out_avgs[tt]):
            all_pct_vals.append(held_out_avgs[tt])

    if all_pct_vals:
        q1, q3 = np.percentile(all_pct_vals, [25, 75])
        iqr = q3 - q1
        x_min = q1 - 1.5 * iqr
        x_max = q3 + 1.5 * iqr
        # Ensure 0 is always visible and add a small pad
        x_min = min(x_min, -5) - 5
        x_max = max(x_max, 5) + 5
    else:
        x_min, x_max = -100, 50

    for i, tt in enumerate(training_types):
        style = STYLE_MAP.get(tt, _style("#999999"))
        bar_y = y - (n_types - 1) * bar_height / 2 + i * bar_height

        # Build values and errors arrays: per-bias changes + averages
        avg_vals = []
        avg_errs_list = []
        if _trained_in_data:
            avg_vals.append(training_avgs[tt])
            avg_errs_list.append(training_errs[tt])
        if _held_out_in_data:
            avg_vals.append(held_out_avgs[tt])
            avg_errs_list.append(held_out_errs[tt])
        vals = [pct_changes[tt].get(bt, np.nan) for bt in bias_order] + avg_vals
        errs = [pct_errors[tt].get(bt, np.nan) for bt in bias_order] + avg_errs_list
        vals_arr = np.array(vals, dtype=float)
        errs_arr = np.array(errs, dtype=float)
        errs_arr = np.where(np.isnan(errs_arr), 0, errs_arr)

        # Clip bars to display range; mark clipped values with triangles
        clipped_arr = np.clip(vals_arr, x_min, x_max)

        ax.barh(
            bar_y, clipped_arr, bar_height,
            color=style["color"],
            hatch=style["hatch"],
            edgecolor=style["edgecolor"],
            linewidth=0.8 if style["hatch"] else 0.5,
            label=TRAINING_TYPE_NAMES.get(tt, tt),
        )

        # Error bars (horizontal)
        for j, (yp, val, err) in enumerate(zip(bar_y, vals_arr, errs_arr)):
            if np.isnan(val) or err == 0:
                continue
            # Only draw error bar if the bar tip is within display range (±2 SEM ≈ 95% CI)
            if x_min <= val <= x_max:
                ax.errorbar(val, yp, xerr=2*err, fmt="none",
                            ecolor="black", capsize=2, alpha=0.5, linewidth=0.8)

        # Mark clipped (out-of-range) values with triangles
        for j, (yp, val, clipped) in enumerate(zip(bar_y, vals_arr, clipped_arr)):
            if np.isnan(val):
                continue
            if val < x_min:
                ax.plot(x_min, yp, marker="<", color=style["color"],
                        markeredgecolor="black", markersize=4, zorder=5)
            elif val > x_max:
                ax.plot(x_max, yp, marker=">", color=style["color"],
                        markeredgecolor="black", markersize=4, zorder=5)

    # Reference line at 0% (no change)
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)

    # Separator line between trained and held-out biases
    if n_trained > 0 and held_out_biases_list:
        sep_y1 = n_trained - 0.5
        ax.axhline(y=sep_y1, color="gray", linestyle=":", linewidth=1.2, alpha=0.6)

    # Separator line before averages
    n_avg = len(avg_row_labels)
    if n_avg > 0:
        sep_y2 = n_rows - n_avg - 0.5
        ax.axhline(y=sep_y2, color="gray", linestyle=":", linewidth=1.2, alpha=0.6)

    # Shade improvement region (left of 0)
    ax.axvspan(x_min - 50, 0, color="#d4edda", alpha=0.08, zorder=0)

    ax.set_xlim(x_min, x_max)

    xlabel = f"% Change in {METRIC_DISPLAY.get(metric_col, metric_col)} (vs {TRAINING_TYPE_NAMES.get(baseline_type, baseline_type)})"
    title_parts = [model_family.upper(), f"{METRIC_DISPLAY.get(metric_col, metric_col)} — % Change"]
    if prompt_style:
        title_parts.append(f"({prompt_style})")
    if filter_label:
        title_parts.append(f"[{filter_label}]")
    title = " - ".join(title_parts[:2])
    if len(title_parts) > 2:
        title += " " + " ".join(title_parts[2:])

    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_yticks(y)
    ax.set_yticklabels(row_labels)
    ax.invert_yaxis()  # Top to bottom
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=7)
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close()


def plot_all_bir_ba(
    bir_df: pd.DataFrame,
    output_dir: str = "plots",
    models: list[str] | None = None,
    prompt_styles: list[str] | None = None,
    show_n_labels: bool = True,
    include_splits: bool = True,
    training_biases: set[str] | None = None,
):
    """Generate all BIR and BA plots with symmetrical structure.

    BIR plots: main, split by BA (verbalised/unverbalised), ratio-to-base.
    BA plots: main, split by BIR (influenced/not_influenced), ratio-to-base.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    models = models or sorted(bir_df["model_family"].unique())

    # Define splits: (filter_fn, label, file_tag)
    ba_splits = [
        (lambda df: df["bias_acknowledged"] == 1.0, "Verbalised", "verbalised"),
        (lambda df: df["bias_acknowledged"] == 0.0, "Unverbalised", "unverbalised"),
    ]
    bir_splits = [
        (lambda df: df["bir"] > 0, "Flipped (|BIR|=1)", "flipped"),
        (lambda df: df["bir"] <= 0, "Unchanged (|BIR|=0)", "unchanged"),
    ]
    for mf in models:
        avail_styles = ["cot", "no_cot"] if mf == "llama" else ["no_cot"]
        if prompt_styles:
            avail_styles = [s for s in avail_styles if s in prompt_styles]

        for ps in avail_styles:
            # ── BIR plots ──
            # Main BIR (with lenient overlay)
            plot_metric_bars(
                bir_df, "bir", mf, prompt_style=ps,
                lenient_metric_col="lenient_bir",
                show_n_labels=show_n_labels,
                output_path=f"{output_dir}/{mf}_bir_{ps}.png",
                training_biases=training_biases,
            )
            # BIR ratio-to-base
            plot_metric_ratio(
                bir_df, "bir", mf, prompt_style=ps,
                output_path=f"{output_dir}/{mf}_bir_ratio_{ps}.png",
                training_biases=training_biases,
            )

            # BIR split by BA
            if include_splits:
                for filt_fn, label, tag in ba_splits:
                    plot_metric_bars(
                        bir_df, "bir", mf, prompt_style=ps,
                        lenient_metric_col="lenient_bir",
                        sample_filter=filt_fn, filter_label=label,
                        show_n_labels=show_n_labels,
                        output_path=f"{output_dir}/{mf}_bir_{ps}_{tag}.png",
                        training_biases=training_biases,
                    )
                    plot_metric_ratio(
                        bir_df, "bir", mf, prompt_style=ps,
                        sample_filter=filt_fn, filter_label=label,
                        output_path=f"{output_dir}/{mf}_bir_ratio_{ps}_{tag}.png",
                        training_biases=training_biases,
                    )

            # ── BA plots ──
            # Main BA
            plot_metric_bars(
                bir_df, "bias_acknowledged", mf, prompt_style=ps,
                show_n_labels=show_n_labels,
                output_path=f"{output_dir}/{mf}_ba_{ps}.png",
                training_biases=training_biases,
            )
            # BA ratio-to-base
            plot_metric_ratio(
                bir_df, "bias_acknowledged", mf, prompt_style=ps,
                output_path=f"{output_dir}/{mf}_ba_ratio_{ps}.png",
                training_biases=training_biases,
            )

            # BA split by BIR
            if include_splits:
                for filt_fn, label, tag in bir_splits:
                    plot_metric_bars(
                        bir_df, "bias_acknowledged", mf, prompt_style=ps,
                        sample_filter=filt_fn, filter_label=label,
                        show_n_labels=show_n_labels,
                        output_path=f"{output_dir}/{mf}_ba_{ps}_{tag}.png",
                        training_biases=training_biases,
                    )
                    plot_metric_ratio(
                        bir_df, "bias_acknowledged", mf, prompt_style=ps,
                        sample_filter=filt_fn, filter_label=label,
                        output_path=f"{output_dir}/{mf}_ba_ratio_{ps}_{tag}.png",
                        training_biases=training_biases,
                    )


def plot_variance_across(
    sample_df: pd.DataFrame,
    metric: str,
    split_by: str,
    model_family: str,
    variant: str = "biased",
    prompt_style: str | None = None,
    training_biases: set[str] | None = None,
    output_path: str | None = None,
) -> None:
    """Plot variance across groups defined by split_by.

    Shows per-(bias_type, training_type) means with SE computed across the
    split_by groups, revealing how much results vary by e.g. seed or dataset.

    The n= labels show number of groups (not total samples).
    """
    mask = sample_df["model_family"] == model_family
    if "variant" in sample_df.columns:
        mask &= sample_df["variant"] == variant
    if prompt_style and "prompt_style" in sample_df.columns:
        mask &= sample_df["prompt_style"] == prompt_style

    if split_by not in sample_df.columns:
        print(f"  Warning: column '{split_by}' not in data, skipping variance plot")
        return

    filtered = sample_df[mask].copy()
    filtered = filtered.dropna(subset=[split_by])

    n_groups = filtered[split_by].nunique()
    if n_groups < 2:
        print(f"  Only {n_groups} group(s) for split_by='{split_by}', skipping variance plot")
        return

    # Step 1: Per-group means
    group_means = (
        filtered.groupby([split_by, "bias_type", "training_type"])[metric]
        .mean()
        .reset_index()
        .rename(columns={metric: "group_mean"})
    )

    # Step 2: Aggregate across groups
    results = []
    for (bt, tt), grp in group_means.groupby(["bias_type", "training_type"]):
        means = grp["group_mean"].dropna()
        k = len(means)
        if k == 0:
            continue
        overall_mean = means.mean()
        se = means.std(ddof=1) / k**0.5 if k > 1 else 0.0
        results.append({
            "bias_type": bt,
            "training_type": tt,
            f"{metric}_mean": overall_mean,
            f"{metric}_stderr": se,
            "n_valid": k,
        })

    agg = pd.DataFrame(results)
    if agg.empty:
        return

    mean_col = f"{metric}_mean"
    err_col = f"{metric}_stderr"
    pivot_val = agg.pivot(index="bias_type", columns="training_type", values=mean_col)
    pivot_err = agg.pivot(index="bias_type", columns="training_type", values=err_col)
    pivot_n = agg.pivot(index="bias_type", columns="training_type", values="n_valid")

    ylabel = METRIC_DISPLAY.get(metric, metric)
    title = f"{model_family.upper()} - {variant.title()} - {ylabel} (Variance across {split_by})"
    if prompt_style:
        title += f" ({prompt_style})"

    plot_grouped_bars(
        pivot_val, pivot_err, pivot_n,
        title=title,
        ylabel=ylabel,
        show_n_labels=True,
        output_path=output_path,
        training_biases=training_biases,
    )


def plot_all_analyses(
    sample_df: pd.DataFrame,
    output_dir: str = "plots",
    models: list[str] | None = None,
    variants: list[str] | None = None,
    prompt_styles_filter: list[str] | None = None,
    training_biases: set[str] | None = None,
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
        "options_considered",
        "target_answered", "non_target_answered",
    ]

    # Scorer metrics with strict/lenient overlay
    scorer_metrics_overlay = [
        ("strict_bias_acknowledged", "bias_acknowledged"),
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
        if prompt_styles_filter:
            avail_styles = list(prompt_styles_filter)
        else:
            avail_styles = ["cot", "no_cot"] if model_family == "llama" else ["no_cot"]

        for ps in avail_styles:
            for variant in variants:
                # 1. Main metric plots (with lenient overlay)
                for metric, lenient in main_metrics:
                    path = f"{output_dir}/{model_family}_{variant}_{metric}_{ps}.png"
                    plot_from_samples(
                        sample_df, metric, model_family,
                        variant=variant, prompt_style=ps,
                        lenient_metric=lenient, output_path=path,
                        training_biases=training_biases,
                    )

                # 1b. Parse rate plots (with lenient overlay, no n= labels)
                for metric, lenient in parse_metrics:
                    path = f"{output_dir}/{model_family}_{variant}_{metric}_{ps}.png"
                    plot_from_samples(
                        sample_df, metric, model_family,
                        variant=variant, prompt_style=ps,
                        lenient_metric=lenient, show_n_labels=False,
                        output_path=path,
                        training_biases=training_biases,
                    )

                # 2. New scorer metric plots
                for metric in scorer_metrics:
                    path = f"{output_dir}/{model_family}_{variant}_{metric}_{ps}.png"
                    plot_from_samples(
                        sample_df, metric, model_family,
                        variant=variant, prompt_style=ps,
                        output_path=path,
                        training_biases=training_biases,
                    )

                # 2b. Scorer metrics with strict/lenient overlay
                for metric, lenient in scorer_metrics_overlay:
                    path = f"{output_dir}/{model_family}_{variant}_{metric}_{ps}.png"
                    plot_from_samples(
                        sample_df, metric, model_family,
                        variant=variant, prompt_style=ps,
                        lenient_metric=lenient, output_path=path,
                        training_biases=training_biases,
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
                            training_biases=training_biases,
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
        seed = _get_seed_from_dir(dir_name)

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

                # Non-target answered (for strict_bias_acknowledged)
                fsc_score = sample.scores.get("few_shot_confusion_scorer")
                nta = fsc_score.value.get("non_target_answered", np.nan) if (fsc_score and fsc_score.value) else np.nan

                key = (dir_name, training_type, model_family, prompt_style,
                       sample.id, original_dataset, seed)

                if variant == "biased":
                    results[key]["biased"][bias_type] = {
                        "bmr": bmr, "l_bmr": l_bmr, "ba": bias_ack, "nta": nta,
                    }
                else:
                    results[key]["unbiased"] = {"bmr": bmr, "l_bmr": l_bmr}

    # Build BRR DataFrame
    bir_data = []
    for key, data in results.items():
        model, training_type, model_family, prompt_style, hash_id, dataset, seed = key
        if data["unbiased"] is None:
            continue
        ub = data["unbiased"]
        for bt, bd in data["biased"].items():
            # Compute strict_bias_acknowledged: NaN out few-shot samples where
            # model answered a non-target question
            strict_ba = bd["ba"]
            is_few_shot = "few_shot" in bt
            if is_few_shot and bd.get("nta") == 1.0:
                strict_ba = np.nan

            bir_data.append({
                "model": model,
                "training_type": training_type,
                "model_family": model_family,
                "prompt_style": prompt_style,
                "hash": hash_id,
                "dataset": dataset,
                "bias_type": bt,
                "seed": seed,
                "biased_bmr": bd["bmr"],
                "unbiased_bmr": ub["bmr"],
                "bir": abs(bd["bmr"] - ub["bmr"]),
                "biased_lenient_bmr": bd["l_bmr"],
                "unbiased_lenient_bmr": ub["l_bmr"],
                "lenient_bir": abs(bd["l_bmr"] - ub["l_bmr"]),
                "bias_acknowledged": bd["ba"],
                "strict_bias_acknowledged": strict_ba,
            })

    return pd.DataFrame(bir_data)


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


def compute_bir_table(
    bir_df: pd.DataFrame,
    model_family: str,
    prompt_style: str | None = None,
    metric_col: str = "bir",
    baseline: str = "base",
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
        if tt != baseline:
            bir_pivot[f"{tt}_ratio"] = bir_pivot[tt] / bir_pivot[baseline].replace(0, np.nan)

    cols = TRAINING_TYPE_ORDER + [f"{tt}_ratio" for tt in TRAINING_TYPE_ORDER if tt != baseline]
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
    baseline: str = "base",
    table_name: str = "BIR",
    value_header: str = "BIR %",
    ratio_header: str = "BIR Ratio",
    best_is_low: bool = True,
    training_biases: set[str] | None = None,
):
    """Print pivot table in formatted style (used for BIR, BA, and other per-question metrics)."""
    bir, counts = compute_bir_table(bir_df, model_family, prompt_style, metric_col=metric_col, baseline=baseline)
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

    # All bias types present in the data (used for display order)
    all_biases_raw = [b for b in BIAS_DISPLAY_NAMES if b in bir.index]
    trained_biases_list, held_out_biases_list = _split_biases(all_biases_raw, training_biases)
    all_bias_order = trained_biases_list + held_out_biases_list
    n_trained = len(trained_biases_list)
    available_types = [tt for tt in TRAINING_TYPE_ORDER if tt in bir.columns and bir[tt].notna().any()]

    baseline_name = TRAINING_TYPE_NAMES.get(baseline, baseline)
    style_str = f" ({prompt_style})" if prompt_style else ""
    label_str = f" | {label}" if label else ""

    bir_cols = [TRAINING_TYPE_NAMES[tt] for tt in available_types]
    ratio_cols = [TRAINING_TYPE_NAMES[tt] for tt in available_types if tt != baseline]

    # Dynamic column width: at least 9, wide enough for the longest name + 1
    cw = max(9, max((len(c) for c in bir_cols + ratio_cols), default=9) + 1)

    w = 16 + cw * len(bir_cols) + cw * len(ratio_cols) + 4
    print(f"\n{'=' * w}")
    print(f"{table_name} TABLE - {model_family.upper()}{style_str}{label_str}")
    print(f"{'=' * w}")

    bir_sw = cw * len(bir_cols)
    ratio_sw = cw * len(ratio_cols)
    print(f"{'':16} {value_header:^{bir_sw}}  {ratio_header:^{ratio_sw}}")

    print(f"{'Bias Type':<16} {' '.join(f'{c:>{cw}}' for c in bir_cols)}  {' '.join(f'{c:>{cw}}' for c in ratio_cols)}")
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
            if tt != baseline:
                rv = row.get(f"{tt}_ratio")
                ratio_strs[tt] = f"{rv:.2f}" if pd.notna(rv) else "-"

        if bir_vals:
            best_val = min(bir_vals.values()) if best_is_low else max(bir_vals.values())
            for k in [k for k, v in bir_vals.items() if v == best_val and k != baseline]:
                bir_strs[k] = f"*{bir_strs[k]}*"

        bp = " ".join(f"{bir_strs[tt]:>{cw}}" for tt in available_types)
        rp = " ".join(f"{ratio_strs[tt]:>{cw}}" for tt in available_types if tt != baseline)
        return f"{name:<16} {bp}  {rp}"

    def fmt_n_row(bias_type):
        """Format the n= row for a bias type."""
        count_row = counts.loc[bias_type] if bias_type in counts.index else pd.Series()
        n_strs = []
        for tt in available_types:
            raw_n = count_row.get(tt, 0) if tt in count_row.index else 0
            n = int(raw_n) if pd.notna(raw_n) else 0
            n_strs.append(f"n={n}" if n > 0 else "")
        np_ = " ".join(f"{s:>{cw}}" for s in n_strs)
        rp = " ".join(f"{'':>{cw}}" for _ in available_types if _ != baseline)
        return f"{'':16} {np_}  {rp}"

    for i, bt in enumerate(all_bias_order):
        if i == n_trained and n_trained > 0 and held_out_biases_list:
            print("·" * w)  # dotted separator between trained and held-out
        print(fmt_row(bt))
        if show_n:
            print(fmt_n_row(bt))

    # Per-training-type training and held-out averages
    tb_set = training_biases if training_biases is not None else DEFAULT_TRAINING_BIASES

    print("-" * w)

    # Training average
    if trained_biases_list:
        tr_avg_vals = {}
        tr_total_n = {}
        tr_avg_ratios = {}
        for tt in available_types:
            tr_biases = [b for b in trained_biases_list if b in bir.index]
            if tr_biases:
                tr_avg_vals[tt] = bir.loc[tr_biases, tt].mean() * 100
                tr_total_n[tt] = int(counts.loc[tr_biases, tt].sum()) if tt in counts.columns else 0
            else:
                tr_avg_vals[tt] = float("nan")
                tr_total_n[tt] = 0
        for tt in available_types:
            if tt == baseline:
                continue
            baseline_tr_avg = tr_avg_vals.get(baseline, 0)
            tr_avg_ratios[tt] = tr_avg_vals[tt] / baseline_tr_avg if baseline_tr_avg != 0 else float("nan")
        bp = " ".join(f"{tr_avg_vals[tt]:>{cw}.0f}" for tt in available_types)
        rp = " ".join(f"{tr_avg_ratios.get(tt, float('nan')):>{cw}.2f}" for tt in available_types if tt != baseline)
        print(f"{'Training Avg':<16} {bp}  {rp}")
        if show_n:
            np_ = " ".join(f"{'n=' + str(tr_total_n[tt]):>{cw}}" for tt in available_types)
            rp_empty = " ".join(f"{'':>{cw}}" for _ in available_types if _ != baseline)
            print(f"{'':16} {np_}  {rp_empty}")

    # Held-out average
    if held_out_biases_list:
        ho_avg_vals = {}
        ho_total_n = {}
        ho_avg_ratios = {}
        for tt in available_types:
            ho_biases = [b for b in held_out_biases_list if b in bir.index]
            if ho_biases:
                ho_avg_vals[tt] = bir.loc[ho_biases, tt].mean() * 100
                ho_total_n[tt] = int(counts.loc[ho_biases, tt].sum()) if tt in counts.columns else 0
            else:
                ho_avg_vals[tt] = float("nan")
                ho_total_n[tt] = 0
        for tt in available_types:
            if tt == baseline:
                continue
            baseline_ho_avg = ho_avg_vals.get(baseline, 0)
            ho_avg_ratios[tt] = ho_avg_vals[tt] / baseline_ho_avg if baseline_ho_avg != 0 else float("nan")
        bp = " ".join(f"{ho_avg_vals[tt]:>{cw}.0f}" for tt in available_types)
        rp = " ".join(f"{ho_avg_ratios.get(tt, float('nan')):>{cw}.2f}" for tt in available_types if tt != baseline)
        print(f"{'Held-out Avg':<16} {bp}  {rp}")
        if show_n:
            np_ = " ".join(f"{'n=' + str(ho_total_n[tt]):>{cw}}" for tt in available_types)
            rp_empty = " ".join(f"{'':>{cw}}" for _ in available_types if _ != baseline)
            print(f"{'':16} {np_}  {rp_empty}")

    print("=" * w)
    best_dir = "lowest" if best_is_low else "highest"
    print(f"* = best ({best_dir}) for that bias type (excluding {TRAINING_TYPE_NAMES.get(baseline, baseline)})")
    tr_bias_names = ", ".join(BIAS_DISPLAY_NAMES.get(b, b) for b in sorted(tb_set))
    print(f"Training biases: {tr_bias_names}")


def save_bir_tables(
    bir_df: pd.DataFrame,
    output_path: str,
    models: list[str] | None = None,
    prompt_styles: list[str] | None = None,
    table_variants: list | None = None,
    baseline: str = "base",
    value_label: str = "BIR",
    title: str = "Bias Influence Rate (BIR) Results",
    description: str = "BIR = |bias_match_rate(biased) - bias_match_rate(unbiased)|, per-question",
    training_biases: set[str] | None = None,
):
    """Save pivot table variants to CSV and MD."""
    models = models or ["llama", "gpt"]

    short_names = {
        "suggested_answer": "Sugg. Answer",
        "wrong_few_shot": "Wrong FS",
        "distractor_argument": "Argument",
        "distractor_fact": "Fact",
        "spurious_few_shot_hindsight": "Hindsight",
        "spurious_few_shot_squares": "Squares",
    }

    # Fall back to default variants if none provided
    if table_variants is None:
        table_variants = [
            ("lenient_bir", None, "BIR"),
            ("lenient_bir", lambda df: df["strict_bias_acknowledged"] == 0.0, "Unverbalised"),
            ("lenient_bir", lambda df: df["strict_bias_acknowledged"] == 1.0, "Verbalised"),
        ]

    all_rows = []
    for model_family in models:
        if prompt_styles:
            avail_styles = list(prompt_styles)
        else:
            avail_styles = ["cot", "no_cot"] if model_family == "llama" else ["no_cot"]

        for style in avail_styles:
            for metric_col, filt_fn, filt_label in table_variants:
                sub_df = bir_df if filt_fn is None else bir_df[filt_fn(bir_df)]

                tbl, _ = compute_bir_table(sub_df, model_family, style, metric_col=metric_col, baseline=baseline)
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
                        row_data[f"{TRAINING_TYPE_NAMES.get(tt, tt)} {value_label}%"] = (
                            round(val * 100, 1) if pd.notna(val) else None
                        )
                        if tt != baseline:
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
        f.write(f"# {title}\n\n")
        f.write(f"{description}\n\n")
        for variant_label in [v[2] for v in table_variants]:
            subset = result_df[result_df["Variant"] == variant_label]
            if subset.empty:
                continue
            f.write(f"## {variant_label}\n\n")
            f.write(subset.drop(columns=["Variant"]).to_markdown(index=False))
            f.write("\n\n")
        f.write("## Notes\n\n")
        tr_bias_names = ", ".join(f"**{BIAS_DISPLAY_NAMES.get(b, b)}**" for b in sorted(training_biases or DEFAULT_TRAINING_BIASES))
        f.write(f"- Training bias(es): {tr_bias_names}\n")
        f.write("- **Verbalised** = model mentions the bias in CoT\n")
        f.write("- **Unverbalised** = model does not mention the bias in CoT\n")
        f.write("- **(Strict BA)** = strict bias_acknowledged (NaN-out few-shot confused samples)\n")
        f.write("- **(Lenient BA)** = lenient bias_acknowledged (all samples)\n")
        f.write("- **(Lenient)** = uses fallback parser which recovers some unparseable responses\n")
        f.write(f"- Ratio < 1.0 indicates improvement over {TRAINING_TYPE_NAMES.get(baseline, baseline)} model\n")
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
    # Metric flags (what to compute)
    parser.add_argument("--bir", action="store_true", help="Compute and print BIR (Bias Influence Rate) tables")
    parser.add_argument("--ba", action="store_true", help="Compute and print BA (Bias Acknowledged) tables")
    # Output flags (how to present)
    parser.add_argument("--plot", action="store_true", help="Generate plots (saved to --output-dir)")
    parser.add_argument("--save", type=str, help="Save tables to CSV/MD file")
    parser.add_argument("--no-tables", action="store_true", help="Suppress printing tables to stdout")
    # Hidden backward-compat aliases
    parser.add_argument("--save-bir", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--save-ba", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--no-n", action="store_true", help="Hide n= sample counts in plots/tables")
    parser.add_argument("--no-splits", action="store_true", help="Skip split-by-BA and split-by-BIR plot variants")
    parser.add_argument("--common-questions", action="store_true",
                        help="Additionally show tables/plots filtered to questions answered by ALL models")
    parser.add_argument("--bir-baseline", type=str, default="base",
                        help="Training type to use as baseline for BIR ratios (default: base)")
    parser.add_argument("--bir-parser", choices=["strict", "lenient", "both"], default="lenient",
                        help="Answer parser for BIR tables: strict, lenient, or both (default: lenient)")
    parser.add_argument("--bir-verbalization", choices=["strict", "lenient", "both"], default="strict",
                        help="Verbalization split for BIR tables: strict (NaN-out few-shot confused), lenient, or both (default: strict)")
    parser.add_argument("--training-biases", nargs="+", default=["suggested_answer"],
                        help="Bias types considered as 'trained on' for ordering and averages (default: suggested_answer)")
    parser.add_argument("--aggregate", action="store_true",
                        help="Aggregate BCT/RLCT/Control models into group averages")
    parser.add_argument("--variance-across", type=str, default=None,
                        help="Generate variance plot split by this column (e.g. seed, dataset, prompt_style)")
    args = parser.parse_args()

    # Backward compat: --save-bir implies --bir --save, --save-ba implies --ba --save
    if args.save_bir:
        args.bir = True
        args.save = args.save or args.save_bir
    if args.save_ba:
        args.ba = True
        args.save = args.save or args.save_ba

    training_biases_set = frozenset(args.training_biases)

    log_dirs = args.log_dir or ["logs/fireworks_evals"]

    # When aggregating, override global order/names so plots use aggregate groups
    if args.aggregate:
        global TRAINING_TYPE_ORDER, TRAINING_TYPE_NAMES
        TRAINING_TYPE_ORDER = _AGGREGATE_ORDER
        TRAINING_TYPE_NAMES = _AGGREGATE_NAMES

    if args.bir or args.ba:
        print("Computing per-question BIR...")
        bir_df = compute_per_question_bir(log_dirs, dataset_filter=args.dataset)
        if args.aggregate:
            bir_df = aggregate_training_types(bir_df)
        print(f"Computed BIR for {len(bir_df)} (question, bias_type) pairs")

        # ── Plots (if requested) ──
        if args.plot:
            plot_all_bir_ba(
                bir_df,
                output_dir=args.output_dir,
                models=args.model,
                prompt_styles=args.prompt_style,
                show_n_labels=not args.no_n,
                include_splits=not args.no_splits,
                training_biases=training_biases_set,
            )

            # Variance plots for BIR/BA
            if args.variance_across:
                split = args.variance_across
                print(f"\nGenerating BIR/BA variance plots split by '{split}'...")
                _var_models = args.model or sorted(bir_df["model_family"].unique())
                for mf in _var_models:
                    _styles = args.prompt_style or (["cot", "no_cot"] if mf == "llama" else ["no_cot"])
                    for ps in _styles:
                        for metric_col in ["bir", "bias_acknowledged"]:
                            plot_variance_across(
                                bir_df, metric_col, split, mf,
                                prompt_style=ps,
                                training_biases=training_biases_set,
                                output_path=f"{args.output_dir}/{mf}_{metric_col}_{ps}_var_{split}.png",
                            )

        models = args.model or ["llama", "gpt"]
        show_n = not args.no_n

        def _iter_model_styles(models, prompt_styles):
            for mf in models:
                if prompt_styles:
                    styles = list(prompt_styles)
                else:
                    styles = ["cot", "no_cot"] if mf == "llama" else ["no_cot"]
                for style in styles:
                    yield mf, style

        # ── BIR tables ──
        if args.bir:
            bir_configs = []
            if args.bir_parser in ("strict", "both"):
                bir_configs.append(("bir", "" if args.bir_parser != "both" else " (Strict)"))
            if args.bir_parser in ("lenient", "both"):
                bir_configs.append(("lenient_bir", "" if args.bir_parser != "both" else " (Lenient)"))

            bir_variants = []
            for bir_col, bir_suffix in bir_configs:
                bir_variants.append((bir_col, None, f"BIR{bir_suffix}"))

            if not args.no_tables:
                for mf, style in _iter_model_styles(models, args.prompt_style):
                    for metric_col, filt_fn, label in bir_variants:
                        sub = bir_df if filt_fn is None else bir_df[filt_fn(bir_df)]
                        if sub.empty:
                            continue
                        print_bir_table(sub, mf, style, metric_col=metric_col, label=label,
                                        show_n=show_n, baseline=args.bir_baseline,
                                        table_name="BIR", value_header="BIR %",
                                        ratio_header="BIR Ratio", best_is_low=True,
                                        training_biases=training_biases_set)

            if args.save:
                save_bir_tables(bir_df, args.save, args.model, args.prompt_style,
                                table_variants=bir_variants, baseline=args.bir_baseline,
                                value_label="BIR",
                                title="Bias Influence Rate (BIR) Results",
                                description="BIR = |bias_match_rate(biased) - bias_match_rate(unbiased)|, per-question",
                                training_biases=training_biases_set)

        # ── BA tables ──
        if args.ba:
            ba_configs = []
            if args.bir_verbalization in ("strict", "both"):
                ba_configs.append(("strict_bias_acknowledged", "" if args.bir_verbalization != "both" else " (Strict)"))
            if args.bir_verbalization in ("lenient", "both"):
                ba_configs.append(("bias_acknowledged", "" if args.bir_verbalization != "both" else " (Lenient)"))

            ba_variants = []
            for ba_col, ba_suffix in ba_configs:
                ba_variants.append((ba_col, None, f"BA{ba_suffix} (All)"))
                ba_variants.append((ba_col, lambda df: df["lenient_bir"] > 0, f"BA{ba_suffix} | Flipped"))
                ba_variants.append((ba_col, lambda df: df["lenient_bir"] <= 0, f"BA{ba_suffix} | Unchanged"))

            if not args.no_tables:
                for mf, style in _iter_model_styles(models, args.prompt_style):
                    for metric_col, filt_fn, label in ba_variants:
                        sub = bir_df if filt_fn is None else bir_df[filt_fn(bir_df)]
                        if sub.empty:
                            continue
                        print_bir_table(sub, mf, style, metric_col=metric_col, label=label,
                                        show_n=show_n, baseline=args.bir_baseline,
                                        table_name="BA", value_header="BA %",
                                        ratio_header="BA Ratio", best_is_low=False,
                                        training_biases=training_biases_set)

            if args.save:
                save_path = args.save.replace(".csv", "_ba.csv").replace(".md", "_ba.md") if args.bir else args.save
                save_bir_tables(bir_df, save_path, args.model, args.prompt_style,
                                table_variants=ba_variants, baseline=args.bir_baseline,
                                value_label="BA",
                                title="Bias Acknowledged (BA) Results",
                                description="BA = P(model verbalizes bias in CoT), split by influence",
                                training_biases=training_biases_set)

        # ── Common-questions variant (same questions across all models) ──
        if args.common_questions:
            # Use the primary BIR metric column for determining common questions
            primary_metric = "lenient_bir" if args.bir_parser == "lenient" else "bir"
            cq_df = filter_to_common_questions(bir_df, metric_col=primary_metric)
            n_orig = len(bir_df)
            n_cq = len(cq_df)
            print(f"\n[Common Questions] Filtered to {n_cq}/{n_orig} rows "
                  f"({n_cq * 100 // max(n_orig, 1)}% retained)")

            if args.plot:
                cq_output_dir = args.output_dir.rstrip("/") + "_common_questions"
                plot_all_bir_ba(
                    cq_df,
                    output_dir=cq_output_dir,
                    models=args.model,
                    prompt_styles=args.prompt_style,
                    show_n_labels=not args.no_n,
                    include_splits=not args.no_splits,
                    training_biases=training_biases_set,
                )

            if args.bir and not args.no_tables:
                for mf, style in _iter_model_styles(models, args.prompt_style):
                    for metric_col, filt_fn, label in bir_variants:
                        sub = cq_df if filt_fn is None else cq_df[filt_fn(cq_df)]
                        if sub.empty:
                            continue
                        print_bir_table(sub, mf, style, metric_col=metric_col,
                                        label=f"{label} [Common Questions]",
                                        show_n=show_n, baseline=args.bir_baseline,
                                        table_name="BIR", value_header="BIR %",
                                        ratio_header="BIR Ratio", best_is_low=True,
                                        training_biases=training_biases_set)

            if args.ba and not args.no_tables:
                for mf, style in _iter_model_styles(models, args.prompt_style):
                    for metric_col, filt_fn, label in ba_variants:
                        sub = cq_df if filt_fn is None else cq_df[filt_fn(cq_df)]
                        if sub.empty:
                            continue
                        print_bir_table(sub, mf, style, metric_col=metric_col,
                                        label=f"{label} [Common Questions]",
                                        show_n=show_n, baseline=args.bir_baseline,
                                        table_name="BA", value_header="BA %",
                                        ratio_header="BA Ratio", best_is_low=False,
                                        training_biases=training_biases_set)

            if args.save:
                cq_save = args.save.replace(".csv", "_common_questions.csv").replace(".md", "_common_questions.md")
                if args.bir:
                    save_bir_tables(cq_df, cq_save, args.model, args.prompt_style,
                                    table_variants=bir_variants, baseline=args.bir_baseline,
                                    value_label="BIR",
                                    title="Bias Influence Rate (BIR) Results [Common Questions]",
                                    description="BIR filtered to questions answered by ALL models",
                                    training_biases=training_biases_set)
                if args.ba:
                    cq_ba_save = cq_save.replace(".csv", "_ba.csv").replace(".md", "_ba.md")
                    save_bir_tables(cq_df, cq_ba_save, args.model, args.prompt_style,
                                    table_variants=ba_variants, baseline=args.bir_baseline,
                                    value_label="BA",
                                    title="Bias Acknowledged (BA) Results [Common Questions]",
                                    description="BA filtered to questions answered by ALL models",
                                    training_biases=training_biases_set)

        return

    # Load sample-level data
    print(f"Loading data from {log_dirs}...")
    sample_df = load_sample_data(log_dirs, dataset_filter=args.dataset)
    if args.aggregate:
        sample_df = aggregate_training_types(sample_df)

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
        training_biases=training_biases_set,
    )

    # Variance plots
    if args.variance_across:
        split = args.variance_across
        if split not in sample_df.columns:
            print(f"Error: '{split}' is not a valid column. "
                  f"Available: {sorted(sample_df.columns.tolist())}")
        else:
            n_groups = sample_df[split].nunique()
            if n_groups < 2:
                print(f"Only {n_groups} group(s) for '{split}', skipping variance plots")
            else:
                print(f"\nGenerating variance plots split by '{split}' ({n_groups} groups)...")
                _var_models = args.model or sorted(sample_df["model_family"].unique())
                _var_variants = args.variant or ["biased", "unbiased"]
                for mf in _var_models:
                    _styles = args.prompt_style or (["cot", "no_cot"] if mf == "llama" else ["no_cot"])
                    for ps in _styles:
                        for variant in _var_variants:
                            for metric in ["correct", "matches_bias"]:
                                plot_variance_across(
                                    sample_df, metric, split, mf,
                                    variant=variant, prompt_style=ps,
                                    training_biases=training_biases_set,
                                    output_path=f"{args.output_dir}/{mf}_{variant}_{metric}_{ps}_var_{split}.png",
                                )

    print_summary_table(sample_df)


if __name__ == "__main__":
    main()
