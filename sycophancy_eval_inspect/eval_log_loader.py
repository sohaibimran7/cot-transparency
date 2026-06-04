"""Shared loading layer for Inspect `.eval` logs.

This is the single source of truth for:
  * parsing model/training-type/seed/family from a log directory name, and
  * iterating samples out of `.eval` logs into normalized rows, and
  * extracting the canonical strict/lenient bias-match metrics from a sample.

It deliberately has NO matplotlib / plotting dependency so it can sit *below*
`visualize_results.py` (which keeps the plotting/style constants) and be imported
by `plot_model_comparison.py`, `extract_bir3.py`, etc. without an import cycle.

Previously the dir/log/sample iteration loop was copy-pasted in four places
(`visualize_results.load_sample_data`, `visualize_results.compute_per_question_bir`,
`plot_model_comparison.compute_noise_floor`, `scripts/tinker_training/extract_bir3.py`)
and could silently drift. Those now drive off `iter_eval_samples` / `extract_bias_metrics`.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator, NamedTuple

import numpy as np


# ── Directory-name → (training_type, model_family, seed) parsing ─────────────

_REGISTRY_PATH = Path(__file__).parent / "model_registry.json"
_MODEL_PREFIXES = ["llama-", "gpt-oss-120b-", "gpt-oss-20b-", "gpt-"]

_DIR_TO_TRAINING_TYPE: dict[str, str | None] = {
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
    "base-tqa-659": "base",
}

_SEED_SUFFIX_RE = re.compile(r"-s(\d+)$")


def read_model_registry() -> dict:
    """Read model_registry.json (or {} if absent)."""
    if not _REGISTRY_PATH.exists():
        return {}
    with open(_REGISTRY_PATH) as f:
        return json.load(f)


def iter_registry_models() -> Iterator[tuple[str, str, dict]]:
    """Yield (dir_suffix, training_type, entry) for each registered model.

    `training_type` is computed identically to how it's stored in
    `_DIR_TO_TRAINING_TYPE`, so the dir-parsing layer (here) and the plotting/style
    layer (visualize_results) agree on the key without duplicating the rule.
    """
    for suffix, entry in read_model_registry().get("models", {}).items():
        yield suffix, entry.get("training_type", suffix.replace("-", "_")), entry


def _load_registry_dirs() -> None:
    """Merge registry model_prefixes + dir→training_type into the module dicts.

    Reads the registry once and applies both fields (the `training_type` rule matches
    iter_registry_models, which visualize_results uses for the style half of the merge).
    """
    registry = read_model_registry()
    for p in registry.get("model_prefixes", []):
        if p not in _MODEL_PREFIXES:
            _MODEL_PREFIXES.append(p)
    for suffix, entry in registry.get("models", {}).items():
        if suffix not in _DIR_TO_TRAINING_TYPE:
            _DIR_TO_TRAINING_TYPE[suffix] = entry.get("training_type", suffix.replace("-", "_"))


_load_registry_dirs()


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
    """Extract model family from directory name (e.g. 'llama-' -> 'llama')."""
    lower = dir_name.lower()
    for prefix in sorted(_MODEL_PREFIXES, key=len, reverse=True):
        if lower.startswith(prefix):
            return prefix.rstrip("-")
    return None


def _iter_model_dirs(log_dirs) -> Iterator[tuple[Path, str]]:
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


# ── Sample iteration + canonical metric extraction ──────────────────────────

class EvalSampleCtx(NamedTuple):
    """One sample plus its resolved log/dir context.

    `dir_name` is the model directory name (the identity used by BIR aggregation);
    `model` is `log.eval.model` (the raw model string used by load_sample_data).
    """
    dir_name: str
    model: str
    training_type: str | None
    model_family: str | None
    seed: int | None
    variant: str | None
    prompt_style: str
    dataset_path: str
    bias_type: str
    dataset: str
    sample: object  # inspect_ai EvalSample


class BiasMetrics(NamedTuple):
    """Canonical bias metrics extracted from a sample's scores.

    `bmr`/`l_bmr` are NaN when the corresponding parser failed. Callers decide
    whether to skip (e.g. compute_per_question_bir drops samples where BOTH parsers
    failed; compute_noise_floor keeps only strict-parsed unbiased samples).
    """
    strict_parsed: float
    bmr: float
    lenient_parsed: float
    l_bmr: float
    bias_ack: float


def extract_bias_metrics(sample) -> BiasMetrics | None:
    """Extract strict/lenient bias-match-rate + bias_acknowledged from a sample.

    Returns None when the primary `mcq_bias_scorer` is missing or empty (same as the
    old `if not score or not score.value: continue`). This is the single definition of
    how a sample's BMR is read — previously duplicated in compute_per_question_bir and
    compute_noise_floor.
    """
    if not sample.scores:
        return None
    score = sample.scores.get("mcq_bias_scorer")
    if not score or not score.value:
        return None

    # Strict parse success: explicit key, else fall back to scorer metadata.
    strict_parsed = score.value.get("answer_parsed")
    if strict_parsed is None:
        strict_parsed = 1.0 if (score.metadata or {}).get("parse_success", True) else 0.0

    # Lenient parse success comes from the fallback scorer, if present.
    l_score = sample.scores.get("mcq_bias_scorer_fallback")
    lenient_parsed = 0.0
    if l_score and l_score.value:
        lenient_parsed = l_score.value.get("lenient_answer_parsed", 0.0)

    # Strict BMR (NaN if strict parse failed).
    if strict_parsed:
        bmr = score.value.get("matches_bias", score.value.get("bias_match_rate", np.nan))
    else:
        bmr = np.nan

    # Lenient BMR (NaN if lenient parse failed).
    if lenient_parsed and l_score and l_score.value:
        l_bmr = l_score.value.get("lenient_matches_bias", np.nan)
    else:
        l_bmr = np.nan

    # Bias acknowledged (separate scorer).
    ba_score = sample.scores.get("bias_acknowledged_scorer")
    bias_ack = ba_score.value.get("bias_acknowledged", np.nan) if (ba_score and ba_score.value) else np.nan

    return BiasMetrics(strict_parsed, bmr, lenient_parsed, l_bmr, bias_ack)


def iter_eval_samples(log_dirs, dataset_filter: list[str] | None = None) -> Iterator[EvalSampleCtx]:
    """Iterate every sample across all `.eval` logs under `log_dirs`, normalized.

    Yields one `EvalSampleCtx` per sample (no None-filtering on training_type /
    model_family — callers apply their own guard, since they historically differ).
    Honors `dataset_filter` on the derived `dataset` name. Unreadable logs are skipped
    with a warning, matching the prior behavior.
    """
    from inspect_ai.log import read_eval_log

    if isinstance(log_dirs, (str, Path)):
        log_dirs = [log_dirs]

    for model_dir, dir_name in _iter_model_dirs(log_dirs):
        training_type = _get_training_type_from_dir(dir_name)
        model_family = _get_model_family_from_dir(dir_name)
        seed = _get_seed_from_dir(dir_name)

        for eval_file in sorted(model_dir.glob("*.eval")):
            try:
                log = read_eval_log(str(eval_file))
            except Exception as e:
                print(f"Warning: Could not read {eval_file}: {e}")
                continue

            variant = log.eval.task_args.get("variant")
            prompt_style = log.eval.task_args.get("prompt_style", "no_cot")
            dataset_path = log.eval.task_args.get("dataset_path", "")
            bias_type = Path(dataset_path).parent.name if dataset_path else "unknown"
            dataset_stem = Path(dataset_path).stem if dataset_path else "unknown"
            dataset = dataset_stem.replace(f"_{bias_type}", "")

            if dataset_filter and dataset not in dataset_filter:
                continue

            for sample in log.samples or []:
                yield EvalSampleCtx(
                    dir_name=dir_name,
                    model=log.eval.model,
                    training_type=training_type,
                    model_family=model_family,
                    seed=seed,
                    variant=variant,
                    prompt_style=prompt_style,
                    dataset_path=dataset_path,
                    bias_type=bias_type,
                    dataset=dataset,
                    sample=sample,
                )
