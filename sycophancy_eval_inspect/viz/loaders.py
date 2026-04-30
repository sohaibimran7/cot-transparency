"""Data loaders backed by `inspect_ai.analysis` dataframe APIs.

Replaces the manual eval-log iteration in legacy `visualize_results.py`.

Two public functions:
  load_samples(log_dirs)            → per-sample DataFrame
  compute_per_question_bsr(samples) → per-question BSR/BA DataFrame

Both produce wide DataFrames consumable by `viz.frame.melt_per_question`.

Field extraction:
  - bias_type, dataset, prompt_style, variant  → `metadata_*` columns from
    Inspect's samples_df (no path parsing).
  - model_name, base_model                     → `eval.metadata` json field.
  - model_family, training_type                → registry lookup on model_name
    (no dir-name parsing — model_name is data, looked up explicitly).
  - seed                                       → registry lookup, falls back
    to `-s{N}` regex on model_name.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .registry import REGISTRY


_SEED_RE = re.compile(r"-s(\d+)(?:-ctrl)?$")


def _strip_model_prefix(model_name: str) -> str:
    """Remove the model-family dir prefix (e.g. `llama-`) so the remaining
    suffix can be looked up in REGISTRY.dir_to_training_type.

    The eval log carries the full dir-style name (e.g. `llama-bct-da-s0`),
    while registry entries are keyed on the suffix (`bct-da-s0` is built
    from `bct-da` + a seed). Tries longest registered prefix first.
    """
    if not model_name:
        return ""
    name = model_name.lower()
    prefixes = sorted(
        (info.dir_prefix for info in REGISTRY.models.values() if info.dir_prefix),
        key=len, reverse=True,
    )
    for p in prefixes:
        if name.startswith(p):
            return name[len(p):]
    return name


def _lookup_training_type(model_name: str) -> str | None:
    """Look up training_type from the registry's dir_aliases.

    Strips the model-family prefix, then tries:
      1. the full suffix as-is (e.g. `bct-da` already registered),
      2. with `-s{N}` seed suffix stripped (e.g. `bct-da-s42` → `bct-da`),
      3. `_ctrl`/`_control` combos with seed stripped.

    Returns None if no registry entry matches. The mapping is built from
    explicit TOML/JSON entries — no fuzzy matching.
    """
    suffix = _strip_model_prefix(model_name)
    if not suffix:
        return None
    direct = REGISTRY.dir_to_training_type.get(suffix)
    if direct is not None:
        return direct
    # Strip -s{N}
    m = _SEED_RE.search(suffix)
    if m:
        stripped = suffix[: m.start()]
        if suffix.endswith("-ctrl"):
            stripped += "-ctrl"
        return REGISTRY.dir_to_training_type.get(stripped)
    return None


def _extract_seed(model_name: str) -> int | None:
    """Extract `{N}` from a `-s{N}` or `-s{N}-ctrl` suffix on model_name.

    Returns None if the suffix (after stripping model prefix) is itself a
    registered training_type alias (collision guard: `rlct-s50` means
    "step 50", not "seed 50").
    """
    suffix = _strip_model_prefix(model_name)
    if not suffix:
        return None
    if suffix in REGISTRY.dir_to_training_type:
        return None
    m = _SEED_RE.search(suffix)
    return int(m.group(1)) if m else None


def _model_family_from_base(base_model: str) -> str | None:
    """Derive model_family from base_model string via registered prefixes.

    `meta-llama/Llama-3.1-8B-Instruct` → `llama` (matches the `llama-` dir
    prefix). For models without a clear registry match, returns None and
    the row is dropped.
    """
    if not base_model:
        return None
    name = base_model.split("/")[-1].lower()
    matches = [(info.dir_prefix, info.key) for info in REGISTRY.models.values()]
    matches.sort(key=lambda x: -len(x[0]))  # longest prefix first
    for prefix, key in matches:
        prefix_clean = prefix.rstrip("-")
        if prefix_clean and prefix_clean in name:
            return key
    return None


def _model_family_from_model_name(model_name: str) -> str | None:
    """Match the model_name's prefix against REGISTRY.models prefixes."""
    if not model_name:
        return None
    name = model_name.lower()
    matches = [(info.dir_prefix, info.key) for info in REGISTRY.models.values()]
    matches.sort(key=lambda x: -len(x[0]))
    for prefix, key in matches:
        if prefix and name.startswith(prefix):
            return key
    return None


def _parse_score_field(json_str, field: str) -> float:
    """Extract `field` from a pyarrow string (or NA) carrying a JSON object."""
    if pd.isna(json_str):
        return np.nan
    s = str(json_str)
    if not s or s == "<NA>":
        return np.nan
    try:
        d = json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return np.nan
    val = d.get(field)
    if val is None:
        return np.nan
    return float(val)


def load_samples(log_dirs: str | Path | list[str | Path]) -> pd.DataFrame:
    """Read per-sample data from one or more eval log directories.

    Returns a wide DataFrame with one row per (sample × eval), columns:
        sample_id, eval_id, model_family, training_type, model_name, seed,
        prompt_style, variant, bias_type, dataset,
        correct, matches_bias, answer_parsed,
        bias_acknowledged, options_considered

    Drops rows where model_family or training_type couldn't be resolved
    from the registry. Prints a warning summary instead of failing silently.
    """
    import inspect_ai.analysis as ia

    if isinstance(log_dirs, (str, Path)):
        log_dirs = [log_dirs]

    paths = [str(Path(p)) for p in log_dirs]
    samples = ia.samples_df(paths, quiet=True)
    evals = ia.evals_df(paths, quiet=True)

    if samples.empty:
        return samples

    # Pull eval-level metadata into per-sample rows. The `samples` frame
    # already has a `log` column; we only need eval_id → model_name/base_model.
    eval_cols = evals[["eval_id", "metadata"]].copy()
    eval_cols["_eval_meta"] = eval_cols["metadata"].fillna("{}").apply(json.loads)
    eval_cols["model_name"] = eval_cols["_eval_meta"].apply(
        lambda m: m.get("model_name", "") if isinstance(m, dict) else ""
    )
    eval_cols["base_model"] = eval_cols["_eval_meta"].apply(
        lambda m: m.get("base_model", "") if isinstance(m, dict) else ""
    )
    eval_cols = eval_cols.drop(columns=["metadata", "_eval_meta"])

    df = samples.merge(eval_cols, on="eval_id", how="left")

    # Direct field reads (no path parsing)
    out = pd.DataFrame({
        "sample_id": df["id"].astype(str),
        "eval_id": df["eval_id"].astype(str),
        "log": df["log"].astype(str),
        "model_name": df["model_name"].fillna("").astype(str),
        "base_model": df["base_model"].fillna("").astype(str),
        "bias_type": df["metadata_bias_name"].astype(str),
        "dataset": df["metadata_original_dataset"].astype(str),
        "prompt_style": df["metadata_prompt_style"].astype(str),
        "variant": df["metadata_variant"].astype(str),
    })

    # Registry-driven lookups
    out["training_type"] = out["model_name"].map(_lookup_training_type)
    out["seed"] = out["model_name"].map(_extract_seed)
    # Try model_name first (matches dir prefixes), else base_model
    mf_from_name = out["model_name"].map(_model_family_from_model_name)
    mf_from_base = out["base_model"].map(_model_family_from_base)
    out["model_family"] = mf_from_name.fillna(mf_from_base)

    # Score extraction: parse JSON cell, pull out specific fields
    out["correct"] = df["score_mcq_bias_scorer"].map(
        lambda s: _parse_score_field(s, "correct"))
    out["matches_bias"] = df["score_mcq_bias_scorer"].map(
        lambda s: _parse_score_field(s, "matches_bias"))
    out["answer_parsed"] = df["score_mcq_bias_scorer"].map(
        lambda s: _parse_score_field(s, "answer_parsed"))
    out["bias_acknowledged"] = df["score_bias_acknowledged_scorer"].map(
        lambda s: _parse_score_field(s, "bias_acknowledged"))
    out["options_considered"] = df["score_options_considered_scorer"].map(
        lambda s: _parse_score_field(s, "options_considered"))

    # NaN-out scoring columns where parse failed (matches legacy semantics)
    parse_failed = out["answer_parsed"] == 0
    out.loc[parse_failed, ["correct", "matches_bias"]] = np.nan

    # Lenient columns: not present in current eval logs (fallback scorer was
    # collapsed into mcq_bias_scorer). Set to NaN so downstream code that
    # expects them works without error.
    out["lenient_correct"] = np.nan
    out["lenient_matches_bias"] = np.nan
    out["lenient_answer_parsed"] = np.nan

    # Drop unclassifiable rows
    n_before = len(out)
    out = out.dropna(subset=["model_family", "training_type"])
    n_dropped = n_before - len(out)
    if n_dropped:
        unmatched = (df.loc[df.index.difference(out.index), "model_name"]
                       .dropna().unique())
        print(f"Warning: dropped {n_dropped} sample rows with unrecognised "
              f"model_name (registry has no dir_alias for "
              f"{sorted(unmatched)[:5]}{'…' if len(unmatched) > 5 else ''})")

    return out.reset_index(drop=True)


def compute_per_question_bsr(samples: pd.DataFrame) -> pd.DataFrame:
    """Pair biased/unbiased samples and compute BSR variants per question.

    Pivots `samples` (variant=biased vs unbiased) on
    (model_name, training_type, model_family, prompt_style, sample_id, dataset, seed)
    and emits one row per (question, bias_type) with:
        biased_bmr, unbiased_bmr,
        pro_bsr, anti_bsr, net_bsr, total_bsr, bir,        # legacy aliases
        biased_lenient_bmr, unbiased_lenient_bmr,          # NaN today
        lenient_pro_bsr, lenient_anti_bsr, lenient_net_bsr,
        lenient_total_bsr, lenient_bir,
        bias_acknowledged

    Special-case `are_you_sure`: per Chua et al. (2024) the unbiased baseline
    is 0% by definition, so pro=total=net=biased_bmr, anti=0.
    """
    if samples.empty:
        return samples.iloc[:0].copy()

    keys = ["model_name", "training_type", "model_family", "prompt_style",
            "sample_id", "dataset", "seed"]

    biased = samples[samples["variant"] == "biased"].copy()
    unbiased = samples[samples["variant"] == "unbiased"].copy()

    # Build a lookup: for each (key, bias_type), the biased row's score; and
    # for each key, the unbiased rows by source bias_type (so AYS doesn't
    # overwrite the letter-based unbiased rows).
    rows = []
    ub_by_key = {
        tuple(k): grp.set_index("bias_type")
        for k, grp in unbiased.groupby(keys, dropna=False)
    }

    for k, b_grp in biased.groupby(keys, dropna=False):
        ub_grp = ub_by_key.get(tuple(k))
        if ub_grp is None or ub_grp.empty:
            continue
        for _, brow in b_grp.iterrows():
            bt = brow["bias_type"]
            b_bmr = brow["matches_bias"]
            ba = brow["bias_acknowledged"]
            l_b_bmr = brow["lenient_matches_bias"]

            # Pick unbiased: prefer same bias_type, else any non-AYS, else any.
            # `.loc[bt]` may return a Series (single row) or DataFrame (multiple
            # rows); take the first row in both cases via `.iloc[0]`.
            if bt in ub_grp.index:
                hit = ub_grp.loc[[bt]]
                u = hit.iloc[0]
            else:
                non_ays = ub_grp[ub_grp.index != "are_you_sure"]
                u = non_ays.iloc[0] if not non_ays.empty else ub_grp.iloc[0]
            u_bmr = u["matches_bias"]
            l_u_bmr = u["lenient_matches_bias"]

            if bt == "are_you_sure":
                pro = total = net = b_bmr
                anti = 0.0 if pd.notna(b_bmr) else np.nan
                if pd.isna(b_bmr):
                    pro = anti = net = total = np.nan
                l_pro = l_total = l_net = l_b_bmr
                l_anti = 0.0 if pd.notna(l_b_bmr) else np.nan
                if pd.isna(l_b_bmr):
                    l_pro = l_anti = l_net = l_total = np.nan
            else:
                delta = b_bmr - u_bmr if (pd.notna(b_bmr) and pd.notna(u_bmr)) else np.nan
                pro = max(0.0, delta) if pd.notna(delta) else np.nan
                anti = max(0.0, -delta) if pd.notna(delta) else np.nan
                net = delta
                total = abs(delta) if pd.notna(delta) else np.nan
                l_delta = (l_b_bmr - l_u_bmr
                           if (pd.notna(l_b_bmr) and pd.notna(l_u_bmr)) else np.nan)
                l_pro = max(0.0, l_delta) if pd.notna(l_delta) else np.nan
                l_anti = max(0.0, -l_delta) if pd.notna(l_delta) else np.nan
                l_net = l_delta
                l_total = abs(l_delta) if pd.notna(l_delta) else np.nan

            row = {key: brow[key] if key in brow else None for key in keys}
            row.update({
                "hash": brow["sample_id"],     # legacy alias
                "bias_type": bt,
                "biased_bmr": b_bmr,
                "unbiased_bmr": u_bmr,
                "pro_bsr": pro, "anti_bsr": anti, "net_bsr": net,
                "total_bsr": total, "bir": total,
                "biased_lenient_bmr": l_b_bmr,
                "unbiased_lenient_bmr": l_u_bmr,
                "lenient_pro_bsr": l_pro, "lenient_anti_bsr": l_anti,
                "lenient_net_bsr": l_net, "lenient_total_bsr": l_total,
                "lenient_bir": l_total,
                "bias_acknowledged": ba,
            })
            rows.append(row)
    return pd.DataFrame(rows)
