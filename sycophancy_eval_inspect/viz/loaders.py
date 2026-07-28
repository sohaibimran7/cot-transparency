"""Data loaders backed by `inspect_ai.analysis` dataframe APIs.

Replaces the manual eval-log iteration in legacy `visualize_results.py`.

Two public functions:
  load_samples(log_dirs)            → per-sample DataFrame
  compute_per_question_bsr(samples) → per-question BSR/BA DataFrame

Both produce wide DataFrames consumable by `viz.frame.melt_per_question`.

Field extraction:
  - bias_type, dataset, prompt_style, variant  → `metadata_*` columns from
    Inspect's samples_df (no path parsing).
  - model_name, base_model                     → `eval.metadata` json field,
    with the registered parent-directory alias as a compatibility fallback.
  - model_family, training_type                → registry lookup on that
    resolved model identity.
  - seed                                       → registry lookup, falls back
    to `-s{N}` regex on model_name.
"""

from __future__ import annotations

import json
import os
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
        key=len,
        reverse=True,
    )
    for p in prefixes:
        if name.startswith(p):
            return name[len(p) :]
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
    if suffix == "base":
        return "base"
    direct = REGISTRY.dir_to_training_type.get(suffix)
    if direct is not None:
        return direct
    # Strip -s{N}
    m = _SEED_RE.search(suffix)
    if m:
        stripped = suffix[: m.start()]
        if suffix.endswith("-ctrl"):
            stripped += "-ctrl"
        if stripped == "base":
            return "base"
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


def _model_dir_name(log_path: object) -> str:
    """Return the immediate model-directory name for a local Inspect log."""
    value = str(log_path or "")
    if value.startswith("file://"):
        value = value.removeprefix("file://")
    return Path(value).parent.name


def _resolve_model_identity(
    metadata_model_name: object,
    dir_name: object,
    base_model: object,
) -> tuple[str, str | None, str | None, int | None]:
    """Resolve (name, training type, family, seed) through the registry.

    Modern logs carry the registered directory-style name in eval metadata.
    Some older logs instead carry an internal checkpoint label (for example
    ``bct_mt_final``), while their parent directory still has the registered
    experiment alias. Prefer registered metadata, then the registered parent
    directory; retain the metadata value in warnings if neither is known.
    """
    metadata_name = str(metadata_model_name or "")
    directory_name = str(dir_name or "")
    base_name = str(base_model or "")

    model_name = metadata_name or directory_name
    training_type = _lookup_training_type(metadata_name)
    if training_type is None:
        directory_training_type = _lookup_training_type(directory_name)
        if directory_training_type is not None:
            model_name = directory_name
            training_type = directory_training_type

    model_family = (
        _model_family_from_model_name(model_name)
        or _model_family_from_model_name(directory_name)
        or _model_family_from_base(base_name)
    )
    seed = _extract_seed(model_name)
    if seed is None and model_name != directory_name:
        seed = _extract_seed(directory_name)
    return model_name, training_type, model_family, seed


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


def _expand_eval_log_paths(log_dirs: list[str | Path]) -> list[str]:
    """Resolve local log dirs to concrete .eval files before calling Inspect.

    Inspect's directory resolver can return an empty table for some nested
    artifact roots and symlinked log dirs. Expanding local directories here
    keeps CLI inputs flexible while still passing Inspect ordinary log files.
    """
    paths: list[str] = []
    for raw in log_dirs:
        path = Path(raw).expanduser()
        resolved = path.resolve()
        if resolved.is_dir():
            evals: list[str] = []
            seen_dirs: set[str] = set()
            for root, dirs, files in os.walk(resolved, followlinks=True):
                dirs[:] = [directory for directory in dirs if directory != "_archive"]
                real_root = os.path.realpath(root)
                if real_root in seen_dirs:
                    dirs[:] = []
                    continue
                seen_dirs.add(real_root)
                evals.extend(str(Path(root) / file) for file in files if file.endswith(".eval"))
            paths.extend(sorted(evals))
        elif resolved.is_file():
            paths.append(str(resolved))
        else:
            paths.append(str(path))
    # Callers sometimes pass both an artifact root and one of its nested log
    # directories. De-duplicate by real path so `dedup="none"` still means
    # one row per physical eval, not one row per overlapping CLI argument.
    unique: list[str] = []
    seen: set[str] = set()
    for item in paths:
        key = os.path.realpath(item) if os.path.exists(item) else os.path.abspath(item)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _read_inspect_frames(ia, paths: list[str], batch_size: int = 64) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read Inspect sample/eval tables in batches to cap peak memory."""
    if not paths:
        return pd.DataFrame(), pd.DataFrame()

    sample_frames = []
    eval_frames = []
    for start in range(0, len(paths), batch_size):
        batch = paths[start : start + batch_size]
        batch_samples = ia.samples_df(batch, quiet=True)
        batch_evals = ia.evals_df(batch, quiet=True)
        if not batch_samples.empty:
            sample_frames.append(batch_samples)
        if not batch_evals.empty:
            eval_frames.append(batch_evals)

    samples = pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame()
    evals = pd.concat(eval_frames, ignore_index=True) if eval_frames else pd.DataFrame()
    return samples, evals


def _as_dict(value) -> dict:
    """Best-effort conversion of Inspect metadata/score values to a dict."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    if hasattr(value, "model_dump"):
        parsed = value.model_dump()
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _number(mapping: dict, *keys: str) -> float:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return np.nan
    return np.nan


def _read_eval_logs(paths: list[str]) -> pd.DataFrame:
    """Compatibility reader for logs unsupported by Inspect's dataframe API."""
    from inspect_ai.log import read_eval_log

    rows = []
    for path_str in paths:
        path = Path(path_str)
        try:
            log = read_eval_log(str(path))
        except Exception as exc:
            print(f"Warning: could not read {path}: {exc}")
            continue

        eval_meta = _as_dict(getattr(log.eval, "metadata", {}))
        task_args = _as_dict(getattr(log.eval, "task_args", {}))
        dir_name = path.parent.name
        base_model = str(eval_meta.get("base_model") or getattr(log.eval, "model", ""))
        model_name, training_type, model_family, seed = _resolve_model_identity(
            eval_meta.get("model_name", ""), dir_name, base_model
        )
        variant_default = str(task_args.get("variant", ""))
        prompt_default = str(task_args.get("prompt_style", "no_cot"))
        dataset_path = str(task_args.get("dataset_path", ""))
        path_bias = Path(dataset_path).parent.name if dataset_path else "unknown"
        path_dataset = Path(dataset_path).stem.replace(f"_{path_bias}", "") if dataset_path else "unknown"

        created = str(getattr(log.eval, "created", "") or path.name)

        for sample in log.samples or []:
            sample_meta = _as_dict(getattr(sample, "metadata", {}))
            scores = getattr(sample, "scores", None) or {}
            primary_score = scores.get("mcq_bias_scorer")
            primary = _as_dict(getattr(primary_score, "value", {}))
            primary_meta = _as_dict(getattr(primary_score, "metadata", {}))
            fallback_score = scores.get("mcq_bias_scorer_fallback")
            fallback = _as_dict(getattr(fallback_score, "value", {}))
            bias_score = scores.get("bias_acknowledged_scorer")
            bias_values = _as_dict(getattr(bias_score, "value", {}))
            options_score = scores.get("options_considered_scorer")
            options_values = _as_dict(getattr(options_score, "value", {}))

            parsed = _number(primary, "answer_parsed")
            if np.isnan(parsed):
                parsed = 1.0 if primary_meta.get("parse_success", True) else 0.0
            lenient_parsed = _number(fallback, "lenient_answer_parsed")
            if np.isnan(lenient_parsed):
                lenient_parsed = 0.0

            correct = _number(primary, "correct", "accuracy")
            matches_bias = _number(primary, "matches_bias", "bias_match_rate")
            lenient_correct = _number(fallback, "lenient_correct")
            lenient_matches_bias = _number(fallback, "lenient_matches_bias")
            if parsed == 0:
                correct = matches_bias = np.nan
            if lenient_parsed == 0:
                lenient_correct = lenient_matches_bias = np.nan

            rows.append(
                {
                    "sample_id": str(sample.id),
                    "eval_id": str(getattr(log.eval, "run_id", "") or path),
                    "created": created,
                    "log": str(path),
                    "model_name": model_name,
                    "base_model": base_model,
                    "bias_type": str(sample_meta.get("bias_name") or path_bias),
                    "dataset": str(sample_meta.get("original_dataset") or path_dataset),
                    "prompt_style": str(sample_meta.get("prompt_style") or prompt_default),
                    "variant": str(sample_meta.get("variant") or variant_default),
                    "training_type": training_type,
                    "seed": seed,
                    "model_family": model_family,
                    "correct": correct,
                    "matches_bias": matches_bias,
                    "answer_parsed": parsed,
                    "bias_acknowledged": _number(bias_values, "bias_acknowledged"),
                    "options_considered": _number(options_values, "options_considered"),
                    "lenient_correct": lenient_correct,
                    "lenient_matches_bias": lenient_matches_bias,
                    "lenient_answer_parsed": lenient_parsed,
                }
            )
    return pd.DataFrame(rows)


def _finalize_samples(out: pd.DataFrame, dedup: str) -> pd.DataFrame:
    if out.empty:
        return out

    unclassified = out[out[["model_family", "training_type"]].isna().any(axis=1)]
    if not unclassified.empty:
        names = sorted(unclassified["model_name"].dropna().astype(str).unique())
        print(
            f"Warning: dropped {len(unclassified)} sample rows with unrecognised "
            f"model_name (registry has no dir_alias for "
            f"{names[:5]}{'…' if len(names) > 5 else ''})"
        )
    out = out.dropna(subset=["model_family", "training_type"])

    dedup_keys = [
        "model_name",
        "dataset",
        "bias_type",
        "variant",
        "sample_id",
        "prompt_style",
    ]
    if dedup == "none":
        return out.reset_index(drop=True)
    if dedup == "last":
        return (
            out.sort_values("created", kind="stable")
            .drop_duplicates(subset=dedup_keys, keep="last")
            .reset_index(drop=True)
        )
    if dedup == "mean":
        score_cols = [
            "correct",
            "matches_bias",
            "answer_parsed",
            "bias_acknowledged",
            "options_considered",
            "lenient_correct",
            "lenient_matches_bias",
            "lenient_answer_parsed",
        ]
        score_cols = [column for column in score_cols if column in out.columns]
        meta_cols = [column for column in out.columns if column not in dedup_keys and column not in score_cols]
        aggregations = {
            **{column: "mean" for column in score_cols},
            **{column: "first" for column in meta_cols},
        }
        return out.groupby(dedup_keys, as_index=False, dropna=False).agg(aggregations).reset_index(drop=True)
    raise ValueError(f"dedup must be 'last', 'mean', or 'none'; got {dedup!r}")


def load_samples(log_dirs: str | Path | list[str | Path], dedup: str = "last") -> pd.DataFrame:
    """Read per-sample data from one or more eval log directories.

    When the same logical `(model, sample, bias, variant, prompt_style)` was
    scored by multiple eval runs, choose how to collapse them:
      - `"last"`  — keep the chronologically latest run only (matches legacy
                    behavior when re-runs replaced earlier waves).
      - `"mean"`  — average each score across runs. For binary 0/1 scores this
                    yields a fractional consensus value (round to get majority
                    vote); for already-fractional metrics it gives the mean.
      - `"none"`  — return all rows without dedup (debugging only; downstream
                    BIR computation will count duplicates).

    Returns a wide DataFrame with one row per logical sample, columns:
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

    paths = _expand_eval_log_paths(log_dirs)
    if not paths:
        return pd.DataFrame()
    try:
        samples, evals = _read_inspect_frames(ia, paths)
    except Exception as exc:
        print(
            "Warning: Inspect dataframe loading failed; falling back to " f"read_eval_log ({type(exc).__name__}: {exc})"
        )
        return _finalize_samples(_read_eval_logs(paths), dedup)

    if samples.empty or evals.empty:
        return _finalize_samples(_read_eval_logs(paths), dedup)

    # Pull eval-level metadata + chronology into per-sample rows.
    eval_cols = evals[["eval_id", "metadata", "created"]].copy()
    eval_cols["_eval_meta"] = eval_cols["metadata"].map(_as_dict)
    eval_cols["model_name"] = eval_cols["_eval_meta"].apply(
        lambda m: m.get("model_name", "") if isinstance(m, dict) else ""
    )
    eval_cols["base_model"] = eval_cols["_eval_meta"].apply(
        lambda m: m.get("base_model", "") if isinstance(m, dict) else ""
    )
    eval_cols = eval_cols.drop(columns=["metadata", "_eval_meta"])

    df = samples.merge(eval_cols, on="eval_id", how="left")

    # Resolve identity once per distinct (metadata name, directory, base)
    # triple instead of repeating registry work for every sample row.
    df["_model_dir_name"] = df["log"].map(_model_dir_name)
    identity_keys = list(
        zip(
            df["model_name"].fillna("").astype(str),
            df["_model_dir_name"].fillna("").astype(str),
            df["base_model"].fillna("").astype(str),
        )
    )
    identity_cache = {key: _resolve_model_identity(*key) for key in set(identity_keys)}
    identities = [identity_cache[key] for key in identity_keys]

    # Direct sample/eval field reads. Directory names are used only by the
    # compatibility identity resolver above, never for dataset metadata.
    out = pd.DataFrame(
        {
            "sample_id": df["id"].astype(str),
            "eval_id": df["eval_id"].astype(str),
            "created": df["created"],
            "log": df["log"].astype(str),
            "model_name": [identity[0] for identity in identities],
            "base_model": df["base_model"].fillna("").astype(str),
            "bias_type": df["metadata_bias_name"].astype(str),
            "dataset": df["metadata_original_dataset"].astype(str),
            "prompt_style": df["metadata_prompt_style"].astype(str),
            "variant": df["metadata_variant"].astype(str),
        }
    )

    out["training_type"] = [identity[1] for identity in identities]
    out["model_family"] = [identity[2] for identity in identities]
    out["seed"] = [identity[3] for identity in identities]

    # Score extraction: parse JSON cells, tolerating scorer columns that are
    # absent from older logs.
    def score_field(column: str, field: str) -> pd.Series:
        if column not in df.columns:
            return pd.Series(np.nan, index=df.index, dtype=float)
        return df[column].map(lambda value: _parse_score_field(value, field))

    out["correct"] = score_field("score_mcq_bias_scorer", "correct").fillna(
        score_field("score_mcq_bias_scorer", "accuracy")
    )
    out["matches_bias"] = score_field("score_mcq_bias_scorer", "matches_bias").fillna(
        score_field("score_mcq_bias_scorer", "bias_match_rate")
    )
    out["answer_parsed"] = score_field("score_mcq_bias_scorer", "answer_parsed")
    out["bias_acknowledged"] = score_field("score_bias_acknowledged_scorer", "bias_acknowledged")
    out["options_considered"] = score_field("score_options_considered_scorer", "options_considered")

    # NaN-out scoring columns where parse failed (matches legacy semantics)
    parse_failed = out["answer_parsed"] == 0
    out.loc[parse_failed, ["correct", "matches_bias"]] = np.nan

    out["lenient_correct"] = score_field("score_mcq_bias_scorer_fallback", "lenient_correct")
    out["lenient_matches_bias"] = score_field("score_mcq_bias_scorer_fallback", "lenient_matches_bias")
    out["lenient_answer_parsed"] = score_field("score_mcq_bias_scorer_fallback", "lenient_answer_parsed")
    lenient_failed = out["lenient_answer_parsed"] == 0
    out.loc[lenient_failed, ["lenient_correct", "lenient_matches_bias"]] = np.nan

    return _finalize_samples(out, dedup)


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

    # The canonical legacy implementation discarded a sample only when both
    # parsers failed.  Keep parse-failed rows in ``load_samples`` so parse-rate
    # plots remain possible, but exclude them before building BIR pairs.  If a
    # caller supplies an already-normalized frame without parser columns, its
    # existing score NaNs continue to determine validity below.
    if {"answer_parsed", "lenient_answer_parsed"}.issubset(samples.columns):
        strict_ok = samples["answer_parsed"].fillna(0).ne(0)
        lenient_ok = samples["lenient_answer_parsed"].fillna(0).ne(0)
        samples = samples[strict_ok | lenient_ok].copy()

    keys = ["model_name", "training_type", "model_family", "prompt_style", "sample_id", "dataset", "seed"]

    samples = samples.copy()

    # NaN values compare unequal even when they identify the same missing key
    # on both sides of a biased/unbiased pair.  Normalize lookup keys without
    # mutating the public output values (notably the missing seed for base
    # models, which legacy callers expect to remain missing).
    missing_key = object()

    def pair_key(values: tuple[object, ...]) -> tuple[object, ...]:
        return tuple(missing_key if pd.isna(value) else value for value in values)

    biased = samples[samples["variant"] == "biased"].copy()
    unbiased = samples[samples["variant"] == "unbiased"].copy()

    # Build a lookup: for each (key, bias_type), the biased row's score; and
    # for each key, the unbiased rows by source bias_type (so AYS doesn't
    # overwrite the letter-based unbiased rows).
    rows = []
    ub_by_key = {pair_key(tuple(k)): grp.set_index("bias_type") for k, grp in unbiased.groupby(keys, dropna=False)}

    for k, b_grp in biased.groupby(keys, dropna=False):
        ub_grp = ub_by_key.get(pair_key(tuple(k)))
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
                l_delta = l_b_bmr - l_u_bmr if (pd.notna(l_b_bmr) and pd.notna(l_u_bmr)) else np.nan
                l_pro = max(0.0, l_delta) if pd.notna(l_delta) else np.nan
                l_anti = max(0.0, -l_delta) if pd.notna(l_delta) else np.nan
                l_net = l_delta
                l_total = abs(l_delta) if pd.notna(l_delta) else np.nan

            row = {key: brow[key] if key in brow else None for key in keys}
            row.update(
                {
                    # Legacy/public compatibility: downstream population-BIR
                    # consumers group on `model`; the loader-native identity
                    # is named `model_name`.
                    "model": brow["model_name"],
                    "hash": brow["sample_id"],  # legacy alias
                    "bias_type": bt,
                    "biased_bmr": b_bmr,
                    "unbiased_bmr": u_bmr,
                    "pro_bsr": pro,
                    "anti_bsr": anti,
                    "net_bsr": net,
                    "total_bsr": total,
                    "bir": total,
                    "biased_lenient_bmr": l_b_bmr,
                    "unbiased_lenient_bmr": l_u_bmr,
                    "lenient_pro_bsr": l_pro,
                    "lenient_anti_bsr": l_anti,
                    "lenient_net_bsr": l_net,
                    "lenient_total_bsr": l_total,
                    "lenient_bir": l_total,
                    "bias_acknowledged": ba,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)
