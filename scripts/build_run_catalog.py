#!/usr/bin/env python3
"""
Build a unified catalog of all training runs across this repo.

Walks experiment configs in standard locations, matches each to its eval
log directory, computes per-run summary metrics (pro-BSR, BIR, accuracy)
plus bootstrap CIs, and writes a single parquet (one row per run).

Output:
    artifacts/run_catalog.parquet
    artifacts/run_catalog.csv

The schema unifies HPs across sweep families (RLCT vs BCT, different
biases, datasets, prompt styles). Inapplicable cells are NaN, not zero.

Downstream consumers: the W&B push script and any local
seaborn/altair visualisation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

CONFIG_GLOBS = [
    "artifacts/runs/*/metadata/config.yaml",
    "artifacts/legacy/experiments/*/config.yaml",
    "scripts/tinker_training/experiment_configs/**/*.yaml",
]


def discover_configs() -> list[Path]:
    """Walk all known config roots, dedupe by config `name`. Prefer the
    runtime snapshot in artifacts/runs/<name>/metadata/config.yaml when
    available (that's the as-executed state); fall back to source configs.
    """
    by_name: dict[str, Path] = {}
    # Prefer artifacts/runs/ snapshots (most authoritative)
    for p in sorted(PROJECT_ROOT.glob("artifacts/runs/*/metadata/config.yaml")):
        if "_archive" in p.parts:
            continue
        name = _quick_name(p)
        if name and name not in by_name:
            by_name[name] = p

    # Then legacy experiments
    for p in sorted(PROJECT_ROOT.glob("artifacts/legacy/experiments/*/config.yaml")):
        if "_archive" in p.parts:
            continue
        name = _quick_name(p)
        if name and name not in by_name:
            by_name[name] = p

    # Then static config sources (ones that haven't been run yet)
    for p in sorted(PROJECT_ROOT.glob("scripts/tinker_training/experiment_configs/**/*.yaml")):
        if "_archive" in p.parts:
            continue
        name = _quick_name(p)
        if name and name not in by_name:
            by_name[name] = p

    return list(by_name.values())


def _quick_name(p: Path) -> str | None:
    try:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line.startswith("name:"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# HP extraction
# ---------------------------------------------------------------------------

@dataclass
class RunRow:
    # Identity
    name: str | None = None
    config_path: str | None = None
    dir_suffix: str | None = None  # registry key
    training_type: str | None = None

    # Faceting (categorical)
    model: str | None = None
    method: str | None = None  # rl | sft | bct | rlct
    training_bias: str | None = None  # canonical (sa, da, da_g4, wfs, sa+wfs, ...)
    training_dataset: str | None = None
    eval_dataset: str | None = None
    prompt_style: str | None = None
    is_control: bool = False

    # RLCT HPs
    lr: float | None = None
    lr_schedule: str | None = None
    lora_rank: int | None = None
    kl_coef: float | None = None
    anchor_weight: float | None = None
    anchor_model: str | None = None
    loss_fn: str | None = None
    n_ref_rollouts: int | None = None
    n_train_rollouts: int | None = None
    n_consistency_rollouts: int | None = None
    n_anchor_rollouts: int | None = None
    temperature: float | None = None
    max_new_tokens: int | None = None
    n_datapoints: int | None = None
    n_epochs: int | None = None
    batch_size: int | None = None
    gradient_accumulation_steps: int | None = None
    refresh_every: int | None = None
    checkpoint_every: int | None = None

    # BCT HPs
    bct_data_path: str | None = None
    bct_n_samples: int | None = None
    bct_with_instruct: bool | None = None

    # Eval
    eval_log_dir: str | None = None
    eval_limit: int | None = None
    eval_max_tasks: int | None = None
    eval_bias_types: str | None = None

    # Outcomes — populated later
    n_eval_questions: int | None = None
    pro_bsr_training: float | None = None
    pro_bsr_held_out_avg: float | None = None
    pro_bsr_ratio_training: float | None = None
    pro_bsr_ratio_held_out_avg: float | None = None
    pro_bsr_ratio_training_lo: float | None = None
    pro_bsr_ratio_training_hi: float | None = None
    pro_bsr_ratio_held_out_lo: float | None = None
    pro_bsr_ratio_held_out_hi: float | None = None
    bir_training: float | None = None
    bir_held_out_avg: float | None = None
    bir_ratio_training: float | None = None
    bir_ratio_held_out_avg: float | None = None
    accuracy_unbiased: float | None = None

    # Catch-all for HPs we don't normalize
    extra: dict = field(default_factory=dict)

    # Internal — the name the eval log dir uses (often == run_name)
    run_name: str | None = None


# Bias name normalisation
_BIAS_NORM = {
    "suggested_answer": "sa",
    "distractor_argument": "da",
    "distractor_argument_g4": "da_g4",
    "distractor_fact": "df",
    "wrong_few_shot": "wfs",
    "spurious_few_shot_squares": "ss",
    "post_hoc": "ph",
    "are_you_sure": "ays",
}


def _norm_bias_set(s: str | list[str] | None) -> str | None:
    if s is None:
        return None
    if isinstance(s, str):
        items = [b.strip() for b in s.split(",") if b.strip()]
    else:
        items = list(s)
    norm = sorted({_BIAS_NORM.get(b, b) for b in items})
    return "+".join(norm) if norm else None


def _norm_dataset(s: str | list[str] | None) -> str | None:
    if s is None:
        return None
    if isinstance(s, str):
        items = [b.strip() for b in s.split(",") if b.strip()]
    else:
        items = list(s)
    return "+".join(sorted(set(items))) if items else None


def parse_config(path: Path) -> RunRow | None:
    try:
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return None

    if not isinstance(cfg, dict):
        return None

    name = cfg.get("name")
    model = cfg.get("model")
    if not name:
        return None

    tr = cfg.get("training", {}) or {}
    ev = cfg.get("evaluation", {}) or {}
    an = cfg.get("analysis", {}) or {}
    viz = cfg.get("viz_registration", {}) or {}

    method = tr.get("method")
    targs = tr.get("args", {}) or {}
    eargs = ev.get("args", {}) or {}
    aargs = an.get("args", {}) or {}

    row = RunRow(
        name=name,
        config_path=str(path.relative_to(PROJECT_ROOT)),
        dir_suffix=viz.get("dir_suffix"),
        training_type=viz.get("training_type"),
        model=model,
        method=method,
        prompt_style=targs.get("prompt_style") or _first(eargs.get("prompt_styles")),
    )

    # Identify the training bias(es)
    if method == "rl":
        row.training_bias = _norm_bias_set(targs.get("bias_types"))
        row.training_dataset = _norm_dataset(targs.get("datasets"))
        row.lr = _to_float(targs.get("lr"))
        row.lr_schedule = targs.get("lr_schedule")
        row.lora_rank = _to_int(targs.get("lora_rank"))
        row.kl_coef = _to_float(targs.get("kl_coef"))
        row.anchor_weight = _to_float(targs.get("anchor_weight"))
        row.anchor_model = targs.get("anchor_model")
        row.loss_fn = targs.get("loss_fn")
        row.n_ref_rollouts = _to_int(targs.get("n_ref_rollouts"))
        row.n_train_rollouts = _to_int(targs.get("n_train_rollouts"))
        row.n_consistency_rollouts = _to_int(targs.get("n_consistency_rollouts"))
        row.n_anchor_rollouts = _to_int(targs.get("n_anchor_rollouts"))
        row.temperature = _to_float(targs.get("temperature"))
        row.max_new_tokens = _to_int(targs.get("max_new_tokens"))
        row.n_datapoints = _to_int(targs.get("n_datapoints"))
        row.n_epochs = _to_int(targs.get("n_epochs"))
        row.batch_size = _to_int(targs.get("batch_size"))
        row.gradient_accumulation_steps = _to_int(targs.get("gradient_accumulation_steps"))
        row.refresh_every = _to_int(targs.get("refresh_every"))
        row.checkpoint_every = _to_int(targs.get("checkpoint_every"))
    elif method == "sft":
        # BCT (SFT). Training bias derived from data path or viz_registration
        row.training_bias = _norm_bias_set(viz.get("training_biases"))
        data_str = targs.get("data")
        if data_str:
            data_path = data_str.rsplit(":", 1)[0] if isinstance(data_str, str) else str(data_str)
            row.bct_data_path = data_path
            row.training_dataset = _infer_dataset_from_path(data_path)
            limit_part = data_str.rsplit(":", 1)[1] if isinstance(data_str, str) and ":" in data_str else None
            row.bct_n_samples = _to_int(limit_part)
        row.bct_with_instruct = _to_bool(targs.get("with_instruct"))
        row.lr = _to_float(targs.get("lr"))
        row.lora_rank = _to_int(targs.get("lora_rank"))
        row.batch_size = _to_int(targs.get("batch_size"))
        row.n_epochs = _to_int(targs.get("epochs") or targs.get("n_epochs"))

    # Eval params
    row.eval_dataset = _norm_dataset(eargs.get("datasets"))
    row.eval_limit = _to_int(eargs.get("limit"))
    row.eval_max_tasks = _to_int(eargs.get("max_tasks"))
    row.eval_bias_types = _norm_bias_set(eargs.get("bias_types"))
    row.eval_log_dir = eargs.get("log_dir") or aargs.get("output_dir")

    # Sweep-specific HPs that don't fit the schema get parked in extras
    row.extra = {
        "experiment_name": targs.get("experiment_name"),
        "run_name": targs.get("run_name"),
    }
    # Surface run_name as a top-level for eval-dir lookups (handles cases like
    # the gpt-oss LR sweep where dir_suffix carries '20b' for registry-collision
    # avoidance but the eval log dir uses run_name).
    row.run_name = targs.get("run_name")  # type: ignore[attr-defined]

    return row


def _to_float(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _to_int(v: Any) -> int | None:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _to_bool(v: Any) -> bool | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("true", "yes", "1")
    return bool(v)


def _first(v: Any) -> Any:
    if isinstance(v, list):
        return v[0] if v else None
    return v


def _infer_dataset_from_path(path: str) -> str | None:
    # heuristic: dataset_dumps/train-from-test-<dataset>/...
    parts = Path(path).parts
    for p in parts:
        if p.startswith("train-from-test-"):
            return p.replace("train-from-test-", "")
    return None


# ---------------------------------------------------------------------------
# Eval log location & metric computation
# ---------------------------------------------------------------------------

def get_run_checkpoint(row: RunRow) -> str | None:
    """Read this run's checkpoint URI from its state.json.

    Searches both the canonical artifacts/runs/<name>/metadata/state.json and
    the legacy artifacts/legacy/experiments/<name>/state.json.
    """
    if not row.name:
        return None
    candidates = [
        PROJECT_ROOT / "artifacts" / "runs" / row.name / "metadata" / "state.json",
        PROJECT_ROOT / "artifacts" / "legacy" / "experiments" / row.name / "state.json",
    ]
    for state_path in candidates:
        if not state_path.exists():
            continue
        try:
            state = json.loads(state_path.read_text())
        except Exception:
            continue
        # New schema: tasks["train:main"].outputs.checkpoint
        ck = (
            state.get("tasks", {})
            .get("train:main", {})
            .get("outputs", {})
            .get("checkpoint")
        )
        if ck:
            return ck
        # Legacy schema: stages.training.outputs.checkpoint (or main_checkpoint)
        train_outputs = (
            state.get("stages", {}).get("training", {}).get("outputs", {})
        )
        ck = train_outputs.get("checkpoint") or train_outputs.get("main_checkpoint")
        if ck:
            return ck
    return None


@dataclass
class EvalDirRecord:
    path: Path
    checkpoint_path: str | None  # None for base evals
    base_model: str | None       # canonical model id from .eval metadata
    n_eval_files: int


def scan_eval_dirs(log_dir: Path) -> list[EvalDirRecord]:
    """Discover every model dir under log_dir and read its identifying
    metadata from the first .eval file.
    """
    from inspect_ai.log import read_eval_log

    if not log_dir.exists():
        return []
    out: list[EvalDirRecord] = []
    for sub in sorted(log_dir.iterdir()):
        if not sub.is_dir():
            continue
        evals = sorted(sub.glob("*.eval"))
        if not evals:
            continue
        try:
            log = read_eval_log(str(evals[0]), header_only=True)
        except Exception:
            continue
        meta = log.eval.metadata or {}
        out.append(EvalDirRecord(
            path=sub,
            checkpoint_path=meta.get("checkpoint_path"),
            base_model=meta.get("base_model"),
            n_eval_files=len(evals),
        ))
    return out


_BIR_CACHE: dict[Path, "pd.DataFrame"] = {}


def compute_metrics_for_run(
    run_eval_dir: Path,
    base_eval_dir: Path | None,
    training_bias_norm: str | None,
) -> dict:
    """Compute per-run summary metrics: pro_bsr_{training, held_out_avg},
    bootstrap CI of ratios, accuracy, n.

    Uses _BIR_CACHE keyed by the parent log_dir so each directory is read at
    most once across the whole catalog build (instead of once per run).
    """
    from sycophancy_eval_inspect.visualize_results import compute_per_question_bir  # type: ignore

    parent = run_eval_dir.parent
    df = _BIR_CACHE.get(parent)
    if df is None:
        df = compute_per_question_bir([parent])
        _BIR_CACHE[parent] = df

    # If base lives in a different parent dir (which happens with the global
    # checkpoint scan), also load that parent and concat. Cached too.
    if base_eval_dir is not None and base_eval_dir.parent != parent:
        base_parent_df = _BIR_CACHE.get(base_eval_dir.parent)
        if base_parent_df is None:
            base_parent_df = compute_per_question_bir([base_eval_dir.parent])
            _BIR_CACHE[base_eval_dir.parent] = base_parent_df
        df = pd.concat([df, base_parent_df], ignore_index=True)

    if df.empty:
        return {}

    # df has 'model' = dir_name. Filter to this run.
    run_name = run_eval_dir.name
    run_df = df[df["model"] == run_name]
    if run_df.empty:
        return {}

    base_name = base_eval_dir.name if base_eval_dir else None
    base_df = df[df["model"] == base_name] if base_name else None

    out: dict = {"n_eval_questions": int(len(run_df))}

    # Per-bias means for this run + base
    run_per_bias = run_df.groupby("bias_type")["pro_bsr"].mean()
    bir_per_bias = run_df.groupby("bias_type")["total_bsr"].mean()  # legacy "bir" = total_bsr
    if base_df is not None and not base_df.empty:
        base_per_bias = base_df.groupby("bias_type")["pro_bsr"].mean()
        base_bir_per_bias = base_df.groupby("bias_type")["total_bsr"].mean()
    else:
        base_per_bias = None
        base_bir_per_bias = None

    # Determine training bias(es) — accept multi-bias strings like "da+wfs"
    training_biases_set = set()
    if training_bias_norm:
        # The catalog's normalized bias names need un-normalizing for eval-log bias_type column
        # which uses canonical names like 'distractor_argument_g4', not 'da_g4'.
        rev_norm = {v: k for k, v in _BIAS_NORM.items()}
        for b in training_bias_norm.split("+"):
            training_biases_set.add(rev_norm.get(b, b))

    biases = list(run_per_bias.index)
    train_biases = [b for b in biases if b in training_biases_set]
    held_out_biases = [b for b in biases if b not in training_biases_set]

    if train_biases:
        out["pro_bsr_training"] = float(run_per_bias[train_biases].mean())
        out["bir_training"] = float(bir_per_bias[train_biases].mean())
        if base_per_bias is not None:
            common = [b for b in train_biases if b in base_per_bias.index]
            if common:
                run_avg = run_per_bias[common].mean()
                base_avg = base_per_bias[common].mean()
                out["pro_bsr_ratio_training"] = float(run_avg / base_avg) if base_avg > 0 else float("nan")
                # Bootstrap CI
                lo, hi = _bootstrap_ratio_ci(
                    run_df[run_df["bias_type"].isin(common)]["pro_bsr"].to_numpy(),
                    base_df[base_df["bias_type"].isin(common)]["pro_bsr"].to_numpy(),
                )
                out["pro_bsr_ratio_training_lo"] = lo
                out["pro_bsr_ratio_training_hi"] = hi
            if "bir_training" in out and base_bir_per_bias is not None:
                common = [b for b in train_biases if b in base_bir_per_bias.index]
                if common:
                    run_avg = bir_per_bias[common].mean()
                    base_avg = base_bir_per_bias[common].mean()
                    out["bir_ratio_training"] = float(run_avg / base_avg) if base_avg > 0 else float("nan")

    if held_out_biases:
        out["pro_bsr_held_out_avg"] = float(run_per_bias[held_out_biases].mean())
        out["bir_held_out_avg"] = float(bir_per_bias[held_out_biases].mean())
        if base_per_bias is not None:
            common = [b for b in held_out_biases if b in base_per_bias.index]
            if common:
                run_avg = run_per_bias[common].mean()
                base_avg = base_per_bias[common].mean()
                out["pro_bsr_ratio_held_out_avg"] = float(run_avg / base_avg) if base_avg > 0 else float("nan")
                lo, hi = _bootstrap_ratio_ci(
                    run_df[run_df["bias_type"].isin(common)]["pro_bsr"].to_numpy(),
                    base_df[base_df["bias_type"].isin(common)]["pro_bsr"].to_numpy(),
                )
                out["pro_bsr_ratio_held_out_lo"] = lo
                out["pro_bsr_ratio_held_out_hi"] = hi
            if base_bir_per_bias is not None:
                common = [b for b in held_out_biases if b in base_bir_per_bias.index]
                if common:
                    run_avg = bir_per_bias[common].mean()
                    base_avg = base_bir_per_bias[common].mean()
                    out["bir_ratio_held_out_avg"] = float(run_avg / base_avg) if base_avg > 0 else float("nan")

    # Accuracy (unbiased) — read directly from eval logs
    out["accuracy_unbiased"] = _compute_unbiased_accuracy(run_eval_dir)

    return out


def _bootstrap_ratio_ci(
    run_pro: np.ndarray,
    base_pro: np.ndarray,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    if len(run_pro) == 0 or len(base_pro) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    ratios = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        r_idx = rng.integers(0, len(run_pro), size=len(run_pro))
        b_idx = rng.integers(0, len(base_pro), size=len(base_pro))
        run_mean = run_pro[r_idx].mean()
        base_mean = base_pro[b_idx].mean()
        ratios[i] = run_mean / base_mean if base_mean > 0 else np.nan
    lo = float(np.nanpercentile(ratios, 100 * alpha / 2))
    hi = float(np.nanpercentile(ratios, 100 * (1 - alpha / 2)))
    return lo, hi


def _compute_unbiased_accuracy(run_eval_dir: Path) -> float | None:
    from inspect_ai.log import read_eval_log
    correct = 0
    total = 0
    for f in sorted(run_eval_dir.glob("*.eval")):
        try:
            log = read_eval_log(str(f))
        except Exception:
            continue
        for s in log.samples or []:
            if (s.metadata or {}).get("variant") != "unbiased":
                continue
            sc = (s.scores or {}).get("mcq_bias_scorer")
            if not sc or not isinstance(sc.value, dict):
                continue
            c = sc.value.get("correct")
            if c is None:
                continue
            total += 1
            if c == 1.0:
                correct += 1
    return (correct / total) if total > 0 else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    paths = discover_configs()
    print(f"Discovered {len(paths)} configs")

    rows: list[RunRow] = []
    for p in paths:
        row = parse_config(p)
        if row is None:
            continue
        rows.append(row)

    print(f"Parsed {len(rows)} runs")

    # Resolve each row's checkpoint URI from its state.json (data-driven, not
    # name-based).
    for row in rows:
        row.extra["checkpoint"] = get_run_checkpoint(row)

    # Globally scan every known log root for .eval files. Build maps from
    # checkpoint URI -> EvalDirRecord (for trained runs) and from base_model
    # -> [EvalDirRecord] (for base evals). This is fully checkpoint-driven:
    # the YAML's log_dir is ignored because legacy migration changed paths.
    log_roots_globs = [
        "artifacts/runs/*/eval_logs",
        "artifacts/eval_suites/*/eval_logs",
        "artifacts/legacy/logs/*",
        "artifacts/legacy/sycophancy_eval_inspect/logs/*",
        "artifacts/legacy/sycophancy_eval_inspect/logs/*/*",
        # Live (non-legacy) eval dirs at the project root
        "sycophancy_eval_inspect/logs/*",
        "sycophancy_eval_inspect/logs/*/*",
    ]
    log_dirs_to_scan: set[Path] = set()
    for glob in log_roots_globs:
        for p in PROJECT_ROOT.glob(glob):
            if p.is_dir() and "_archive" not in p.parts:
                log_dirs_to_scan.add(p)

    print(f"Scanning {len(log_dirs_to_scan)} log roots for .eval files...")

    by_checkpoint: dict[str, EvalDirRecord] = {}
    base_by_model: dict[str, list[EvalDirRecord]] = {}
    for log_dir in log_dirs_to_scan:
        for rec in scan_eval_dirs(log_dir):
            if rec.checkpoint_path:
                if rec.checkpoint_path not in by_checkpoint:
                    by_checkpoint[rec.checkpoint_path] = rec
            elif rec.base_model:
                base_by_model.setdefault(rec.base_model, []).append(rec)

    print(f"Indexed {len(by_checkpoint)} trained eval dirs, "
          f"{sum(len(v) for v in base_by_model.values())} base eval dirs")

    n_with_metrics = 0
    for r in rows:
        ck = r.extra.get("checkpoint")
        run_record = by_checkpoint.get(ck) if ck else None
        if run_record is None:
            continue
        # Pick the base eval dir for this run's base model. If multiple bases,
        # prefer one in the same parent directory as the run's eval dir.
        base_candidates = base_by_model.get(r.model or "", [])
        base_record = None
        if base_candidates:
            same_parent = [b for b in base_candidates if b.path.parent == run_record.path.parent]
            base_record = same_parent[0] if same_parent else base_candidates[0]
        try:
            metrics = compute_metrics_for_run(
                run_record.path,
                base_record.path if base_record else None,
                r.training_bias,
            )
        except Exception as e:
            print(f"  WARN metrics failed for {r.name}: {e}")
            continue
        for k, v in metrics.items():
            setattr(r, k, v)
        if r.pro_bsr_ratio_training is not None or r.pro_bsr_ratio_held_out_avg is not None:
            n_with_metrics += 1

    print(f"Computed metrics for {n_with_metrics}/{len(rows)} runs")

    # Convert to DataFrame, drop the dataclass `extra` dict to a JSON column
    records = []
    for r in rows:
        d = r.__dict__.copy()
        d["extra"] = json.dumps(d.get("extra") or {})
        records.append(d)

    df = pd.DataFrame.from_records(records)

    out_parquet = PROJECT_ROOT / "artifacts" / "run_catalog.parquet"
    out_csv = PROJECT_ROOT / "artifacts" / "run_catalog.csv"
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_parquet} ({len(df)} rows)")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
