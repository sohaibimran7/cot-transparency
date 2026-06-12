#!/usr/bin/env python3
"""
Push the run catalog to a unified W&B project for combinatorial visualisation.

One W&B run per catalog row. Each run carries:
- config: all HP fields (lr, kl_coef, n_datapoints, batch_size, etc.)
- summary: outcome metrics (pro_bsr_ratio_training, ..._held_out_avg, CIs,
  accuracy_unbiased)
- tags: model_short, method, training_bias, has_metrics

Once pushed, W&B's parallel-coords / scatter / table panels give you the
combinatorial dashboard for free.

Usage:
    python scripts/push_catalog_to_wandb.py
    python scripts/push_catalog_to_wandb.py --project my-project
    python scripts/push_catalog_to_wandb.py --dry-run
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "artifacts" / "run_catalog.parquet"

# Columns that hold outcome metrics (go to wandb.summary, not config)
METRIC_COLS = [
    "n_eval_questions",
    "pro_bsr_training",
    "pro_bsr_held_out_avg",
    "pro_bsr_ratio_training",
    "pro_bsr_ratio_held_out_avg",
    "pro_bsr_ratio_training_lo",
    "pro_bsr_ratio_training_hi",
    "pro_bsr_ratio_held_out_lo",
    "pro_bsr_ratio_held_out_hi",
    "bir_training",
    "bir_held_out_avg",
    "bir_ratio_training",
    "bir_ratio_held_out_avg",
    "accuracy_unbiased",
]

# Identity / metadata columns — go to wandb.summary (or skipped), NOT
# wandb.config. Path strings and identifiers must not appear in config or
# they pollute Sweep parameter auto-detection (W&B treats config keys as
# tunable HPs and rejects "auto" as a categorical value).
METADATA_COLS = {
    "name",
    "config_path",
    "extra",
    "run_name",
    "dir_suffix",
    "training_type",
    "bct_data_path",       # path string, not an HP
    "eval_log_dir",        # path string
    "eval_bias_types",     # eval-set spec, not a tunable HP
    "eval_limit",
    "eval_max_tasks",
    "is_control",
    "anchor_model",        # 'base' | path — categorical, but rarely tuned
}

# Column → tag mapping (for low-cardinality categoricals)
TAG_COLS = ["method", "training_bias", "training_dataset", "eval_dataset", "prompt_style"]


def model_short(model: str | None) -> str | None:
    if not model:
        return None
    if "Llama" in model:
        return "llama-3.1-8b"
    if "gpt-oss-120b" in model:
        return "gpt-oss-120b"
    if "gpt-oss-20b" in model:
        return "gpt-oss-20b"
    if "Qwen3" in model:
        return "qwen3-30b"
    return model.split("/")[-1].lower()


def is_clean(v) -> bool:
    """Whether a value is suitable for wandb config/summary (not NaN/None/inf)."""
    if v is None:
        return False
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="consistency-training-catalog")
    ap.add_argument("--entity", default=None, help="Override W&B entity (default: user default)")
    ap.add_argument("--catalog", default=str(CATALOG_PATH))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="Push only first N rows")
    ap.add_argument("--only-with-metrics", action="store_true",
                    help="Skip rows with no metrics computed")
    args = ap.parse_args()

    df = pd.read_parquet(args.catalog)
    print(f"Loaded {len(df)} rows from {args.catalog}")

    if args.only_with_metrics:
        df = df[df["pro_bsr_ratio_training"].notna() | df["pro_bsr_ratio_held_out_avg"].notna()]
        print(f"After filter: {len(df)} rows")

    if args.limit:
        df = df.head(args.limit)

    if args.dry_run:
        print(f"[DRY RUN] would push {len(df)} runs to project '{args.project}'")
        for _, row in df.iterrows():
            print(f"  - {row['name']}  ({model_short(row['model'])}/{row['method']}/{row['training_bias']})")
        return

    import wandb

    pushed = 0
    skipped = 0
    for _, row in df.iterrows():
        run_name = row["name"]
        if not run_name:
            skipped += 1
            continue

        # Build config from tunable HPs only (no paths / identifiers)
        config: dict = {}
        metadata_summary: dict = {}
        for col, val in row.items():
            if col in METRIC_COLS:
                continue
            if not is_clean(val):
                continue
            v = val.item() if hasattr(val, "item") else val
            if col in METADATA_COLS:
                metadata_summary[col] = v
            else:
                config[col] = v

        # Pull a few key fields from `extra` JSON, surface as metadata
        try:
            import json
            extra = json.loads(row.get("extra") or "{}")
            if extra.get("checkpoint"):
                metadata_summary["checkpoint"] = extra["checkpoint"]
        except Exception:
            pass

        config["model_short"] = model_short(row["model"])
        has_metrics = is_clean(row.get("pro_bsr_ratio_training")) or is_clean(row.get("pro_bsr_ratio_held_out_avg"))

        tags = [config["model_short"]] if config.get("model_short") else []
        for c in TAG_COLS:
            v = row.get(c)
            if is_clean(v):
                tags.append(f"{c}={v}")
        tags.append("has_metrics" if has_metrics else "no_metrics")

        wandb_run = wandb.init(
            project=args.project,
            entity=args.entity,
            name=run_name,
            id=f"catalog-{run_name}".replace("/", "-")[-128:],
            config=config,
            tags=tags,
            reinit=True,
            settings=wandb.Settings(silent=True),
            resume="allow",
        )

        # Push outcome metrics to summary
        for col in METRIC_COLS:
            v = row.get(col)
            if is_clean(v):
                wandb_run.summary[col] = v.item() if hasattr(v, "item") else v
        # Push metadata / identifiers to summary (so they're filterable but
        # not treated as tunable hyperparameters)
        for k, v in metadata_summary.items():
            wandb_run.summary[k] = v
        wandb_run.summary["has_metrics"] = has_metrics

        wandb.finish()
        pushed += 1
        if pushed % 10 == 0:
            print(f"  pushed {pushed}/{len(df)}")

    print(f"Done: pushed {pushed} runs, skipped {skipped}")
    print(f"View at: https://wandb.ai/<entity>/{args.project}")


if __name__ == "__main__":
    main()
