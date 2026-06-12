"""Batch re-score eval logs with all scorers."""

import asyncio
import os
import sys
import time
import zipfile
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from inspect_ai import score
from inspect_ai.log import read_eval_log, write_eval_log

from sycophancy_eval_inspect.mcq.scorer import (
    mcq_bias_scorer,
    options_considered_scorer,
    bias_acknowledged_scorer,
    few_shot_confusion_scorer,
)

ALL_SCORERS = [
    mcq_bias_scorer(),
    options_considered_scorer(),
    bias_acknowledged_scorer(),
    few_shot_confusion_scorer(),
]

KEEP_BIASES = {
    "suggested_answer",
    "distractor_argument",
    "distractor_fact",
    "wrong_few_shot",
    "spurious_few_shot_squares",
}


def get_log_info(path: str) -> dict | None:
    """Extract variant/bias/dataset from eval log metadata."""
    try:
        with zipfile.ZipFile(path) as z:
            with z.open("_journal/start.json") as f:
                meta = json.load(f)
                ta = meta["eval"]["task_args"]
                dp = ta.get("dataset_path", "")
                variant = ta.get("variant", "")
                parts = dp.split("/")
                bias = parts[2] if len(parts) > 2 else "?"
                dataset = parts[3].split("_")[0] if len(parts) > 3 else "?"
                return {
                    "path": path,
                    "variant": variant,
                    "bias": bias,
                    "dataset": dataset,
                }
    except Exception as e:
        print(f"  Error reading {path}: {e}", file=sys.stderr)
        return None


def rescore_log(path: str) -> str:
    """Re-score a single eval log with all scorers. Returns status message."""
    try:
        log = read_eval_log(path)
        result = score(log, ALL_SCORERS, action="overwrite")
        write_eval_log(result, path)

        # Summarize metrics
        lines = []
        for s in result.results.scores:
            for mname, m in s.metrics.items():
                lines.append(f"    {s.scorer}/{s.name}/{mname}: {m.value:.4f}")
        return f"OK {path}\n" + "\n".join(lines)
    except Exception as e:
        return f"FAIL {path}: {e}"


def collect_logs(base_dir: str, filter_dataset: str | None = None) -> list[str]:
    """Collect eval log paths matching the 5 biases + unbiased."""
    targets = []
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if not f.endswith(".eval"):
                continue
            path = os.path.join(root, f)
            info = get_log_info(path)
            if info is None:
                continue
            if info["variant"] == "unbiased" or info["bias"] in KEEP_BIASES:
                if filter_dataset and info["dataset"] != filter_dataset:
                    continue
                targets.append(path)
    return targets


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", help="Base directory with eval logs")
    parser.add_argument("--dataset", default=None, help="Filter to specific dataset")
    parser.add_argument("--model", default=None, help="Filter to specific model subdir")
    parser.add_argument("--workers", type=int, default=6, help="Parallel workers")
    parser.add_argument("--dry-run", action="store_true", help="List logs without scoring")
    args = parser.parse_args()

    base = args.base_dir
    if args.model:
        base = os.path.join(base, args.model)

    targets = collect_logs(base, args.dataset)
    print(f"Found {len(targets)} logs to re-score")

    if args.dry_run:
        for t in targets:
            info = get_log_info(t)
            print(f"  {info['variant']:10s} {info['bias']:30s} {info['dataset']:15s} {t}")
        sys.exit(0)

    start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(rescore_log, t): t for t in targets}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            print(f"[{i}/{len(targets)}] {result}")
            print()

    elapsed = time.time() - start
    print(f"\nDone. {len(targets)} logs in {elapsed:.1f}s")
