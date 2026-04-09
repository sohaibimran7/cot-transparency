#!/usr/bin/env python3
"""
Run sycophancy evaluation suite with logs structured like dataset_dumps/test/:
    logs/<experiment_name>/<bias>/<dataset>_<bias>_<variant>/

Usage:
    python -m sycophancy_eval_inspect.run_suite --experiment my_exp --model openai/gpt-4o
    python -m sycophancy_eval_inspect.run_suite --experiment reasoning_test --model openai/o1 --prompt-style no_cot
    python -m sycophancy_eval_inspect.run_suite --experiment my_exp --model openai/gpt-4o --bias-types suggested_answer,are_you_sure

Hash-based filtering is enabled by default to ensure consistent samples across bias types
for proper BRR (Biased Reasoning Rate) calculation. Use --skip-hash-filter to disable.
"""

import argparse
import subprocess
from pathlib import Path
from typing import Literal

from sycophancy_eval_inspect.eval_common import (
    add_hash_filter_args,
    compute_hash_filter,
    discover_mcq_datasets,
    discover_positional_datasets,
    get_unique_original_datasets,
)


def get_log_dir(
    experiment_name: str,
    dataset_path: Path,
    variant: str | None = None,
) -> str:
    """
    Compute log directory to match dataset_dumps structure.

    dataset_dumps/test/suggested_answer/mmlu_suggested_answer.jsonl
    -> logs/<experiment>/suggested_answer/mmlu_suggested_answer_biased/
    """
    bias_name = dataset_path.parent.name  # e.g., "suggested_answer"
    file_stem = dataset_path.stem  # e.g., "mmlu_suggested_answer"

    if variant:
        log_subdir = f"{file_stem}_{variant}"
    else:
        log_subdir = file_stem

    return f"logs/{experiment_name}/{bias_name}/{log_subdir}"


def run_mcq_eval(
    dataset_path: Path,
    variant: Literal["biased", "unbiased"],
    model: str,
    experiment_name: str,
    prompt_style: str = "cot",
    limit: int | None = None,
    hash_filter_file: Path | None = None,
) -> int:
    """Run a single MCQ evaluation. Returns subprocess return code."""
    log_dir = get_log_dir(experiment_name, dataset_path, variant)

    cmd = [
        "inspect",
        "eval",
        "sycophancy_eval_inspect/mcq/task.py@mcq_bias_eval",
        "-T",
        f"dataset_path={dataset_path}",
        "-T",
        f"variant={variant}",
        "-T",
        f"prompt_style={prompt_style}",
        "--model",
        model,
        "--log-dir",
        log_dir,
    ]
    if limit:
        cmd.extend(["-T", f"limit={limit}"])
    if hash_filter_file:
        cmd.extend(["-T", f"hash_filter_file={hash_filter_file}"])

    print(f"\n{'='*60}")
    print(f"MCQ Eval: {dataset_path.parent.name}/{dataset_path.stem} [{variant}]")
    print(f"Log dir: {log_dir}")
    if hash_filter_file:
        print(f"Hash filter: {hash_filter_file}")
    print(f"{'='*60}")

    result = subprocess.run(cmd)
    return result.returncode


def run_positional_eval(
    dataset_path: Path,
    model: str,
    experiment_name: str,
    limit: int | None = None,
) -> int:
    """Run positional bias evaluation. Returns subprocess return code."""
    log_dir = get_log_dir(experiment_name, dataset_path)

    cmd = [
        "inspect",
        "eval",
        "sycophancy_eval_inspect/positional/task.py@positional_bias_eval",
        "-T",
        f"dataset_path={dataset_path}",
        "--model",
        model,
        "--log-dir",
        log_dir,
    ]
    if limit:
        cmd.extend(["-T", f"limit={limit}"])

    print(f"\n{'='*60}")
    print(f"Positional Eval: {dataset_path.name}")
    print(f"Log dir: {log_dir}")
    print(f"{'='*60}")

    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run sycophancy evaluation suite")
    parser.add_argument("--experiment", required=True, help="Experiment name for log directory")
    parser.add_argument("--model", required=True, help="Model to evaluate (e.g., openai/gpt-4o)")
    parser.add_argument("--prompt-style", default="cot", choices=["cot", "no_cot"], help="CoT or no-CoT prompting")
    parser.add_argument("--variants", default="biased,unbiased", help="Comma-separated: biased,unbiased")
    parser.add_argument("--bias-types", default=None, help="Filter to specific bias types (comma-separated)")
    parser.add_argument("--dataset-dir", default="dataset_dumps/test", help="Base directory for datasets")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples per eval")
    parser.add_argument("--skip-mcq", action="store_true", help="Skip MCQ evals")
    parser.add_argument("--skip-positional", action="store_true", help="Skip positional bias eval")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    add_hash_filter_args(parser)
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",")]
    total_evals = 0
    failed_evals = 0

    # MCQ biases
    if not args.skip_mcq:
        mcq_datasets = discover_mcq_datasets(args.dataset_dir)

        if args.bias_types:
            allowed = set(args.bias_types.split(","))
            mcq_datasets = [d for d in mcq_datasets if d.parent.name in allowed]

        print(f"Found {len(mcq_datasets)} MCQ dataset files")
        print(f"Variants: {variants}")

        # Compute common hashes for consistent sampling
        hash_filter_file = None
        if not args.skip_hash_filter and mcq_datasets:
            print("\nComputing common hashes across bias types...")
            try:
                hash_filter_dir = Path(f"logs/{args.experiment}")
                hash_filter = compute_hash_filter(
                    datasets=mcq_datasets,
                    limit=args.limit,
                    save_to=hash_filter_dir / "common_hashes.json",
                    print_report=True,
                )
                hash_filter_file = hash_filter.hash_filter_file
                for original, hashes in hash_filter.common_hashes.items():
                    print(f"  {original}: {len(hashes)} common samples")
            except ValueError as e:
                print(f"\nERROR: {e}")
                print("Use --skip-hash-filter to disable hash filtering (not recommended for BRR)")
                return 1

        # For unbiased, only need one file per original dataset
        if "unbiased" in variants:
            unbiased_datasets = get_unique_original_datasets(mcq_datasets)
        else:
            unbiased_datasets = []

        # Count evals
        biased_count = len(mcq_datasets) if "biased" in variants else 0
        unbiased_count = len(unbiased_datasets) if "unbiased" in variants else 0
        print(f"MCQ evals to run: {biased_count} biased + {unbiased_count} unbiased = {biased_count + unbiased_count}")

        if not args.dry_run:
            # Run biased evals
            if "biased" in variants:
                for dataset_path in mcq_datasets:
                    total_evals += 1
                    ret = run_mcq_eval(
                        dataset_path=dataset_path,
                        variant="biased",
                        model=args.model,
                        experiment_name=args.experiment,
                        prompt_style=args.prompt_style,
                        limit=args.limit,
                        hash_filter_file=hash_filter_file,
                    )
                    if ret != 0:
                        failed_evals += 1

            # Run unbiased evals (one per original dataset)
            if "unbiased" in variants:
                for dataset_path in unbiased_datasets:
                    total_evals += 1
                    ret = run_mcq_eval(
                        dataset_path=dataset_path,
                        variant="unbiased",
                        model=args.model,
                        experiment_name=args.experiment,
                        prompt_style=args.prompt_style,
                        limit=args.limit,
                        hash_filter_file=hash_filter_file,
                    )
                    if ret != 0:
                        failed_evals += 1

    # Positional bias
    if not args.skip_positional:
        positional_datasets = discover_positional_datasets(args.dataset_dir)

        if args.bias_types and "positional_bias" not in args.bias_types.split(","):
            positional_datasets = []

        print(f"\nFound {len(positional_datasets)} positional bias files")

        if not args.dry_run:
            for dataset_path in positional_datasets:
                total_evals += 1
                ret = run_positional_eval(
                    dataset_path=dataset_path,
                    model=args.model,
                    experiment_name=args.experiment,
                    limit=args.limit,
                )
                if ret != 0:
                    failed_evals += 1

    print(f"\n{'='*60}")
    print(f"Suite complete!")
    print(f"Total evals: {total_evals}, Failed: {failed_evals}")
    print(f"Logs at: logs/{args.experiment}/")
    print(f"{'='*60}")

    return 1 if failed_evals > 0 else 0


if __name__ == "__main__":
    exit(main())
