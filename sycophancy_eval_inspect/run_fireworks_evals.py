#!/usr/bin/env python3
"""
Run sycophancy evaluations on Fireworks AI models.

Usage:
    python -m sycophancy_eval_inspect.run_fireworks_evals --limit 2
    python -m sycophancy_eval_inspect.run_fireworks_evals --models llama
    python -m sycophancy_eval_inspect.run_fireworks_evals --models gpt --biased-only

Hash-based filtering is enabled by default to ensure consistent samples across bias types
for proper BRR (Biased Reasoning Rate) calculation. Use --skip-hash-filter to disable.
"""

import argparse
import asyncio
import logging
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

from inspect_ai import eval_async
from inspect_ai.model import GenerateConfig, get_model

from sycophancy_eval_inspect.mcq.task import mcq_bias_eval
from sycophancy_eval_inspect.eval_common import (
    add_common_eval_args,
    get_original_dataset_name,
    prepare_evaluation,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    name: str
    model_id: str
    prompt_styles: list[str]


MODEL_CONFIGS = [
    # Llama models
    ModelConfig("llama-fireworks-base", "accounts/sohaib/deployments/dh6uv0yu", ["cot", "no_cot"]),
    ModelConfig("llama-fireworks-bct", "accounts/sohaib/models/llama-bct-train#accounts/sohaib/deployments/dh6uv0yu", ["cot", "no_cot"]),
    ModelConfig("llama-fireworks-control", "accounts/sohaib/models/llama-bct-control#accounts/sohaib/deployments/dh6uv0yu", ["cot", "no_cot"]),
    # GPT models (trained on all datasets)
    ModelConfig("gpt-fireworks-base", "accounts/sohaib/deployments/dftlbcrz", ["no_cot"]),
    ModelConfig("gpt-fireworks-bct", "accounts/sohaib/models/gpt-bct-train-cleaned#accounts/sohaib/deployments/hahpsd3w", ["no_cot"]),
    ModelConfig("gpt-fireworks-control", "accounts/sohaib/models/gpt-bct-control-cleaned#accounts/sohaib/deployments/fggq4iw1", ["no_cot"]),
    # GPT models (trained on logiqa only)
    ModelConfig("gpt-logiqa-bct", "accounts/sohaib/models/bct-logiqa-train#accounts/sohaib/deployments/dgwgw67g", ["no_cot"]),
    ModelConfig("gpt-logiqa-control", "accounts/sohaib/models/bct-logiqa-control#accounts/sohaib/deployments/uf9yup8s", ["no_cot"]),
]


async def run(args):
    # Filter model configs
    configs = MODEL_CONFIGS
    if args.models:
        filter_terms = args.models.split(",")
        configs = [c for c in configs if any(m in c.name for m in filter_terms)]

    if not configs:
        logger.error("No models matched the filter")
        return

    log_dir = args.log_dir

    # Filter datasets if specified
    dataset_filter = None
    if args.datasets:
        dataset_filter = set(args.datasets.split(","))

    # Prepare evaluation with hash filtering
    try:
        dataset_configs, hash_filter = prepare_evaluation(
            args,
            log_base_dir=log_dir,
        )
    except ValueError as e:
        logger.error(f"Failed to prepare evaluation: {e}")
        return

    # Apply dataset filter (hellaswag, logiqa, etc.)
    if dataset_filter:
        dataset_configs = [
            dc for dc in dataset_configs
            if get_original_dataset_name(dc.path) in dataset_filter
        ]
        logger.info(f"Filtered to {len(dataset_configs)} dataset configs for: {dataset_filter}")

    # Filter prompt styles if specified
    prompt_style_filter = None
    if args.prompt_styles:
        prompt_style_filter = set(args.prompt_styles.split(","))

    # Run evaluations for each model
    for config in configs:
        model = get_model(
            f"fireworks/{config.model_id}",
            config=GenerateConfig(temperature=1.0, max_tokens=args.max_tokens),
        )
        tasks = []

        # Filter prompt styles for this model
        prompt_styles = config.prompt_styles
        if prompt_style_filter:
            prompt_styles = [ps for ps in prompt_styles if ps in prompt_style_filter]
        if not prompt_styles:
            logger.info(f"Skipping {config.name} - no matching prompt styles")
            continue

        for prompt_style in prompt_styles:
            for dc in dataset_configs:
                task = mcq_bias_eval(
                    dataset_path=str(dc.path),
                    variant=dc.variant,
                    prompt_style=prompt_style,
                    limit=args.limit,
                    allowed_hashes=dc.allowed_hashes,
                )
                tasks.append(task)

        print(f"\n{'='*60}")
        print(f"{config.name}: {len(tasks)} tasks")
        print(f"{'='*60}")

        if not args.dry_run:
            eval_kwargs = dict(
                model=model,
                log_dir=f"{log_dir}/{config.name}",
                max_tasks=args.max_tasks,
                fail_on_error=False,
            )
            if args.max_connections is not None:
                eval_kwargs["max_connections"] = args.max_connections
            await eval_async(tasks, **eval_kwargs)


def main():
    parser = argparse.ArgumentParser(description="Run sycophancy evals on Fireworks models")
    parser.add_argument("--models", help="Filter models (comma-separated substrings)")
    parser.add_argument("--datasets", help="Filter to specific original datasets (comma-separated, e.g., hellaswag,logiqa)")
    parser.add_argument("--prompt-styles", help="Filter prompt styles (comma-separated: cot,no_cot)")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Max tokens for generation")
    parser.add_argument("--max-tasks", type=int, default=10, help="Max parallel tasks")
    parser.add_argument("--max-connections", type=int, default=None, help="Max concurrent model API connections (controls per-model concurrency)")
    parser.add_argument("--dry-run", action="store_true", help="Print tasks without running")
    parser.add_argument("--log-dir", default="logs/fireworks_evals", help="Base directory for logs")
    add_common_eval_args(parser)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
