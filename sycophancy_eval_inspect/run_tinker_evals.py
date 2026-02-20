#!/usr/bin/env python3
"""
Run sycophancy evaluations on Tinker model checkpoints.

Usage:
    # Run on a specific checkpoint with specific datasets and bias types
    python -m sycophancy_eval_inspect.run_tinker_evals \
        --checkpoint tinker://... \
        --base-model meta-llama/Llama-3.1-8B-Instruct \
        --bias-types distractor_argument,distractor_fact \
        --datasets hellaswag,logiqa \
        --limit 50

    # Run base model only (no checkpoint)
    python -m sycophancy_eval_inspect.run_tinker_evals \
        --base-model meta-llama/Llama-3.1-8B-Instruct

    # Dry run to see what would be evaluated
    python -m sycophancy_eval_inspect.run_tinker_evals --dry-run
"""

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

import tinker
from inspect_ai import eval_async
from inspect_ai.model import GenerateConfig, Model
from tinker_cookbook.eval.inspect_utils import InspectAPIFromTinkerSampling
from tinker_cookbook.model_info import get_recommended_renderer_name

from sycophancy_eval_inspect.mcq.task import mcq_bias_eval
from sycophancy_eval_inspect.eval_common import (
    add_common_eval_args,
    prepare_evaluation,
    get_original_dataset_name,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    name: str
    base_model: str
    checkpoint_path: str | None
    prompt_styles: list[str]

    @property
    def renderer(self) -> str:
        """Get recommended renderer for the base model."""
        return get_recommended_renderer_name(self.base_model)


# Default model configs (can be overridden via CLI)
LLAMA_BASE = "meta-llama/Llama-3.1-8B-Instruct"

DEFAULT_CONFIGS = [
    ModelConfig(
        name="llama-3.1-8b-rlct",
        base_model=LLAMA_BASE,
        checkpoint_path="tinker://0be61a5c-2978-5a83-b26b-5fea4f4d0bab:train:0/sampler_weights/bct_suggested_answer_llama8b_v3",
        prompt_styles=["cot", "no_cot"],
    ),
]


def create_tinker_model(
    config: ModelConfig,
    service_client: tinker.ServiceClient,
    temperature: float = 1.0,
    max_tokens: int = 8192,
) -> Model:
    """Create an Inspect Model backed by Tinker sampling."""
    sampling_client = service_client.create_sampling_client(
        model_path=config.checkpoint_path,
        base_model=config.base_model,
    )

    api = InspectAPIFromTinkerSampling(
        renderer_name=config.renderer,
        model_name=config.base_model,
        sampling_client=sampling_client,
        verbose=False,
    )

    return Model(
        api=api,
        config=GenerateConfig(
            temperature=temperature,
            max_tokens=max_tokens,
        ),
    )


async def run(args):
    # Build model config from CLI args or use defaults
    if args.checkpoint or args.base_model:
        # CLI-specified model
        configs = [ModelConfig(
            name=args.name or "custom-model",
            base_model=args.base_model or LLAMA_BASE,
            checkpoint_path=args.checkpoint,
            prompt_styles=args.prompt_styles.split(",") if args.prompt_styles else ["cot", "no_cot"],
        )]
    else:
        configs = DEFAULT_CONFIGS

    # Filter configs if --models specified
    if args.models:
        filter_terms = args.models.split(",")
        configs = [c for c in configs if any(m in c.name for m in filter_terms)]

    if not configs:
        logger.error("No models to evaluate")
        return {"success": 0, "failed": 0, "skipped": 0}

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
        return {"success": 0, "failed": 0, "skipped": 0}

    # Apply dataset filter (hellaswag, logiqa, etc.)
    if dataset_filter:
        dataset_configs = [
            dc for dc in dataset_configs
            if get_original_dataset_name(dc.path) in dataset_filter
        ]
        logger.info(f"Filtered to {len(dataset_configs)} dataset configs for: {dataset_filter}")

    if not dataset_configs:
        logger.error("No datasets matched filters")
        return {"success": 0, "failed": 0, "skipped": 0}

    # Create Tinker service client
    service_client = tinker.ServiceClient()

    results = {"success": 0, "failed": 0, "skipped": 0}

    for config in configs:
        logger.info(f"Creating model: {config.name}")
        logger.info(f"  Base model: {config.base_model}")
        logger.info(f"  Checkpoint: {config.checkpoint_path}")
        logger.info(f"  Renderer: {config.renderer}")

        if not args.dry_run:
            model = create_tinker_model(config, service_client, max_tokens=args.max_tokens)

        tasks = []

        for prompt_style in config.prompt_styles:
            for dc in dataset_configs:
                task = mcq_bias_eval(
                    dataset_path=str(dc.path),
                    variant=dc.variant,
                    prompt_style=prompt_style,
                    limit=args.limit,
                    allowed_hashes=dc.allowed_hashes,
                    metadata={
                        "checkpoint_path": config.checkpoint_path,
                        "base_model": config.base_model,
                        "model_name": config.name,
                    },
                )
                tasks.append(task)

        print(f"\n{'='*60}")
        print(f"{config.name}: {len(tasks)} tasks")
        print(f"{'='*60}")

        if args.dry_run:
            for dc in dataset_configs[:5]:
                hash_info = f", {len(dc.allowed_hashes)} hashes" if dc.allowed_hashes else ""
                logger.info(f"  {dc.path.name} ({dc.variant}{hash_info})")
            if len(dataset_configs) > 5:
                logger.info(f"  ... and {len(dataset_configs) - 5} more")
            results["skipped"] += len(tasks)
        else:
            try:
                eval_kwargs = dict(
                    model=model,
                    log_dir=f"{log_dir}/{config.name}",
                    max_tasks=args.max_tasks,
                    fail_on_error=False,
                )
                if args.max_connections is not None:
                    eval_kwargs["max_connections"] = args.max_connections
                await eval_async(tasks, **eval_kwargs)
                results["success"] += len(tasks)
            except Exception as e:
                logger.error(f"Batch eval failed for {config.name}: {e}")
                results["failed"] += len(tasks)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run sycophancy evals on Tinker checkpoints")

    # Model specification (CLI overrides defaults)
    parser.add_argument("--checkpoint", help="Tinker checkpoint path (tinker://...)")
    parser.add_argument("--base-model", help="Base model name (e.g., meta-llama/Llama-3.1-8B-Instruct)")
    parser.add_argument("--name", help="Model name for logging")
    parser.add_argument("--prompt-styles", help="Prompt styles (comma-separated: cot,no_cot)")

    # Filtering
    parser.add_argument("--models", help="Filter model configs by name (comma-separated)")
    parser.add_argument("--datasets", help="Filter to specific original datasets (comma-separated, e.g., hellaswag,logiqa)")

    # Execution
    parser.add_argument("--max-tokens", type=int, default=8192, help="Max tokens for generation")
    parser.add_argument("--max-tasks", type=int, default=12, help="Max parallel tasks")
    parser.add_argument("--max-connections", type=int, default=None, help="Max concurrent model API connections (controls per-model concurrency)")
    parser.add_argument("--dry-run", action="store_true", help="Print tasks without running")
    parser.add_argument("--log-dir", default="logs/tinker_evals", help="Base directory for logs")
    parser.add_argument("--log-level", default="INFO", help="Logging level")

    add_common_eval_args(parser)

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    results = asyncio.run(run(args))

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Success: {results['success']}")
    print(f"Failed: {results['failed']}")
    print(f"Skipped: {results['skipped']}")

    return 0 if results["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
