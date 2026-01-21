"""
Unified SFT training script using Tinker API.

Trains on arbitrary data files with flexible mixing and hyperparameters.

Usage:
    # List available models
    python scripts/tinker_training/train_sft.py --list-models

    # BCT training on Llama (all file types)
    python scripts/tinker_training/train_sft.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --data instruct.jsonl:10 bct_cot.jsonl:5 bct_non_cot.jsonl:5 \
        --experiment-name bct_llama \
        --checkpoint-prefix bct

    # GPT OSS (non-cot only, half instruct for 1:1 ratio)
    python scripts/tinker_training/train_sft.py \
        --model gpt-oss-120b \
        --data instruct.jsonl:5 bct_non_cot.jsonl:5 \
        --experiment-name bct_gpt

    # With interleaving for mixed batches
    python scripts/tinker_training/train_sft.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --data instruct.jsonl:10 bct_cot.jsonl:5 bct_non_cot.jsonl:5 \
        --interleave \
        --experiment-name bct_llama_interleaved
"""

import argparse
from pathlib import Path

import tinker

from cot_transparency.apis.tinker.finetune import TinkerSFTConfig, TinkerSFTTrainer
from cot_transparency.apis.tinker.common import TinkerLoRAConfig, TinkerAdamParams, CheckpointConfig
from cot_transparency.apis.openai.finetune import FinetuneSample, WandbSyncer
from cot_transparency.json_utils.read_write import read_jsonl_file_into_basemodel


def list_available_models() -> list[str]:
    """Query Tinker API for available models."""
    service_client = tinker.ServiceClient()
    capabilities = service_client.get_server_capabilities()
    return [
        model.model_name
        for model in capabilities.supported_models
        if model.model_name is not None
    ]


def print_available_models() -> None:
    """Print available models from Tinker API."""
    print("Querying Tinker API for available models...")
    print()
    try:
        models = list_available_models()
        print("Available models:")
        print("-" * 50)
        for model in sorted(models):
            print(f"  {model}")
        print("-" * 50)
        print(f"Total: {len(models)} models")
    except Exception as e:
        print(f"Error querying models: {e}")
        print()
        print("Common models (may not all be available):")
        print("  meta-llama/Llama-3.1-8B-Instruct")
        print("  meta-llama/Llama-3.1-70B-Instruct")
        print("  gpt-oss-120b")


def parse_file_spec(spec: str) -> tuple[Path, int | None]:
    """Parse file:limit spec. Returns (path, limit) where limit may be None."""
    if ":" in spec:
        path_str, limit_str = spec.rsplit(":", 1)
        return Path(path_str), int(limit_str)
    return Path(spec), None


def load_samples(file_path: Path, limit: int | None = None) -> list[FinetuneSample]:
    """Load samples from a jsonl file with optional limit."""
    all_samples = read_jsonl_file_into_basemodel(file_path, FinetuneSample)
    if limit is not None:
        return all_samples[:limit]
    return all_samples


def load_and_combine(
    file_specs: list[tuple[Path, int | None]],
    interleave: bool,
) -> list[FinetuneSample]:
    """Load samples from multiple files and combine them.

    Args:
        file_specs: List of (path, limit) tuples
        interleave: If True, round-robin across files. If False, concatenate.

    Returns:
        Combined list of samples
    """
    # Load all files into memory
    all_file_samples = []
    for path, limit in file_specs:
        samples = load_samples(path, limit)
        print(f"  {path.name}: {len(samples)} samples")
        all_file_samples.append(samples)

    if not interleave:
        # Simple concatenation
        return [s for samples in all_file_samples for s in samples]

    # Round-robin interleave
    result: list[FinetuneSample] = []
    iterators = [iter(samples) for samples in all_file_samples]
    while iterators:
        for it in list(iterators):  # copy to allow removal during iteration
            try:
                result.append(next(it))
            except StopIteration:
                iterators.remove(it)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Unified SFT training script using Tinker API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model selection
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models from Tinker API and exit",
    )
    parser.add_argument(
        "--model",
        help="Model name (e.g., meta-llama/Llama-3.1-8B-Instruct)",
    )

    # Data
    parser.add_argument(
        "--data",
        nargs="+",
        metavar="FILE[:N]",
        help="Data files with optional sample limits (e.g., instruct.jsonl:10 cot.jsonl:5)",
    )
    parser.add_argument(
        "--interleave",
        action="store_true",
        help="Round-robin interleave samples across files instead of concatenating",
    )

    # Naming
    parser.add_argument(
        "--experiment-name",
        default="sft_experiment",
        help="Experiment name for WandB grouping (default: sft_experiment)",
    )
    parser.add_argument(
        "--checkpoint-prefix",
        default="sft",
        help="Prefix for checkpoint file names (default: sft)",
    )

    # Hyperparameters
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate (default: 1e-5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size (default: 10)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs (default: 1)",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=32,
        help="LoRA rank (default: 32)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save checkpoint every N steps (default: 1)",
    )
    parser.add_argument(
        "--for-resuming",
        action="store_true",
        help="Save full state for training resumption (default: save for sampling)",
    )
    parser.add_argument(
        "--skip-near-final",
        type=int,
        default=0,
        help="Skip intermediate checkpoints within N steps of final (default: 0)",
    )

    # Execution
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip confirmation prompts",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging",
    )

    args = parser.parse_args()

    # Handle --list-models
    if args.list_models:
        print_available_models()
        return

    # Validate required args
    if not args.model:
        parser.error("--model is required unless using --list-models")
    if not args.data:
        parser.error("--data is required")

    # Parse file specs
    file_specs = [parse_file_spec(spec) for spec in args.data]

    # Validate files exist
    for path, _ in file_specs:
        if not path.exists():
            parser.error(f"File not found: {path}")

    # Load and combine samples
    print("Loading samples...")
    all_samples = load_and_combine(file_specs, args.interleave)
    print(f"Total: {len(all_samples)} samples")
    if args.interleave:
        print("(interleaved)")

    # Build config
    config = TinkerSFTConfig(
        experiment_name=args.experiment_name,
        model=args.model,
        lora=TinkerLoRAConfig(rank=args.lora_rank),
        optimizer=TinkerAdamParams(learning_rate=args.lr),
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint=CheckpointConfig(
            save_every_n_steps=args.save_every,
            save_full_state=args.for_resuming,
            checkpoint_prefix=args.checkpoint_prefix,
            skip_near_final_steps=args.skip_near_final,
        ),
    )

    # Print summary
    print()
    print(f"Model: {config.model}")
    print(f"Experiment: {config.experiment_name}")
    print(f"Checkpoint prefix: {config.checkpoint.checkpoint_prefix}")
    print(f"Hyperparams: lr={args.lr}, batch={args.batch_size}, epochs={args.epochs}, lora_rank={args.lora_rank}")
    print(f"Save every: {args.save_every} steps")

    n_steps = (len(all_samples) + args.batch_size - 1) // args.batch_size
    print(f"Expected: {n_steps} steps, {n_steps} checkpoints")

    if not args.yes:
        if input("\nProceed? (y/n): ").lower() != "y":
            print("Cancelled.")
            return

    # Setup WandB
    syncer = None
    if not args.no_wandb:
        syncer = WandbSyncer.create(
            project_name="consistency-training",
            name=f"{config.experiment_name}",
            notes=f"SFT training: {', '.join(p.name for p, _ in file_specs)}",
        )

    # Train
    trainer = TinkerSFTTrainer(config=config)
    trainer.setup()
    final_checkpoint = trainer.train(samples=all_samples, syncer=syncer)

    if syncer:
        syncer.end()

    print(f"\nDone! Final checkpoint: {final_checkpoint}")


if __name__ == "__main__":
    main()
