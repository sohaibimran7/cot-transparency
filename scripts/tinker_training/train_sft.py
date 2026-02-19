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
        --experiment-name bct_llama

    # With interleaving for mixed batches
    python scripts/tinker_training/train_sft.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --data instruct.jsonl:10 bct_cot.jsonl:5 bct_non_cot.jsonl:5 \
        --interleave \
        --experiment-name bct_llama_interleaved
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

import asyncio  # noqa: E402

import tinker  # noqa: E402

from cot_transparency.apis.tinker.finetune import SFTConfig, train_sft  # noqa: E402
from cot_transparency.apis.tinker.common import (  # noqa: E402
    LoRAConfig,
    AdamConfig,
    CheckpointConfig,
)


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


def load_and_combine(
    file_specs: list[tuple[Path, int | None]],
    interleave: bool,
) -> list[dict]:
    """Load samples from multiple files and combine them."""
    all_file_samples: list[list[dict]] = []
    for path, limit in file_specs:
        samples = []
        with open(path) as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                samples.append(json.loads(line))
        print(f"  {path.name}: {len(samples)} samples")
        all_file_samples.append(samples)

    if not interleave:
        return [s for samples in all_file_samples for s in samples]

    # Round-robin interleave
    result: list[dict] = []
    iterators = [iter(samples) for samples in all_file_samples]
    while iterators:
        for it in list(iterators):
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

    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--model", help="Model name")

    # Data
    parser.add_argument("--data", nargs="+", metavar="FILE[:N]",
                        help="Data files with optional sample limits")
    parser.add_argument("--interleave", action="store_true",
                        help="Round-robin interleave samples across files")

    # Naming
    parser.add_argument("--experiment-name", default="sft_experiment")
    parser.add_argument("--run-name", default="default")

    # Hyperparameters
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (default: auto-detect from model)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--save-state", action="store_true",
                        help="Save full optimizer state for intermediate checkpoints (for resuming)")
    parser.add_argument("--skip-near-final", type=int, default=0,
                        help="Skip intermediate checkpoints within N steps of final")

    # Resume from checkpoint
    parser.add_argument("--resume-from", default=None,
                        help="Tinker checkpoint path to load before training (tinker://...)")

    # Execution
    parser.add_argument("-y", "--yes", action="store_true")

    args = parser.parse_args()

    if args.list_models:
        print_available_models()
        return

    if not args.model:
        parser.error("--model is required unless using --list-models")
    if not args.data:
        parser.error("--data is required")

    # Parse and validate files
    file_specs = [parse_file_spec(spec) for spec in args.data]
    for path, _ in file_specs:
        if not path.exists():
            parser.error(f"File not found: {path}")

    # Load samples
    print("Loading samples...")
    all_samples = load_and_combine(file_specs, args.interleave)
    n_samples = len(all_samples)
    print(f"Total: {n_samples} samples")

    # If multiple files or limits, write combined data to temp file
    if len(file_specs) > 1 or any(limit is not None for _, limit in file_specs):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        for sample in all_samples:
            tmp.write(json.dumps(sample) + "\n")
        tmp.close()
        data_path = Path(tmp.name)
        print(f"Combined data written to {data_path}")
    else:
        data_path = file_specs[0][0]

    # Build config
    config = SFTConfig(
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        model=args.model,
        lora=LoRAConfig(rank=args.lora_rank),
        optimizer=AdamConfig(learning_rate=args.lr),
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint=CheckpointConfig(
            save_every_n_steps=args.save_every,
            save_state=args.save_state,
            skip_near_final_steps=args.skip_near_final,
        ),
    )

    # Print summary
    n_steps = n_samples // args.batch_size
    n_ckpts = n_steps // args.save_every
    print()
    print(f"Model: {config.model}")
    print(f"Experiment: {config.experiment_name} / {config.run_name}")
    print(f"Hyperparams: lr={args.lr or 'auto'}, batch={args.batch_size}, "
          f"epochs={args.epochs}, lora_rank={args.lora_rank}")
    print(f"Steps: {n_steps}, checkpoints: ~{n_ckpts} intermediate + 1 final")
    print(f"Final checkpoint: always saves full state (for resuming)")
    if args.save_state:
        print(f"Intermediate checkpoints: also saving full state")
    if args.resume_from:
        print(f"Resuming from: {args.resume_from}")

    if not args.yes:
        if input("\nProceed? (y/n): ").lower() != "y":
            print("Cancelled.")
            return

    # Train
    final_checkpoint = asyncio.run(train_sft(data_path, config, resume_from=args.resume_from))
    print(f"\nDone! Final checkpoint: {final_checkpoint}")


if __name__ == "__main__":
    main()
