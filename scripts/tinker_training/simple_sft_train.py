"""
Simple SFT training script based on Tinker cookbook.

Usage:
    conda activate RLconsistencytraining

    # Train BCT model
    python scripts/tinker_training/simple_sft_train.py \
        --data dataset_dumps/train_seed_42/llama-3-1-8b-instruct/mixed.jsonl \
        --name llama-bct

    # Train Control model
    python scripts/tinker_training/simple_sft_train.py \
        --data dataset_dumps/control_seed_42/llama-3-1-8b-instruct/llama-control-mixed.jsonl \
        --name llama-control
"""

import argparse
import json
import random
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import tinker
from tinker import types
from tinker_cookbook import renderers, model_info
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.supervised.common import datum_from_tokens_weights, compute_mean_nll
from tqdm import tqdm


def load_data(file_path: Path) -> list[dict]:
    """Load training data from JSONL file."""
    samples = []
    with open(file_path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def get_renderer_for_model(model: str):
    """Get renderer and tokenizer for model."""
    tokenizer = get_tokenizer(model)
    renderer_name = model_info.get_recommended_renderer_name(model)
    return renderers.get_renderer(renderer_name, tokenizer), tokenizer


def create_datum(sample: dict, renderer, max_length: int | None = None) -> types.Datum:
    """Convert a sample to Tinker Datum format using cookbook helper."""
    messages = sample["messages"]
    tokens, weights = renderer.build_supervised_example(messages)
    return datum_from_tokens_weights(tokens, weights, max_length)


def train(
    data_path: Path,
    name: str,
    model: str = "meta-llama/Llama-3.1-8B-Instruct",
    batch_size: int = 128,
    lr: float = 1e-4,
    lora_rank: int = 32,
    checkpoint_every: int | None = None,
    max_samples: int | None = None,
) -> str:
    """
    Run simple SFT training (matching Tinker cookbook hyperparameters).

    Args:
        data_path: Path to training data JSONL
        name: Experiment name (used for checkpoint naming)
        model: Base model
        batch_size: Batch size (cookbook default: 128)
        lr: Base learning rate (cookbook default: 1e-4, with linear decay)
        lora_rank: LoRA rank (cookbook default: 32)
        checkpoint_every: Save checkpoint every N steps (None = only final)
        max_samples: Limit number of training samples (None = use all)

    Returns:
        Final checkpoint path
    """
    print(f"Loading data from {data_path}")
    samples = load_data(data_path)
    if max_samples is not None and max_samples < len(samples):
        samples = samples[:max_samples]
        print(f"Using {len(samples)} samples (limited from full dataset)")
    else:
        print(f"Loaded {len(samples)} samples")

    # Calculate number of training steps (1 epoch through data)
    n_train_batches = len(samples) // batch_size
    print(f"Training for {n_train_batches} steps (1 epoch)")

    # Setup
    print(f"\nInitializing training for {model}")
    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=model,
        rank=lora_rank,
        train_mlp=True,
        train_attn=True,
        train_unembed=True,
    )
    renderer, tokenizer = get_renderer_for_model(model)

    print(f"\nTraining config (Tinker cookbook defaults):")
    print(f"  Model: {model}")
    print(f"  Samples: {len(samples)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Steps: {n_train_batches}")
    print(f"  Base learning rate: {lr} (with linear decay)")
    print(f"  Adam beta1: 0.9, beta2: 0.95, eps: 1e-8")
    print(f"  LoRA rank: {lora_rank}")
    print(f"  Checkpoint every: {checkpoint_every if checkpoint_every else 'final only'}")

    checkpoint_paths = []

    # Training loop
    print("\nStarting training...")
    pbar = tqdm(range(n_train_batches), desc="Training")

    for step in pbar:
        # Linear LR decay (cookbook style)
        lr_mult = max(0.0, 1.0 - step / n_train_batches)
        current_lr = lr * lr_mult

        # Adam params with cookbook defaults (beta2=0.95)
        adam_params = types.AdamParams(
            learning_rate=current_lr,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
        )

        # Sample batch
        batch_samples = random.choices(samples, k=batch_size)
        batch_data = [create_datum(s, renderer) for s in batch_samples]

        # Forward/backward pass
        fwd_bwd_result = training_client.forward_backward(
            batch_data, loss_fn="cross_entropy"
        ).result()

        # Optimizer step
        training_client.optim_step(adam_params).result()

        # Compute proper per-token loss using cookbook helper
        logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
        weights = [d.loss_fn_inputs["weights"] for d in batch_data]
        train_nll = compute_mean_nll(logprobs, weights)
        pbar.set_postfix({"nll": f"{train_nll:.4f}", "lr": f"{current_lr:.2e}"})

        # Checkpoint
        step_num = step + 1  # 1-indexed for checkpoint naming
        if checkpoint_every and step_num % checkpoint_every == 0:
            ckpt_name = f"{name}-step{step_num}"
            result = training_client.save_weights_for_sampler(name=ckpt_name).result()
            ckpt_path = result.path if hasattr(result, 'path') else ckpt_name
            checkpoint_paths.append(ckpt_path)
            print(f"\n  Saved checkpoint: {ckpt_path}")

    # Final checkpoint
    final_name = f"{name}-final"
    result = training_client.save_weights_for_sampler(name=final_name).result()
    final_path = result.path if hasattr(result, 'path') else final_name
    checkpoint_paths.append(final_path)

    print(f"\n\nTraining complete!")
    print(f"Final checkpoint: {final_path}")
    print(f"\nAll checkpoints:")
    for p in checkpoint_paths:
        print(f"  {p}")

    return final_path


def main():
    parser = argparse.ArgumentParser(description="Simple SFT training with Tinker (cookbook defaults)")
    parser.add_argument("--data", required=True, help="Path to training data JSONL")
    parser.add_argument("--name", required=True, help="Experiment name")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (cookbook default: 128)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Base learning rate (cookbook default: 1e-4)")
    parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank (cookbook default: 32)")
    parser.add_argument("--checkpoint-every", type=int, default=None, help="Checkpoint every N steps (None = only final)")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of training samples")

    args = parser.parse_args()

    train(
        data_path=Path(args.data),
        name=args.name,
        model=args.model,
        batch_size=args.batch_size,
        lr=args.lr,
        lora_rank=args.lora_rank,
        checkpoint_every=args.checkpoint_every,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
