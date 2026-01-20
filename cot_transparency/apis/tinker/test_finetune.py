"""
Test script for Tinker API finetuning.

Finetunes llama 3.1 8b on:
- 10 samples from instruct_samples.jsonl
- 5 samples from bct_cot.jsonl
- 5 samples from bct_non_cot.jsonl

Trains 2 checkpoints (saves every 10 samples).
"""

import argparse
from pathlib import Path

from cot_transparency.apis.tinker.finetune import TinkerSFTConfig, TinkerSFTTrainer
from cot_transparency.apis.tinker.common import TinkerLoRAConfig, TinkerAdamParams, CheckpointConfig
from cot_transparency.apis.openai.finetune import FinetuneSample
from cot_transparency.json_utils.read_write import read_jsonl_file_into_basemodel


def load_samples(file_path: Path, n_samples: int) -> list[FinetuneSample]:
    """Load first n samples from a jsonl file."""
    all_samples = read_jsonl_file_into_basemodel(file_path, FinetuneSample)
    return all_samples[:n_samples]


def main(skip_confirmation: bool = False):
    data_dir = Path("dataset_dumps/train_seed_42")

    # Load samples
    print("Loading samples...")
    instruct_samples = load_samples(data_dir / "instruct_samples.jsonl", n_samples=10)
    bct_cot_samples = load_samples(data_dir / "bct_cot.jsonl", n_samples=5)
    bct_non_cot_samples = load_samples(data_dir / "bct_non_cot.jsonl", n_samples=5)
    all_samples = instruct_samples + bct_cot_samples + bct_non_cot_samples

    print(f"Loaded {len(instruct_samples)} instruct, {len(bct_cot_samples)} CoT, {len(bct_non_cot_samples)} non-CoT")
    print(f"Total samples: {len(all_samples)}")

    # Configure: 20 samples, batch_size=10 → 2 steps, save every step = 2 checkpoints
    config = TinkerSFTConfig(
        model="meta-llama/Llama-3.1-8B-Instruct",
        lora=TinkerLoRAConfig(rank=32),
        optimizer=TinkerAdamParams(learning_rate=1e-5),
        n_epochs=1,
        batch_size=10,
        checkpoint=CheckpointConfig(
            save_every_n_steps=1,
            save_full_state=True,
            checkpoint_prefix="sft-checkpoint",
            final_checkpoint_name="sft-final",
        ),
    )

    print(f"\nConfig: {config.model}, lr={config.optimizer.learning_rate}, batch={config.batch_size}")
    print(f"Expected: 2 checkpoints + final")

    if not skip_confirmation:
        if input("Proceed? (y/n): ").lower() != "y":
            print("Cancelled.")
            return

    trainer = TinkerSFTTrainer(config=config)
    trainer.setup()
    final_checkpoint = trainer.train(samples=all_samples)

    print(f"\nDone! Final: {final_checkpoint}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Tinker API finetuning")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    args = parser.parse_args()
    main(skip_confirmation=args.yes)
