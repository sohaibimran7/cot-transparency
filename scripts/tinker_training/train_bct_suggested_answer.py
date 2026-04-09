#!/usr/bin/env python3
"""Train BCT models on suggested-answer data."""

import asyncio
from pathlib import Path

from cot_transparency.apis.tinker.finetune import train_sft, SFTConfig
from cot_transparency.apis.tinker.common import CheckpointConfig, AdamConfig, LoRAConfig


DATA_DIR = Path("dataset_dumps/train-from-test-mmlu-truthfulqa/suggested-answer")


async def train_job(
    file_path: Path,
    experiment_name: str,
    run_name: str,
    batch_size: int,
    save_every: int,
    skip_near_final: int,
):
    """Run a single training job."""
    config = SFTConfig(
        experiment_name=experiment_name,
        run_name=run_name,
        model="meta-llama/Llama-3.1-8B-Instruct",
        lora=LoRAConfig(rank=8),
        optimizer=AdamConfig(lr_schedule="linear"),
        batch_size=batch_size,
        n_epochs=1,
        checkpoint=CheckpointConfig(
            save_every_n_steps=save_every,
            skip_near_final_steps=skip_near_final,
        ),
    )

    print(f"\n{'='*60}")
    print(f"Training: {run_name}")
    print(f"  File: {file_path}")
    print(f"  Batch size: {batch_size}")
    print(f"  Save every: {save_every} steps")
    print(f"  Skip near final: {skip_near_final} steps")
    print(f"{'='*60}\n")

    checkpoint = await train_sft(file_path, config=config)
    print(f"\nCompleted {run_name}: {checkpoint}")
    return checkpoint


async def main():
    jobs = [
        # Only the remaining bs16 job
        {
            "file_path": DATA_DIR / "bct_cot.jsonl",
            "experiment_name": "bct-suggested-answer",
            "run_name": "bct_mmlu_truthfulqa_only_bs16_r8",
            "batch_size": 16,
            "save_every": 5,
            "skip_near_final": 63,  # 2000/16=125 steps, skip latter half
        },
    ]

    # Run jobs sequentially (wandb doesn't support parallel runs in same process)
    checkpoints = []
    for job in jobs:
        checkpoint = await train_job(**job)
        checkpoints.append(checkpoint)

    print("\n" + "="*60)
    print("All training complete!")
    print("="*60)
    for i, (job, ckpt) in enumerate(zip(jobs, checkpoints)):
        print(f"{i+1}. {job['run_name']}: {ckpt}")


if __name__ == "__main__":
    asyncio.run(main())
