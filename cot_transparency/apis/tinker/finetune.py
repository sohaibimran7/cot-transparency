"""
Tinker SFT (Supervised Fine-Tuning).

Usage:
    from cot_transparency.apis.tinker.finetune import train_sft, SFTConfig

    # Basic usage with defaults
    checkpoint = asyncio.run(train_sft(Path("data/train.jsonl")))

    # With custom config
    config = SFTConfig(
        experiment_name="bct_debug",
        run_name="control",
        model="meta-llama/Llama-3.1-8B-Instruct",
        optimizer=AdamConfig(learning_rate=1e-4, lr_schedule="linear"),
        batch_size=128,
        n_epochs=1,
    )
    checkpoint = asyncio.run(train_sft(Path("data/train.jsonl"), config=config))

Training data format (JSONL):
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, ...]}
"""

import asyncio
import json
import random
from pathlib import Path
from typing import Optional

import tinker
from tinker import types
from pydantic import BaseModel
from tqdm import tqdm

from tinker_cookbook.supervised.common import datum_from_tokens_weights, compute_mean_nll
from tinker_cookbook.utils.lr_scheduling import compute_schedule_lr_multiplier
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.utils.ml_log import setup_logging

from cot_transparency.apis.tinker.common import (
    LoRAConfig,
    AdamConfig,
    CheckpointConfig,
    build_checkpoint_name,
    build_log_dir,
    get_renderer_and_tokenizer,
    get_recommended_lr,
)


class SFTConfig(BaseModel):
    """SFT training configuration."""
    experiment_name: str = "sft"
    run_name: str = "default"
    model: str = "meta-llama/Llama-3.1-8B-Instruct"
    lora: LoRAConfig = LoRAConfig()
    optimizer: AdamConfig = AdamConfig()
    n_epochs: int = 1
    batch_size: int = 128
    checkpoint: CheckpointConfig = CheckpointConfig()
    log_base_dir: str = "logs"


def load_samples(file_path: Path) -> list[dict]:
    """
    Load training samples from JSONL file.

    Each line should be a JSON object with a "messages" field containing
    a list of message dicts with "role" and "content" fields.
    """
    samples = []
    with open(file_path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


async def train_sft(
    file_path: Path,
    config: Optional[SFTConfig] = None,
    max_samples: Optional[int] = None,
) -> str:
    """
    Run SFT training from JSONL file.

    Args:
        file_path: Path to JSONL file with {"messages": [...]} format
        config: Training configuration (uses defaults if not provided)
        max_samples: Limit number of samples (None = use all)

    Returns:
        Final checkpoint path
    """
    cfg = config or SFTConfig()

    # Build log directory: logs/{experiment_name}/{run_name}/
    log_dir = Path(build_log_dir(cfg.log_base_dir, cfg.experiment_name, cfg.run_name))
    log_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging (writes to files + WandB with experiment_name as project, run_name as name)
    logger = setup_logging(
        log_dir=str(log_dir),
        wandb_project=cfg.experiment_name,
        wandb_name=cfg.run_name,
        config=cfg.model_dump(),
    )

    # Load training data
    samples = load_samples(file_path)
    if max_samples and max_samples < len(samples):
        samples = samples[:max_samples]

    n_samples = len(samples)
    steps_per_epoch = n_samples // cfg.batch_size
    total_steps = steps_per_epoch * cfg.n_epochs

    # Determine learning rate: use configured value or get recommended LR for model
    base_lr: float = cfg.optimizer.learning_rate if cfg.optimizer.learning_rate is not None else get_recommended_lr(cfg.model)

    print(f"SFT Training: {n_samples} samples, batch={cfg.batch_size}, {total_steps} steps, lr={base_lr:.2e}")
    logger.log_hparams({"n_samples": n_samples, "total_steps": total_steps, "file": str(file_path), "base_lr": base_lr})

    # Initialize Tinker client
    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=cfg.model,
        **cfg.lora.model_dump(),
    )
    renderer, _ = get_renderer_and_tokenizer(cfg.model)

    checkpoint_paths: list[str] = []
    global_step = 0

    # Training loop
    for epoch in range(cfg.n_epochs):
        # Shuffle samples each epoch
        epoch_samples = list(samples)
        random.shuffle(epoch_samples)
        epoch_loss = 0.0
        n_steps = 0

        pbar = tqdm(range(0, n_samples, cfg.batch_size), desc=f"Epoch {epoch+1}")
        for batch_start in pbar:
            batch_samples = epoch_samples[batch_start:batch_start + cfg.batch_size]

            # Create datums with proper token shifting for next-token prediction
            batch_data = []
            for sample in batch_samples:
                tokens, weights = renderer.build_supervised_example(sample["messages"])
                batch_data.append(datum_from_tokens_weights(tokens, weights))

            # Compute LR with schedule
            lr_mult = compute_schedule_lr_multiplier(
                lr_schedule=cfg.optimizer.lr_schedule,
                step=global_step,
                total_steps=total_steps,
            )
            current_lr = base_lr * lr_mult

            adam_params = types.AdamParams(
                learning_rate=current_lr,
                beta1=cfg.optimizer.beta1,
                beta2=cfg.optimizer.beta2,
                eps=cfg.optimizer.eps,
            )

            # Async: enqueue forward_backward and optim_step before awaiting results (overlapping pattern)
            fwd_bwd_future = await training_client.forward_backward_async(
                batch_data, loss_fn="cross_entropy"
            )
            optim_future = await training_client.optim_step_async(adam_params)

            # Await results
            fwd_bwd_result = await fwd_bwd_future.result_async()
            await optim_future.result_async()

            # Compute proper per-token NLL
            logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
            weights = [d.loss_fn_inputs["weights"] for d in batch_data]
            nll = compute_mean_nll(logprobs, weights)

            epoch_loss += nll
            n_steps += 1
            global_step += 1

            pbar.set_postfix({"nll": f"{nll:.4f}", "lr": f"{current_lr:.2e}"})
            logger.log_metrics({"train/nll": nll, "train/lr": current_lr}, step=global_step)

            # Intermediate checkpoint (skip if near final to avoid duplicates)
            ckpt_cfg = cfg.checkpoint
            steps_remaining = total_steps - global_step
            near_final = steps_remaining <= ckpt_cfg.skip_near_final_steps

            if ckpt_cfg.save_every_n_steps and global_step % ckpt_cfg.save_every_n_steps == 0 and not near_final:
                name = build_checkpoint_name(cfg.experiment_name, cfg.run_name, step=global_step)
                kind = "both" if ckpt_cfg.save_state else "sampler"
                paths = await checkpoint_utils.save_checkpoint_async(
                    training_client,
                    name=name,
                    log_path=str(log_dir),
                    loop_state={"epoch": epoch, "step": global_step},
                    kind=kind,
                )
                checkpoint_path = paths.get("sampler_path") or paths.get("state_path")
                checkpoint_paths.append(checkpoint_path)
                logger.log_metrics({"checkpoint": checkpoint_path}, step=global_step)

        # Epoch summary
        if n_steps > 0:
            avg_loss = epoch_loss / n_steps
            print(f"Epoch {epoch+1} avg NLL: {avg_loss:.4f}")
            logger.log_metrics({"train/epoch_nll": avg_loss, "train/epoch": epoch + 1}, step=global_step)

    # Final checkpoint (no step suffix)
    final_name = build_checkpoint_name(cfg.experiment_name, cfg.run_name)
    kind = "both" if cfg.checkpoint.save_state else "sampler"
    paths = await checkpoint_utils.save_checkpoint_async(
        training_client,
        name=final_name,
        log_path=str(log_dir),
        loop_state={"epoch": cfg.n_epochs, "step": global_step, "final": True},
        kind=kind,
    )
    final_path = paths.get("sampler_path") or paths.get("state_path")
    checkpoint_paths.append(final_path)

    print(f"\nTraining complete. Final checkpoint: {final_path}")
    logger.log_metrics({"final_checkpoint": final_path}, step=global_step)
    logger.log_hparams({"final_checkpoint": final_path, "all_checkpoints": checkpoint_paths})
    logger.close()

    return final_path


def train_sft_sync(
    file_path: Path,
    config: Optional[SFTConfig] = None,
    max_samples: Optional[int] = None,
) -> str:
    """Synchronous wrapper for train_sft."""
    return asyncio.run(train_sft(file_path, config, max_samples))
