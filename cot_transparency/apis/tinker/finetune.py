"""
Tinker SFT (Supervised Fine-Tuning) implementation.
"""

import random
from pathlib import Path
from typing import Optional, Sequence, Callable

import tinker
from tinker import types
from pydantic import BaseModel
from tqdm import tqdm

from cot_transparency.apis.openai.finetune import FinetuneSample, WandbSyncer, confirm_to_continue
from cot_transparency.apis.tinker.common import (
    TinkerLoRAConfig,
    TinkerAdamParams,
    CheckpointConfig,
    build_checkpoint_name,
    get_renderer_for_model,
    messages_to_dict,
    extract_loss_from_result,
    save_checkpoint,
)
from cot_transparency.json_utils.read_write import read_jsonl_file_into_basemodel


class TinkerSFTConfig(BaseModel):
    """Full SFT training configuration."""
    experiment_name: Optional[str] = None  # For grouping runs in WandB
    model: str = "meta-llama/Llama-3.1-8B-Instruct"
    lora: TinkerLoRAConfig = TinkerLoRAConfig()
    optimizer: TinkerAdamParams = TinkerAdamParams()
    n_epochs: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    checkpoint: CheckpointConfig = CheckpointConfig()


class TinkerSFTTrainer:
    """SFT Trainer using Tinker API."""

    def __init__(self, config: TinkerSFTConfig):
        self.config = config
        self.service_client = tinker.ServiceClient()
        self.training_client = None
        self.renderer = None
        self.tokenizer = None

    def setup(self) -> None:
        """Initialize training client and renderer."""
        self.training_client = self.service_client.create_lora_training_client(
            base_model=self.config.model,
            **self.config.lora.model_dump(),
        )
        self.renderer, self.tokenizer = get_renderer_for_model(self.config.model)
        print(f"Initialized Tinker training client for {self.config.model}")

    def _create_datum(self, sample: FinetuneSample) -> types.Datum:
        """Convert a FinetuneSample to Tinker Datum format."""
        msg_dicts = messages_to_dict(sample.messages)
        model_input, weights = self.renderer.build_supervised_example(msg_dicts)
        tokens = model_input.tolist()
        weights = weights.tolist()

        return types.Datum(
            model_input=types.ModelInput.from_ints(tokens=tokens),
            loss_fn_inputs=dict(target_tokens=tokens, weights=weights),
        )

    def train(
        self,
        samples: Sequence[FinetuneSample],
        syncer: Optional[WandbSyncer] = None,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> str:
        """
        Run SFT training.

        Args:
            samples: Training samples
            syncer: Optional WandB syncer for logging
            progress_callback: Optional callback(step, total_steps, loss)

        Returns:
            Checkpoint path of the trained model
        """
        if self.training_client is None:
            self.setup()

        # Log config to WandB
        if syncer:
            syncer.update_parameters_with_dict({
                "training_type": "sft",
                "experiment_name": self.config.experiment_name,
                "config": self.config.model_dump(),
                "n_samples": len(samples),
            })

        checkpoint_paths: list[str] = []  # Track all saved checkpoints

        adam_params = types.AdamParams(**self.config.optimizer.model_dump())
        n_samples = len(samples)
        effective_batch_size = self.config.batch_size * self.config.gradient_accumulation_steps
        steps_per_epoch = n_samples // effective_batch_size
        total_steps = steps_per_epoch * self.config.n_epochs

        print(f"Starting SFT training{f' [{self.config.experiment_name}]' if self.config.experiment_name else ''}:")
        print(f"  Samples: {n_samples}, Batch size: {self.config.batch_size}")
        print(f"  Gradient accumulation: {self.config.gradient_accumulation_steps}, Effective batch size: {effective_batch_size}")
        print(f"  Steps per epoch: {steps_per_epoch}, Total steps: {total_steps}")

        global_step = 0
        accumulated_grads = 0
        accumulated_loss = 0.0

        for epoch in range(self.config.n_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.n_epochs}")

            epoch_samples = list(samples)
            random.shuffle(epoch_samples)
            epoch_loss = 0.0
            n_steps_in_epoch = 0

            pbar = tqdm(range(0, n_samples, self.config.batch_size), desc=f"Epoch {epoch+1}")

            for batch_start in pbar:
                batch_end = min(batch_start + self.config.batch_size, n_samples)
                batch_samples = epoch_samples[batch_start:batch_end]
                batch_data = [self._create_datum(s) for s in batch_samples]

                fwd_bwd_result = self.training_client.forward_backward(
                    batch_data, loss_fn="cross_entropy"
                ).result()

                batch_loss = extract_loss_from_result(fwd_bwd_result, len(batch_samples))
                accumulated_loss += batch_loss
                accumulated_grads += 1

                if accumulated_grads >= self.config.gradient_accumulation_steps:
                    self.training_client.optim_step(adam_params).result()

                    avg_loss = accumulated_loss / accumulated_grads
                    epoch_loss += avg_loss
                    n_steps_in_epoch += 1
                    global_step += 1

                    pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

                    if progress_callback:
                        progress_callback(global_step, total_steps, avg_loss)

                    if syncer:
                        syncer.run.log({"train/loss": avg_loss, "train/step": global_step})

                    # Save checkpoint if requested
                    ckpt_cfg = self.config.checkpoint
                    if ckpt_cfg.save_every_n_steps and global_step % ckpt_cfg.save_every_n_steps == 0:
                        ckpt_name = build_checkpoint_name(self.config.experiment_name, ckpt_cfg.checkpoint_prefix, step=global_step)
                        ckpt_path = save_checkpoint(self.training_client, ckpt_name, ckpt_cfg.save_full_state)
                        checkpoint_paths.append(ckpt_path)
                        print(f"\nSaved checkpoint: {ckpt_path}")
                        if syncer:
                            syncer.run.log({"checkpoint/path": ckpt_path, "checkpoint/step": global_step})

                    accumulated_grads = 0
                    accumulated_loss = 0.0

            avg_epoch_loss = epoch_loss / n_steps_in_epoch if n_steps_in_epoch > 0 else 0.0
            print(f"Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}")

            if syncer:
                syncer.run.log({"train/epoch": epoch + 1, "train/epoch_loss": avg_epoch_loss})

        # Final optimizer step if any remaining gradients
        if accumulated_grads > 0:
            self.training_client.optim_step(adam_params).result()

        # Save final checkpoint (no step suffix)
        final_ckpt_name = build_checkpoint_name(self.config.experiment_name, self.config.checkpoint.checkpoint_prefix)
        final_ckpt_path = save_checkpoint(
            self.training_client, final_ckpt_name, self.config.checkpoint.save_full_state
        )
        checkpoint_paths.append(final_ckpt_path)
        print(f"\nTraining complete. Final checkpoint: {final_ckpt_path}")

        # Log all checkpoint paths to WandB config for easy lookup
        if syncer:
            syncer.run.config.update({
                "final_checkpoint_path": final_ckpt_path,
                "all_checkpoint_paths": checkpoint_paths,
            })

        return final_ckpt_path


def finetune_sft_tinker(
    model: str,
    samples: Sequence[FinetuneSample],
    lora_config: Optional[TinkerLoRAConfig] = None,
    optimizer_config: Optional[TinkerAdamParams] = None,
    n_epochs: int = 1,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 1,
    checkpoint_config: Optional[CheckpointConfig] = None,
    syncer: Optional[WandbSyncer] = None,
    ask_to_validate_training: bool = True,
) -> str:
    """
    Run SFT finetuning with Tinker.

    Returns:
        Checkpoint path of the trained model
    """
    if ask_to_validate_training:
        response = input(f"About to train on {len(samples)} samples. Continue? (y/n): ")
        if response.lower() != "y":
            print("Training cancelled.")
            return ""

    config = TinkerSFTConfig(
        model=model,
        lora=lora_config or TinkerLoRAConfig(),
        optimizer=optimizer_config or TinkerAdamParams(),
        n_epochs=n_epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        checkpoint=checkpoint_config or CheckpointConfig(),
    )

    trainer = TinkerSFTTrainer(config=config)
    trainer.setup()

    return trainer.train(samples=samples, syncer=syncer)


def finetune_sft_tinker_from_file(
    model: str,
    file_path: Path,
    lora_config: Optional[TinkerLoRAConfig] = None,
    optimizer_config: Optional[TinkerAdamParams] = None,
    n_epochs: int = 1,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 1,
    checkpoint_config: Optional[CheckpointConfig] = None,
    syncer: Optional[WandbSyncer] = None,
    ask_to_validate_training: bool = True,
) -> str:
    """Load samples from file and run SFT finetuning."""
    samples = read_jsonl_file_into_basemodel(file_path, FinetuneSample)

    if ask_to_validate_training:
        confirm_to_continue(file_path)

    return finetune_sft_tinker(
        model=model,
        samples=samples,
        lora_config=lora_config,
        optimizer_config=optimizer_config,
        n_epochs=n_epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        checkpoint_config=checkpoint_config,
        syncer=syncer,
        ask_to_validate_training=False,
    )
