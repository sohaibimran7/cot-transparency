"""
Tinker SFT (Supervised Fine-Tuning) implementation.

This module provides SFT training via the Tinker API, compatible with the
existing BCT training pipeline.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Callable, Any
from dotenv import load_dotenv

import tinker
from tinker import types
from tinker_cookbook import renderers, model_info
from tinker_cookbook.tokenizer_utils import get_tokenizer
from pydantic import BaseModel
from tqdm import tqdm

from cot_transparency.apis.openai.finetune import (
    FinetuneSample,
    WandbSyncer,
    confirm_to_continue,
)
from cot_transparency.data_models.messages import StrictMessageRole
from cot_transparency.json_utils.read_write import (
    read_jsonl_file_into_basemodel,
    write_jsonl_file_from_basemodel,
)

load_dotenv()


def get_renderer_for_model(model: str):
    """Get the appropriate renderer for a model."""
    tokenizer = get_tokenizer(model)
    renderer_name = model_info.get_recommended_renderer_name(model)
    return renderers.get_renderer(renderer_name, tokenizer), tokenizer


class TinkerLoRAConfig(BaseModel):
    """LoRA configuration for Tinker training."""
    rank: int = 32
    train_mlp: bool = True
    train_attn: bool = True
    train_unembed: bool = True
    seed: Optional[int] = None


class TinkerAdamParams(BaseModel):
    """Adam optimizer parameters for Tinker training."""
    learning_rate: float = 1e-5
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0


class TinkerSFTConfig(BaseModel):
    """Full SFT training configuration."""
    model: str = "meta-llama/Llama-3.1-8B-Instruct"
    lora: TinkerLoRAConfig = TinkerLoRAConfig()
    optimizer: TinkerAdamParams = TinkerAdamParams()
    n_epochs: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    save_every_n_steps: Optional[int] = None
    eval_every_n_steps: Optional[int] = None


@dataclass
class TinkerSFTTrainer:
    """
    SFT Trainer using Tinker API.

    Implements supervised fine-tuning with cross-entropy loss,
    compatible with BCT training data format.
    """
    config: TinkerSFTConfig
    service_client: tinker.ServiceClient = field(default=None)
    training_client: tinker.TrainingClient = field(default=None)
    renderer: Any = field(default=None)
    tokenizer: Any = field(default=None)

    def __post_init__(self):
        if self.service_client is None:
            self.service_client = tinker.ServiceClient()

    def setup(self) -> None:
        """Initialize training client and renderer."""
        self.training_client = self.service_client.create_lora_training_client(
            base_model=self.config.model,
            **self.config.lora.model_dump(),
        )
        # Get renderer and tokenizer for this model
        self.renderer, self.tokenizer = get_renderer_for_model(self.config.model)
        print(f"Initialized Tinker training client for {self.config.model}")

    def _messages_to_dict(self, messages: list) -> list[dict]:
        """Convert StrictChatMessage list to dict format for renderer."""
        return [{"role": msg.role.value, "content": msg.content} for msg in messages]

    def _format_chat_to_tokens(self, sample: FinetuneSample) -> tuple[list[int], list[float]]:
        """
        Convert a FinetuneSample to tokens and weights using renderer.

        Returns:
            tokens: Full sequence of token IDs
            weights: Loss weights (0 for prompt, 1 for completion)
        """
        msg_dicts = self._messages_to_dict(sample.messages)
        model_input, weights = self.renderer.build_supervised_example(msg_dicts)
        tokens = model_input.to_ints()
        return tokens, weights

    def _create_datum(self, sample: FinetuneSample) -> types.Datum:
        """Convert a FinetuneSample to Tinker Datum format."""
        tokens, weights = self._format_chat_to_tokens(sample)

        return types.Datum(
            model_input=types.ModelInput.from_ints(tokens=tokens),
            loss_fn_inputs=dict(
                target_tokens=tokens,
                weights=weights,
            )
        )

    def train(
        self,
        samples: Sequence[FinetuneSample],
        val_samples: Optional[Sequence[FinetuneSample]] = None,
        syncer: Optional[WandbSyncer] = None,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> str:
        """
        Run SFT training.

        Args:
            samples: Training samples
            val_samples: Optional validation samples
            syncer: Optional WandB syncer for logging
            progress_callback: Optional callback(step, total_steps, loss)

        Returns:
            Checkpoint name of the trained model
        """
        if self.training_client is None:
            self.setup()

        adam_params = types.AdamParams(**self.config.optimizer.model_dump())

        n_samples = len(samples)
        steps_per_epoch = n_samples // self.config.batch_size
        total_steps = steps_per_epoch * self.config.n_epochs

        print(f"Starting SFT training:")
        print(f"  Samples: {n_samples}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Steps per epoch: {steps_per_epoch}")
        print(f"  Total steps: {total_steps}")

        global_step = 0

        for epoch in range(self.config.n_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.n_epochs}")

            # Shuffle samples each epoch
            import random
            epoch_samples = list(samples)
            random.shuffle(epoch_samples)

            epoch_loss = 0.0
            n_batches = 0

            pbar = tqdm(range(0, n_samples, self.config.batch_size), desc=f"Epoch {epoch+1}")

            for batch_start in pbar:
                batch_end = min(batch_start + self.config.batch_size, n_samples)
                batch_samples = epoch_samples[batch_start:batch_end]

                # Convert to Tinker format
                batch_data = [self._create_datum(s) for s in batch_samples]

                # Forward-backward pass
                fwd_bwd_future = self.training_client.forward_backward(
                    batch_data,
                    loss_fn="cross_entropy"
                )

                # Optimizer step
                optim_future = self.training_client.optim_step(adam_params)

                # Wait for results
                fwd_bwd_result = fwd_bwd_future.result()
                optim_result = optim_future.result()

                # Track loss
                batch_loss = fwd_bwd_result.loss if hasattr(fwd_bwd_result, 'loss') else 0.0
                epoch_loss += batch_loss
                n_batches += 1

                global_step += 1

                pbar.set_postfix({"loss": f"{batch_loss:.4f}"})

                if progress_callback:
                    progress_callback(global_step, total_steps, batch_loss)

                if syncer:
                    syncer.run.log({"train/loss": batch_loss, "train/step": global_step})

                # Save checkpoint if requested
                if (self.config.save_every_n_steps and
                    global_step % self.config.save_every_n_steps == 0):
                    ckpt_name = f"checkpoint-{global_step}"
                    self.training_client.save_weights(name=ckpt_name)
                    print(f"\nSaved checkpoint: {ckpt_name}")

            avg_epoch_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
            print(f"Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}")

            if syncer:
                syncer.run.log({
                    "train/epoch": epoch + 1,
                    "train/epoch_loss": avg_epoch_loss,
                })

        # Save final checkpoint
        final_ckpt = "final-checkpoint"
        self.training_client.save_weights(name=final_ckpt)
        print(f"\nTraining complete. Final checkpoint: {final_ckpt}")

        return final_ckpt

    def save_and_get_sampling_client(self, name: str = "trained") -> tinker.SamplingClient:
        """Save weights and return a sampling client for inference."""
        return self.training_client.save_weights_and_get_sampling_client(name=name)


def finetune_sft_tinker(
    model: str,
    samples: Sequence[FinetuneSample],
    val_samples: Optional[Sequence[FinetuneSample]] = None,
    lora_config: Optional[TinkerLoRAConfig] = None,
    optimizer_config: Optional[TinkerAdamParams] = None,
    n_epochs: int = 1,
    batch_size: int = 4,
    syncer: Optional[WandbSyncer] = None,
    ask_to_validate_training: bool = True,
) -> str:
    """
    Convenience function to run SFT finetuning with Tinker.

    Args:
        model: Base model name (e.g., "meta-llama/Llama-3.1-8B-Instruct")
        samples: Training samples in FinetuneSample format
        val_samples: Optional validation samples
        lora_config: LoRA configuration
        optimizer_config: Adam optimizer configuration
        n_epochs: Number of training epochs
        batch_size: Training batch size
        syncer: Optional WandB syncer
        ask_to_validate_training: Whether to ask for confirmation before training

    Returns:
        Checkpoint name of the trained model
    """
    if ask_to_validate_training:
        print(f"About to train on {len(samples)} samples. Continue? (y/n)")
        response = input()
        if response.lower() != "y":
            print("Training cancelled.")
            return ""

    config = TinkerSFTConfig(
        model=model,
        lora=lora_config or TinkerLoRAConfig(),
        optimizer=optimizer_config or TinkerAdamParams(),
        n_epochs=n_epochs,
        batch_size=batch_size,
    )

    trainer = TinkerSFTTrainer(config=config)
    trainer.setup()

    return trainer.train(samples=samples, val_samples=val_samples, syncer=syncer)


def finetune_sft_tinker_from_file(
    model: str,
    file_path: Path,
    val_file_path: Optional[Path] = None,
    lora_config: Optional[TinkerLoRAConfig] = None,
    optimizer_config: Optional[TinkerAdamParams] = None,
    n_epochs: int = 1,
    batch_size: int = 4,
    syncer: Optional[WandbSyncer] = None,
    ask_to_validate_training: bool = True,
) -> str:
    """Load samples from file and run SFT finetuning."""
    samples = read_jsonl_file_into_basemodel(file_path, FinetuneSample)
    val_samples = None
    if val_file_path:
        val_samples = read_jsonl_file_into_basemodel(val_file_path, FinetuneSample)

    if ask_to_validate_training:
        confirm_to_continue(file_path)

    return finetune_sft_tinker(
        model=model,
        samples=samples,
        val_samples=val_samples,
        lora_config=lora_config,
        optimizer_config=optimizer_config,
        n_epochs=n_epochs,
        batch_size=batch_size,
        syncer=syncer,
        ask_to_validate_training=False,  # Already confirmed
    )


if __name__ == "__main__":
    # Example usage
    from cot_transparency.data_models.messages import StrictChatMessage, StrictMessageRole

    # Create sample data
    samples = [
        FinetuneSample(
            messages=[
                StrictChatMessage(role=StrictMessageRole.user, content="What is 2+2?"),
                StrictChatMessage(role=StrictMessageRole.assistant, content="2+2 equals 4."),
            ]
        )
    ] * 10

    checkpoint = finetune_sft_tinker(
        model="meta-llama/Llama-3.1-8B-Instruct",
        samples=samples,
        n_epochs=1,
        batch_size=2,
    )
    print(f"Trained model checkpoint: {checkpoint}")
