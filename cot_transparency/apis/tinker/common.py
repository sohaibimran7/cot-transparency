"""
Common utilities for Tinker API training modules.

Shared code between SFT and RL training.
"""

from typing import Optional

import tinker
from tinker import types
from tinker_cookbook import renderers, model_info
from tinker_cookbook.tokenizer_utils import get_tokenizer
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# Configuration Classes
# =============================================================================

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


class CheckpointConfig(BaseModel):
    """Configuration for checkpointing during training."""
    save_every_n_steps: Optional[int] = None
    save_full_state: bool = False  # If True, use save_state() for resumability; else save_weights_for_sampler()
    checkpoint_prefix: str = "checkpoint"
    skip_near_final_steps: int = 0  # Skip intermediate checkpoints within N steps of final (avoids duplicates)


def build_checkpoint_name(
    experiment_name: Optional[str],
    checkpoint_prefix: str,
    step: Optional[int] = None,
) -> str:
    """
    Build checkpoint name from experiment_name and prefix.

    Naming convention:
    - Final: {experiment_name}_{prefix} or {prefix}
    - Intermediate: {experiment_name}_{prefix}-{step} or {prefix}-{step}
    """
    base = f"{experiment_name}_{checkpoint_prefix}" if experiment_name else checkpoint_prefix
    return f"{base}-{step}" if step is not None else base


# =============================================================================
# Utility Functions
# =============================================================================

def get_renderer_for_model(model: str):
    """Get the appropriate renderer and tokenizer for a model."""
    tokenizer = get_tokenizer(model)
    renderer_name = model_info.get_recommended_renderer_name(model)
    return renderers.get_renderer(renderer_name, tokenizer), tokenizer


def messages_to_dict(messages: list) -> list[dict]:
    """Convert StrictChatMessage list to dict format for renderer."""
    return [{"role": msg.role.value, "content": msg.content} for msg in messages]


def extract_loss_from_result(fwd_bwd_result: types.ForwardBackwardOutput, batch_size: int = 1) -> float:
    """Extract normalized per-sample loss from ForwardBackwardOutput."""
    total_loss = fwd_bwd_result.metrics.get('loss:sum', 0.0)
    return total_loss / batch_size if batch_size > 0 else total_loss


def save_checkpoint(
    training_client: tinker.TrainingClient,
    name: str,
    save_full_state: bool = False,
) -> str:
    """
    Save a checkpoint.

    Args:
        training_client: Tinker training client
        name: Checkpoint name
        save_full_state: If True, save with optimizer state for resumability

    Returns:
        Checkpoint path
    """
    if save_full_state:
        result = training_client.save_state(name=name).result()
    else:
        result = training_client.save_weights_for_sampler(name=name).result()
    return result.path if hasattr(result, 'path') else name
