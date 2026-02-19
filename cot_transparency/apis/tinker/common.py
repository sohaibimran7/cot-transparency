"""
Common utilities for Tinker API training modules.

Shared code between SFT and RL training.
"""

from typing import Optional

from pydantic import BaseModel

from tinker_cookbook import renderers, model_info, hyperparam_utils
from tinker_cookbook.tokenizer_utils import get_tokenizer


# =============================================================================
# Configuration Classes
# =============================================================================

class LoRAConfig(BaseModel):
    """LoRA adapter configuration."""
    rank: int = 32
    train_mlp: bool = True
    train_attn: bool = True
    train_unembed: bool = True
    seed: Optional[int] = None


class AdamConfig(BaseModel):
    """Adam optimizer and learning rate schedule configuration."""
    learning_rate: Optional[float] = None  # None = use get_recommended_lr(model)
    lr_schedule: str = "linear"  # "linear", "cosine", or "constant"
    beta1: float = 0.9
    beta2: float = 0.95  # cookbook default
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0


class CheckpointConfig(BaseModel):
    """Checkpointing configuration."""
    save_every_n_steps: Optional[int] = None
    save_state: bool = False  # If True, save optimizer state for resumability
    skip_near_final_steps: int = 0  # Skip intermediate checkpoints within N steps of final


# =============================================================================
# Utility Functions
# =============================================================================

def get_renderer_and_tokenizer(model: str):
    """
    Get the appropriate renderer and tokenizer for a model.

    The renderer handles chat template formatting and knows how to:
    - Build supervised examples (tokens + loss weights)
    - Build generation prompts
    - Parse responses
    """
    tokenizer = get_tokenizer(model)
    renderer_name = model_info.get_recommended_renderer_name(model)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    return renderer, tokenizer


def build_checkpoint_name(
    experiment_name: str,
    run_name: str,
    step: Optional[int] = None,
) -> str:
    """
    Build checkpoint name from experiment and run names.

    Examples:
        - Final: "bct_debug_control"
        - Intermediate: "bct_debug_control_step100"
    """
    base = f"{experiment_name}_{run_name}"
    return f"{base}_step{step}" if step is not None else base


def build_log_dir(base_dir: str, experiment_name: str, run_name: str) -> str:
    """
    Build log directory path.

    Example: "logs/bct_debug/control/"
    """
    return f"{base_dir}/{experiment_name}/{run_name}"


def get_recommended_lr(model: str, is_lora: bool = True, fallback: float = 1e-4) -> float:
    """
    Get recommended learning rate for a model using Tinker's hyperparam_utils.

    Falls back to default if model not in hyperparam_utils.
    """
    try:
        return hyperparam_utils.get_lr(model, is_lora=is_lora)
    except (KeyError, AssertionError, OSError):
        return fallback
