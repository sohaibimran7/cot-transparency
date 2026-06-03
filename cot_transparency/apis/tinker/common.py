"""
Common utilities for Tinker API training modules.

Shared code between SFT and RL training.
"""

import logging
import subprocess
from typing import Literal, Optional

from pydantic import BaseModel

from tinker_cookbook import renderers, model_info, hyperparam_utils, checkpoint_utils
from tinker_cookbook.tokenizer_utils import get_tokenizer


_log = logging.getLogger(__name__)


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
    lr_schedule: Literal["constant", "linear", "cosine"] = "linear"  # shared SFT+RL default; train_sft/train_rl/train_evalaware CLIs mirror it
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


# Checkpoint save helpers shared by the SFT (finetune.py) and RL (rl_training.py) loops.
def _checkpoint_path(paths: dict) -> str:
    return paths.get("sampler_path") or paths.get("state_path")


async def save_intermediate_checkpoint(
    training_client,
    *,
    experiment_name: str,
    run_name: str,
    checkpoint_cfg,
    global_step: int,
    total_steps: int,
    epoch: int,
    log_dir,
    checkpoint_paths: list,
    logger,
) -> Optional[str]:
    """Save an intermediate checkpoint when the schedule fires (and we're not near final).

    Returns the checkpoint path, or None if nothing was saved. Appends to `checkpoint_paths`
    and logs `{"checkpoint": path}` as a side effect, matching the prior inline behaviour.
    """
    steps_remaining = total_steps - global_step
    near_final = steps_remaining <= checkpoint_cfg.skip_near_final_steps
    if not (checkpoint_cfg.save_every_n_steps
            and global_step % checkpoint_cfg.save_every_n_steps == 0
            and not near_final):
        return None
    name = build_checkpoint_name(experiment_name, run_name, step=global_step)
    kind = "both" if checkpoint_cfg.save_state else "sampler"
    paths = await checkpoint_utils.save_checkpoint_async(
        training_client,
        name=name,
        log_path=str(log_dir),
        loop_state={"epoch": epoch, "step": global_step},
        kind=kind,
    )
    path = _checkpoint_path(paths)
    checkpoint_paths.append(path)
    logger.log_metrics({"checkpoint": path}, step=global_step)
    return path


async def finalize_checkpoint(
    training_client,
    *,
    experiment_name: str,
    run_name: str,
    n_epochs: int,
    save_state: bool,
    global_step: int,
    log_dir,
    checkpoint_paths: list,
    logger,
) -> str:
    """Save the final (no step-suffix) checkpoint, log it, and close the logger. Returns the path."""
    final_name = build_checkpoint_name(experiment_name, run_name)
    kind = "both" if save_state else "sampler"
    paths = await checkpoint_utils.save_checkpoint_async(
        training_client,
        name=final_name,
        log_path=str(log_dir),
        loop_state={"epoch": n_epochs, "step": global_step, "final": True},
        kind=kind,
    )
    final_path = _checkpoint_path(paths)
    checkpoint_paths.append(final_path)
    print(f"\nTraining complete. Final checkpoint: {final_path}")
    logger.log_metrics({"final_checkpoint": final_path}, step=global_step)
    logger.log_hparams({"final_checkpoint": final_path, "all_checkpoints": checkpoint_paths})
    logger.close()
    return final_path


def get_git_state() -> dict:
    """Capture current git state for reproducibility logging.

    Returns a dict with commit SHA, branch, dirty flag, changed files list,
    and the full diff of uncommitted changes (truncated to 50k chars).
    Degrades gracefully if not in a git repo.
    """
    def _run(args: list[str]) -> str:
        r = subprocess.run(args, capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else ""

    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        if not sha:
            return {"git_error": "not a git repository"}
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        dirty_files = _run(["git", "status", "--short"])
        diff = _run(["git", "diff"])
        max_diff = 50_000
        return {
            "git_sha": sha,
            "git_branch": branch,
            "git_dirty": len(dirty_files) > 0,
            "git_dirty_files": dirty_files,
            "git_diff": diff[:max_diff] + ("\n... (truncated)" if len(diff) > max_diff else ""),
        }
    except Exception as e:
        return {"git_error": str(e)}


def warn_if_dirty(git_state: dict) -> None:
    """Print a prominent warning if the git working tree is dirty."""
    if git_state.get("git_dirty"):
        files = git_state.get("git_dirty_files", "")
        n_files = len([l for l in files.splitlines() if l.strip()])
        print(
            f"\n{'='*60}\n"
            f"WARNING: Git working tree is DIRTY ({n_files} file(s) changed)\n"
            f"Commit: {git_state.get('git_sha', 'unknown')}\n"
            f"Branch: {git_state.get('git_branch', 'unknown')}\n"
            f"Changed files:\n{files}\n"
            f"The diff is logged to WandB for reproducibility.\n"
            f"{'='*60}\n"
        )


def get_recommended_lr(model: str, is_lora: bool = True, fallback: float = 1e-4) -> float:
    """
    Get recommended learning rate for a model using Tinker's hyperparam_utils.

    Falls back to default if model not in hyperparam_utils.
    """
    try:
        return hyperparam_utils.get_lr(model, is_lora=is_lora)
    except Exception as e:
        # get_lr raises ConfigurationError (a ValueError subclass, NOT in the old
        # KeyError/AssertionError/NotImplementedError/OSError tuple) for any model that is
        # neither Llama/Qwen nor in its explicit list — so an arbitrary base model with no
        # explicit --lr used to crash LR resolution instead of falling back. The documented
        # contract is "fall back for unknown models", so catch broadly and warn.
        _log.warning(
            "get_recommended_lr(%s) failed (%s: %s); using fallback lr=%g",
            model, type(e).__name__, e, fallback,
        )
        return fallback
