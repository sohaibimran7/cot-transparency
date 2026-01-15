"""Tinker API pricing data.

All prices are in USD per million tokens.
Source: Tinker pricing documentation.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelPricing:
    """Pricing for a single model."""
    prefill: float  # USD per million tokens
    sample: float   # USD per million tokens
    train: float    # USD per million tokens


# Pricing data as of 2025
# Format: model_name -> ModelPricing(prefill, sample, train)
TINKER_PRICING: dict[str, ModelPricing] = {
    # Qwen models
    "Qwen/Qwen3-4B-Instruct-2507": ModelPricing(0.07, 0.22, 0.22),
    "Qwen/Qwen3-8B": ModelPricing(0.13, 0.40, 0.40),
    "Qwen/Qwen3-30B-A3B": ModelPricing(0.12, 0.30, 0.36),
    "Qwen/Qwen3-VL-30B-A3B-Instruct": ModelPricing(0.18, 0.44, 0.53),
    "Qwen/Qwen3-32B": ModelPricing(0.49, 1.47, 1.47),
    "Qwen/Qwen3-235B-Instruct-2507": ModelPricing(0.68, 1.70, 2.04),
    "Qwen/Qwen3-VL-235B-A22B-Instruct": ModelPricing(1.02, 2.56, 3.07),

    # Llama models
    "meta-llama/Llama-3.2-1B": ModelPricing(0.03, 0.09, 0.09),
    "meta-llama/Llama-3.2-3B": ModelPricing(0.06, 0.18, 0.18),
    "meta-llama/Llama-3.1-8B": ModelPricing(0.13, 0.40, 0.40),
    "meta-llama/Llama-3.1-8B-Instruct": ModelPricing(0.13, 0.40, 0.40),
    "meta-llama/Llama-3.1-70B": ModelPricing(1.05, 3.16, 3.16),
    "meta-llama/Llama-3.1-70B-Instruct": ModelPricing(1.05, 3.16, 3.16),

    # DeepSeek
    "deepseek-ai/DeepSeek-V3.1": ModelPricing(1.13, 2.81, 3.38),

    # GPT-OSS
    "GPT-OSS-120B": ModelPricing(0.18, 0.44, 0.52),
    "GPT-OSS-20B": ModelPricing(0.12, 0.30, 0.36),

    # Kimi
    "Kimi-K2-Thinking": ModelPricing(0.98, 2.44, 2.93),
}


def get_pricing(model: str) -> Optional[ModelPricing]:
    """Get pricing for a model, or None if not found."""
    return TINKER_PRICING.get(model)


def estimate_sampling_cost(
    model: str,
    n_samples: int,
    avg_prompt_tokens: int,
    avg_response_tokens: int,
) -> Optional[float]:
    """
    Estimate cost for sampling responses.

    Args:
        model: Model name
        n_samples: Number of samples to generate
        avg_prompt_tokens: Average prompt length in tokens
        avg_response_tokens: Average response length in tokens

    Returns:
        Estimated cost in USD, or None if model pricing not found.
    """
    pricing = get_pricing(model)
    if pricing is None:
        return None

    prefill_tokens = n_samples * avg_prompt_tokens
    sample_tokens = n_samples * avg_response_tokens

    prefill_cost = (prefill_tokens / 1_000_000) * pricing.prefill
    sample_cost = (sample_tokens / 1_000_000) * pricing.sample

    return prefill_cost + sample_cost


def estimate_training_cost(
    model: str,
    n_samples: int,
    avg_sequence_tokens: int,
) -> Optional[float]:
    """
    Estimate cost for training (forward-backward pass).

    Args:
        model: Model name
        n_samples: Number of samples in training batch
        avg_sequence_tokens: Average full sequence length (prompt + response)

    Returns:
        Estimated cost in USD, or None if model pricing not found.
    """
    pricing = get_pricing(model)
    if pricing is None:
        return None

    total_tokens = n_samples * avg_sequence_tokens
    return (total_tokens / 1_000_000) * pricing.train


@dataclass
class TrainingCostEstimate:
    """Detailed cost estimate for a training run."""
    # Initial reference rate estimation
    initial_reference_sampling_cost: float

    # Per-epoch costs
    training_sampling_cost_per_epoch: float
    training_gradient_cost_per_epoch: float

    # Totals
    total_sampling_cost: float
    total_training_cost: float
    total_cost: float

    # Metadata
    n_epochs: int
    n_situations: int
    model: str

    def __str__(self) -> str:
        return f"""Training Cost Estimate for {self.model}
{'='*50}
Initial reference rate estimation: ${self.initial_reference_sampling_cost:.4f}

Per epoch:
  - Sampling (rate estimation): ${self.training_sampling_cost_per_epoch:.4f}
  - Training (gradient):        ${self.training_gradient_cost_per_epoch:.4f}

Total ({self.n_epochs} epochs, {self.n_situations} situations):
  - Sampling: ${self.total_sampling_cost:.4f}
  - Training: ${self.total_training_cost:.4f}
  - TOTAL:    ${self.total_cost:.4f}
"""
