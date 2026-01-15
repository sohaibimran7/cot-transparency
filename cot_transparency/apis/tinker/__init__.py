"""Tinker API integration for SFT and RL training."""

from cot_transparency.apis.tinker.finetune import (
    TinkerSFTTrainer,
    TinkerSFTConfig,
    TinkerLoRAConfig,
    TinkerAdamParams,
    finetune_sft_tinker,
    finetune_sft_tinker_from_file,
)
from cot_transparency.apis.tinker.rl_training import (
    # Trainer
    TinkerRLTrainer,
    # Configs
    RLConfig,
    RateEstimationConfig,
    TrainingSamplingConfig,
    TrainingLoopConfig,
    GenerationConfig,
    SampleCacheConfig,
    CheckpointConfig,
    CostEstimationConfig,
    # Data classes
    Sample,
    CachedSample,
    # Reward functions
    RewardFunction,
    ConsistencyReward,
    # Cost estimation
    estimate_training_run_cost,
    # Convenience functions
    train_consistency_rl,
)
from cot_transparency.apis.tinker.pricing import (
    ModelPricing,
    TINKER_PRICING,
    get_pricing,
    estimate_sampling_cost,
    estimate_training_cost,
    TrainingCostEstimate,
)
from cot_transparency.apis.tinker.inference import (
    TinkerSamplingClient,
    SamplingConfig as InferenceSamplingConfig,
    SamplingResult,
    sample_from_tinker,
)

__all__ = [
    # SFT
    "TinkerSFTTrainer",
    "TinkerSFTConfig",
    "TinkerLoRAConfig",
    "TinkerAdamParams",
    "finetune_sft_tinker",
    "finetune_sft_tinker_from_file",
    # RL - Trainer
    "TinkerRLTrainer",
    # RL - Configs
    "RLConfig",
    "RateEstimationConfig",
    "TrainingSamplingConfig",
    "TrainingLoopConfig",
    "GenerationConfig",
    "SampleCacheConfig",
    "CheckpointConfig",
    "CostEstimationConfig",
    # RL - Data classes
    "Sample",
    "CachedSample",
    # RL - Reward functions
    "RewardFunction",
    "ConsistencyReward",
    # RL - Cost estimation
    "estimate_training_run_cost",
    # RL - Convenience functions
    "train_consistency_rl",
    # Pricing
    "ModelPricing",
    "TINKER_PRICING",
    "get_pricing",
    "estimate_sampling_cost",
    "estimate_training_cost",
    "TrainingCostEstimate",
    # Inference
    "TinkerSamplingClient",
    "InferenceSamplingConfig",
    "SamplingResult",
    "sample_from_tinker",
]
