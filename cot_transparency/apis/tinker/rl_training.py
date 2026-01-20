"""
RL Consistency Training via Tinker API.

Implements the GRPO rate-matching algorithm for consistency training.
"""

import hashlib
import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Callable, Any

import tinker
from tinker import types
from tinker.types import TensorData
from pydantic import BaseModel
import torch
from tqdm import tqdm
from dotenv import load_dotenv

from cot_transparency.apis.openai.finetune import FinetuneSample, WandbSyncer
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
from cot_transparency.apis.tinker.pricing import (
    get_pricing,
    estimate_sampling_cost,
    estimate_training_cost,
    TrainingCostEstimate,
)

load_dotenv()


def _resolve_indices(indices: list[int] | str, n_total: int) -> list[int]:
    """Resolve perturbation indices - 'all' or explicit list."""
    if indices == "all":
        return list(range(n_total))
    return list(indices)


# =============================================================================
# Configuration Classes
# =============================================================================

class RateEstimationConfig(BaseModel):
    """Configuration for rate estimation (reference or perturbation rates)."""
    perturbation_indices: list[int] | str = [0]  # Which perturbations to use, or "all"
    n_samples: int = 64  # Samples per perturbation for rate estimation
    aggregation: Optional[str] = "mean"  # How to aggregate rates: "mean", "min", "max", or None

    def get_indices(self, n_perturbations: int) -> list[int]:
        return _resolve_indices(self.perturbation_indices, n_perturbations)

    def aggregate_rates(self, rates: list[float]) -> float:
        if self.aggregation == "mean":
            return sum(rates) / len(rates)
        elif self.aggregation == "min":
            return min(rates)
        elif self.aggregation == "max":
            return max(rates)
        raise ValueError(f"Unknown aggregation: {self.aggregation}")


class TrainingSamplingConfig(BaseModel):
    """Configuration for training sampling."""
    perturbation_indices: list[int] | str = [1, 2, 3]  # Perturbations to train on
    n_samples_for_rate: int = 64  # Samples for estimating current trait rate
    n_samples_for_gradient: Optional[int] = 16  # Subset used for gradient (None = use all)
    gradient_sample_selection: str = "random"  # "random" or "stratified" (preserves rate ratio)

    def get_indices(self, n_perturbations: int) -> list[int]:
        return _resolve_indices(self.perturbation_indices, n_perturbations)


class TrainingLoopConfig(BaseModel):
    """Configuration for the training loop."""
    situations_per_group: int = 1  # Situations to batch before computing rewards/gradients
    gradient_accumulation_steps: int = 1  # Groups to accumulate before optimizer step
    refresh_policy_every_n_steps: int = 10  # How often to sync sampling policy with training weights
    n_epochs: int = 1


class GenerationConfig(BaseModel):
    """Configuration for text generation."""
    max_new_tokens: int = 512
    temperature: float = 0.7


class SampleCacheConfig(BaseModel):
    """Configuration for sample caching (for debugging)."""
    enabled: bool = False
    cache_dir: str = "cache/rl_samples"


class CostEstimationConfig(BaseModel):
    """Configuration for cost estimation."""
    enabled: bool = True
    avg_prompt_tokens: int = 200
    avg_response_tokens: int = 100


class RLConfig(BaseModel):
    """Full RL training configuration."""
    experiment_name: Optional[str] = None  # For grouping runs in WandB
    model: str = "meta-llama/Llama-3.1-8B-Instruct"
    lora: TinkerLoRAConfig = TinkerLoRAConfig()
    optimizer: TinkerAdamParams = TinkerAdamParams()
    reference_rate: RateEstimationConfig = RateEstimationConfig(perturbation_indices=[0], n_samples=64)
    training: TrainingSamplingConfig = TrainingSamplingConfig(perturbation_indices=[1, 2, 3])
    loop: TrainingLoopConfig = TrainingLoopConfig()
    generation: GenerationConfig = GenerationConfig()
    cache: SampleCacheConfig = SampleCacheConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    cost_estimation: CostEstimationConfig = CostEstimationConfig()
    kl_coef: float = 0.05
    loss_fn: str = "ppo"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Sample:
    """A single sampled response."""
    tokens: list[int]
    logprobs: list[float]
    text: str
    trait_value: float
    perturbation_idx: int


@dataclass
class CachedSample:
    """A cached raw sample (before trait classification)."""
    tokens: list[int]
    logprobs: list[float]
    text: str

    def to_dict(self) -> dict:
        return {"tokens": self.tokens, "logprobs": self.logprobs, "text": self.text}

    @classmethod
    def from_dict(cls, d: dict) -> "CachedSample":
        return cls(tokens=d["tokens"], logprobs=d["logprobs"], text=d["text"])


# =============================================================================
# Sample Cache
# =============================================================================

class SampleCache:
    """Cache for trajectory samples to avoid regenerating during debugging."""

    def __init__(self, cache_dir: str, model: str, temperature: float, max_tokens: int):
        self.cache_dir = Path(cache_dir)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, prompt_content: str, perturbation_id: str) -> str:
        prompt_hash = hashlib.md5(prompt_content.encode()).hexdigest()[:16]
        model_safe = self.model.replace("/", "_")
        return f"{model_safe}/t{self.temperature}_m{self.max_tokens}/{prompt_hash}_{perturbation_id}"

    def _get_cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.json"

    def load_samples(self, prompt_content: str, perturbation_id: str) -> list[CachedSample]:
        cache_path = self._get_cache_path(self._get_cache_key(prompt_content, perturbation_id))
        if not cache_path.exists():
            return []
        try:
            with open(cache_path) as f:
                data = json.load(f)
            return [CachedSample.from_dict(s) for s in data.get("samples", [])]
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to load cache {cache_path}: {e}")
            return []

    def save_samples(self, prompt_content: str, perturbation_id: str, samples: list[CachedSample]) -> None:
        cache_key = self._get_cache_key(prompt_content, perturbation_id)
        cache_path = self._get_cache_path(cache_key)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "perturbation_id": perturbation_id,
            "samples": [s.to_dict() for s in samples],
        }
        with open(cache_path, "w") as f:
            json.dump(data, f, indent=2)

    def get_samples_with_delta(self, prompt_content: str, perturbation_id: str, n_needed: int) -> tuple[list[CachedSample], int]:
        cached = self.load_samples(prompt_content, perturbation_id)
        n_cached = len(cached)
        if n_cached > 0:
            print(f"    Cache hit: {n_cached}/{n_needed} samples for {perturbation_id}")
        return cached[:n_needed], max(0, n_needed - n_cached)

    def append_samples(self, prompt_content: str, perturbation_id: str, new_samples: list[CachedSample]) -> None:
        existing = self.load_samples(prompt_content, perturbation_id)
        self.save_samples(prompt_content, perturbation_id, existing + new_samples)


# =============================================================================
# Reward Functions
# =============================================================================

class RewardFunction(ABC):
    """Abstract base class for reward computation."""

    @abstractmethod
    def compute_rewards(
        self,
        samples: list[Sample],
        p_hat: dict[int, float],
        p_ref: float,
        p_ref_initial: float,
    ) -> list[float]:
        """
        Compute rewards for samples.

        Args:
            samples: List of samples with trait values (0 or 1)
            p_hat: Current estimated rates per perturbation {pert_idx: rate}
            p_ref: Current reference rate (may change during training)
            p_ref_initial: Initial reference rate from base model (fixed)

        Returns:
            List of rewards corresponding to each sample
        """
        ...


class ConsistencyReward(RewardFunction):
    """
    Consistency reward for rate matching.

    reward = (1 - anchor_weight) * r_consistency + anchor_weight * r_anchor
    where:
        r_consistency = -(p_hat[f] - p_ref) * (T - p_ref)
        r_anchor = -(p_ref - p_ref_initial) * (T - p_ref_initial)
    """

    def __init__(self, anchor_weight: float = 0.5):
        self.anchor_weight = anchor_weight

    def compute_rewards(
        self,
        samples: list[Sample],
        p_hat: dict[int, float],
        p_ref: float,
        p_ref_initial: float,
    ) -> list[float]:
        a = self.anchor_weight
        rewards = []
        for sample in samples:
            f, T = sample.perturbation_idx, sample.trait_value
            r_consistency = -(p_hat[f] - p_ref) * (T - p_ref)
            r_anchor = -(p_ref - p_ref_initial) * (T - p_ref_initial)
            rewards.append((1 - a) * r_consistency + a * r_anchor)
        return rewards


# =============================================================================
# RL Trainer
# =============================================================================

class TinkerRLTrainer:
    """RL Trainer using Tinker API for consistency training."""

    def __init__(self, config: RLConfig, reward_function: RewardFunction = None):
        self.config = config
        self.reward_function = reward_function or ConsistencyReward()
        self.service_client = tinker.ServiceClient()
        self.training_client: tinker.TrainingClient = None
        self.sampling_client: tinker.SamplingClient = None
        self.base_sampling_client: tinker.SamplingClient = None
        self.renderer: Any = None
        self.tokenizer: Any = None
        self.sample_cache: Optional[SampleCache] = None

    def setup(self) -> None:
        """Initialize training client, sampling client, renderer, and cache."""
        self.training_client = self.service_client.create_lora_training_client(
            base_model=self.config.model,
            **self.config.lora.model_dump(),
        )
        self.renderer, self.tokenizer = get_renderer_for_model(self.config.model)

        # Sampling client for rollouts (will be refreshed during training)
        self.sampling_client = self.training_client.save_weights_and_get_sampling_client(
            name=f"{self.config.checkpoint.checkpoint_prefix}_sampler"
        )

        # Base model sampling client for initial reference rate estimation
        self.base_sampling_client = self.service_client.create_sampling_client(
            base_model=self.config.model
        )

        if self.config.cache.enabled:
            self.sample_cache = SampleCache(
                cache_dir=self.config.cache.cache_dir,
                model=self.config.model,
                temperature=self.config.generation.temperature,
                max_tokens=self.config.generation.max_new_tokens,
            )
            print(f"Sample cache enabled: {self.config.cache.cache_dir}")

        print(f"Initialized Tinker RL training client for {self.config.model}")

    def _format_prompt(self, messages: list) -> types.ModelInput:
        """Convert messages to ModelInput for generation."""
        return self.renderer.build_generation_prompt(messages_to_dict(messages))

    def _sample_from_client(self, client: tinker.SamplingClient, prompt: types.ModelInput, n_samples: int) -> list[CachedSample]:
        """Sample responses from a given client."""
        result = client.sample(
            prompt=prompt,
            sampling_params=types.SamplingParams(
                max_tokens=self.config.generation.max_new_tokens,
                temperature=self.config.generation.temperature,
                stop=self.renderer.get_stop_sequences(),
            ),
            num_samples=n_samples,
        ).result()

        samples = []
        for seq in result.sequences:
            tokens = list(seq.tokens)
            logprobs = list(seq.logprobs) if seq.logprobs else [0.0] * len(tokens)
            parsed_msg, _ = self.renderer.parse_response(tokens)
            text = parsed_msg.get("content", "") if parsed_msg else self.tokenizer.decode(tokens)
            samples.append(CachedSample(tokens=tokens, logprobs=logprobs, text=text))
        return samples

    def _collect_samples_for_step(
        self,
        situation: dict,
        perturbation_fns: list[Callable[[dict], FinetuneSample]],
        trait_classifier: Callable[[str, dict], float],
        use_base_model: bool = False,
    ) -> dict[int, list[Sample]]:
        """
        Collect all samples needed for one training step.

        Samples from the union of training and reference perturbations to avoid
        redundant computation when they overlap (e.g., if perturbation 0 is used
        for both reference and training).
        """
        n_perturbations = len(perturbation_fns)
        training_indices = set(self.config.training.get_indices(n_perturbations))
        reference_indices = set(self.config.reference_rate.get_indices(n_perturbations))
        all_indices = training_indices | reference_indices

        # Use max samples needed across training/reference to avoid re-sampling
        n_samples_per_pert = {}
        for idx in all_indices:
            n_training = self.config.training.n_samples_for_rate if idx in training_indices else 0
            n_reference = self.config.reference_rate.n_samples if idx in reference_indices else 0
            n_samples_per_pert[idx] = max(n_training, n_reference)

        client = self.base_sampling_client if use_base_model else self.sampling_client
        cache_prefix = "base_" if use_base_model else ""

        all_samples: dict[int, list[Sample]] = {}

        for idx in all_indices:
            n_needed = n_samples_per_pert[idx]
            perturb_fn = perturbation_fns[idx]
            perturbation_id = f"{cache_prefix}f{idx}"

            sample_obj = perturb_fn(situation)
            prompt = self._format_prompt(sample_obj.messages)
            prompt_content = json.dumps(messages_to_dict(sample_obj.messages), sort_keys=True)

            # Check cache
            cached_samples, n_to_generate = [], n_needed
            if self.sample_cache is not None:
                cached_samples, n_to_generate = self.sample_cache.get_samples_with_delta(
                    prompt_content, perturbation_id, n_needed
                )

            # Generate additional samples if needed
            new_samples = []
            if n_to_generate > 0:
                new_samples = self._sample_from_client(client, prompt, n_to_generate)
                if self.sample_cache is not None:
                    self.sample_cache.append_samples(prompt_content, perturbation_id, new_samples)

            # Combine and classify
            classified_samples = []
            for s in cached_samples + new_samples:
                classified_samples.append(Sample(
                    tokens=s.tokens,
                    logprobs=s.logprobs,
                    text=s.text,
                    trait_value=float(trait_classifier(s.text, situation)),
                    perturbation_idx=idx,
                ))
            all_samples[idx] = classified_samples

        return all_samples

    def _compute_rates(self, samples: dict[int, list[Sample]], perturbation_indices: list[int]) -> dict[int, float]:
        """Compute rates for specified perturbations from samples."""
        rates = {}
        for idx in perturbation_indices:
            if idx in samples:
                trait_values = [s.trait_value for s in samples[idx]]
                rates[idx] = sum(trait_values) / len(trait_values) if trait_values else 0.5
            else:
                rates[idx] = 0.5
        return rates

    def _select_gradient_samples(self, samples: dict[int, list[Sample]], training_indices: list[int]) -> list[Sample]:
        """Select subset of samples for gradient computation."""
        n_gradient = self.config.training.n_samples_for_gradient
        selection = self.config.training.gradient_sample_selection
        gradient_samples = []

        for idx in training_indices:
            pert_samples = samples.get(idx, [])
            if n_gradient is None or n_gradient >= len(pert_samples):
                gradient_samples.extend(pert_samples)
            elif selection == "stratified":
                positive = [s for s in pert_samples if s.trait_value == 1]
                negative = [s for s in pert_samples if s.trait_value == 0]
                n_pos = round(n_gradient * len(positive) / len(pert_samples)) if pert_samples else 0
                gradient_samples.extend(random.sample(positive, min(n_pos, len(positive))))
                gradient_samples.extend(random.sample(negative, min(n_gradient - n_pos, len(negative))))
            else:
                gradient_samples.extend(random.sample(pert_samples, n_gradient))
        return gradient_samples

    def _normalize_advantages(self, rewards: list[float]) -> list[float]:
        """Normalize rewards to get advantages (GRPO style)."""
        if not rewards:
            return rewards
        mean_r = sum(rewards) / len(rewards)
        std_r = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5
        if std_r < 1e-8:
            return [0.0] * len(rewards)
        return [(r - mean_r) / std_r for r in rewards]

    def _compute_kl_penalty(self, data: list[tuple[list[int], Sample, float]]) -> list[float]:
        """
        Compute KL penalty for each sample using base model logprobs.

        Returns adjusted advantages: advantage + kl_coef * (avg_kl - sample_kl)
        This penalizes samples that deviate more from the base model than average,
        encouraging the policy to stay close to the base model distribution.
        """
        if self.config.kl_coef <= 0:
            return [adv for _, _, adv in data]

        kl_diffs = []
        for prompt_tokens, sample, _ in data:
            full_tokens = prompt_tokens + sample.tokens
            base_logprobs = self.base_sampling_client.compute_logprobs(
                types.ModelInput.from_ints(tokens=full_tokens)
            ).result()

            # KL diff = current_logprob - base_logprob (for response tokens only)
            n_response = len(sample.tokens)
            base_response_logprobs = base_logprobs[-n_response:]
            sample_kl_diff = sum(curr - base for curr, base in zip(sample.logprobs, base_response_logprobs))
            kl_diffs.append(sample_kl_diff)

        avg_kl_diff = sum(kl_diffs) / len(kl_diffs) if kl_diffs else 0.0
        return [adv + self.config.kl_coef * (avg_kl_diff - kl) for (_, _, adv), kl in zip(data, kl_diffs)]

    def _create_rl_datum(self, prompt_tokens: list[int], sample: Sample, advantage: float) -> types.Datum:
        """Create a Tinker Datum for RL training."""
        full_tokens = prompt_tokens + sample.tokens
        full_logprobs = [0.0] * len(prompt_tokens) + sample.logprobs
        full_advantages = [0.0] * len(prompt_tokens) + [advantage] * len(sample.tokens)

        return types.Datum(
            model_input=types.ModelInput.from_ints(tokens=full_tokens),
            loss_fn_inputs={
                "target_tokens": TensorData.from_torch(torch.tensor(full_tokens)),
                "logprobs": TensorData.from_torch(torch.tensor(full_logprobs)),
                "advantages": TensorData.from_torch(torch.tensor(full_advantages)),
            }
        )

    def estimate_initial_reference_rates(
        self,
        situations: Sequence[dict],
        perturbation_fns: list[Callable[[dict], FinetuneSample]],
        trait_classifier: Callable[[str, dict], float],
    ) -> dict[int, float]:
        """Estimate initial reference rates from base model."""
        print("Estimating initial reference rates from base model...")
        ref_config = self.config.reference_rate
        ref_indices = ref_config.get_indices(len(perturbation_fns))
        initial_rates = {}

        for sit_idx, situation in enumerate(tqdm(situations, desc="Initial reference rates")):
            samples = self._collect_samples_for_step(situation, perturbation_fns, trait_classifier, use_base_model=True)
            pert_rates = self._compute_rates(samples, ref_indices)
            rate_values = [pert_rates[idx] for idx in ref_indices if idx in pert_rates]

            if ref_config.aggregation and rate_values:
                initial_rates[sit_idx] = ref_config.aggregate_rates(rate_values)
            else:
                initial_rates[sit_idx] = sum(rate_values) / len(rate_values) if rate_values else 0.5

        return initial_rates

    def train(
        self,
        situations: Sequence[dict],
        perturbation_fns: list[Callable[[dict], FinetuneSample]],
        trait_classifier: Callable[[str, dict], float],
        initial_reference_rates: Optional[dict[int, float]] = None,
        syncer: Optional[WandbSyncer] = None,
        progress_callback: Optional[Callable[[int, int, dict], None]] = None,
    ) -> str:
        """Run RL consistency training."""
        if self.training_client is None:
            self.setup()

        n_situations = len(situations)
        n_perturbations = len(perturbation_fns)
        training_indices = self.config.training.get_indices(n_perturbations)
        reference_indices = self.config.reference_rate.get_indices(n_perturbations)

        if initial_reference_rates is None:
            initial_reference_rates = self.estimate_initial_reference_rates(situations, perturbation_fns, trait_classifier)

        situations_per_group = self.config.loop.situations_per_group
        n_groups = (n_situations + situations_per_group - 1) // situations_per_group

        print(f"\nStarting RL consistency training{f' [{self.config.experiment_name}]' if self.config.experiment_name else ''}:")
        print(f"  Model: {self.config.model}, Situations: {n_situations}, Groups: {n_groups}")
        print(f"  Training perturbations: {training_indices}, Reference: {reference_indices}")
        print(f"  Samples for rate: {self.config.training.n_samples_for_rate}, for gradient: {self.config.training.n_samples_for_gradient}")
        print(f"  KL coef: {self.config.kl_coef}, Epochs: {self.config.loop.n_epochs}")

        # Log config to WandB
        if syncer:
            syncer.update_parameters_with_dict({
                "training_type": "rl_consistency",
                "experiment_name": self.config.experiment_name,
                "config": self.config.model_dump(),
                "n_situations": n_situations,
                "n_perturbations": n_perturbations,
                "reward_function": type(self.reward_function).__name__,
            })

        checkpoint_paths: list[str] = []  # Track all saved checkpoints

        adam_params = types.AdamParams(**self.config.optimizer.model_dump())
        total_steps = n_groups * self.config.loop.n_epochs
        global_step = 0
        accumulated_grads = 0

        for epoch in range(self.config.loop.n_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.loop.n_epochs}")

            shuffled_indices = list(range(n_situations))
            random.shuffle(shuffled_indices)
            situation_groups = [shuffled_indices[i:i + situations_per_group] for i in range(0, n_situations, situations_per_group)]

            epoch_metrics = {"loss": 0.0, "rate_variance": 0.0, "n_steps": 0}
            pbar = tqdm(situation_groups, desc=f"Epoch {epoch + 1}")

            for group_indices in pbar:
                # === STEP 1: Collect samples and compute rates ===
                # For each situation, sample from current policy and estimate trait rates
                group_samples = []
                for sit_idx in group_indices:
                    situation = situations[sit_idx]
                    p_ref_initial = initial_reference_rates.get(sit_idx, 0.5)

                    samples = self._collect_samples_for_step(situation, perturbation_fns, trait_classifier, use_base_model=False)
                    p_hat = self._compute_rates(samples, training_indices)

                    ref_rates = self._compute_rates(samples, reference_indices)
                    rate_values = [ref_rates[idx] for idx in reference_indices if idx in ref_rates]
                    p_ref = self.config.reference_rate.aggregate_rates(rate_values) if self.config.reference_rate.aggregation and rate_values else (sum(rate_values) / len(rate_values) if rate_values else p_ref_initial)

                    gradient_samples = self._select_gradient_samples(samples, training_indices)
                    group_samples.append((sit_idx, situation, gradient_samples, p_hat, p_ref, p_ref_initial))

                # === STEP 2: Compute rewards using consistency reward function ===
                # Rewards encourage matching the reference rate across perturbations
                all_rewards, all_gradient_data = [], []
                for sit_idx, situation, gradient_samples, p_hat, p_ref, p_ref_initial in group_samples:
                    rewards = self.reward_function.compute_rewards(gradient_samples, p_hat, p_ref, p_ref_initial)
                    all_rewards.extend(rewards)
                    for sample in gradient_samples:
                        prompt = self._format_prompt(perturbation_fns[sample.perturbation_idx](situation).messages)
                        all_gradient_data.append((prompt.to_ints(), sample, situation))

                # === STEP 3: Normalize advantages (GRPO style) and apply KL penalty ===
                advantages = self._normalize_advantages(all_rewards)
                if self.config.kl_coef > 0:
                    advantages = self._compute_kl_penalty([(pt, s, adv) for (pt, s, _), adv in zip(all_gradient_data, advantages)])

                # === STEP 4: Create training data and run forward-backward ===
                batch_data = [self._create_rl_datum(pt, s, adv) for (pt, s, _), adv in zip(all_gradient_data, advantages)]
                fwd_bwd_future = self.training_client.forward_backward(batch_data, loss_fn=self.config.loss_fn)
                accumulated_grads += 1

                # === STEP 5: Optimizer step (after gradient accumulation) ===
                if accumulated_grads >= self.config.loop.gradient_accumulation_steps:
                    self.training_client.optim_step(adam_params).result()
                    accumulated_grads = 0

                # Get metrics
                fwd_bwd_result = fwd_bwd_future.result()
                batch_loss = extract_loss_from_result(fwd_bwd_result, len(batch_data))

                all_p_hats = [p_hat for _, _, _, p_hat, _, _ in group_samples]
                avg_rate_variance = sum(
                    sum((r - sum(p.values()) / len(p)) ** 2 for r in p.values()) / len(p) if p else 0
                    for p in all_p_hats
                ) / len(all_p_hats) if all_p_hats else 0
                avg_p_ref = sum(p_ref for _, _, _, _, p_ref, _ in group_samples) / len(group_samples)

                epoch_metrics["loss"] += batch_loss
                epoch_metrics["rate_variance"] += avg_rate_variance
                epoch_metrics["n_steps"] += 1
                global_step += 1

                pbar.set_postfix({"loss": f"{batch_loss:.4f}", "rate_var": f"{avg_rate_variance:.4f}", "p_ref": f"{avg_p_ref:.3f}"})

                if progress_callback:
                    progress_callback(global_step, total_steps, {"loss": batch_loss, "rate_variance": avg_rate_variance, "p_ref": avg_p_ref})

                if syncer:
                    syncer.run.log({"train/loss": batch_loss, "train/rate_variance": avg_rate_variance, "train/p_ref": avg_p_ref, "train/step": global_step})

                # Refresh policy periodically
                refresh_interval = self.config.loop.refresh_policy_every_n_steps
                if refresh_interval and global_step % refresh_interval == 0:
                    self.sampling_client = self.training_client.save_weights_and_get_sampling_client(
                        name=f"{self.config.checkpoint.checkpoint_prefix}_sampler_{global_step}"
                    )

                # Save checkpoint
                save_interval = self.config.checkpoint.save_every_n_steps
                if save_interval and global_step % save_interval == 0:
                    ckpt_name = build_checkpoint_name(self.config.experiment_name, self.config.checkpoint.checkpoint_prefix, step=global_step)
                    ckpt_path = save_checkpoint(self.training_client, ckpt_name, self.config.checkpoint.save_full_state)
                    checkpoint_paths.append(ckpt_path)
                    print(f"\nSaved checkpoint: {ckpt_path}")
                    if syncer:
                        syncer.run.log({"checkpoint/path": ckpt_path, "checkpoint/step": global_step})

            # Epoch summary
            n = epoch_metrics["n_steps"]
            print(f"Epoch {epoch + 1}: avg_loss={epoch_metrics['loss'] / n:.4f}, avg_rate_variance={epoch_metrics['rate_variance'] / n:.4f}")
            if syncer:
                syncer.run.log({"train/epoch": epoch + 1, "train/epoch_loss": epoch_metrics["loss"] / n, "train/epoch_rate_variance": epoch_metrics["rate_variance"] / n})

        # Final
        if accumulated_grads > 0:
            self.training_client.optim_step(adam_params).result()

        final_ckpt_name = build_checkpoint_name(self.config.experiment_name, self.config.checkpoint.checkpoint_prefix)
        final_ckpt_path = save_checkpoint(self.training_client, final_ckpt_name, self.config.checkpoint.save_full_state)
        checkpoint_paths.append(final_ckpt_path)
        print(f"\nTraining complete. Final checkpoint: {final_ckpt_path}")

        # Log all checkpoint paths to WandB config for easy lookup
        if syncer:
            syncer.run.config.update({
                "final_checkpoint_path": final_ckpt_path,
                "all_checkpoint_paths": checkpoint_paths,
            })

        return final_ckpt_path


# =============================================================================
# Cost Estimation
# =============================================================================

def estimate_training_run_cost(config: RLConfig, n_situations: int, n_perturbations: int) -> Optional[TrainingCostEstimate]:
    """Estimate the cost of a training run."""
    pricing = get_pricing(config.model)
    if pricing is None:
        print(f"Warning: No pricing data for model {config.model}")
        return None

    avg_prompt = config.cost_estimation.avg_prompt_tokens
    avg_response = config.cost_estimation.avg_response_tokens

    training_indices = config.training.get_indices(n_perturbations)
    reference_indices = config.reference_rate.get_indices(n_perturbations)
    all_indices = set(training_indices) | set(reference_indices)

    # Initial reference rate estimation
    n_ref_samples = len(reference_indices) * config.reference_rate.n_samples * n_situations
    initial_ref_cost = estimate_sampling_cost(config.model, n_ref_samples, avg_prompt, avg_response)

    # Per-situation sampling
    n_samples_per_pert = {
        idx: max(
            config.training.n_samples_for_rate if idx in training_indices else 0,
            config.reference_rate.n_samples if idx in reference_indices else 0,
        )
        for idx in all_indices
    }
    training_sampling_cost = estimate_sampling_cost(config.model, sum(n_samples_per_pert.values()) * n_situations, avg_prompt, avg_response)

    # Gradient samples
    n_gradient = config.training.n_samples_for_gradient or config.training.n_samples_for_rate
    training_gradient_cost = estimate_training_cost(config.model, len(training_indices) * n_gradient * n_situations, avg_prompt + avg_response)

    n_epochs = config.loop.n_epochs
    total_sampling = initial_ref_cost + training_sampling_cost * n_epochs
    total_training = training_gradient_cost * n_epochs

    return TrainingCostEstimate(
        initial_reference_sampling_cost=initial_ref_cost,
        training_sampling_cost_per_epoch=training_sampling_cost,
        training_gradient_cost_per_epoch=training_gradient_cost,
        total_sampling_cost=total_sampling,
        total_training_cost=total_training,
        total_cost=total_sampling + total_training,
        n_epochs=n_epochs,
        n_situations=n_situations,
        model=config.model,
    )


# =============================================================================
# Convenience Function
# =============================================================================

def train_consistency_rl(
    model: str,
    situations: Sequence[dict],
    perturbation_fns: list[Callable[[dict], FinetuneSample]],
    trait_classifier: Callable[[str, dict], float],
    experiment_name: str = "consistency_rl",
    n_epochs: int = 1,
    reference_rate_config: Optional[RateEstimationConfig] = None,
    training_config: Optional[TrainingSamplingConfig] = None,
    loop_config: Optional[TrainingLoopConfig] = None,
    generation_config: Optional[GenerationConfig] = None,
    lora_config: Optional[TinkerLoRAConfig] = None,
    optimizer_config: Optional[TinkerAdamParams] = None,
    cache_config: Optional[SampleCacheConfig] = None,
    checkpoint_config: Optional[CheckpointConfig] = None,
    cost_estimation_config: Optional[CostEstimationConfig] = None,
    reward_function: Optional[RewardFunction] = None,
    initial_reference_rates: Optional[dict[int, float]] = None,
    syncer: Optional[WandbSyncer] = None,
) -> str:
    """Convenience function to run RL consistency training."""
    final_loop_config = loop_config or TrainingLoopConfig()
    final_loop_config = TrainingLoopConfig(
        situations_per_group=final_loop_config.situations_per_group,
        gradient_accumulation_steps=final_loop_config.gradient_accumulation_steps,
        refresh_policy_every_n_steps=final_loop_config.refresh_policy_every_n_steps,
        n_epochs=n_epochs,
    )

    # Default checkpoint prefix from experiment_name if not provided
    final_checkpoint_config = checkpoint_config or CheckpointConfig()
    final_checkpoint_config = CheckpointConfig(
        save_every_n_steps=final_checkpoint_config.save_every_n_steps,
        save_full_state=final_checkpoint_config.save_full_state,
        checkpoint_prefix=final_checkpoint_config.checkpoint_prefix,
    )

    config = RLConfig(
        experiment_name=experiment_name,
        model=model,
        lora=lora_config or TinkerLoRAConfig(),
        optimizer=optimizer_config or TinkerAdamParams(),
        reference_rate=reference_rate_config or RateEstimationConfig(),
        training=training_config or TrainingSamplingConfig(),
        loop=final_loop_config,
        generation=generation_config or GenerationConfig(),
        cache=cache_config or SampleCacheConfig(),
        checkpoint=final_checkpoint_config,
        cost_estimation=cost_estimation_config or CostEstimationConfig(),
    )

    if config.cost_estimation.enabled:
        cost_estimate = estimate_training_run_cost(config, len(situations), len(perturbation_fns))
        if cost_estimate:
            print(cost_estimate)
            if syncer:
                syncer.update_parameters_with_dict({"cost_estimate": cost_estimate.__dict__})

    trainer = TinkerRLTrainer(config=config, reward_function=reward_function)
    trainer.setup()

    return trainer.train(
        situations=situations,
        perturbation_fns=perturbation_fns,
        trait_classifier=trait_classifier,
        initial_reference_rates=initial_reference_rates,
        syncer=syncer,
    )
