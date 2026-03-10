"""
RL Consistency Training via Tinker API.

Implements GRPO rate-matching for consistency training.

Usage:
    from cot_transparency.apis.tinker.rl_training import train_consistency_rl, RLConfig

    # Define perturbation functions that return {"messages": [...]}
    def neutral_prompt(situation: dict) -> dict:
        return {"messages": [{"role": "user", "content": situation["question"]}]}

    def biased_prompt(situation: dict) -> dict:
        return {"messages": [{"role": "user", "content": f"I think the answer is A. {situation['question']}"}]}

    # Define trait classifier
    def classifier(response: str, situation: dict) -> float:
        return 1.0 if "A" in response else 0.0

    checkpoint = asyncio.run(train_consistency_rl(
        model="meta-llama/Llama-3.1-8B-Instruct",
        situations=[{"question": "What is 2+2?"}],
        perturbation_fns=[neutral_prompt, biased_prompt],
        trait_classifier=classifier,
    ))
"""

import asyncio
import hashlib
import json
import logging
import random
import sys
import traceback
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

from tinker_cookbook import checkpoint_utils
from tinker_cookbook.utils.ml_log import setup_logging
from tinker_cookbook.rl.metrics import compute_kl_sample_train, incorporate_kl_penalty
from tinker_cookbook.rl.data_processing import trajectory_to_data
from tinker_cookbook.rl.types import Trajectory, Transition
from tinker_cookbook.rl.train import forward_backward as cookbook_forward_backward, remove_mask
from tinker_cookbook.completers import TokensWithLogprobs

from cot_transparency.apis.tinker.common import (
    LoRAConfig,
    AdamConfig,
    CheckpointConfig,
    get_renderer_and_tokenizer,
    build_checkpoint_name,
    build_log_dir,
    get_recommended_lr,
)
from cot_transparency.apis.tinker.pricing import (
    get_pricing,
    estimate_sampling_cost,
    estimate_training_cost,
    TrainingCostEstimate,
)


_log = logging.getLogger(__name__)


class _SafeFileWrapper:
    """Wraps a file object to silently handle BrokenPipeError.

    When running as a background process, the parent may close its pipe
    (e.g., Claude Code session refresh). tqdm and print calls then raise
    BrokenPipeError. This wrapper swallows those errors so training can
    continue uninterrupted.
    """
    def __init__(self, fp):
        self._fp = fp

    def write(self, s):
        try:
            return self._fp.write(s)
        except BrokenPipeError:
            return 0

    def flush(self):
        try:
            self._fp.flush()
        except BrokenPipeError:
            pass

    def __getattr__(self, name):
        return getattr(self._fp, name)


def _resolve_indices(indices: list[int] | str, n_total: int) -> list[int]:
    if indices == "all":
        return list(range(n_total))
    return list(indices)


# =============================================================================
# Configuration Classes
# =============================================================================

class RateEstimationConfig(BaseModel):
    """Rate estimation config (reference or perturbation rates)."""
    perturbation_indices: list[int] | str = [0]
    n_samples: int = 64
    aggregation: Optional[str] = "mean"

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
    """Training sampling config."""
    perturbation_indices: list[int] | str = [1, 2, 3]
    n_samples_for_rate: int = 64
    n_samples_for_gradient: Optional[int] = 16
    gradient_sample_selection: str = "random"

    def get_indices(self, n_perturbations: int) -> list[int]:
        return _resolve_indices(self.perturbation_indices, n_perturbations)


class TrainingLoopConfig(BaseModel):
    """Training loop config."""
    situations_per_group: int = 1
    gradient_accumulation_steps: int = 1
    refresh_policy_every_n_steps: int = 10
    max_concurrent_base_samples: int = 2  # Max base sampling tasks in flight at once
    n_epochs: int = 1


class GenerationConfig(BaseModel):
    """Generation config."""
    max_new_tokens: int = 8192  # High default - model stops at EOS, avoids truncation
    temperature: float = 0.7


class SampleCacheConfig(BaseModel):
    """Sample caching config (for debugging)."""
    enabled: bool = False
    cache_dir: str = "cache/rl_samples"


class CostEstimationConfig(BaseModel):
    """Cost estimation config."""
    enabled: bool = True
    avg_prompt_tokens: int = 200
    avg_response_tokens: int = 100


class RLConfig(BaseModel):
    """Full RL training configuration."""
    experiment_name: str = "rl"
    run_name: str = "default"
    model: str = "meta-llama/Llama-3.1-8B-Instruct"
    lora: LoRAConfig = LoRAConfig()
    optimizer: AdamConfig = AdamConfig()
    reference_rate: RateEstimationConfig = RateEstimationConfig(perturbation_indices=[0], n_samples=64)
    training: TrainingSamplingConfig = TrainingSamplingConfig(perturbation_indices=[1, 2, 3])
    loop: TrainingLoopConfig = TrainingLoopConfig()
    generation: GenerationConfig = GenerationConfig()
    cache: SampleCacheConfig = SampleCacheConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    cost_estimation: CostEstimationConfig = CostEstimationConfig()
    kl_coef: float = 0.05  # KL penalty coefficient for incorporate_kl_penalty
    kl_discount_factor: float = 0.0  # Discount factor for KL penalty (0.0 = no discounting)
    loss_fn: str = "ppo"
    log_base_dir: str = "logs"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Sample:
    """A sampled response."""
    tokens: list[int]
    logprobs: list[float]
    text: str
    trait_value: float
    perturbation_idx: int
    parsed_successfully: bool = True  # Track if answer was parsed


@dataclass
class SampleResult:
    """Pre-computed result from _collect_samples.
    Rates are computed inside _collect_samples so the full sample data
    can be freed early. Only gradient samples retain tokens/logprobs."""
    grad_samples: list["Sample"]          # Gradient samples (with tokens/logprobs)
    rates: dict[int, float]               # Trait rate per perturbation index
    rate_counts: dict[int, int]           # Number of parsed samples per perturbation
    n_total: int                          # Total raw samples (all perturbations)
    n_parsed: int                         # Parsed samples (all perturbations)


@dataclass
class CachedSample:
    """Cached raw sample (before classification)."""
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
    """Cache for trajectory samples."""

    def __init__(self, cache_dir: str, model: str, temperature: float, max_tokens: int):
        self.cache_dir = Path(cache_dir)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, prompt_content: str, perturbation_id: str) -> Path:
        prompt_hash = hashlib.md5(prompt_content.encode()).hexdigest()[:16]
        model_safe = self.model.replace("/", "_")
        return self.cache_dir / f"{model_safe}/t{self.temperature}_m{self.max_tokens}/{prompt_hash}_{perturbation_id}.json"

    def load(self, prompt_content: str, perturbation_id: str) -> list[CachedSample]:
        path = self._cache_path(prompt_content, perturbation_id)
        if not path.exists():
            return []
        try:
            with open(path) as f:
                return [CachedSample.from_dict(s) for s in json.load(f).get("samples", [])]
        except (json.JSONDecodeError, KeyError):
            return []

    def save(self, prompt_content: str, perturbation_id: str, samples: list[CachedSample]) -> None:
        path = self._cache_path(prompt_content, perturbation_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"samples": [s.to_dict() for s in samples]}, f)

    def get_with_delta(self, prompt_content: str, perturbation_id: str, n_needed: int) -> tuple[list[CachedSample], int]:
        cached = self.load(prompt_content, perturbation_id)
        return cached[:n_needed], max(0, n_needed - len(cached))

    def append(self, prompt_content: str, perturbation_id: str, new_samples: list[CachedSample]) -> None:
        self.save(prompt_content, perturbation_id, self.load(prompt_content, perturbation_id) + new_samples)


# =============================================================================
# Reward Functions
# =============================================================================

class RewardFunction(ABC):
    @abstractmethod
    def compute_rewards(self, samples: list[Sample], p_hat: dict[int, float], p_ref: float, p_ref_initial: float) -> list[float]:
        ...


class ConsistencyReward(RewardFunction):
    """Consistency reward: r = (1-a)*r_consistency + a*r_anchor"""

    def __init__(self, anchor_weight: float = 0.5):
        self.anchor_weight = anchor_weight

    def compute_rewards(self, samples: list[Sample], p_hat: dict[int, float], p_ref: float, p_ref_initial: float) -> list[float]:
        a = self.anchor_weight
        rewards = []
        for s in samples:
            r_consistency = -(p_hat[s.perturbation_idx] - p_ref) * (s.trait_value - p_ref)
            r_anchor = -(p_ref - p_ref_initial) * (s.trait_value - p_ref_initial)
            rewards.append((1 - a) * r_consistency + a * r_anchor)
        return rewards


# =============================================================================
# RL Trainer
# =============================================================================

class RLTrainer:
    """RL Trainer for consistency training."""

    def __init__(self, config: RLConfig, reward_function: Optional[RewardFunction] = None, resume_from: Optional[str] = None):
        self.config = config
        self.reward_function = reward_function or ConsistencyReward()
        self.resume_from = resume_from
        self.service_client = tinker.ServiceClient()
        self.training_client: tinker.TrainingClient | None = None
        self.sampling_client: tinker.SamplingClient | None = None
        self.base_sampling_client: tinker.SamplingClient | None = None
        self.renderer: Any = None
        self.tokenizer: Any = None
        self.sample_cache: Optional[SampleCache] = None

    def setup(self) -> None:
        """Initialize clients and renderer."""
        self.training_client = self.service_client.create_lora_training_client(
            base_model=self.config.model,
            **self.config.lora.model_dump(),
        )

        # Load checkpoint if resuming from a previous training run
        if self.resume_from:
            if "/weights/" in self.resume_from and "/sampler_weights/" not in self.resume_from:
                print(f"Loading full state (weights + optimizer) from: {self.resume_from}")
                self.training_client.load_state_with_optimizer(self.resume_from).result()
            else:
                print(f"Loading weights (optimizer will reset) from: {self.resume_from}")
                self.training_client.load_state(self.resume_from).result()
            print("Checkpoint loaded successfully")

        self.renderer, self.tokenizer = get_renderer_and_tokenizer(self.config.model)
        self.sampling_client = self.training_client.save_weights_and_get_sampling_client(
            name=f"{self.config.experiment_name}_{self.config.run_name}_sampler"
        )
        self.base_sampling_client = self.service_client.create_sampling_client(
            base_model=self.config.model
        )
        if self.config.cache.enabled:
            self.sample_cache = SampleCache(
                self.config.cache.cache_dir,
                self.config.model,
                self.config.generation.temperature,
                self.config.generation.max_new_tokens,
            )

    def _format_prompt(self, messages: list[dict]) -> types.ModelInput:
        """Convert messages to a generation prompt for sampling."""
        return self.renderer.build_generation_prompt(messages)

    async def _sample_from_client(self, client: tinker.SamplingClient, prompt: types.ModelInput, n_samples: int) -> list[CachedSample]:
        result = await client.sample_async(
            prompt=prompt,
            sampling_params=types.SamplingParams(
                max_tokens=self.config.generation.max_new_tokens,
                temperature=self.config.generation.temperature,
                stop=self.renderer.get_stop_sequences(),
            ),
            num_samples=n_samples,
        )

        samples = []
        for seq in result.sequences:
            tokens = list(seq.tokens)
            logprobs = list(seq.logprobs) if seq.logprobs else [0.0] * len(tokens)
            parsed_msg, _ = self.renderer.parse_response(tokens)
            text = parsed_msg.get("content", "") if parsed_msg else self.tokenizer.decode(tokens)
            samples.append(CachedSample(tokens=tokens, logprobs=logprobs, text=text))
        return samples

    async def _collect_samples(
        self,
        situation: dict,
        perturbation_fns: list[Callable[[dict], dict]],
        trait_classifier: Callable[[str, dict], float],
        use_base_model: bool = False,
        answer_parser: Optional[Callable[[str], Optional[str]]] = None,
        rates_only: bool = False,
    ) -> SampleResult:
        """Collect samples and compute rates in one pass.

        Rates are computed internally so the full sample data (tokens/logprobs
        for all 64 samples) can be freed immediately. Only the gradient-selected
        subset retains tokens/logprobs.

        Args:
            rates_only: If True, skip gradient sample selection and return empty
                grad_samples. Used for base sampling where only rates are needed.
        """
        n_perts = len(perturbation_fns)
        training_idx = set(self.config.training.get_indices(n_perts))
        ref_idx = set(self.config.reference_rate.get_indices(n_perts))
        all_idx = training_idx | ref_idx

        n_samples_per = {
            idx: max(
                self.config.training.n_samples_for_rate if idx in training_idx else 0,
                self.config.reference_rate.n_samples if idx in ref_idx else 0,
            )
            for idx in all_idx
        }

        client = self.base_sampling_client if use_base_model else self.sampling_client
        cache_prefix = "base_" if use_base_model else ""

        # Prepare sampling tasks for parallel execution
        async def sample_for_perturbation(idx: int) -> tuple[int, list[Sample]]:
            pert_result = perturbation_fns[idx](situation)
            messages = pert_result["messages"]
            prompt = self._format_prompt(messages)
            prompt_content = json.dumps(messages, sort_keys=True)
            pert_id = f"{cache_prefix}f{idx}"

            cached, n_to_gen = [], n_samples_per[idx]
            if self.sample_cache:
                cached, n_to_gen = self.sample_cache.get_with_delta(prompt_content, pert_id, n_samples_per[idx])

            new_samples = []
            if n_to_gen > 0:
                new_samples = await self._sample_from_client(client, prompt, n_to_gen)
                if self.sample_cache:
                    self.sample_cache.append(prompt_content, pert_id, new_samples)

            samples = []
            for s in cached + new_samples:
                parsed_ok = True
                if answer_parser is not None:
                    parsed_ok = answer_parser(s.text) is not None
                samples.append(Sample(
                    tokens=s.tokens,
                    logprobs=s.logprobs,
                    text=s.text,
                    trait_value=float(trait_classifier(s.text, situation)),
                    perturbation_idx=idx,
                    parsed_successfully=parsed_ok,
                ))
            return idx, samples

        # Run all perturbation sampling in parallel
        results = await asyncio.gather(*[sample_for_perturbation(idx) for idx in all_idx])
        all_samples = dict(results)

        # Compute rates from the full sample set
        rates, rate_counts = self._compute_rates(all_samples, list(all_idx))

        # Count totals before dropping full sample data
        n_total = sum(len(s_list) for s_list in all_samples.values())
        n_parsed = sum(sum(1 for s in s_list if s.parsed_successfully) for s_list in all_samples.values())

        # Select gradient samples (only these keep tokens/logprobs)
        if rates_only:
            grad_samples = []
        else:
            grad_samples = self._select_gradient_samples(all_samples, list(training_idx))

        # all_samples (with full token/logprob data for all 64×n_perts samples)
        # goes out of scope here and can be GC'd immediately.
        return SampleResult(
            grad_samples=grad_samples,
            rates=rates,
            rate_counts=rate_counts,
            n_total=n_total,
            n_parsed=n_parsed,
        )

    def _compute_rates(self, samples: dict[int, list[Sample]], indices: list[int]) -> tuple[dict[int, float], dict[int, int]]:
        """Compute trait rates, only using successfully parsed samples.

        Returns:
            rates: dict mapping perturbation index to trait rate
            counts: dict mapping perturbation index to number of parsed samples used
        """
        rates = {}
        counts = {}
        for idx in indices:
            parsed = [s for s in samples.get(idx, []) if s.parsed_successfully]
            counts[idx] = len(parsed)
            rates[idx] = sum(s.trait_value for s in parsed) / len(parsed) if parsed else 0.5
        return rates, counts

    def _select_gradient_samples(self, samples: dict[int, list[Sample]], training_indices: list[int]) -> list[Sample]:
        n_gradient = self.config.training.n_samples_for_gradient
        selection = self.config.training.gradient_sample_selection
        result = []

        for idx in training_indices:
            # Only use samples that parsed successfully - unparsed samples have unreliable trait values
            pert_samples = [s for s in samples.get(idx, []) if s.parsed_successfully]
            if n_gradient is None or n_gradient >= len(pert_samples):
                result.extend(pert_samples)
            elif selection == "stratified":
                pos = [s for s in pert_samples if s.trait_value == 1]
                neg = [s for s in pert_samples if s.trait_value == 0]
                n_pos = round(n_gradient * len(pos) / len(pert_samples)) if pert_samples else 0
                result.extend(random.sample(pos, min(n_pos, len(pos))))
                result.extend(random.sample(neg, min(n_gradient - n_pos, len(neg))))
            else:
                result.extend(random.sample(pert_samples, n_gradient))
        return result

    def _normalize_advantages(self, rewards: list[float]) -> list[float]:
        if not rewards:
            return rewards
        mean_r = sum(rewards) / len(rewards)
        var = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
        std_r = var ** 0.5
        if std_r < 1e-8:
            return [0.0] * len(rewards)
        return [(r - mean_r) / std_r for r in rewards]

    def _create_rl_datum(self, prompt_input: types.ModelInput, sample: Sample, advantage: float) -> types.Datum:
        """Create RL datum using cookbook's trajectory_to_data for proper token shifting.

        Uses Tinker cookbook's trajectory_to_data which handles token shifting, mask creation,
        and all the proper alignment of logprobs/advantages with target tokens.
        """
        # Create a single-turn trajectory: prompt -> response
        action = TokensWithLogprobs(
            tokens=sample.tokens,
            maybe_logprobs=sample.logprobs,
        )
        transition = Transition(
            ob=prompt_input,
            ac=action,
            reward=0.0,  # We use advantages directly, not per-step rewards
            episode_done=True,
        )
        trajectory = Trajectory(
            transitions=[transition],
            final_ob=types.ModelInput.from_ints(tokens=[]),  # Empty final observation
        )

        # trajectory_to_data handles all the token shifting and mask creation
        datums = trajectory_to_data(trajectory, traj_advantage=advantage)

        # For single-turn, we get exactly one datum
        assert len(datums) == 1, f"Expected 1 datum, got {len(datums)}"
        return datums[0]

    async def estimate_initial_reference_rates(
        self,
        situations: Sequence[dict],
        perturbation_fns: list[Callable[[dict], dict]],
        trait_classifier: Callable[[str, dict], float],
        answer_parser: Optional[Callable[[str], Optional[str]]] = None,
    ) -> dict[int, float]:
        """Estimate initial reference rates from base model (async, parallelized)."""
        print(f"Estimating initial reference rates for {len(situations)} situations...")
        ref_cfg = self.config.reference_rate
        ref_indices = ref_cfg.get_indices(len(perturbation_fns))

        # Track parse stats across all situations
        total_samples = 0
        parsed_samples = 0

        async def estimate_for_situation(sit_idx: int, situation: dict, parser: Optional[Callable[[str], Optional[str]]] = None) -> tuple[int, float, int, int]:
            result = await self._collect_samples(situation, perturbation_fns, trait_classifier,
                                                 use_base_model=True, answer_parser=parser,
                                                 rates_only=True)
            rate_values = [result.rates[idx] for idx in ref_indices if idx in result.rates]
            rate = ref_cfg.aggregate_rates(rate_values) if ref_cfg.aggregation and rate_values else (
                sum(rate_values) / len(rate_values) if rate_values else 0.5
            )
            return sit_idx, rate, result.n_total, result.n_parsed

        # Submit all tasks and let Tinker handle queueing
        results = await asyncio.gather(*[
            estimate_for_situation(sit_idx, situations[sit_idx], answer_parser)
            for sit_idx in range(len(situations))
        ])

        # Aggregate parse stats and report
        rate_dict = {}
        for sit_idx, rate, n_total, n_parsed in results:
            rate_dict[sit_idx] = rate
            total_samples += n_total
            parsed_samples += n_parsed

        if total_samples > 0 and answer_parser is not None:
            parse_rate = parsed_samples / total_samples
            print(f"Initial reference rate estimation parse rate: {parse_rate:.1%} ({parsed_samples}/{total_samples})")
            if parse_rate < 0.8:
                print(f"\n⚠️  WARNING: Low parse rate ({parse_rate:.1%}) during initial reference rate estimation. "
                      f"Consider increasing max_new_tokens (currently {self.config.generation.max_new_tokens}).")

        return rate_dict

    async def train(
        self,
        situations: Sequence[dict],
        perturbation_fns: list[Callable[[dict], dict]],
        trait_classifier: Callable[[str, dict], float],
        initial_reference_rates: Optional[dict[int, float]] = None,
        answer_parser: Optional[Callable[[str], Optional[str]]] = None,
    ) -> str:
        """Run RL consistency training. Returns final checkpoint path.

        Args:
            answer_parser: Optional function to parse answers from responses.
                          If provided, parse rate will be tracked and logged.
                          Should return the parsed answer string or None if parsing failed.
        """
        if self.training_client is None:
            self.setup()

        # Setup logging: logs/{experiment_name}/{run_name}/
        log_dir = Path(build_log_dir(
            self.config.log_base_dir,
            self.config.experiment_name,
            self.config.run_name,
        ))
        log_dir.mkdir(parents=True, exist_ok=True)
        logger = setup_logging(
            log_dir=str(log_dir),
            wandb_project=self.config.experiment_name,
            wandb_name=self.config.run_name,
            config=self.config.model_dump(),
        )

        try:
            return await self._train_loop(
                logger, log_dir, situations, perturbation_fns,
                trait_classifier, initial_reference_rates, answer_parser,
            )
        except Exception:
            tb = traceback.format_exc()
            _log.error("Training failed with exception:\n%s", tb)
            # Log error to WandB so it's visible in the dashboard
            try:
                logger.log_metrics({"train/error": tb}, step=None)
            except Exception:
                pass
            # Ensure WandB marks this run as failed (exit_code=1)
            try:
                import wandb
                if wandb.run is not None:
                    wandb.finish(exit_code=1)
            except Exception:
                pass
            raise

    async def _train_loop(
        self,
        logger,
        log_dir: Path,
        situations: list[dict],
        perturbation_fns: list[Callable],
        trait_classifier: Callable,
        initial_reference_rates: dict | None,
        answer_parser: Callable | None,
    ):
        """Inner training loop, separated for error handling in train()."""
        # Protect stdout/stderr from BrokenPipeError when running as a
        # background process whose parent has closed its pipe.
        sys.stdout = _SafeFileWrapper(sys.stdout)
        sys.stderr = _SafeFileWrapper(sys.stderr)

        n_situations = len(situations)
        n_perts = len(perturbation_fns)
        training_idx = self.config.training.get_indices(n_perts)
        ref_idx = self.config.reference_rate.get_indices(n_perts)

        # Compute initial reference rates upfront from the base model.
        # This ensures every situation has a correct p_ref_init before
        # training begins, avoiding the 0.5 fallback that distorts the
        # anchor reward signal in early steps.
        if initial_reference_rates is None:
            initial_reference_rates = await self.estimate_initial_reference_rates(
                situations, perturbation_fns, trait_classifier, answer_parser
            )
        ref_cfg = self.config.reference_rate

        sit_per_group = self.config.loop.situations_per_group
        n_groups = (n_situations + sit_per_group - 1) // sit_per_group
        total_steps = n_groups * self.config.loop.n_epochs

        # Determine learning rate: use configured value or get recommended LR for model
        base_lr: float = (
            self.config.optimizer.learning_rate
            if self.config.optimizer.learning_rate is not None
            else get_recommended_lr(self.config.model)
        )

        print(f"RL Training: {n_situations} situations, {n_groups} groups, {total_steps} steps, lr={base_lr:.2e}")
        logger.log_hparams({
            "n_situations": n_situations,
            "total_steps": total_steps,
            "n_perturbations": n_perts,
            "base_lr": base_lr,
        })

        checkpoint_paths: list[str] = []
        adam_params = types.AdamParams(
            learning_rate=base_lr,
            beta1=self.config.optimizer.beta1,
            beta2=self.config.optimizer.beta2,
            eps=self.config.optimizer.eps,
        )
        global_step = 0
        accumulated_grads = 0

        for epoch in range(self.config.loop.n_epochs):
            shuffled = list(range(n_situations))
            random.shuffle(shuffled)
            groups = [shuffled[i:i + sit_per_group] for i in range(0, n_situations, sit_per_group)]

            # ── Background base sampling ──────────────────────────────────
            # Base sampling computes p_ref_init (one-time per situation).
            # To avoid competing with policy sampling for GPU time, base
            # tasks are fired during training waits (when the GPU is busy
            # with fwd_bwd) and resolved lazily when their result is needed.
            pending_base_tasks: dict[int, asyncio.Task] = {}

            def _resolve_base_rate(sit_idx: int, base_result: SampleResult):
                """Extract and cache p_ref_init from base model sample result.
                Rates are already computed inside _collect_samples, so this just
                aggregates the reference perturbation rates into a single float."""
                base_rate_vals = [base_result.rates[i] for i in ref_idx if i in base_result.rates]
                initial_reference_rates[sit_idx] = (
                    ref_cfg.aggregate_rates(base_rate_vals) if ref_cfg.aggregation and base_rate_vals
                    else (sum(base_rate_vals) / len(base_rate_vals) if base_rate_vals else 0.5)
                )

            # Target situation list for base watcher — updated as training progresses
            base_target_sits: list[int] = []

            def _drain_completed_base_tasks():
                """Prune completed base tasks and resolve their rates."""
                done_keys = [k for k, t in pending_base_tasks.items() if t.done()]
                for k in done_keys:
                    _resolve_base_rate(k, pending_base_tasks.pop(k).result())

            def _fill_base_slots():
                """Launch base tasks up to the concurrency limit.
                Iterates base_target_sits and fills any free slots."""
                if global_step == 0:
                    return
                max_concurrent = self.config.loop.max_concurrent_base_samples
                for idx in base_target_sits:
                    if len(pending_base_tasks) >= max_concurrent:
                        break
                    if idx not in initial_reference_rates and idx not in pending_base_tasks:
                        pending_base_tasks[idx] = asyncio.create_task(
                            self._collect_samples(situations[idx], perturbation_fns, trait_classifier,
                                                  use_base_model=True, answer_parser=answer_parser,
                                                  rates_only=True)
                        )

            async def _base_watcher():
                """Background task: auto-refills base sampling slots as tasks complete.
                Runs continuously so that during long training awaits, completed base
                tasks are immediately replaced without waiting for the next call site."""
                while True:
                    _drain_completed_base_tasks()
                    _fill_base_slots()
                    await asyncio.sleep(0.001)  # 1ms poll interval

            base_watcher_task: asyncio.Task | None = None

            # Sampling helper (extracted for pipelining)
            async def collect_for_situation(sit_idx: int):
                situation = situations[sit_idx]

                # Step 0: policy == base model, so policy samples give the
                # exact base rate. No extra API call needed.
                need_base = sit_idx not in initial_reference_rates
                if need_base and global_step == 0:
                    result = await self._collect_samples(situation, perturbation_fns, trait_classifier, answer_parser=answer_parser)
                    init_rate_vals = [result.rates[i] for i in ref_idx if i in result.rates]
                    initial_reference_rates[sit_idx] = (
                        ref_cfg.aggregate_rates(init_rate_vals) if ref_cfg.aggregation and init_rate_vals
                        else (sum(init_rate_vals) / len(init_rate_vals) if init_rate_vals else 0.5)
                    )
                else:
                    # Policy-only sampling. Base sampling (if needed) runs
                    # separately during training waits — see _launch_base_tasks.
                    result = await self._collect_samples(situation, perturbation_fns, trait_classifier, answer_parser=answer_parser)

                    # If a base task completed in the background, resolve it now
                    if need_base and sit_idx in pending_base_tasks:
                        base_task = pending_base_tasks.pop(sit_idx)
                        if base_task.done():
                            _resolve_base_rate(sit_idx, base_task.result())
                        else:
                            # Not done yet — will be awaited before reward computation
                            pending_base_tasks[sit_idx] = base_task

                # p_ref_init may not be available yet (base task still running).
                # Return None as placeholder; resolved before reward computation.
                p_ref_init = initial_reference_rates.get(sit_idx, None)
                p_hat = {i: result.rates[i] for i in training_idx if i in result.rates}
                n_hat = {i: result.rate_counts[i] for i in training_idx if i in result.rate_counts}
                ref_rate_vals = [result.rates[i] for i in ref_idx if i in result.rates]
                p_ref = ref_cfg.aggregate_rates(ref_rate_vals) if ref_cfg.aggregation and ref_rate_vals else (
                    sum(ref_rate_vals) / len(ref_rate_vals) if ref_rate_vals else (p_ref_init if p_ref_init is not None else 0.5)
                )
                n_ref_total = sum(result.rate_counts[i] for i in ref_idx if i in result.rate_counts)
                n_hat_total = sum(n_hat.values())
                return (sit_idx, situation, result.grad_samples, p_hat, p_ref, p_ref_init,
                        result.n_total, result.n_parsed, n_ref_total, n_hat_total)

            async def sample_group(group_idx):
                return await asyncio.gather(*[collect_for_situation(sit_idx) for sit_idx in group_idx])

            # Pipelined loop: prefetch up to `refresh_policy_every_n_steps`
            # groups ahead. Samples in the queue are at most queue_size steps
            # off-policy, which never exceeds the refresh interval.
            # We never discard samples — instead we pause prefetching when
            # the queue is full, and refill after each step or refresh.
            max_prefetch = self.config.loop.refresh_policy_every_n_steps or len(groups)
            prefetch_queue: list[asyncio.Task] = []  # tasks producing group_data

            def _fill_prefetch_queue(from_group: int) -> int:
                """Kick off sampling tasks up to the staleness budget."""
                while len(prefetch_queue) < max_prefetch and from_group < len(groups):
                    prefetch_queue.append(
                        asyncio.create_task(sample_group(groups[from_group]))
                    )
                    from_group += 1
                return from_group

            next_to_prefetch = _fill_prefetch_queue(0)

            pbar = tqdm(groups, desc=f"Epoch {epoch + 1}")
            for i_group, group_idx in enumerate(pbar):
                # Pop the next ready group from the queue
                group_data = await prefetch_queue.pop(0)

                # Calculate parse rate across all RAW samples (before filtering)
                total_samples = 0
                parsed_samples = 0
                total_n_ref = 0
                total_n_hat = 0
                for _, _, _, _, _, _, n_total, n_parsed, n_ref, n_hat in group_data:
                    total_samples += n_total
                    parsed_samples += n_parsed
                    total_n_ref += n_ref
                    total_n_hat += n_hat

                parse_rate = parsed_samples / total_samples if total_samples > 0 else 1.0

                # Warn if parse rate is low (but don't log yet - consolidate at end)
                if parse_rate < 0.8:
                    print(f"\n⚠️  WARNING: Low parse rate ({parse_rate:.1%}) at step {global_step + 1}. "
                          f"Consider increasing max_new_tokens (currently {self.config.generation.max_new_tokens}).")

                # Resolve any pending base tasks for situations in this group.
                # Base tasks run during training waits; if one hasn't finished,
                # await it now (rare — training is typically 10x slower).
                resolved_group_data = []
                for row in group_data:
                    sit_idx, situation, grad_samples, p_hat, p_ref, p_ref_init, n_total, n_parsed, n_ref_total, n_hat_total = row
                    if p_ref_init is None:
                        if sit_idx in pending_base_tasks:
                            base_samples = await pending_base_tasks.pop(sit_idx)
                            _resolve_base_rate(sit_idx, base_samples)
                        p_ref_init = initial_reference_rates[sit_idx]
                    resolved_group_data.append((sit_idx, situation, grad_samples, p_hat, p_ref, p_ref_init, n_total, n_parsed, n_ref_total, n_hat_total))
                group_data = resolved_group_data

                # Compute rewards and advantages
                all_rewards, all_grad_data = [], []
                for sit_idx, situation, grad_samples, p_hat, p_ref, p_ref_init, _, _, _, _ in group_data:
                    rewards = self.reward_function.compute_rewards(grad_samples, p_hat, p_ref, p_ref_init)
                    all_rewards.extend(rewards)
                    for sample in grad_samples:
                        pert_result = perturbation_fns[sample.perturbation_idx](situation)
                        prompt = self._format_prompt(pert_result["messages"])
                        all_grad_data.append((prompt, sample))  # Keep as ModelInput, not tokens

                advantages = self._normalize_advantages(all_rewards)

                # Skip empty batches: if all advantages are zero (constant rewards),
                # forward_backward produces zero gradients — waste of compute.
                if not advantages or all(abs(a) < 1e-8 for a in advantages):
                    global_step += 1
                    logger.log_metrics({"train/skipped_empty_batch": 1}, step=global_step)
                    # Still save checkpoints on schedule even when batch is empty
                    ckpt_cfg = self.config.checkpoint
                    steps_remaining = total_steps - global_step
                    near_final = steps_remaining <= ckpt_cfg.skip_near_final_steps
                    if ckpt_cfg.save_every_n_steps and global_step % ckpt_cfg.save_every_n_steps == 0 and not near_final:
                        name = build_checkpoint_name(self.config.experiment_name, self.config.run_name, step=global_step)
                        kind = "both" if ckpt_cfg.save_state else "sampler"
                        paths = await checkpoint_utils.save_checkpoint_async(
                            self.training_client,
                            name=name,
                            log_path=str(log_dir),
                            loop_state={"epoch": epoch, "step": global_step},
                            kind=kind,
                        )
                        checkpoint_path = paths.get("sampler_path") or paths.get("state_path")
                        checkpoint_paths.append(checkpoint_path)
                        logger.log_metrics({"checkpoint": checkpoint_path}, step=global_step)
                    # No training happened so no policy change. Refill queue
                    # since skipped steps don't consume staleness budget.
                    next_to_prefetch = _fill_prefetch_queue(next_to_prefetch)
                    # Update base watcher target — it will auto-fill slots
                    base_target_sits = [s for g in groups[i_group + 1:] for s in g]
                    continue

                # Create training batch using cookbook's trajectory_to_data
                batch_data = [self._create_rl_datum(prompt, s, adv) for (prompt, s), adv in zip(all_grad_data, advantages)]

                # Apply KL penalty from base model (modifies advantages in-place)
                kl_penalty_metrics = {}
                if self.config.kl_coef > 0 and self.base_sampling_client is not None:
                    kl_penalty_metrics = await incorporate_kl_penalty(
                        data_D=batch_data,
                        base_sampling_client=self.base_sampling_client,
                        kl_penalty_coef=self.config.kl_coef,
                        kl_discount_factor=self.config.kl_discount_factor,
                    )

                # Submit forward_backward (removes mask, computes gradients)
                fwd_bwd_future = await self.training_client.forward_backward_async(
                    [remove_mask(d) for d in batch_data], loss_fn=self.config.loss_fn
                )
                accumulated_grads += 1

                # Pipeline: submit optim_step on the SAME clock cycle as fwd_bwd.
                # The server queues optim_step behind fwd_bwd, so both execute in
                # one round-trip instead of two sequential ones.
                optim_future = None
                if accumulated_grads >= self.config.loop.gradient_accumulation_steps:
                    optim_future = await self.training_client.optim_step_async(adam_params)
                    accumulated_grads = 0

                # Refill prefetch queue NOW — while fwd_bwd + optim run on the
                # server, new sampling tasks can execute concurrently.
                next_to_prefetch = _fill_prefetch_queue(next_to_prefetch)

                # Update base watcher target and ensure it's running.
                # The watcher auto-refills base slots as tasks complete, so
                # during long training awaits, freed slots get filled immediately
                # instead of waiting for the next explicit call site.
                base_target_sits = [s for g in groups[i_group + 1:] for s in g]
                if base_watcher_task is None or base_watcher_task.done():
                    base_watcher_task = asyncio.create_task(_base_watcher())

                # Await results — both complete in one clock cycle.
                # During these awaits, the base watcher keeps running and
                # auto-replenishes completed base tasks.
                fwd_bwd_result = await fwd_bwd_future.result_async()
                if optim_future is not None:
                    await optim_future.result_async()

                # Extract training logprobs from fwd_bwd result
                training_logprobs = [
                    output["logprobs"].to_torch()
                    for output in fwd_bwd_result.loss_fn_outputs
                ]

                global_step += 1

                # Compute KL divergence metrics between sampling and training distributions
                kl_sample_train_metrics = compute_kl_sample_train(batch_data, training_logprobs)

                # Rate variance metric - across ALL perturbations (reference + training)
                # This measures consistency: how different are rates between biased and unbiased
                all_rates = []
                for _, _, _, p_hat, p_ref, _, _, _, _, _ in group_data:
                    all_rates.extend(p_hat.values())  # Training perturbation rates
                    all_rates.append(p_ref)  # Reference rate
                if all_rates:
                    mean_rate = sum(all_rates) / len(all_rates)
                    rate_var = sum((r - mean_rate) ** 2 for r in all_rates) / len(all_rates)
                else:
                    rate_var = 0.0

                # Compute average rates across the group
                avg_p_ref = sum(p_ref for _, _, _, _, p_ref, _, _, _, _, _ in group_data) / len(group_data)
                avg_p_ref_init = sum(p_ref_init for _, _, _, _, _, p_ref_init, _, _, _, _ in group_data) / len(group_data)
                avg_p_hat = {}
                for _, _, _, p_hat, _, _, _, _, _, _ in group_data:
                    for pert_idx, rate in p_hat.items():
                        avg_p_hat[pert_idx] = avg_p_hat.get(pert_idx, 0.0) + rate / len(group_data)

                # Compute reward stats
                reward_mean = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
                reward_std = (sum((r - reward_mean)**2 for r in all_rewards) / len(all_rewards)) ** 0.5 if all_rewards else 0.0

                # KL metrics
                kl_v1 = kl_sample_train_metrics.get("optim/kl_sample_train_v1", 0.0)

                # CONSOLIDATED LOGGING - one call with all metrics
                step_metrics = {
                    "train/parse_rate": parse_rate,
                    "train/p_ref": avg_p_ref,
                    "train/p_ref_init": avg_p_ref_init,
                    "train/n_ref": total_n_ref,
                    "train/n_hat": total_n_hat,
                    "train/rate_var": rate_var,
                    "train/reward_mean": reward_mean,
                    "train/reward_std": reward_std,
                    "train/reward_min": min(all_rewards) if all_rewards else 0.0,
                    "train/reward_max": max(all_rewards) if all_rewards else 0.0,
                }
                # Add p_hat for each training perturbation
                for pert_idx, rate in avg_p_hat.items():
                    step_metrics[f"train/p_hat_{pert_idx}"] = rate

                # Add KL metrics from cookbook and penalty
                step_metrics.update(kl_sample_train_metrics)
                for k, v in kl_penalty_metrics.items():
                    step_metrics[f"train/{k}"] = v

                # Single log call for all step metrics
                logger.log_metrics(step_metrics, step=global_step)

                # Update progress bar
                p_hat_display = list(avg_p_hat.values())[0] if avg_p_hat else 0.0
                pbar.set_postfix({"kl": f"{kl_v1:.4f}", "rate_var": f"{rate_var:.4f}", "p_hat": f"{p_hat_display:.2f}"})

                # Refresh policy (update sampling client with current weights)
                if self.config.loop.refresh_policy_every_n_steps and global_step % self.config.loop.refresh_policy_every_n_steps == 0:
                    self.sampling_client = await self.training_client.save_weights_and_get_sampling_client_async(
                        name=f"{self.config.experiment_name}_{self.config.run_name}_sampler_{global_step}"
                    )

                # Intermediate checkpoint (skip if near final to avoid duplicates)
                ckpt_cfg = self.config.checkpoint
                steps_remaining = total_steps - global_step
                near_final = steps_remaining <= ckpt_cfg.skip_near_final_steps

                if ckpt_cfg.save_every_n_steps and global_step % ckpt_cfg.save_every_n_steps == 0 and not near_final:
                    name = build_checkpoint_name(self.config.experiment_name, self.config.run_name, step=global_step)
                    kind = "both" if ckpt_cfg.save_state else "sampler"
                    paths = await checkpoint_utils.save_checkpoint_async(
                        self.training_client,
                        name=name,
                        log_path=str(log_dir),
                        loop_state={"epoch": epoch, "step": global_step},
                        kind=kind,
                    )
                    checkpoint_path = paths.get("sampler_path") or paths.get("state_path")
                    checkpoint_paths.append(checkpoint_path)
                    logger.log_metrics({"checkpoint": checkpoint_path}, step=global_step)

            # End of epoch — cancel the base watcher
            if base_watcher_task is not None and not base_watcher_task.done():
                base_watcher_task.cancel()
                try:
                    await base_watcher_task
                except asyncio.CancelledError:
                    pass
                base_watcher_task = None

        # Final optimizer step for any remaining gradients
        if accumulated_grads > 0:
            optim_future = await self.training_client.optim_step_async(adam_params)
            await optim_future.result_async()

        # Final checkpoint (no step suffix)
        ckpt_cfg = self.config.checkpoint
        final_name = build_checkpoint_name(self.config.experiment_name, self.config.run_name)
        kind = "both" if ckpt_cfg.save_state else "sampler"
        paths = await checkpoint_utils.save_checkpoint_async(
            self.training_client,
            name=final_name,
            log_path=str(log_dir),
            loop_state={"epoch": self.config.loop.n_epochs, "step": global_step, "final": True},
            kind=kind,
        )
        final_path = paths.get("sampler_path") or paths.get("state_path")
        checkpoint_paths.append(final_path)

        print(f"\nTraining complete. Final checkpoint: {final_path}")
        logger.log_metrics({"final_checkpoint": final_path}, step=global_step)
        logger.log_hparams({"final_checkpoint": final_path, "all_checkpoints": checkpoint_paths})
        logger.close()

        return final_path


# =============================================================================
# Cost Estimation
# =============================================================================

def estimate_rl_cost(config: RLConfig, n_situations: int, n_perturbations: int) -> Optional[TrainingCostEstimate]:
    """Estimate training run cost."""
    pricing = get_pricing(config.model)
    if not pricing:
        return None

    avg_prompt = config.cost_estimation.avg_prompt_tokens
    avg_response = config.cost_estimation.avg_response_tokens

    training_idx = config.training.get_indices(n_perturbations)
    ref_idx = config.reference_rate.get_indices(n_perturbations)
    all_idx = set(training_idx) | set(ref_idx)

    n_ref_samples = len(ref_idx) * config.reference_rate.n_samples * n_situations
    initial_ref_cost = estimate_sampling_cost(config.model, n_ref_samples, avg_prompt, avg_response)

    n_samples_per = {
        idx: max(
            config.training.n_samples_for_rate if idx in training_idx else 0,
            config.reference_rate.n_samples if idx in ref_idx else 0,
        )
        for idx in all_idx
    }
    training_sampling_cost = estimate_sampling_cost(config.model, sum(n_samples_per.values()) * n_situations, avg_prompt, avg_response)

    n_gradient = config.training.n_samples_for_gradient or config.training.n_samples_for_rate
    training_gradient_cost = estimate_training_cost(config.model, len(training_idx) * n_gradient * n_situations, avg_prompt + avg_response)

    n_epochs = config.loop.n_epochs
    return TrainingCostEstimate(
        initial_reference_sampling_cost=initial_ref_cost,
        training_sampling_cost_per_epoch=training_sampling_cost,
        training_gradient_cost_per_epoch=training_gradient_cost,
        total_sampling_cost=initial_ref_cost + training_sampling_cost * n_epochs,
        total_training_cost=training_gradient_cost * n_epochs,
        total_cost=initial_ref_cost + training_sampling_cost * n_epochs + training_gradient_cost * n_epochs,
        n_epochs=n_epochs,
        n_situations=n_situations,
        model=config.model,
    )


# =============================================================================
# Convenience Function
# =============================================================================

async def train_consistency_rl(
    model: str,
    situations: Sequence[dict],
    perturbation_fns: list[Callable[[dict], dict]],
    trait_classifier: Callable[[str, dict], float],
    config: Optional[RLConfig] = None,
    reward_function: Optional[RewardFunction] = None,
    initial_reference_rates: Optional[dict[int, float]] = None,
    show_cost_estimate: bool = True,
    answer_parser: Optional[Callable[[str], Optional[str]]] = None,
    resume_from: Optional[str] = None,
) -> str:
    """
    Run RL consistency training.

    Args:
        model: Base model to fine-tune
        situations: List of situation dicts (each containing data for the task)
        perturbation_fns: Functions that transform a situation into {"messages": [...]}
        trait_classifier: Function that scores a response (0.0 or 1.0) for a situation
        config: Full RL config (uses defaults if not provided)
        reward_function: Custom reward function (uses ConsistencyReward if not provided)
        initial_reference_rates: Pre-computed reference rates (estimated if not provided)
        show_cost_estimate: Whether to print cost estimate before training
        answer_parser: Optional function to parse answers from responses.
                      If provided, parse rate will be tracked and logged.
                      Should return the parsed answer string or None if parsing failed.
        resume_from: Tinker checkpoint path to load weights from before training.

    Returns:
        Path to final checkpoint
    """
    cfg = config or RLConfig(model=model)
    if cfg.model != model:
        cfg = cfg.model_copy(update={"model": model})

    if show_cost_estimate and cfg.cost_estimation.enabled:
        cost = estimate_rl_cost(cfg, len(situations), len(perturbation_fns))
        if cost:
            print(cost)

    trainer = RLTrainer(config=cfg, reward_function=reward_function, resume_from=resume_from)
    trainer.setup()
    return await trainer.train(situations, perturbation_fns, trait_classifier, initial_reference_rates, answer_parser)


def train_consistency_rl_sync(
    model: str,
    situations: Sequence[dict],
    perturbation_fns: list[Callable[[dict], dict]],
    trait_classifier: Callable[[str, dict], float],
    config: Optional[RLConfig] = None,
    reward_function: Optional[RewardFunction] = None,
    initial_reference_rates: Optional[dict[int, float]] = None,
    show_cost_estimate: bool = True,
    answer_parser: Optional[Callable[[str], Optional[str]]] = None,
    resume_from: Optional[str] = None,
) -> str:
    """Synchronous wrapper for train_consistency_rl."""
    return asyncio.run(train_consistency_rl(
        model=model,
        situations=situations,
        perturbation_fns=perturbation_fns,
        trait_classifier=trait_classifier,
        config=config,
        reward_function=reward_function,
        initial_reference_rates=initial_reference_rates,
        show_cost_estimate=show_cost_estimate,
        answer_parser=answer_parser,
        resume_from=resume_from,
    ))
