"""
RL Consistency Training via Tinker API.

Implements GRPO rate-matching for consistency training.

Usage:
    from cot_transparency.apis.tinker.rl_training import RLConfig, RLTrainer

    config = RLConfig(model="meta-llama/Llama-3.1-8B-Instruct", experiment_name="rl", run_name="test")
    trainer = RLTrainer(config=config)
    trainer.setup()
    checkpoint = asyncio.run(trainer.train(
        datapoints=[{"question": "What is 2+2?"}],
        perturbation_fns=[neutral_prompt, biased_prompt],
        trait_classifier=classifier,
    ))
"""

import asyncio
from collections import defaultdict, deque
import json
import logging
import random
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Callable, Any

import tinker
from tinker import types
from pydantic import BaseModel
import torch
from tqdm import tqdm

from tinker_cookbook import checkpoint_utils
from tinker_cookbook.utils.ml_log import setup_logging
from tinker_cookbook.rl.metrics import compute_kl_sample_train, incorporate_kl_penalty
from tinker_cookbook.rl.data_processing import trajectory_to_data
from tinker_cookbook.rl.types import Trajectory, Transition
from tinker_cookbook.rl.train import remove_mask
from tinker_cookbook.completers import TokensWithLogprobs

from cot_transparency.apis.tinker.common import (
    LoRAConfig,
    AdamConfig,
    CheckpointConfig,
    get_renderer_and_tokenizer,
    build_checkpoint_name,
    build_log_dir,
    get_recommended_lr,
    get_git_state,
    warn_if_dirty,
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


def _select_rollouts(rollouts: dict[int, list[Rollout]], indices: list[int], n_gradient: int | None) -> list[Rollout]:
    """Select up to n_gradient parsed rollouts per perturbation index."""
    result = []
    for idx in indices:
        parsed = [r for r in rollouts.get(idx, []) if r.parsed_successfully]
        if n_gradient is None or n_gradient >= len(parsed):
            result.extend(parsed)
        else:
            result.extend(random.sample(parsed, n_gradient))
    return result


# =============================================================================
# Configuration Classes
# =============================================================================

class RateEstimationConfig(BaseModel):
    """Rate estimation config (reference or perturbation rates)."""
    perturbation_indices: list[int] | str = [0]
    n_rollouts: int = 64
    aggregation: Optional[str] = "mean"


class TrainingSamplingConfig(BaseModel):
    """Training sampling config."""
    perturbation_indices: list[int] | str = [1, 2, 3]
    n_rollouts_for_rate: int = 64
    n_rollouts_for_consistency: Optional[int] = 16   # Consistency gradient rollouts (None = all parsed)
    n_rollouts_for_anchor: Optional[int] = None      # Anchor gradient rollouts (None = all parsed)


class TrainingLoopConfig(BaseModel):
    """Training loop config."""
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    refresh_policy_every_n_steps: int = 10
    n_epochs: int = 1


class GenerationConfig(BaseModel):
    """Generation config."""
    max_new_tokens: int = 8192
    temperature: float = 0.7


class RLConfig(BaseModel):
    """Full RL training configuration."""
    experiment_name: str = "rl"
    run_name: str = "default"
    model: str = "meta-llama/Llama-3.1-8B-Instruct"
    lora: LoRAConfig = LoRAConfig()
    optimizer: AdamConfig = AdamConfig()
    reference_rate: RateEstimationConfig = RateEstimationConfig(perturbation_indices=[0], n_rollouts=64)
    training: TrainingSamplingConfig = TrainingSamplingConfig(perturbation_indices=[1, 2, 3])
    loop: TrainingLoopConfig = TrainingLoopConfig()
    generation: GenerationConfig = GenerationConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    kl_coef: float = 0.05
    kl_discount_factor: float = 0.0
    loss_fn: str = "ppo"
    anchor_weight: float = 0.5
    log_base_dir: str = "logs"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Rollout:
    """A single rollout (sampled response)."""
    tokens: list[int]
    logprobs: list[float]
    text: str
    trait_value: float
    perturbation_idx: int
    parsed_successfully: bool = True
    prompt: Optional[types.ModelInput] = None


@dataclass
class RolloutResult:
    """Pre-computed result from _collect_rollouts.
    Rates are computed internally so the full rollout data (tokens/logprobs
    for all rollouts) can be freed early. Only gradient rollouts retained."""
    train_rollouts: list["Rollout"]         # Training perturbation gradient rollouts
    anchor_rollouts: list["Rollout"]        # Reference perturbation gradient rollouts (for anchor)
    rates: dict[int, float | None]          # Trait rate per perturbation index (None if 0 parsed)
    rate_counts: dict[int, int]             # Number of parsed rollouts per perturbation
    n_total: int                            # Total raw rollouts (all perturbations)
    n_parsed: int                           # Parsed rollouts (all perturbations)


@dataclass
class BatchItem:
    """One datapoint's rollout results, ready for reward computation."""
    datapoint_idx: int
    datapoint: dict
    train_rollouts: list[Rollout]
    anchor_rollouts: list[Rollout]
    p_hat: dict[int, float]         # Per-perturbation trait rates (training)
    p_ref: float                    # Reference perturbation rate
    p_ref_init: float | None        # Initial (base/anchor) reference rate
    n_total: int                    # Total raw rollouts
    n_parsed: int                   # Parsed rollouts
    n_ref_parsed: int               # Parsed ref rollouts
    n_training_parsed: int          # Parsed training rollouts


# =============================================================================
# Reward
# =============================================================================

class ConsistencyReward:
    """Consistency reward for training perturbation rollouts.

    Pushes p_hat toward p_ref (privileged) using variance-optimal baseline p_hat.
    r = -(p_hat - p_ref) * (trait - p_hat)

    Anchor reward is computed separately on reference perturbation rollouts.
    """

    def compute_rewards(self, rollouts: list[Rollout], p_hat: dict[int, float], p_ref: float) -> list[float]:
        """Consistency-only rewards for training perturbation rollouts.

        r = -(p_hat[pert] - p_ref) * (trait - p_hat[pert])
        """
        return [-(p_hat[r.perturbation_idx] - p_ref) * (r.trait_value - p_hat[r.perturbation_idx])
                for r in rollouts]

    def compute_anchor_rewards(self, ref_rollouts: list[Rollout], p_ref: float, p_ref_initial: float) -> list[float]:
        """Anchor rewards for reference perturbation rollouts.

        r = -(p_ref - p_ref_initial) * (trait - p_ref)
        """
        return [-(p_ref - p_ref_initial) * (r.trait_value - p_ref)
                for r in ref_rollouts]


# =============================================================================
# RL Trainer
# =============================================================================

class RLTrainer:
    """RL Trainer for consistency training."""

    def __init__(self, config: RLConfig, reward_function: Optional[ConsistencyReward] = None, resume_from: Optional[str] = None, resume_with_optimizer: bool = False):
        self.config = config
        self.reward_function = reward_function or ConsistencyReward()
        self.resume_from = resume_from
        self.resume_with_optimizer = resume_with_optimizer
        self.service_client = tinker.ServiceClient()
        self.training_client: tinker.TrainingClient | None = None
        self.sampling_client: tinker.SamplingClient | None = None
        self.base_sampling_client: tinker.SamplingClient | None = None
        self.renderer: Any = None
        self.tokenizer: Any = None

    def setup(self) -> None:
        """Initialize clients and renderer."""
        self.training_client = self.service_client.create_lora_training_client(
            base_model=self.config.model,
            **self.config.lora.model_dump(),
        )

        if self.resume_from:
            if self.resume_with_optimizer:
                print(f"Loading weights + optimizer from: {self.resume_from}")
                self.training_client.load_state_with_optimizer(self.resume_from).result()
            else:
                print(f"Loading weights from: {self.resume_from}")
                self.training_client.load_state(self.resume_from).result()
            print("Checkpoint loaded successfully")

        self.renderer, self.tokenizer = get_renderer_and_tokenizer(self.config.model)
        self.sampling_client = self.training_client.save_weights_and_get_sampling_client(
            name=f"{self.config.experiment_name}_{self.config.run_name}_sampler"
        )
        self.base_sampling_client = self.service_client.create_sampling_client(
            base_model=self.config.model
        )

    async def _sample_from_client(self, client: tinker.SamplingClient, prompt: types.ModelInput, n_samples: int) -> list[tuple[list[int], list[float], str]]:
        """Sample from a client and return (tokens, logprobs, text) tuples."""
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
            if seq.logprobs:
                logprobs = list(seq.logprobs)
            else:
                _log.warning("Missing logprobs for sample, using zeros — KL penalty will be inaccurate")
                logprobs = [0.0] * len(tokens)
            parsed_msg, _ = self.renderer.parse_response(tokens)
            text = parsed_msg.get("content", "") if parsed_msg else self.tokenizer.decode(tokens)
            samples.append((tokens, logprobs, text))
        return samples

    async def _collect_rollouts(
        self,
        datapoint: dict,
        perturbation_fns: list[Callable[[dict], dict]],
        trait_classifier: Callable[[str, dict], float],
        use_base_model: bool = False,
        answer_parser: Optional[Callable[[str], Optional[str]]] = None,
        rates_only: bool = False,
    ) -> RolloutResult:
        """Collect rollouts and compute rates in one pass.

        Rates are computed internally so the full rollout data (tokens/logprobs
        for all rollouts) can be freed immediately. Only the gradient-selected
        subset retains tokens/logprobs.

        Args:
            rates_only: If True, skip gradient rollout selection and return empty
                rollout lists. Used for base sampling where only rates are needed.
        """
        n_perts = len(perturbation_fns)
        training_idx = set(_resolve_indices(self.config.training.perturbation_indices, n_perts))
        ref_idx = set(_resolve_indices(self.config.reference_rate.perturbation_indices, n_perts))
        all_idx = training_idx | ref_idx

        n_rollouts_per = {
            idx: max(
                self.config.training.n_rollouts_for_rate if idx in training_idx else 0,
                self.config.reference_rate.n_rollouts if idx in ref_idx else 0,
            )
            for idx in all_idx
        }

        client = self.base_sampling_client if use_base_model else self.sampling_client

        async def rollout_perturbation(idx: int) -> tuple[int, list[Rollout]]:
            pert_result = perturbation_fns[idx](datapoint)
            messages = pert_result["messages"]
            prompt = self.renderer.build_generation_prompt(messages)

            raw = await self._sample_from_client(client, prompt, n_rollouts_per[idx])

            rollouts = []
            for tokens, logprobs, text in raw:
                parsed_ok = answer_parser(text) is not None if answer_parser else True
                rollouts.append(Rollout(
                    tokens=tokens,
                    logprobs=logprobs,
                    text=text,
                    trait_value=float(trait_classifier(text, datapoint)),
                    perturbation_idx=idx,
                    parsed_successfully=parsed_ok,
                    prompt=prompt,
                ))
            return idx, rollouts

        results = await asyncio.gather(*[rollout_perturbation(idx) for idx in all_idx])
        all_rollouts = dict(results)

        rates, rate_counts = self._compute_rates(all_rollouts, list(all_idx))

        n_total = sum(len(r_list) for r_list in all_rollouts.values())
        n_parsed = sum(sum(1 for r in r_list if r.parsed_successfully) for r_list in all_rollouts.values())

        if rates_only:
            train_rollouts = []
            anchor_rollouts = []
        else:
            train_rollouts = _select_rollouts(all_rollouts, list(training_idx), self.config.training.n_rollouts_for_consistency)
            anchor_rollouts = _select_rollouts(all_rollouts, list(ref_idx), self.config.training.n_rollouts_for_anchor)

        return RolloutResult(
            train_rollouts=train_rollouts,
            anchor_rollouts=anchor_rollouts,
            rates=rates,
            rate_counts=rate_counts,
            n_total=n_total,
            n_parsed=n_parsed,
        )

    def _compute_rates(self, rollouts: dict[int, list[Rollout]], indices: list[int]) -> tuple[dict[int, float | None], dict[int, int]]:
        """Compute trait rates from parsed rollouts only."""
        rates: dict[int, float | None] = {}
        counts = {}
        for idx in indices:
            parsed = [r for r in rollouts.get(idx, []) if r.parsed_successfully]
            counts[idx] = len(parsed)
            rates[idx] = sum(r.trait_value for r in parsed) / len(parsed) if parsed else None
        return rates, counts

    def _aggregate_ref_rates(self, rates: dict[int, float | None], ref_idx: list[int]) -> float | None:
        """Aggregate reference perturbation rates into a single scalar."""
        valid: list[float] = [rates[i] for i in ref_idx if i in rates and rates[i] is not None]  # type: ignore[misc]
        if not valid:
            return None
        agg = self.config.reference_rate.aggregation or "mean"
        if agg == "mean":
            return sum(valid) / len(valid)
        elif agg == "min":
            return min(valid)
        elif agg == "max":
            return max(valid)
        raise ValueError(f"Unknown aggregation: {agg}")

    def _normalize_advantages(self, rewards: list[float]) -> list[float]:
        if not rewards:
            return rewards
        mean_r = sum(rewards) / len(rewards)
        var = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
        std_r = var ** 0.5
        if std_r < 1e-8:
            return [0.0] * len(rewards)
        return [(r - mean_r) / std_r for r in rewards]

    def _create_rl_datum(self, prompt_input: types.ModelInput, rollout: Rollout, advantage: float) -> types.Datum:
        """Create RL datum using cookbook's trajectory_to_data for proper token shifting."""
        action = TokensWithLogprobs(
            tokens=rollout.tokens,
            maybe_logprobs=rollout.logprobs,
        )
        transition = Transition(
            ob=prompt_input,
            ac=action,
            reward=0.0,
            episode_done=True,
        )
        trajectory = Trajectory(
            transitions=[transition],
            final_ob=types.ModelInput.from_ints(tokens=[]),
        )
        datums = trajectory_to_data(trajectory, traj_advantage=advantage)
        assert len(datums) == 1, f"Expected 1 datum, got {len(datums)}" #change for multi-turn
        return datums[0]

    def _build_training_batch(
        self,
        batch_items: list[BatchItem],
    ) -> tuple[list, list[float], list[float], list[float], list[tuple]] | None:
        """Compute rewards, normalize advantages, and build training data.

        Returns (grad_datums, consistency_rewards, anchor_rewards, advantages, policy_grad_data)
        or None if the batch should be skipped (empty/zero advantages).
        """
        anchor_weight = self.config.anchor_weight

        # Consistency: training perturbation rollouts
        consistency_rewards, consistency_data = [], []
        for item in batch_items:
            rewards = self.reward_function.compute_rewards(item.train_rollouts, item.p_hat, item.p_ref)
            consistency_rewards.extend(rewards)
            for rollout in item.train_rollouts:
                consistency_data.append((rollout.prompt, rollout))

        # Anchor: reference perturbation rollouts
        anchor_rewards, anchor_data = [], []
        if anchor_weight > 0:
            for item in batch_items:
                rewards = self.reward_function.compute_anchor_rewards(item.anchor_rollouts, item.p_ref, item.p_ref_init)
                anchor_rewards.extend(rewards)
                for rollout in item.anchor_rollouts:
                    anchor_data.append((rollout.prompt, rollout))

        # Normalize each population separately, then scale by weight
        consistency_adv = self._normalize_advantages(consistency_rewards)
        anchor_adv = self._normalize_advantages(anchor_rewards)
        consistency_adv = [a * (1 - anchor_weight) for a in consistency_adv]
        anchor_adv = [a * anchor_weight for a in anchor_adv]

        all_rewards = consistency_rewards + anchor_rewards
        policy_grad_data = consistency_data + anchor_data
        advantages = consistency_adv + anchor_adv

        # Skip empty batches
        if not advantages or all(abs(a) < 1e-8 for a in advantages):
            return None

        grad_datums = [self._create_rl_datum(prompt, r, adv) for (prompt, r), adv in zip(policy_grad_data, advantages)]
        return grad_datums, consistency_rewards, anchor_rewards, advantages, policy_grad_data

    async def train(
        self,
        datapoints: Sequence[dict],
        perturbation_fns: list[Callable[[dict], dict]],
        trait_classifier: Callable[[str, dict], float],
        initial_reference_rates: Optional[dict[int, float]] = None,
        answer_parser: Optional[Callable[[str], Optional[str]]] = None,
    ) -> str:
        """Run RL consistency training. Returns final checkpoint path."""
        if self.training_client is None:
            self.setup()

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

        git_state = get_git_state()
        warn_if_dirty(git_state)
        logger.log_hparams({"git": git_state})

        try:
            return await self._train_loop(
                logger, log_dir, datapoints, perturbation_fns,
                trait_classifier, initial_reference_rates, answer_parser,
            )
        except Exception:
            tb = traceback.format_exc()
            _log.error("Training failed with exception:\n%s", tb)
            try:
                logger.log_metrics({"train/error": tb}, step=None)
            except Exception:
                pass
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
        datapoints: list[dict],
        perturbation_fns: list[Callable],
        trait_classifier: Callable,
        initial_reference_rates: dict | None,
        answer_parser: Callable | None,
    ):
        """Inner training loop."""
        original_stdout, original_stderr = sys.stdout, sys.stderr
        sys.stdout = _SafeFileWrapper(sys.stdout)
        sys.stderr = _SafeFileWrapper(sys.stderr)

        try:
            return await self._train_loop_inner(
                logger, log_dir, datapoints, perturbation_fns,
                trait_classifier, initial_reference_rates, answer_parser,
            )
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    async def _train_loop_inner(
        self,
        logger,
        log_dir: Path,
        datapoints: list[dict],
        perturbation_fns: list[Callable],
        trait_classifier: Callable,
        initial_reference_rates: dict | None,
        answer_parser: Callable | None,
    ):
        n_datapoints = len(datapoints)
        n_perts = len(perturbation_fns)
        training_idx = _resolve_indices(self.config.training.perturbation_indices, n_perts)
        ref_idx = _resolve_indices(self.config.reference_rate.perturbation_indices, n_perts)

        if initial_reference_rates is None:
            initial_reference_rates = {}
        need_p_ref_init = self.config.anchor_weight > 0

        batch_size = self.config.loop.batch_size
        n_steps_per_epoch = (n_datapoints + batch_size - 1) // batch_size
        total_steps = n_steps_per_epoch * self.config.loop.n_epochs

        base_lr: float = (
            self.config.optimizer.learning_rate
            if self.config.optimizer.learning_rate is not None
            else get_recommended_lr(self.config.model)
        )

        print(f"RL Training: {n_datapoints} datapoints, {n_steps_per_epoch} steps/epoch, {total_steps} total steps, lr={base_lr:.2e}")
        logger.log_hparams({
            "n_datapoints": n_datapoints,
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
            shuffled = list(range(n_datapoints))
            random.shuffle(shuffled)
            batches = [shuffled[i:i + batch_size] for i in range(0, n_datapoints, batch_size)]

            # ── Sampling helpers ──────────────────────────────────────────

            async def collect_for_datapoint(dp_idx: int) -> BatchItem:
                dp = datapoints[dp_idx]

                # Step 0: policy == base model, so policy rollouts give base rate too
                need_base = need_p_ref_init and dp_idx not in initial_reference_rates
                step_snapshot = global_step  # Capture before await; prefetched coroutines read this after yield
                result = await self._collect_rollouts(dp, perturbation_fns, trait_classifier, answer_parser=answer_parser)

                if need_base and step_snapshot == 0:
                    # At step 0, policy IS base model — extract p_ref_init directly
                    p_ref_init_val = self._aggregate_ref_rates(result.rates, ref_idx)
                    if p_ref_init_val is not None:
                        initial_reference_rates[dp_idx] = p_ref_init_val

                p_ref_init = initial_reference_rates.get(dp_idx, None)
                p_hat = {i: result.rates[i] for i in training_idx
                         if i in result.rates and result.rates[i] is not None}
                training_counts = {i: result.rate_counts[i] for i in training_idx if i in result.rate_counts}
                p_ref = self._aggregate_ref_rates(result.rates, ref_idx)
                if p_ref is None:
                    _log.warning("All ref rollouts failed to parse for datapoint %d, falling back to p_ref_init=%s", dp_idx, p_ref_init)
                    p_ref = p_ref_init  # fallback

                n_ref_parsed = sum(result.rate_counts[i] for i in ref_idx if i in result.rate_counts)
                n_training_parsed = sum(training_counts.values())

                return BatchItem(
                    datapoint_idx=dp_idx,
                    datapoint=dp,
                    train_rollouts=result.train_rollouts,
                    anchor_rollouts=result.anchor_rollouts,
                    p_hat=p_hat,
                    p_ref=p_ref,
                    p_ref_init=p_ref_init,
                    n_total=result.n_total,
                    n_parsed=result.n_parsed,
                    n_ref_parsed=n_ref_parsed,
                    n_training_parsed=n_training_parsed,
                )

            async def sample_batch(batch_indices):
                return await asyncio.gather(*[collect_for_datapoint(idx) for idx in batch_indices])

            # ── Prefetch queue ────────────────────────────────────────────
            # Pipeline: prefetch next step's sampling while current step's
            # fwd_bwd runs on the server (~5-7s overlap).
            max_prefetch = self.config.loop.refresh_policy_every_n_steps or len(batches)
            prefetch_queue: deque[asyncio.Task] = deque()

            def _fill_prefetch_queue(from_batch: int) -> int:
                while len(prefetch_queue) < max_prefetch and from_batch < len(batches):
                    prefetch_queue.append(asyncio.create_task(sample_batch(batches[from_batch])))
                    from_batch += 1
                return from_batch

            next_to_prefetch = _fill_prefetch_queue(0)

            pbar = tqdm(batches, desc=f"Epoch {epoch + 1}")
            for i_batch, batch_indices in enumerate(pbar):
                batch_items: list[BatchItem] = await prefetch_queue.popleft()

                # Parse rate
                total_samples = sum(b.n_total for b in batch_items)
                parsed_samples = sum(b.n_parsed for b in batch_items)
                total_n_ref_parsed = sum(b.n_ref_parsed for b in batch_items)
                total_n_training_parsed = sum(b.n_training_parsed for b in batch_items)
                parse_rate = parsed_samples / total_samples if total_samples > 0 else 1.0

                if parse_rate < 0.8:
                    print(f"\n⚠️  Low parse rate ({parse_rate:.1%}) at step {global_step + 1}")

                # ── Resolve p_ref_init for anchor (if needed) ─────────────
                resolved_items = []
                for item in batch_items:
                    if item.p_ref_init is None and need_p_ref_init:
                        # Launch on-demand base sampling
                        if item.datapoint_idx not in initial_reference_rates:
                            print(f"  ⏳ Base rate missing for datapoint {item.datapoint_idx}, sampling now...")
                            base_result = await self._collect_rollouts(
                                datapoints[item.datapoint_idx], perturbation_fns, trait_classifier,
                                use_base_model=True, answer_parser=answer_parser, rates_only=True,
                            )
                            p_ref_init_val = self._aggregate_ref_rates(base_result.rates, ref_idx)
                            if p_ref_init_val is None:
                                print(f"  ⚠️  Could not compute base rate for datapoint {item.datapoint_idx}, skipping")
                                continue
                            initial_reference_rates[item.datapoint_idx] = p_ref_init_val
                        item.p_ref_init = initial_reference_rates[item.datapoint_idx]

                    if item.p_ref is None:
                        if item.p_ref_init is not None:
                            item.p_ref = item.p_ref_init
                        else:
                            print(f"  ⚠️  No parsed ref rollouts for datapoint {item.datapoint_idx}, skipping")
                            continue
                    resolved_items.append(item)
                batch_items = resolved_items

                # ── Compute rewards and advantages ────────────────────────
                batch_result = self._build_training_batch(batch_items)
                if batch_result is None:
                    global_step += 1
                    logger.log_metrics({"train/skipped_empty_batch": 1}, step=global_step)
                    await self._maybe_save_checkpoint(global_step, total_steps, epoch, log_dir, checkpoint_paths, logger)
                    next_to_prefetch = _fill_prefetch_queue(next_to_prefetch)
                    continue

                grad_datums, consistency_rewards, anchor_rewards, advantages, policy_grad_data = batch_result
                all_rewards = consistency_rewards + anchor_rewards

                # KL penalty
                kl_penalty_metrics = {}
                if self.config.kl_coef > 0 and self.base_sampling_client is not None:
                    kl_penalty_metrics = await incorporate_kl_penalty(
                        data_D=grad_datums,
                        base_sampling_client=self.base_sampling_client,
                        kl_penalty_coef=self.config.kl_coef,
                        kl_discount_factor=self.config.kl_discount_factor,
                    )

                # Submit fwd_bwd
                fwd_bwd_future = await self.training_client.forward_backward_async(
                    [remove_mask(d) for d in grad_datums], loss_fn=self.config.loss_fn
                )
                accumulated_grads += 1

                # Pipeline: submit optim_step immediately (server queues behind fwd_bwd)
                optim_future = None
                if accumulated_grads >= self.config.loop.gradient_accumulation_steps:
                    optim_future = await self.training_client.optim_step_async(adam_params)
                    accumulated_grads = 0

                # Refill prefetch queue while fwd_bwd runs on server
                next_to_prefetch = _fill_prefetch_queue(next_to_prefetch)

                # Await training results
                fwd_bwd_result = await fwd_bwd_future.result_async()
                if optim_future is not None:
                    await optim_future.result_async()

                training_logprobs = [
                    output["logprobs"].to_torch()
                    for output in fwd_bwd_result.loss_fn_outputs
                ]

                global_step += 1

                # ── Logging ───────────────────────────────────────────────
                fwd_bwd_metrics = {}
                if hasattr(fwd_bwd_result, 'metrics') and fwd_bwd_result.metrics:
                    fwd_bwd_metrics = {f"train/{k}": v for k, v in fwd_bwd_result.metrics.items()}

                self._log_step_metrics(
                    logger, global_step, epoch, batch_items, grad_datums,
                    consistency_rewards, anchor_rewards, advantages,
                    policy_grad_data, all_rewards, training_logprobs,
                    {**kl_penalty_metrics, **fwd_bwd_metrics},
                    parse_rate, total_n_ref_parsed, total_n_training_parsed,
                    training_idx, need_p_ref_init, pbar,
                )

                # Refresh policy
                if self.config.loop.refresh_policy_every_n_steps and global_step % self.config.loop.refresh_policy_every_n_steps == 0:
                    self.sampling_client = await self.training_client.save_weights_and_get_sampling_client_async(
                        name=f"{self.config.experiment_name}_{self.config.run_name}_sampler_{global_step}"
                    )
                    # Flush stale prefetch tasks that sampled from the old policy
                    for task in prefetch_queue:
                        task.cancel()
                    prefetch_queue.clear()
                    next_to_prefetch = _fill_prefetch_queue(i_batch + 1)

                # Intermediate checkpoint
                await self._maybe_save_checkpoint(global_step, total_steps, epoch, log_dir, checkpoint_paths, logger)

        # Final optimizer step for remaining gradients
        if accumulated_grads > 0:
            optim_future = await self.training_client.optim_step_async(adam_params)
            await optim_future.result_async()

        # Final checkpoint
        final_name = build_checkpoint_name(self.config.experiment_name, self.config.run_name)
        kind = "both" if self.config.checkpoint.save_state else "sampler"
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

    async def _maybe_save_checkpoint(self, global_step, total_steps, epoch, log_dir, checkpoint_paths, logger):
        """Save intermediate checkpoint if schedule says so."""
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

    def _log_step_metrics(
        self, logger, global_step, epoch, batch_items, grad_datums,
        consistency_rewards, anchor_rewards, advantages,
        policy_grad_data, all_rewards, training_logprobs,
        kl_penalty_metrics, parse_rate, total_n_ref_parsed, total_n_training_parsed,
        training_idx, need_p_ref_init, pbar,
    ):
        """Compute and log all metrics for a training step."""
        kl_sample_train_metrics = compute_kl_sample_train(grad_datums, training_logprobs)

        # Rate variance (consistency measure)
        all_rates = []
        for item in batch_items:
            all_rates.extend(item.p_hat.values())
            all_rates.append(item.p_ref)
        rate_var = 0.0
        if all_rates:
            mean_rate = sum(all_rates) / len(all_rates)
            rate_var = sum((r - mean_rate) ** 2 for r in all_rates) / len(all_rates)

        avg_p_ref = sum(item.p_ref for item in batch_items) / len(batch_items)
        avg_p_ref_init = (
            sum(item.p_ref_init for item in batch_items) / len(batch_items)
            if need_p_ref_init else None
        )
        avg_p_hat = {}
        for item in batch_items:
            for pert_idx, rate in item.p_hat.items():
                avg_p_hat[pert_idx] = avg_p_hat.get(pert_idx, 0.0) + rate / len(batch_items)

        cons_reward_mean = sum(consistency_rewards) / len(consistency_rewards) if consistency_rewards else 0.0
        anchor_reward_mean = sum(anchor_rewards) / len(anchor_rewards) if anchor_rewards else 0.0

        reward_by_pert_trait = defaultdict(list)
        for (prompt, rollout), reward in zip(policy_grad_data, all_rewards):
            reward_by_pert_trait[(rollout.perturbation_idx, rollout.trait_value)].append(reward)

        adv_abs_mean = sum(abs(a) for a in advantages) / len(advantages) if advantages else 0.0
        avg_response_len = sum(len(r.tokens) for _, r in policy_grad_data) / len(policy_grad_data) if policy_grad_data else 0.0
        kl_v1 = kl_sample_train_metrics.get("optim/kl_sample_train_v1", 0.0)

        step_metrics = {
            "train/epoch": epoch,
            "train/parse_rate": parse_rate,
            "train/n_consistency_rollouts": len(consistency_rewards),
            "train/n_anchor_rollouts": len(anchor_rewards),
            "train/avg_response_length": avg_response_len,
            "train/p_ref": avg_p_ref,
            **({"train/p_ref_init": avg_p_ref_init,
               "train/p_ref_drift": avg_p_ref - avg_p_ref_init} if avg_p_ref_init is not None else {}),
            "train/n_ref_parsed": total_n_ref_parsed,
            "train/n_training_parsed": total_n_training_parsed,
            "train/rate_var": rate_var,
            "train/consistency_reward_mean": cons_reward_mean,
            "train/consistency_reward_std": (sum((r - cons_reward_mean)**2 for r in consistency_rewards) / len(consistency_rewards)) ** 0.5 if consistency_rewards else 0.0,
            "train/anchor_reward_mean": anchor_reward_mean,
            "train/anchor_reward_std": (sum((r - anchor_reward_mean)**2 for r in anchor_rewards) / len(anchor_rewards)) ** 0.5 if anchor_rewards else 0.0,
            "train/advantage_abs_mean": adv_abs_mean,
        }
        for pert_idx, rate in avg_p_hat.items():
            step_metrics[f"train/p_hat_{pert_idx}"] = rate
            step_metrics[f"train/consistency_gap_{pert_idx}"] = rate - avg_p_ref

        for (pert_idx, trait_val), rewards in reward_by_pert_trait.items():
            trait_key = f"{trait_val:.0f}" if trait_val == int(trait_val) else f"{trait_val:.2f}"
            step_metrics[f"train/reward_pert{pert_idx}_trait{trait_key}_mean"] = sum(rewards) / len(rewards)
            step_metrics[f"train/reward_pert{pert_idx}_trait{trait_key}_count"] = len(rewards)

        step_metrics.update(kl_sample_train_metrics)
        for k, v in kl_penalty_metrics.items():
            step_metrics[f"train/{k}"] = v

        logger.log_metrics(step_metrics, step=global_step)

        p_hat_display = list(avg_p_hat.values())[0] if avg_p_hat else 0.0
        gap_display = list(avg_p_hat.values())[0] - avg_p_ref if avg_p_hat else 0.0
        pbar.set_postfix({"kl": f"{kl_v1:.4f}", "gap": f"{gap_display:.3f}", "p_hat": f"{p_hat_display:.2f}", "adv": f"{adv_abs_mean:.3f}"})
