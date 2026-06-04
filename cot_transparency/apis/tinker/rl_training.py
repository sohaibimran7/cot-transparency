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
from __future__ import annotations

import asyncio
from collections import defaultdict, deque
import inspect
import json
import logging
import random
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Sequence, Callable, Any

import tinker
from tinker import types
from pydantic import BaseModel
import torch
from tqdm import tqdm

from tinker_cookbook.utils.ml_log import setup_logging
from tinker_cookbook.utils.lr_scheduling import compute_schedule_lr_multiplier
from tinker_cookbook.rl.metrics import compute_kl_sample_train, incorporate_kl_penalty
from tinker_cookbook.rl.data_processing import trajectory_to_data
from tinker_cookbook.rl.types import Trajectory, Transition
from tinker_cookbook.rl.train import _remove_mask as remove_mask
from tinker_cookbook.completers import TokensWithLogprobs

from cot_transparency.apis.tinker.inference import decode_response

from cot_transparency.apis.tinker.common import (
    LoRAConfig,
    AdamConfig,
    CheckpointConfig,
    get_renderer_and_tokenizer,
    build_log_dir,
    get_recommended_lr,
    get_git_state,
    warn_if_dirty,
    save_intermediate_checkpoint,
    finalize_checkpoint,
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


def _is_async_callable(fn: Callable) -> bool:
    """True if ``fn`` (a function or a callable object) is a coroutine callable.

    Detects both ``async def`` functions and callable instances whose ``__call__``
    is ``async`` (e.g. a judge object). Lets ``trait_classifier`` be sync or async.
    """
    if inspect.iscoroutinefunction(fn):
        return True
    call = getattr(fn, "__call__", None)
    return call is not None and inspect.iscoroutinefunction(call)


async def _classify_traits(fn: Callable, raw: list, datapoint: dict) -> list[float]:
    """Run a (sync or async) classifier over sampled rollouts, returning floats.

    ``raw`` holds the sampler's ``(tokens, logprobs, full_text, answer_text)`` tuples.
    Async classifiers (e.g. an LLM judge for eval-awareness misalignment) are awaited
    concurrently so network judges don't serialize the rollout loop; sync ones (e.g. an
    answer parser) are called inline.
    """
    if _is_async_callable(fn):
        values = await asyncio.gather(*[fn(answer_text, datapoint) for _, _, _, answer_text in raw])
    else:
        values = [fn(answer_text, datapoint) for _, _, _, answer_text in raw]
    return [float(v) for v in values]


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
    aggregation: Optional[Literal["mean", "min", "max"]] = "mean"


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
    refresh_policy_every_n_steps: int = 1  # 1 = fully on-policy (cookbook contract); all configs set this explicitly
    n_epochs: int = 1
    normalize: Literal["pooled", "per_item"] = "pooled"  # advantage standardization: whole batch vs per-item group


class GenerationConfig(BaseModel):
    """Generation config."""
    max_new_tokens: int = 16384
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
    loss_fn: Literal["ppo", "importance_sampling"] = "ppo"  # SDK LossFnType; "reinforce" is NOT valid (would crash forward_backward)
    anchor_weight: float = 0.5
    anchor_model: Literal["base", "initial_policy"] = "base"
    # Advantage construction:
    #   "grpo_normalized" — standardize each population to unit variance. With one
    #     datapoint/perturbation per step this divides out (p_hat - p_ref) entirely,
    #     so only the sign of the gap survives (chases sampling noise near convergence).
    #   "snr_scaling" — the SAME unit-variance GRPO advantage, multiplied per item by the SNR
    #     shrink factor snr/(snr+z^2) ∈ [0,1] (snr=(gap/SE)^2). Full GRPO step when the gap
    #     clears sampling noise, tapering to 0 within it. Keeps GRPO's scale-stability (LR
    #     transfers, batch-invariant) AND adds anti-overshoot — strictly = grpo × confidence-gate.
    #   "matched_pair" — pool the cued rate across the WHOLE cue family for an item into
    #     one gap (p_hat_pool - p_ref) and score every cued rollout against the neutral
    #     reference p_ref (the matched control), not its own per-cue mean. Lets each cue
    #     use ~1-2 rollouts: cue diversity (not per-cue depth) de-biases the per-item gap.
    #     Keeps the shrunk gap magnitude (snr_normalizer below); z=0 => faithful gap.
    advantage_estimator: Literal["grpo_normalized", "snr_scaling", "matched_pair"] = "grpo_normalized"
    snr_mode: Literal["soft", "hard"] = "soft"
    snr_z: float = 2.0   # half-weight (soft) / cutoff (hard) at |gap| = z·SE; z=0 => gate≡1 (faithful GRPO)
    # snr_normalizer applies to matched_pair only (snr_scaling now uses unit-variance normalization):
    #   "trait_std" — divide the score by per-rollout Bernoulli std sqrt(p(1-p)+floor).
    #   "none" — bare A = -shrunk_gap * (T - p_hat), no division.
    snr_normalizer: Literal["trait_std", "none"] = "trait_std"
    # Non-answer handling during rollout collection (the answer-parse-failure / hedge case):
    #   "discard"  — unparsed (or logprob-less) rollouts are dropped from BOTH the rate
    #     (p_hat = #biased / #parsed, so the denominator shrinks) and the gradient. The default;
    #     hedging stays visible via train/parse_rate. (Distinct from an "ignore" rate of
    #     #biased / #total that would keep non-answers in the denominator as not-biased — not used.)
    #   "resample" — re-sample each slot until its answer parses AND it has logprobs, up to
    #     max_resample_attempts rounds, so the rate is over committed answers (hedging can't
    #     lower it). Logs train/resample_amplif_* (drawn/target ratio) and train/resample_gaveup_*
    #     (slots never filled) — these become the hedge signal since parse_rate then ~= 1.
    unparsed_handling: Literal["discard", "resample"] = "discard"
    max_resample_attempts: int = 4
    # Optional benign-helpfulness GRPO term mixed into the gradient: rewards COMPLETING
    # benign tasks (helpfulness ∈ [0,1], 1=helped), so the model can't satisfy the
    # consistency objective by refusing everything (the refuse-all reward hack). 0 = off.
    helpfulness_weight: float = 0.0
    n_helpfulness_rollouts: int = 16
    # reward  : maximise the benign trait (push accuracy/helpfulness up — anti refuse-all).
    # anchor  : MATCH the benign trait to the base model's per-prompt level (preserve capability,
    #           don't distort). Target = per-prompt base accuracy measured at startup.
    # distill : forward-KL to base on held-out benign data (sample from BASE, imitate) — judge-free,
    #           keeps the output distribution near base where over-refusal would otherwise grow.
    helpfulness_mode: Literal["reward", "anchor", "distill"] = "reward"
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
    resample_stats: dict = field(default_factory=dict)  # {ref,train}_{want,drawn,gave_up} (resample mode)


@dataclass
class BatchItem:
    """One datapoint's rollout results, ready for reward computation."""
    datapoint_idx: int
    datapoint: dict
    train_rollouts: list[Rollout]
    anchor_rollouts: list[Rollout]
    p_hat: dict[int, float]         # Per-perturbation trait rates (training)
    p_hat_counts: dict[int, int]    # Per-perturbation parsed rollout counts (for gap SE)
    p_ref: float                    # Reference perturbation rate
    p_ref_init: float | None        # Initial (base/anchor) reference rate
    n_total: int                    # Total raw rollouts
    n_parsed: int                   # Parsed rollouts
    n_ref_parsed: int               # Parsed ref rollouts
    n_training_parsed: int          # Parsed training rollouts
    n_ref_init_parsed: int | None = None  # Parsed rollouts behind p_ref_init (for anchor-gap SE)
    resample_stats: dict = field(default_factory=dict)  # {ref,train}_{want,drawn,gave_up} (resample mode)


# =============================================================================
# Reward
# =============================================================================

class ConsistencyReward:
    """Consistency reward for training perturbation rollouts.

    Pushes p_hat toward p_ref (privileged) using variance-optimal baseline p_hat.
    r = -(p_hat - p_ref) * (trait - p_hat)

    Anchor reward is computed separately on reference perturbation rollouts.
    """

    def compute_rewards(self, rollouts: list[Rollout], p_hat: dict[int, float], p_ref: float,
                        gaps: Optional[dict[int, float]] = None,
                        baseline: Optional[float] = None) -> list[float]:
        """Consistency-only rewards for training perturbation rollouts.

        r = -gap[pert] * (trait - baseline)

        ``gaps`` lets the caller substitute a transformed gap (e.g. SNR-scaled)
        per perturbation. When None, the raw gap (p_hat[pert] - p_ref) is used.
        ``baseline`` overrides the per-perturbation mean p_hat[pert] used to centre the
        score term; the matched-pair estimator passes p_ref (the neutral control) so every
        cued rollout is judged against the reference, not its own per-cue mean. Any constant
        baseline leaves E[grad] = -gap * ∇p_hat unchanged (it only affects variance).
        """
        if gaps is None:
            gaps = {pert: rate - p_ref for pert, rate in p_hat.items()}
        return [-gaps[r.perturbation_idx] * (r.trait_value - (p_hat[r.perturbation_idx] if baseline is None else baseline))
                for r in rollouts]

    def compute_anchor_rewards(self, ref_rollouts: list[Rollout], p_ref: float, p_ref_initial: Optional[float],
                               gap: Optional[float] = None) -> list[float]:
        """Anchor rewards for reference perturbation rollouts.

        r = -gap * (trait - p_ref), gap = (p_ref - p_ref_initial) by default.

        ``gap`` lets the caller substitute a transformed gap. Returns zeros when no
        anchor target is available (p_ref_initial is None and no gap supplied).
        """
        if gap is None:
            if p_ref_initial is None:
                return [0.0] * len(ref_rollouts)
            gap = p_ref - p_ref_initial
        return [-gap * (r.trait_value - p_ref)
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
        self.anchor_sampling_client: tinker.SamplingClient | None = None
        self.renderer: Any = None
        self.tokenizer: Any = None
        self._snr_metrics: dict[str, float] = {}

    def setup(self) -> None:
        """Initialize clients and renderer."""
        # Reproducibility: seed the epoch shuffle (_train_loop_inner) and gradient-rollout
        # subsampling (_select_rollouts). LoRA init is seeded separately by the SDK via
        # lora.seed. NOTE: stochastic sampling (temperature) is deliberately NOT seeded —
        # threading a fixed SamplingParams seed would force identical rollouts every step.
        if self.config.lora.seed is not None:
            random.seed(self.config.lora.seed)
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
        # Anchor client: used for computing p_ref_init (the anchor target rate).
        # "base" = frozen base model; "initial_policy" = policy at init (base + any resumed ckpt).
        if self.config.anchor_model == "initial_policy":
            self.anchor_sampling_client = self.sampling_client
        else:
            self.anchor_sampling_client = self.base_sampling_client

    async def _sample_from_client(self, client: tinker.SamplingClient, prompt: types.ModelInput, n_samples: int) -> list[tuple[list[int], list[float], str, str]]:
        """Sample from a client and return (tokens, logprobs, full_text, answer_text) tuples.

        full_text includes thinking/reasoning (for Rollout.text and logging).
        answer_text is text-only without thinking (for answer parsing and trait classification).
        """
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
            # Missing logprobs used to be filled with zeros, which poison the importance
            # ratio and KL penalty (sampled_logprob=0 vs a very-negative training logprob).
            # Return None instead; the caller marks such rollouts unparsed so they are
            # excluded from both the gradient set and the rate estimate.
            logprobs = list(seq.logprobs) if seq.logprobs else None
            if logprobs is None:
                _log.warning("Missing logprobs for a sample; excluding it from gradients/rates")
            full_text = self.tokenizer.decode(tokens)
            # decode_response guards parse_response itself (RendererError for gpt-oss when a
            # sequence carries >1 stop token) AND get_text_content; a bare parse_response
            # here would abort the whole training step.
            answer_text = decode_response(self.renderer, self.tokenizer, tokens)
            samples.append((tokens, logprobs, full_text, answer_text))
        return samples

    async def _collect_rollouts(
        self,
        datapoint: dict,
        perturbation_fns: list[Callable[[dict], dict]],
        trait_classifier: Callable[[str, dict], float],
        sampling_client: Optional[tinker.SamplingClient] = None,
        answer_parser: Optional[Callable[[str], Optional[str]]] = None,
        rates_only: bool = False,
    ) -> RolloutResult:
        """Collect rollouts and compute rates in one pass.

        Rates are computed internally so the full rollout data (tokens/logprobs
        for all rollouts) can be freed immediately. Only the gradient-selected
        subset retains tokens/logprobs.

        Args:
            sampling_client: Explicit client to sample from. Defaults to
                self.sampling_client (current policy).
            rates_only: If True, skip gradient rollout selection and return empty
                rollout lists. Used for anchor/base sampling where only rates are needed.
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

        client = sampling_client if sampling_client is not None else self.sampling_client

        resample = self.config.unparsed_handling == "resample" and answer_parser is not None

        async def rollout_perturbation(idx: int) -> tuple[int, list[Rollout], dict]:
            pert_result = perturbation_fns[idx](datapoint)
            messages = pert_result["messages"]
            prompt = self.renderer.build_generation_prompt(messages)
            n_want = n_rollouts_per[idx]

            # A rollout is usable only if its answer parses AND it has logprobs (needed for
            # the importance ratio); missing either => exclude from gradient/rate.
            def _usable(s) -> bool:  # s = (tokens, logprobs, full_text, answer_text)
                return answer_parser(s[3]) is not None and s[1] is not None

            n_drawn, gave_up = 0, 0
            if resample and n_want > 0:
                # Re-sample each slot until its answer is usable, up to max_resample_attempts
                # rounds. Hedging can't lower the rate; its cost shows up as n_drawn > n_want.
                raw, rounds = [], 0
                while len(raw) < n_want and rounds < self.config.max_resample_attempts:
                    need = n_want - len(raw)
                    batch = await self._sample_from_client(client, prompt, need)
                    n_drawn += need
                    raw.extend(s for s in batch if _usable(s))
                    rounds += 1
                raw = raw[:n_want]
                gave_up = n_want - len(raw)  # slots never filled within the attempt budget
            else:
                raw = await self._sample_from_client(client, prompt, n_want)
                n_drawn = n_want

            trait_values = await _classify_traits(trait_classifier, raw, datapoint)

            rollouts = []
            for (tokens, logprobs, full_text, answer_text), trait_value in zip(raw, trait_values):
                answer_ok = answer_parser(answer_text) is not None if answer_parser else True
                parsed_ok = answer_ok and logprobs is not None
                rollouts.append(Rollout(
                    tokens=tokens,
                    logprobs=logprobs if logprobs is not None else [],
                    text=full_text,
                    trait_value=trait_value,
                    perturbation_idx=idx,
                    parsed_successfully=parsed_ok,
                    prompt=prompt,
                ))
            return idx, rollouts, {"n_want": n_want, "n_drawn": n_drawn, "gave_up": gave_up}

        results = await asyncio.gather(*[rollout_perturbation(idx) for idx in all_idx])
        all_rollouts = {idx: rolls for idx, rolls, _ in results}

        # Aggregate resample stats into ref/train buckets (resample mode only).
        resample_stats: dict = {}
        if resample:
            for name, idxs in (("ref", ref_idx), ("train", training_idx)):
                w = d = g = 0
                for idx, _rolls, s in results:
                    if idx in idxs:
                        w += s["n_want"]; d += s["n_drawn"]; g += s["gave_up"]
                resample_stats.update({f"{name}_want": w, f"{name}_drawn": d, f"{name}_gave_up": g})

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
            resample_stats=resample_stats,
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

    @staticmethod
    def _trait_std(p: float, var_floor: float = 0.01) -> float:
        """Per-rollout Bernoulli std, used to standardize trait noise (NOT the gap).

        Dividing the SNR-scaling reward by this — rather than by the full reward std —
        normalizes only the per-rollout noise and leaves the gap magnitude intact, so
        |advantage| ~ |SNR-scaled gap| (comparable scale to grpo's unit advantages).
        """
        return (p * (1.0 - p) + var_floor) ** 0.5

    @staticmethod
    def _binom_var(p: float, n: int, pseudocount: float = 1.0) -> float:
        """Laplace/Beta(pseudocount, pseudocount)-smoothed binomial variance p(1-p)/n.

        Smoothing keeps the boundary (p=0 or p=1) from looking like zero uncertainty: the
        raw normal-approx variance collapses to 0 at an observed rate of exactly 0 (common
        for p_ref in sycophancy: 0/N ref rollouts pick the biased option), which would make
        a tiny empirical gap look infinitely significant. The rule of three says 0/n carries
        uncertainty ~3/n, not 0. Interior proportions are ~unchanged.
        """
        n_eff = max(n, 1)
        n_smooth = n_eff + 2.0 * pseudocount
        p_smooth = (p * n_eff + pseudocount) / n_smooth   # shrink toward 0.5
        return p_smooth * (1.0 - p_smooth) / n_smooth

    @staticmethod
    def _gap_se(p1: float, n1: int, p2: float, n2: int, pseudocount: float = 1.0) -> float:
        """Standard error of the gap (p1 - p2) under independent binomial sampling."""
        return (RLTrainer._binom_var(p1, n1, pseudocount)
                + RLTrainer._binom_var(p2, n2, pseudocount)) ** 0.5

    @staticmethod
    def _matched_pair_gap_se(p_hat: dict[int, float], p_hat_counts: dict[int, int],
                             p_ref: float, n_ref: int, pseudocount: float = 1.0) -> float:
        """SE of (p_pool - p_ref), where p_pool is the count-weighted mean of the per-cue
        rates across the cue family.

        The cue family is a FIXED set of heterogeneous paraphrases, so the cued side's
        sampling variance is the STRATIFIED (cluster) variance ``sum_c (n_c/n_pool)^2 ·
        Var(p_c)`` — NOT a single pooled binomial ``p_pool(1-p_pool)/n_pool``. The pooled
        binomial counts the genuine between-cue spread as if it were sampling noise (mixture
        variance ≥ mean within-cue variance), inflating the SE and over-shrinking real gaps
        — directly contradicting the "cue diversity supplies the low-variance gap" design.
        """
        n_pool = sum(p_hat_counts.get(c, 0) for c in p_hat)
        if n_pool > 0:
            cued_var = sum((p_hat_counts.get(c, 0) / n_pool) ** 2
                           * RLTrainer._binom_var(rate, p_hat_counts.get(c, 0), pseudocount)
                           for c, rate in p_hat.items())
        else:
            # No parsed counts: treat each cue as one observation of the cue-mean.
            k = max(len(p_hat), 1)
            cued_var = sum(RLTrainer._binom_var(rate, 1, pseudocount) for rate in p_hat.values()) / (k * k)
        return (cued_var + RLTrainer._binom_var(p_ref, n_ref, pseudocount)) ** 0.5

    def _snr_scale_gap(self, d: float, se: float) -> float:
        """Scale an empirical gap toward 0 by its sampling SNR = (d/se)^2.

        soft: d * snr / (snr + z^2)     — smooth, half-weight at |d| = z·SE
        hard: d * max(0, 1 - z^2/snr)   — positive-part gate, zero below |d| = z·SE

        Larger sampling budget → smaller SE → less SNR scaling (you trust smaller gaps).
        """
        if d == 0.0:
            return 0.0
        if se <= 0.0:
            return d
        snr = (d * d) / (se * se)
        z2 = self.config.snr_z ** 2
        if self.config.snr_mode == "hard":
            factor = max(0.0, 1.0 - z2 / snr)
        else:
            factor = snr / (snr + z2)
        return d * factor

    def _snr_shrink_factor(self, d: float, se: float) -> float:
        """The SNR gate in [0,1] such that _snr_scale_gap(d, se) == d * factor.

        snr_scaling multiplies the unit-variance GRPO advantage by this per-item factor:
        ~1 when the gap clears sampling noise (full GRPO step), tapering to 0 within it
        (anti-overshoot). z=0 => factor 1 (faithful GRPO).
        """
        if d == 0.0:
            return 0.0
        if se <= 0.0:
            return 1.0
        snr = (d * d) / (se * se)
        z2 = self.config.snr_z ** 2
        if self.config.snr_mode == "hard":
            return max(0.0, 1.0 - z2 / snr)
        return snr / (snr + z2)

    def _normalize_advantages(self, rewards: list[float]) -> list[float]:
        if not rewards:
            return rewards
        mean_r = sum(rewards) / len(rewards)
        var = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
        std_r = var ** 0.5
        if std_r < 1e-8:
            return [0.0] * len(rewards)
        return [(r - mean_r) / std_r for r in rewards]

    def _normalize_grouped(self, rewards: list[float], slices: list[tuple[int, int]]) -> list[float]:
        """Unit-variance standardization, pooled over the whole batch or per-item (per group).

        per_item: each item's rollouts are standardized on their own, so the per-item gap (a
        constant within the group) cancels — leaving sign(gap)·standardized(trait−p_hat); the SNR
        gate is then the sole magnitude signal and no single big-gap item dominates the batch.
        An item with zero within-group variance (e.g. all rollouts refuse) correctly gets 0.
        pooled: standardize the whole list together → gap magnitude becomes cross-item weighting.
        """
        if self.config.loop.normalize != "per_item":
            return self._normalize_advantages(rewards)
        adv = list(rewards)
        for s, e in slices:
            adv[s:e] = self._normalize_advantages(rewards[s:e])
        return adv

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

    async def _collect_helpfulness_datums(self, dps: list[dict]) -> tuple[list, float]:
        """GRPO datums rewarding the policy for COMPLETING benign tasks.

        For each benign datapoint, sample n_helpfulness_rollouts on its prompt, score
        helpfulness ∈ [0,1] (1 = helped, 0 = refused) via self._help_cls, and form a
        per-prompt GRPO advantage (standardised within the prompt). Scaled by
        config.helpfulness_weight. This penalises drifting toward refuse-everything,
        which is the degenerate equaliser of the pure consistency objective.

        Prompts are sampled and judged concurrently. Returns (datums, mean_score).
        """
        weight = self.config.helpfulness_weight
        n = self.config.n_helpfulness_rollouts

        async def one(dp: dict) -> tuple[list, list[float]]:
            messages = self._help_fn(dp)["messages"]
            prompt = self.renderer.build_generation_prompt(messages)
            if self.config.helpfulness_mode == "distill":
                # Imitate BASE on this benign prompt (forward-KL): sample from base, reinforce
                # those tokens with a constant +weight advantage → behaviour-clone base here.
                raw = await self._sample_from_client(self.base_sampling_client, prompt, n)
                datums = [
                    self._create_rl_datum(
                        prompt,
                        Rollout(tokens=tokens, logprobs=logprobs, text=full_text,
                                trait_value=0.0, perturbation_idx=0, parsed_successfully=True, prompt=prompt),
                        weight,
                    )
                    for (tokens, logprobs, full_text, _ans) in raw
                    if logprobs is not None
                ]
                return datums, [1.0] * len(datums)  # logged as helpfulness_mean=1.0 (imitating base)
            raw = await self._sample_from_client(self.sampling_client, prompt, n)
            scores = await _classify_traits(self._help_cls, raw, dp)
            if self.config.helpfulness_mode == "anchor":
                # MATCH this prompt's accuracy to its base level: standardise the scores, then
                # flip sign by drift direction (p>p_base → discourage high = push down; p<p_base →
                # encourage high = push up), tapering to 0 within a 0.05 accuracy band of base.
                p = (sum(scores) / len(scores)) if scores else 0.0
                p_base = dp.get("_help_base_acc", p)
                d = p - p_base
                taper = min(abs(d) / 0.05, 1.0)
                sign = -1.0 if d > 0 else (1.0 if d < 0 else 0.0)
                adv = [sign * taper * a for a in self._normalize_advantages(scores)]
            else:
                adv = self._normalize_advantages(scores)  # reward mode: per-prompt GRPO baseline (maximise)
            datums = [
                self._create_rl_datum(
                    prompt,
                    Rollout(tokens=tokens, logprobs=logprobs, text=full_text,
                            trait_value=0.0, perturbation_idx=0, parsed_successfully=True, prompt=prompt),
                    weight * a,
                )
                for (tokens, logprobs, full_text, _ans), a in zip(raw, adv)
                if logprobs is not None  # skip missing-logprob samples (would poison the IS ratio)
            ]
            return datums, scores

        results = await asyncio.gather(*[one(dp) for dp in dps])
        datums = [d for ds, _ in results for d in ds]
        all_scores = [s for _, scores in results for s in scores]
        mean_score = (sum(all_scores) / len(all_scores)) if all_scores else 0.0
        return datums, mean_score

    async def _measure_help_base_accuracy(self) -> None:
        """Anchor target: measure the BASE model's per-prompt accuracy on the helpfulness
        prompts, stored on each dp as ``_help_base_acc``. Run once at startup (anchor mode)."""
        n = self.config.n_helpfulness_rollouts

        async def one(dp: dict) -> None:
            prompt = self.renderer.build_generation_prompt(self._help_fn(dp)["messages"])
            raw = await self._sample_from_client(self.base_sampling_client, prompt, n)
            scores = await _classify_traits(self._help_cls, raw, dp)
            dp["_help_base_acc"] = (sum(scores) / len(scores)) if scores else 0.0

        await asyncio.gather(*[one(dp) for dp in self._help_dps])
        accs = [dp.get("_help_base_acc", 0.0) for dp in self._help_dps]
        mean = sum(accs) / len(accs) if accs else float("nan")
        print(f"  IFEval anchor: base accuracy measured on {len(accs)} prompts, mean={mean:.3f}")

    def _build_training_batch(
        self,
        batch_items: list[BatchItem],
    ) -> tuple[list, list[float], list[float], list[float], list[tuple]] | None:
        """Compute rewards, normalize advantages, and build training data.

        Returns (grad_datums, consistency_rewards, anchor_rewards, advantages, policy_grad_data)
        or None if the batch should be skipped (empty/zero advantages).
        """
        anchor_weight = self.config.anchor_weight
        use_snr = self.config.advantage_estimator == "snr_scaling"
        use_matched = self.config.advantage_estimator == "matched_pair"

        # Consistency: training perturbation rollouts
        consistency_rewards, consistency_data, consistency_div = [], [], []
        consistency_snr = []   # snr_scaling: per-rollout shrink factor gating the unit-var advantage (1.0 otherwise)
        consistency_slices = []   # (start, end) per item, for per-item (per-group) normalization
        raw_gap_abs, shrunk_gap_abs, gap_n = 0.0, 0.0, 0
        for item in batch_items:
            gaps = None
            baseline = None        # matched_pair: centre the score on p_ref, not per-cue p_hat
            div_p = None           # matched_pair: per-rollout trait-std argument (pooled rate)
            snr_f = {}             # snr_scaling: per-perturbation shrink factor (applied after normalization)
            if use_snr:
                # snr_scaling = unit-variance GRPO advantage × SNR gate. Build the reward with the
                # RAW gap (same as grpo) so normalization sees the real signal; the gate (in [0,1])
                # is applied per item AFTER unit-variance normalization, below.
                gaps = {}
                for pert, rate in item.p_hat.items():
                    raw = rate - item.p_ref
                    se = self._gap_se(rate, item.p_hat_counts.get(pert, 0), item.p_ref, item.n_ref_parsed)
                    f = self._snr_shrink_factor(raw, se)
                    gaps[pert] = raw
                    snr_f[pert] = f
                    raw_gap_abs += abs(raw); shrunk_gap_abs += abs(raw * f); gap_n += 1
            elif use_matched and item.p_hat:
                # Pool the cued rate across the whole cue family into ONE gap vs p_ref, and
                # score every cued rollout against the neutral control p_ref. Cue diversity
                # (not per-cue depth) supplies the low-variance per-item gap. Skipped when
                # p_hat is empty (no parsed training rollouts): that item has no train
                # rollouts to score, and counting its 0-gap would dilute the logged metrics.
                n_pool = sum(item.p_hat_counts.get(p, 0) for p in item.p_hat)
                p_pool = (sum(rate * item.p_hat_counts.get(p, 0) for p, rate in item.p_hat.items()) / n_pool
                          if n_pool > 0 else sum(item.p_hat.values()) / len(item.p_hat))
                raw = p_pool - item.p_ref
                # Stratified (cluster-robust) SE: the cued side's variance is the
                # between-cue-weighted within-cue variance, NOT one pooled binomial over
                # n_pool (which would over-shrink by counting cue heterogeneity as noise).
                se = self._matched_pair_gap_se(item.p_hat, item.p_hat_counts, item.p_ref, item.n_ref_parsed)
                g = self._snr_scale_gap(raw, se)  # SE taper ON by default (snr_z=2.0); set snr_z=0 for the faithful pooled gap
                gaps = {pert: g for pert in item.p_hat}
                baseline = item.p_ref
                div_p = p_pool
                raw_gap_abs += abs(raw); shrunk_gap_abs += abs(g); gap_n += 1
            rewards = self.reward_function.compute_rewards(
                item.train_rollouts, item.p_hat, item.p_ref, gaps=gaps, baseline=baseline)
            slice_start = len(consistency_rewards)
            consistency_rewards.extend(rewards)
            consistency_slices.append((slice_start, len(consistency_rewards)))
            for rollout in item.train_rollouts:
                consistency_data.append((rollout.prompt, rollout))
                p_for_div = div_p if div_p is not None else item.p_hat.get(rollout.perturbation_idx, 0.0)
                consistency_div.append(self._trait_std(p_for_div))
                consistency_snr.append(snr_f.get(rollout.perturbation_idx, 1.0))

        # Anchor: reference perturbation rollouts
        anchor_rewards, anchor_data, anchor_div = [], [], []
        anchor_snr = []
        anchor_slices = []
        if anchor_weight > 0:
            for item in batch_items:
                anchor_gap = None
                anchor_f = 1.0
                if (use_snr or use_matched) and item.p_ref_init is not None:
                    raw = item.p_ref - item.p_ref_init
                    # p_ref_init was sampled with its OWN parsed count (step-0 policy or the
                    # anchor model), not this step's n_ref_parsed; use the tracked count and
                    # fall back to the configured budget when unavailable.
                    n_init = item.n_ref_init_parsed or self.config.reference_rate.n_rollouts
                    se = self._gap_se(item.p_ref, item.n_ref_parsed, item.p_ref_init, n_init)
                    if use_snr:
                        anchor_f = self._snr_shrink_factor(raw, se)  # gate applied after normalization
                        anchor_gap = raw
                    else:  # matched_pair: keep the shrunk gap in the reward (div path below)
                        anchor_gap = self._snr_scale_gap(raw, se)
                rewards = self.reward_function.compute_anchor_rewards(
                    item.anchor_rollouts, item.p_ref, item.p_ref_init, gap=anchor_gap)
                a_start = len(anchor_rewards)
                anchor_rewards.extend(rewards)
                anchor_slices.append((a_start, len(anchor_rewards)))
                for rollout in item.anchor_rollouts:
                    anchor_data.append((rollout.prompt, rollout))
                    anchor_div.append(self._trait_std(item.p_ref))
                    anchor_snr.append(anchor_f)

        # Form advantages.
        #  - grpo_normalized: standardize each population to unit variance (drops gap magnitude).
        #  - snr_scaling: the SAME unit-variance base, then gate each rollout by its item's SNR
        #    shrink factor in [0,1] — full GRPO step when the gap clears sampling noise, tapering
        #    to 0 within it. Scale-stable (LR transfers, batch-invariant) AND anti-overshoot.
        #  - matched_pair: keep the (pooled) gap magnitude, dividing only the per-rollout Bernoulli
        #    noise by trait std (snr_normalizer; "none" = bare, no division).
        if use_matched:
            if self.config.snr_normalizer == "none":
                consistency_adv = list(consistency_rewards)
                anchor_adv = list(anchor_rewards)
            else:
                consistency_adv = [r / d for r, d in zip(consistency_rewards, consistency_div)]
                anchor_adv = [r / d for r, d in zip(anchor_rewards, anchor_div)]
        else:
            consistency_adv = self._normalize_grouped(consistency_rewards, consistency_slices)
            anchor_adv = self._normalize_grouped(anchor_rewards, anchor_slices)
            if use_snr:  # gate the unit-variance advantage by the per-rollout SNR shrink factor
                consistency_adv = [f * a for f, a in zip(consistency_snr, consistency_adv)]
                anchor_adv = [f * a for f, a in zip(anchor_snr, anchor_adv)]
        consistency_adv = [a * (1 - anchor_weight) for a in consistency_adv]
        anchor_adv = [a * anchor_weight for a in anchor_adv]

        self._snr_metrics = (
            {"train/gap_raw_abs_mean": raw_gap_abs / gap_n,
             "train/gap_snr_scaled_abs_mean": shrunk_gap_abs / gap_n,
             "train/gap_snr_scale_factor": (shrunk_gap_abs / raw_gap_abs) if raw_gap_abs > 1e-9 else 0.0}
            if (use_snr or use_matched) and gap_n > 0 else {}
        )

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
        helpfulness_datapoints: Optional[Sequence[dict]] = None,
        helpfulness_perturbation_fn: Optional[Callable[[dict], dict]] = None,
        helpfulness_classifier: Optional[Callable[[str, dict], float]] = None,
    ) -> str:
        """Run RL consistency training. Returns final checkpoint path.

        If ``helpfulness_weight > 0`` and ``helpfulness_datapoints`` are given, a benign
        GRPO term (reward = ``helpfulness_classifier`` ∈ [0,1]) is mixed into every step's
        gradient — the anti-refuse-all signal.
        """
        if self.training_client is None:
            self.setup()

        # Helpfulness term state (anti refuse-all). Cycled through across steps.
        self._help_dps = list(helpfulness_datapoints) if helpfulness_datapoints else []
        self._help_fn = helpfulness_perturbation_fn
        self._help_cls = helpfulness_classifier
        self._help_idx = 0
        if self._help_dps and self.config.helpfulness_weight > 0 and self.config.helpfulness_mode == "anchor":
            await self._measure_help_base_accuracy()  # per-prompt anchor target (base accuracy)

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
        # Parsed-rollout count behind each cached p_ref_init, for a correctly-sized anchor SE.
        initial_reference_counts: dict[int, int] = {}
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

        if self.config.loop.gradient_accumulation_steps > 1:
            print("⚠️  gradient_accumulation_steps>1: advantages are normalized per-microbatch "
                  "(not across the accumulated group) and are NOT rescaled by N, so the effective "
                  "LR/baseline depends on the server-side loss reduction (unverified). Prefer batch_size.")

        checkpoint_paths: list[str] = []

        def _adam_params_for_step(step: int) -> types.AdamParams:
            # Apply the configured LR schedule per optim step. RL previously built AdamParams
            # once and silently ignored lr_schedule (SFT applies it); this restores parity.
            # max(0.0, ...) guards against a negative multiplier past the last step.
            lr_mult = max(0.0, compute_schedule_lr_multiplier(
                lr_schedule=self.config.optimizer.lr_schedule, step=step, total_steps=total_steps,
            ))
            return types.AdamParams(
                learning_rate=base_lr * lr_mult,
                beta1=self.config.optimizer.beta1,
                beta2=self.config.optimizer.beta2,
                eps=self.config.optimizer.eps,
                weight_decay=self.config.optimizer.weight_decay,
                grad_clip_norm=self.config.optimizer.grad_clip_norm,
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

                # Step 0 optimization: when the initial policy IS the anchor model,
                # we can extract p_ref_init from policy rollouts directly (no extra API call).
                # True for "initial_policy" always; true for "base" only when not resuming.
                need_anchor = need_p_ref_init and dp_idx not in initial_reference_rates
                step0_is_anchor = (
                    self.config.anchor_model == "initial_policy"
                    or self.resume_from is None
                )
                step_snapshot = global_step  # Capture before await; prefetched coroutines read this after yield
                result = await self._collect_rollouts(dp, perturbation_fns, trait_classifier, answer_parser=answer_parser)

                if need_anchor and step_snapshot == 0 and step0_is_anchor:
                    # At step 0, policy matches anchor model — extract p_ref_init directly
                    p_ref_init_val = self._aggregate_ref_rates(result.rates, ref_idx)
                    if p_ref_init_val is not None:
                        initial_reference_rates[dp_idx] = p_ref_init_val
                        initial_reference_counts[dp_idx] = sum(
                            result.rate_counts.get(i, 0) for i in ref_idx)

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
                    p_hat_counts=training_counts,
                    p_ref=p_ref,
                    p_ref_init=p_ref_init,
                    n_total=result.n_total,
                    n_parsed=result.n_parsed,
                    n_ref_parsed=n_ref_parsed,
                    n_training_parsed=n_training_parsed,
                    n_ref_init_parsed=initial_reference_counts.get(dp_idx),
                    resample_stats=result.resample_stats,
                )

            async def sample_batch(batch_indices):
                return await asyncio.gather(*[collect_for_datapoint(idx) for idx in batch_indices])

            # ── Prefetch queue ────────────────────────────────────────────
            # Pipeline: prefetch next step's sampling while current step's
            # fwd_bwd runs on the server (~5-7s overlap). Cap the fill at the
            # next refresh boundary so we never create tasks that would be
            # cancelled by a refresh.
            refresh_every = self.config.loop.refresh_policy_every_n_steps
            max_prefetch = refresh_every or len(batches)
            prefetch_queue: deque[asyncio.Task] = deque()

            def _fill_prefetch_queue(from_batch: int) -> int:
                cap = len(batches)
                if refresh_every:
                    cap = min(cap, ((global_step // refresh_every) + 1) * refresh_every)
                while len(prefetch_queue) < max_prefetch and from_batch < cap:
                    prefetch_queue.append(asyncio.create_task(sample_batch(batches[from_batch])))
                    from_batch += 1
                return from_batch

            async def _maybe_refresh_and_refill(i_batch: int, next_to_prefetch: int) -> int:
                """Refresh the sampling client on schedule, flushing+awaiting stale prefetch.

                Called on BOTH the normal and the empty-batch-skip path so a refresh that
                falls on a skipped step is not dropped (which would extend off-policy
                staleness). Reads the current global_step from the enclosing scope.
                """
                refresh_every = self.config.loop.refresh_policy_every_n_steps
                if not (refresh_every and global_step % refresh_every == 0):
                    return next_to_prefetch
                self.sampling_client = await self.training_client.save_weights_and_get_sampling_client_async(
                    name=f"{self.config.experiment_name}_{self.config.run_name}_sampler_{global_step}"
                )
                # Cancel stale prefetch (sampled from the old policy) AND await them so the
                # in-flight sampling is released — avoids "Task was destroyed but pending".
                for task in prefetch_queue:
                    task.cancel()
                await asyncio.gather(*prefetch_queue, return_exceptions=True)
                prefetch_queue.clear()
                return _fill_prefetch_queue(i_batch + 1)

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

                # Resample diagnostics: in resample mode parse_rate ~= 1, so the hedge signal
                # moves here — amplification (rollouts drawn / target) and give-up counts, split
                # ref vs train so cue-triggered hedging is distinguishable from general hedging.
                if self.config.unparsed_handling == "resample":
                    agg: dict[str, int] = {}
                    for b in batch_items:
                        for k, v in (b.resample_stats or {}).items():
                            agg[k] = agg.get(k, 0) + v
                    amp_tr = (agg.get("train_drawn", 0) / agg["train_want"]) if agg.get("train_want") else 1.0
                    amp_rf = (agg.get("ref_drawn", 0) / agg["ref_want"]) if agg.get("ref_want") else 1.0
                    logger.log_metrics({
                        "train/resample_amplif_train": amp_tr,
                        "train/resample_amplif_ref": amp_rf,
                        "train/resample_gaveup_train": agg.get("train_gave_up", 0),
                        "train/resample_gaveup_ref": agg.get("ref_gave_up", 0),
                    }, step=global_step + 1)
                    if amp_tr > 2.0 or agg.get("train_gave_up", 0) > 0:
                        print(f"\n⚠️  Resample amplification train={amp_tr:.1f}x, gave_up "
                              f"train={agg.get('train_gave_up', 0)} ref={agg.get('ref_gave_up', 0)} "
                              f"at step {global_step + 1} (model hedging)")

                # ── Resolve p_ref_init for anchor (if needed) ─────────────
                # Launch all missing anchor rate samples concurrently
                if need_p_ref_init:
                    items_needing_anchor = [
                        item for item in batch_items
                        if item.p_ref_init is None and item.datapoint_idx not in initial_reference_rates
                    ]
                    if items_needing_anchor:
                        print(f"  ⏳ Anchor rate missing for {len(items_needing_anchor)} datapoint(s), sampling from {self.config.anchor_model}...")

                        async def _sample_anchor(item: BatchItem) -> tuple[BatchItem, Optional[float], int]:
                            result = await self._collect_rollouts(
                                datapoints[item.datapoint_idx], perturbation_fns, trait_classifier,
                                sampling_client=self.anchor_sampling_client, answer_parser=answer_parser, rates_only=True,
                            )
                            n_init = sum(result.rate_counts.get(i, 0) for i in ref_idx)
                            return item, self._aggregate_ref_rates(result.rates, ref_idx), n_init

                        anchor_results = await asyncio.gather(*[_sample_anchor(item) for item in items_needing_anchor])
                        for item, p_ref_init_val, n_init in anchor_results:
                            if p_ref_init_val is None:
                                print(f"  ⚠️  Could not compute base rate for datapoint {item.datapoint_idx}, skipping")
                            else:
                                initial_reference_rates[item.datapoint_idx] = p_ref_init_val
                                initial_reference_counts[item.datapoint_idx] = n_init

                resolved_items = []
                for item in batch_items:
                    if item.p_ref_init is None and need_p_ref_init:
                        if item.datapoint_idx in initial_reference_rates:
                            item.p_ref_init = initial_reference_rates[item.datapoint_idx]
                            item.n_ref_init_parsed = initial_reference_counts.get(item.datapoint_idx)

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
                    # Still honor the refresh schedule on a skipped step, then top up.
                    next_to_prefetch = await _maybe_refresh_and_refill(i_batch, next_to_prefetch)
                    next_to_prefetch = _fill_prefetch_queue(next_to_prefetch)
                    continue

                grad_datums, consistency_rewards, anchor_rewards, advantages, policy_grad_data = batch_result
                all_rewards = consistency_rewards + anchor_rewards

                # ── Benign-helpfulness GRPO term (anti refuse-all) ────────────
                if self.config.helpfulness_weight > 0 and self._help_dps:
                    k = max(1, self.config.loop.batch_size)
                    hdps = [self._help_dps[(self._help_idx + j) % len(self._help_dps)] for j in range(k)]
                    self._help_idx += k
                    help_datums, help_mean = await self._collect_helpfulness_datums(hdps)
                    grad_datums = grad_datums + help_datums
                    logger.log_metrics({"train/helpfulness_mean": help_mean}, step=global_step + 1)

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
                    optim_future = await self.training_client.optim_step_async(_adam_params_for_step(global_step))
                    accumulated_grads = 0

                # Refill prefetch while fwd_bwd runs — but skip it when the upcoming step
                # will refresh the policy (those rollouts would be sampled from the old
                # weights and immediately discarded), saving wasted sampling.
                refresh_every = self.config.loop.refresh_policy_every_n_steps
                refresh_imminent = bool(refresh_every) and ((global_step + 1) % refresh_every == 0)
                if not refresh_imminent:
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

                # Refresh policy (also flushes+awaits stale prefetch). Shared with the
                # empty-batch-skip path so a due refresh is never dropped.
                next_to_prefetch = await _maybe_refresh_and_refill(i_batch, next_to_prefetch)

                # Intermediate checkpoint
                await self._maybe_save_checkpoint(global_step, total_steps, epoch, log_dir, checkpoint_paths, logger)

        # Final optimizer step for remaining gradients
        if accumulated_grads > 0:
            optim_future = await self.training_client.optim_step_async(_adam_params_for_step(max(0, global_step - 1)))
            await optim_future.result_async()

        # Final checkpoint
        final_path = await finalize_checkpoint(
            self.training_client,
            experiment_name=self.config.experiment_name,
            run_name=self.config.run_name,
            n_epochs=self.config.loop.n_epochs,
            save_state=self.config.checkpoint.save_state,
            global_step=global_step,
            log_dir=log_dir,
            checkpoint_paths=checkpoint_paths,
            logger=logger,
        )
        return final_path

    async def _maybe_save_checkpoint(self, global_step, total_steps, epoch, log_dir, checkpoint_paths, logger):
        """Save intermediate checkpoint if the schedule says so (delegates to common)."""
        await save_intermediate_checkpoint(
            self.training_client,
            experiment_name=self.config.experiment_name,
            run_name=self.config.run_name,
            checkpoint_cfg=self.config.checkpoint,
            global_step=global_step,
            total_steps=total_steps,
            epoch=epoch,
            log_dir=log_dir,
            checkpoint_paths=checkpoint_paths,
            logger=logger,
        )

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
        # p_ref_init can be None for an item even when need_p_ref_init (anchor on) — e.g.
        # the anchor model failed to parse all ref rollouts for that datapoint. Average only
        # the resolved ones; None if none resolved. (The reward path already handles None.)
        _p_ref_inits = [item.p_ref_init for item in batch_items if item.p_ref_init is not None]
        avg_p_ref_init = (
            (sum(_p_ref_inits) / len(_p_ref_inits) if _p_ref_inits else None)
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
        step_metrics.update(self._snr_metrics)
        for k, v in kl_penalty_metrics.items():
            step_metrics[f"train/{k}"] = v

        logger.log_metrics(step_metrics, step=global_step)

        p_hat_display = list(avg_p_hat.values())[0] if avg_p_hat else 0.0
        gap_display = list(avg_p_hat.values())[0] - avg_p_ref if avg_p_hat else 0.0
        pbar.set_postfix({"kl": f"{kl_v1:.4f}", "gap": f"{gap_display:.3f}", "p_hat": f"{p_hat_display:.2f}", "adv": f"{adv_abs_mean:.3f}"})
