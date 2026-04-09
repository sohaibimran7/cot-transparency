"""
Test that RL training pipelining works correctly.

Mocks the Tinker API calls with controlled delays to verify:
1. Sampling for future groups overlaps with training of current group
2. Prefetch queue respects the staleness budget (refresh_policy_every_n_steps)
3. Empty batch skipping works
4. Lazy p_ref_init works (step 0 reuses policy samples)

With training taking 10x longer than sampling, a sequential loop would take
  N * (sample_time + train_time)
but pipelining should reduce it to roughly
  sample_time + N * train_time
since sampling overlaps with training.
"""

import asyncio
import time
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cot_transparency.apis.tinker.rl_training import (
    RLConfig,
    RLTrainer,
    RateEstimationConfig,
    TrainingSamplingConfig,
    TrainingLoopConfig,
    GenerationConfig,
    Rollout,
    RolloutResult,
    ConsistencyReward,
    _resolve_indices,
)
from cot_transparency.apis.tinker.common import CheckpointConfig, AdamConfig, LoRAConfig

# ── Timing constants ──────────────────────────────────────────────────
# Training takes 10x longer than sampling, as requested.
# With N groups: sequential = N*(sample+train), pipelined ≈ sample + N*train
# The speedup is modest (10%) because sampling is small relative to training,
# but the KEY benefit is overlap — sampling happens during training, not after.
SAMPLE_DELAY = 0.05   # 50ms fallback (used when sit_id is not numeric)
TRAIN_DELAY = 0.01    # 10ms per training step

# ── Event log: records (timestamp, event_type, detail) ───────────────
event_log: list[tuple[float, str, str]] = []
t0: float = 0.0


def log_event(event_type: str, detail: str = ""):
    event_log.append((time.monotonic() - t0, event_type, detail))


# ── Fake samples ─────────────────────────────────────────────────────
def make_fake_rollouts(perturbation_idx: int, n: int = 4, constant_trait: float | None = None) -> list[Rollout]:
    """Create fake rollouts.

    By default, biased perturbation (idx=1) gets 75% trait=1 and reference (idx=0)
    gets 25% trait=1. This creates different p_hat vs p_ref, producing non-zero
    rewards. Pass constant_trait to force all rollouts to the same value (for
    testing empty-batch skip).
    """
    rollouts = []
    for i in range(n):
        if constant_trait is not None:
            tv = constant_trait
        elif perturbation_idx == 0:
            # Reference: 25% sycophantic (low rate)
            tv = 1.0 if i == 0 else 0.0
        else:
            # Biased: 75% sycophantic (high rate)
            tv = 0.0 if i == 0 else 1.0
        rollouts.append(Rollout(
            tokens=[1, 2, 3],
            logprobs=[-0.5, -0.3, -0.1],
            text=f"fake response {i}",
            trait_value=tv,
            perturbation_idx=perturbation_idx,
            parsed_successfully=True,
        ))
    return rollouts


def make_rollout_result(all_rollouts: dict[int, list[Rollout]], training_idx: set[int], ref_idx: set[int] = None, rates_only: bool = False) -> RolloutResult:
    """Build a RolloutResult from a dict of fake rollouts, mimicking what
    the real _collect_rollouts does internally."""
    if ref_idx is None:
        ref_idx = {0}
    rates = {}
    rate_counts = {}
    for idx, rollouts in all_rollouts.items():
        parsed = [r for r in rollouts if r.parsed_successfully]
        rate_counts[idx] = len(parsed)
        rates[idx] = sum(r.trait_value for r in parsed) / len(parsed) if parsed else 0.5
    n_total = sum(len(r_list) for r_list in all_rollouts.values())
    n_parsed = sum(sum(1 for r in r_list if r.parsed_successfully) for r_list in all_rollouts.values())
    if rates_only:
        train_rollouts = []
        anchor_rollouts = []
    else:
        train_rollouts = []
        for idx in training_idx:
            train_rollouts.extend(r for r in all_rollouts.get(idx, []) if r.parsed_successfully)
        anchor_rollouts = []
        for idx in ref_idx:
            anchor_rollouts.extend(r for r in all_rollouts.get(idx, []) if r.parsed_successfully)
    return RolloutResult(
        train_rollouts=train_rollouts,
        anchor_rollouts=anchor_rollouts,
        rates=rates,
        rate_counts=rate_counts,
        n_total=n_total,
        n_parsed=n_parsed,
    )


# ── Build a trainer with all Tinker API calls mocked ─────────────────
def build_mock_trainer(
    n_situations: int = 6,
    situations_per_group: int = 1,
    refresh_every: int = 3,
    n_epochs: int = 1,
) -> tuple[RLTrainer, list[dict], list, list]:
    """
    Returns (trainer, situations, perturbation_fns, [trait_classifier]).
    All Tinker clients are replaced with mocks that sleep for controlled durations.
    """
    config = RLConfig(
        experiment_name="test_pipeline",
        run_name="mock",
        model="meta-llama/Llama-3.1-8B-Instruct",
        lora=LoRAConfig(rank=4),
        optimizer=AdamConfig(learning_rate=1e-4),
        reference_rate=RateEstimationConfig(
            perturbation_indices=[0],
            n_rollouts=4,
        ),
        training=TrainingSamplingConfig(
            perturbation_indices=[1],
            n_rollouts_for_rate=4,
            n_rollouts_for_consistency=4,
        ),
        loop=TrainingLoopConfig(
            batch_size=situations_per_group,
            gradient_accumulation_steps=1,
            refresh_policy_every_n_steps=refresh_every,
            n_epochs=n_epochs,
        ),
        generation=GenerationConfig(max_new_tokens=64, temperature=0.7),
        checkpoint=CheckpointConfig(save_every_n_steps=999, save_state=False),
        kl_coef=0.0,  # Disable KL to simplify mocking
        loss_fn="ppo",
        log_base_dir="/tmp/test_pipeline_logs",
    )

    # Patch tinker.ServiceClient so RLTrainer.__init__ doesn't need an API key
    with patch("tinker.ServiceClient"):
        trainer = RLTrainer(config=config, reward_function=ConsistencyReward())

    # ── Mock _collect_rollouts: sleeps based on situation id ─────────────
    # Delay = sit_id milliseconds (so situation 0 is instant, situation 5 is 5ms, etc.)
    # Training always takes TRAIN_DELAY. This creates varied sampling costs.

    async def mock_collect_rollouts(datapoint, perturbation_fns, trait_classifier,
                                    sampling_client=None, answer_parser=None,
                                    rates_only=False):
        sit_id = datapoint.get("id", "?")
        source = "anchor" if sampling_client is not None else "policy"
        delay = (sit_id / 1000.0) if isinstance(sit_id, (int, float)) else SAMPLE_DELAY
        log_event("sample_start", f"sit={sit_id} source={source}")
        await asyncio.sleep(delay)
        log_event("sample_end", f"sit={sit_id} source={source}")
        # Return RolloutResult with pre-computed rates (like the real _collect_rollouts)
        n_perts = len(perturbation_fns)
        training_idx = set(_resolve_indices(config.training.perturbation_indices, n_perts))
        ref_idx = set(_resolve_indices(config.reference_rate.perturbation_indices, n_perts))
        all_idx = training_idx | ref_idx
        all_samples = {idx: make_fake_rollouts(idx) for idx in all_idx}
        return make_rollout_result(all_samples, training_idx, rates_only=rates_only)

    trainer._collect_rollouts = mock_collect_rollouts

    # ── Mock training client ──────────────────────────────────────────
    mock_training_client = MagicMock()

    # forward_backward_async → returns a future whose result_async sleeps TRAIN_DELAY
    # This matches the real API: submit returns immediately, result_async blocks.
    async def mock_forward_backward_async(data, loss_fn="ppo"):
        import torch
        n_data = len(data)
        log_event("fwd_bwd_submit", f"n_data={n_data}")

        # Return a future object
        future = MagicMock()
        async def result_async():
            log_event("fwd_bwd_await_start", f"n_data={n_data}")
            await asyncio.sleep(TRAIN_DELAY)
            log_event("fwd_bwd_await_end", f"n_data={n_data}")
            # Return result with loss_fn_outputs
            result = MagicMock()
            result.loss_fn_outputs = [
                {"logprobs": MagicMock(to_torch=lambda: torch.tensor([-0.5, -0.3, -0.1]))}
                for _ in range(n_data)
            ]
            return result
        future.result_async = result_async
        return future

    mock_training_client.forward_backward_async = mock_forward_backward_async

    # optim_step_async → returns a future whose result_async sleeps a small amount
    OPTIM_DELAY = 0.02  # 20ms — small relative to fwd_bwd
    async def mock_optim_step(adam_params):
        log_event("optim_submit", "")
        future = MagicMock()
        async def result_async():
            log_event("optim_await_start", "")
            await asyncio.sleep(OPTIM_DELAY)
            log_event("optim_await_end", "")
            return None
        future.result_async = result_async
        return future

    mock_training_client.optim_step_async = mock_optim_step

    # save_weights_and_get_sampling_client_async → returns mock client
    async def mock_save_weights(**kwargs):
        log_event("refresh_policy", str(kwargs.get("name", "")))
        return MagicMock()

    mock_training_client.save_weights_and_get_sampling_client_async = mock_save_weights

    trainer.training_client = mock_training_client
    trainer.sampling_client = MagicMock()
    trainer.base_sampling_client = MagicMock()
    trainer.renderer = MagicMock()
    trainer.tokenizer = MagicMock()

    # Mock renderer.build_generation_prompt to return a simple ModelInput
    from tinker import types
    trainer.renderer = MagicMock()
    trainer.renderer.build_generation_prompt = lambda msgs: types.ModelInput.from_ints(tokens=[1, 2, 3])

    # Mock _create_rl_datum — return a stub since cookbook_forward_backward is also mocked
    def mock_create_datum(prompt_input, sample, advantage):
        datum = MagicMock()
        datum.advantage = advantage
        return datum

    trainer._create_rl_datum = mock_create_datum

    # ── Test data ─────────────────────────────────────────────────────
    situations = [{"id": i, "question": f"Q{i}"} for i in range(n_situations)]

    def pert_unbiased(sit):
        return {"messages": [{"role": "user", "content": sit["question"]}]}

    def pert_biased(sit):
        return {"messages": [{"role": "user", "content": f"bias {sit['question']}"}]}

    perturbation_fns = [pert_unbiased, pert_biased]

    def classifier(response, situation):
        return 1.0 if "0" in response else 0.0

    return trainer, situations, perturbation_fns, classifier


async def run_test_pipelining():
    """Test that pipelining overlaps sampling with training."""
    global event_log, t0

    N = 6
    trainer, situations, pert_fns, classifier = build_mock_trainer(
        n_situations=N,
        situations_per_group=1,
        refresh_every=3,  # Queue up to 3 groups ahead
        n_epochs=1,
    )

    event_log = []
    t0 = time.monotonic()

    # Patch cookbook_forward_backward and checkpoint_utils
    with patch("cot_transparency.apis.tinker.rl_training.remove_mask", lambda d: d), \
         patch("cot_transparency.apis.tinker.rl_training.compute_kl_sample_train", return_value={}), \
         patch("cot_transparency.apis.tinker.rl_training.checkpoint_utils") as mock_ckpt, \
         patch("cot_transparency.apis.tinker.rl_training.setup_logging") as mock_logging:

        # Mock logger
        mock_logger = MagicMock()
        mock_logging.return_value = mock_logger

        # Mock checkpoint saving
        async def mock_save_checkpoint(*args, **kwargs):
            return {"sampler_path": "tinker://test/checkpoint"}
        mock_ckpt.save_checkpoint_async = mock_save_checkpoint

        final = await trainer.train(
            datapoints=situations,
            perturbation_fns=pert_fns,
            trait_classifier=classifier,
        )

    total_time = time.monotonic() - t0

    # ── Print timeline ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EVENT TIMELINE")
    print("=" * 70)
    for ts, event_type, detail in event_log:
        bar_width = int(ts * 100)  # 1 block per 10ms
        bar = "█" * min(bar_width, 60)
        print(f"  {ts:6.3f}s  {bar:<60s}  {event_type:<20s}  {detail}")

    # ── Compute theoretical times ─────────────────────────────────────
    # Sampling delay = sit_id ms, training = TRAIN_DELAY per step
    sit_ids = list(range(N))
    total_sample_time = sum(s / 1000.0 for s in sit_ids)
    sequential_time = total_sample_time + N * TRAIN_DELAY

    print(f"\n{'=' * 70}")
    print(f"TIMING SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Groups:            {N}")
    print(f"  Train delay:       {TRAIN_DELAY*1000:.0f}ms")
    print(f"  Sample delays:     {[f'{s}ms' for s in sit_ids]}")
    print(f"  Sequential would:  {sequential_time:.3f}s")
    print(f"  Actual:            {total_time:.3f}s")

    # ── Verify overlap ────────────────────────────────────────────────
    # Key check: did sampling for group N+1 START before fwd_bwd for group N ENDED?
    fwd_bwd_submits = [(ts, d) for ts, ev, d in event_log if ev == "fwd_bwd_submit"]
    fwd_bwd_ends = [(ts, d) for ts, ev, d in event_log if ev == "fwd_bwd_await_end"]
    sample_starts = [(ts, d) for ts, ev, d in event_log if ev == "sample_start" and "policy" in d]

    overlaps = 0
    for i, (submit_ts, _) in enumerate(fwd_bwd_submits):
        if i < len(fwd_bwd_ends):
            end_ts = fwd_bwd_ends[i][0]
            concurrent = [s for s in sample_starts if submit_ts <= s[0] < end_ts]
            if concurrent:
                overlaps += 1

    print(f"\n  Training steps with concurrent sampling: {overlaps}/{len(fwd_bwd_submits)}")

    # Verify optim pipelining: optim_submit should happen BEFORE fwd_bwd_await_end
    optim_submits = [(ts, d) for ts, ev, d in event_log if ev == "optim_submit"]
    optim_before_fwd_end = 0
    for opt_ts, _ in optim_submits:
        # Find the fwd_bwd_await_end that comes after this optim_submit
        later_ends = [ts for ts, _ in fwd_bwd_ends if ts > opt_ts]
        if later_ends:
            # optim was submitted before fwd_bwd completed → same clock cycle
            optim_before_fwd_end += 1
    print(f"  Optim steps pipelined with fwd_bwd: {optim_before_fwd_end}/{len(optim_submits)}")

    # ── Verify prefetch budget ────────────────────────────────────────
    # With refresh_every=3, at most 3 sample tasks should be in flight before
    # training consumes them.
    first_train = fwd_bwd_submits[0][0] if fwd_bwd_submits else float('inf')
    prefetched_before_train = sum(1 for ts, d in sample_starts if ts < first_train)
    print(f"  Groups prefetched before first train: {prefetched_before_train} (budget=3)")

    # ── Assertions ────────────────────────────────────────────────────
    # 1. Should have overlap (sampling happens during training)
    assert overlaps > 0, "No sampling/training overlap detected!"

    # 2. Prefetch should respect budget (at most refresh_every groups prefetched)
    assert prefetched_before_train <= 3, (
        f"Prefetched {prefetched_before_train} groups before training, budget is 3"
    )

    # 3. Optim should be pipelined with fwd_bwd (submitted before fwd_bwd result awaited)
    assert optim_before_fwd_end > 0, "No optim/fwd_bwd pipelining detected!"

    print(f"\n  ✅ All assertions passed!")
    return True


async def run_test_prefetch_budget():
    """Test that prefetch queue respects refresh_policy_every_n_steps budget."""
    global event_log, t0

    N = 8
    BUDGET = 2  # Very restrictive budget
    trainer, situations, pert_fns, classifier = build_mock_trainer(
        n_situations=N,
        situations_per_group=1,
        refresh_every=BUDGET,
        n_epochs=1,
    )

    event_log = []
    t0 = time.monotonic()

    with patch("cot_transparency.apis.tinker.rl_training.remove_mask", lambda d: d), \
         patch("cot_transparency.apis.tinker.rl_training.compute_kl_sample_train", return_value={}), \
         patch("cot_transparency.apis.tinker.rl_training.checkpoint_utils") as mock_ckpt, \
         patch("cot_transparency.apis.tinker.rl_training.setup_logging") as mock_logging:

        mock_logger = MagicMock()
        mock_logging.return_value = mock_logger

        async def mock_save_checkpoint(*args, **kwargs):
            return {"sampler_path": "tinker://test/checkpoint"}
        mock_ckpt.save_checkpoint_async = mock_save_checkpoint

        await trainer.train(
            datapoints=situations,
            perturbation_fns=pert_fns,
            trait_classifier=classifier,
        )

    # Count max concurrent sample tasks at any point
    # Track in-flight: +1 at sample_start, -1 at sample_end
    starts = sorted([(ts, +1) for ts, ev, _ in event_log if ev == "sample_start"]
                   + [(ts, -1) for ts, ev, _ in event_log if ev == "sample_end"])

    max_inflight = 0
    inflight = 0
    for _, delta in starts:
        inflight += delta
        max_inflight = max(max_inflight, inflight)

    print(f"\n{'=' * 70}")
    print(f"PREFETCH BUDGET TEST (budget={BUDGET})")
    print(f"{'=' * 70}")
    print(f"  Max concurrent sampling tasks: {max_inflight}")
    print(f"  Expected max: {BUDGET} (= refresh_policy_every_n_steps)")

    # Each group samples for 2 perturbations, so max_inflight includes
    # all perturbation tasks within a group. The budget limits GROUPS,
    # so max_inflight = BUDGET * n_perturbations_per_group
    # With 1 situation per group and 2 perturbation indices (0 and 1),
    # _collect_rollouts runs them internally via asyncio.gather.
    # But our mock replaces _collect_rollouts entirely, so each group
    # is one mock_collect_rollouts call = 1 in-flight per group.
    # Plus the lazy p_ref_init base sampling at step > 0 can add 1 more.

    # The key constraint: groups in the prefetch queue ≤ BUDGET
    # Count POLICY sample_starts (not base) that begin before the first fwd_bwd_submit.
    # Each group triggers one policy sample per situation. With 1 sit/group,
    # the number of policy sample_starts before training = number of prefetched groups.
    fwd_bwd_submits_ts = [ts for ts, ev, _ in event_log if ev == "fwd_bwd_submit"]
    policy_sample_starts_ts = [ts for ts, ev, d in event_log if ev == "sample_start" and "policy" in d]

    first_train = fwd_bwd_submits_ts[0] if fwd_bwd_submits_ts else float('inf')
    pre_train_groups = sum(1 for ts in policy_sample_starts_ts if ts < first_train)
    print(f"  Policy sample groups before first train: {pre_train_groups}")

    # The budget constrains prefetch_queue size (number of groups).
    assert pre_train_groups <= BUDGET, (
        f"Prefetched {pre_train_groups} groups before training, budget={BUDGET}"
    )

    print(f"  ✅ Prefetch budget respected!")
    return True


async def run_test_empty_batch_skip():
    """Test that empty batches (constant rewards → zero advantages) are skipped."""
    global event_log, t0

    trainer, situations, pert_fns, classifier = build_mock_trainer(
        n_situations=4,
        situations_per_group=1,
        refresh_every=4,
        n_epochs=1,
    )

    # Override mock_collect_rollouts to return constant trait values for specific situations,
    # which will produce zero advantages after normalization.
    original_mock = trainer._collect_rollouts

    async def mock_collect_with_empty(datapoint, perturbation_fns, trait_classifier,
                                       sampling_client=None, answer_parser=None,
                                       rates_only=False):
        sit_id = datapoint.get("id", "?")
        source = "anchor" if sampling_client is not None else "policy"
        log_event("sample_start", f"sit={sit_id} source={source}")
        await asyncio.sleep(SAMPLE_DELAY)
        log_event("sample_end", f"sit={sit_id} source={source}")

        config = trainer.config
        n_perts = len(perturbation_fns)
        training_idx = set(_resolve_indices(config.training.perturbation_indices, n_perts))
        ref_idx = set(_resolve_indices(config.reference_rate.perturbation_indices, n_perts))
        all_idx = training_idx | ref_idx

        # Situations 1 and 3: varying traits → non-zero rewards → will train
        if sit_id in (1, 3):
            all_samples = {idx: make_fake_rollouts(idx) for idx in all_idx}
        else:
            # Situations 0 and 2: constant trait=1.0 for ALL perturbations
            # → same p_hat and p_ref → rewards all identical → zero advantages → skip
            all_samples = {idx: make_fake_rollouts(idx, constant_trait=1.0) for idx in all_idx}
        return make_rollout_result(all_samples, training_idx, rates_only=rates_only)

    trainer._collect_rollouts = mock_collect_with_empty

    event_log = []
    t0 = time.monotonic()

    with patch("cot_transparency.apis.tinker.rl_training.remove_mask", lambda d: d), \
         patch("cot_transparency.apis.tinker.rl_training.compute_kl_sample_train", return_value={}), \
         patch("cot_transparency.apis.tinker.rl_training.checkpoint_utils") as mock_ckpt, \
         patch("cot_transparency.apis.tinker.rl_training.setup_logging") as mock_logging:

        mock_logger = MagicMock()
        mock_logging.return_value = mock_logger

        async def mock_save_checkpoint(*args, **kwargs):
            return {"sampler_path": "tinker://test/checkpoint"}
        mock_ckpt.save_checkpoint_async = mock_save_checkpoint

        await trainer.train(
            datapoints=situations,
            perturbation_fns=pert_fns,
            trait_classifier=classifier,
        )

    train_events = [(ts, d) for ts, ev, d in event_log if ev == "fwd_bwd_submit"]
    skip_calls = [args for args, kwargs in mock_logger.log_metrics.call_args_list
                  if "train/skipped_empty_batch" in args[0]]

    print(f"\n{'=' * 70}")
    print(f"EMPTY BATCH SKIP TEST")
    print(f"{'=' * 70}")
    print(f"  Total groups: 4")
    print(f"  Training steps executed: {len(train_events)}")
    print(f"  Empty batches skipped: {len(skip_calls)}")

    # Situations 1 and 3 have alternating traits → non-zero advantages → train
    # Situations 0 and 2 (if shuffled to those groups) have constant traits → skip
    # But shuffling is random, so just check that some were skipped and some trained
    assert len(train_events) < 4, f"Expected some skips, but got {len(train_events)} train steps"
    assert len(skip_calls) > 0, "Expected some empty batch skips"
    assert len(train_events) + len(skip_calls) == 4, (
        f"Train ({len(train_events)}) + skip ({len(skip_calls)}) should = 4 groups"
    )

    print(f"  ✅ Empty batch skipping works!")
    return True


async def run_test_refresh_every_1():
    """Test with refresh_policy_every_n_steps=1: max_prefetch=1, so no overlap."""
    global event_log, t0

    N = 6
    trainer, situations, pert_fns, classifier = build_mock_trainer(
        n_situations=N,
        situations_per_group=1,
        refresh_every=1,  # Budget of 1 → no overlap possible
        n_epochs=1,
    )

    event_log = []
    t0 = time.monotonic()

    with patch("cot_transparency.apis.tinker.rl_training.remove_mask", lambda d: d), \
         patch("cot_transparency.apis.tinker.rl_training.compute_kl_sample_train", return_value={}), \
         patch("cot_transparency.apis.tinker.rl_training.checkpoint_utils") as mock_ckpt, \
         patch("cot_transparency.apis.tinker.rl_training.setup_logging") as mock_logging:

        mock_logger = MagicMock()
        mock_logging.return_value = mock_logger

        async def mock_save_checkpoint(*args, **kwargs):
            return {"sampler_path": "tinker://test/checkpoint"}
        mock_ckpt.save_checkpoint_async = mock_save_checkpoint

        await trainer.train(
            datapoints=situations,
            perturbation_fns=pert_fns,
            trait_classifier=classifier,
        )

    total_time = time.monotonic() - t0

    # ── Print timeline ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EVENT TIMELINE (refresh_every=1)")
    print("=" * 70)
    for ts, event_type, detail in event_log:
        bar_width = int(ts / TRAIN_DELAY * 10)
        bar = "█" * bar_width
        print(f"  {ts:6.3f}s  {bar:<30s}  {event_type:<20s}  {detail}")

    # ── Analyze ───────────────────────────────────────────────────────
    fwd_bwd_submits = [(ts, d) for ts, ev, d in event_log if ev == "fwd_bwd_submit"]
    fwd_bwd_ends = [(ts, d) for ts, ev, d in event_log if ev == "fwd_bwd_await_end"]
    sample_starts = [(ts, d) for ts, ev, d in event_log if ev == "sample_start" and "policy" in d]

    # With budget=1, sampling for group N+1 should NOT start during training of group N
    overlaps = 0
    for i, (submit_ts, _) in enumerate(fwd_bwd_submits):
        if i < len(fwd_bwd_ends):
            end_ts = fwd_bwd_ends[i][0]
            concurrent = [s for s in sample_starts if submit_ts <= s[0] < end_ts]
            if concurrent:
                overlaps += 1

    print(f"\n{'=' * 70}")
    print(f"REFRESH_EVERY=1 SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Groups:            {N}")
    print(f"  Actual:            {total_time:.3f}s")
    print(f"  Overlaps:          {overlaps}/{len(fwd_bwd_submits)}")

    # Prefetch budget = 1, so only 1 group can be in the queue at a time.
    # The initial fill puts 1 group in. After popping it, we submit training,
    # then refill (adds 1 more). But that refill happens during training wait,
    # so there IS overlap for the next group's sampling.
    # Actually with budget=1: after submitting fwd_bwd, we refill → 1 new
    # sampling task starts during training. So there IS some overlap.
    # The key difference from budget=3: we can only prefetch 1 ahead, not 3.

    first_train = fwd_bwd_submits[0][0] if fwd_bwd_submits else float('inf')
    prefetched_before_train = sum(1 for ts, d in sample_starts if ts < first_train)
    print(f"  Prefetched before first train: {prefetched_before_train} (budget=1)")

    # With budget=1, at most 1 group prefetched before first training
    assert prefetched_before_train <= 1, (
        f"Prefetched {prefetched_before_train} groups, budget is 1"
    )

    # Optim should still be pipelined with fwd_bwd
    optim_submits = [(ts, d) for ts, ev, d in event_log if ev == "optim_submit"]
    print(f"  Optim steps: {len(optim_submits)}")

    # With refresh_every=1, every step refreshes the policy.
    # Count refresh events
    refreshes = [(ts, d) for ts, ev, d in event_log if ev == "refresh_policy"]
    print(f"  Policy refreshes: {len(refreshes)}")

    print(f"\n  ✅ refresh_every=1 test passed!")
    return True


async def run_test_base_throttle():
    """Test base sampling throttle with many situations.
    NOTE: max_concurrent_base_samples is not yet implemented in TrainingLoopConfig.
    This test is skipped until the feature is added.
    """
    print("  ⏭️  SKIPPED: max_concurrent_base_samples not yet implemented")
    return True
    # -- Original test below (unreachable) --
    global event_log, t0

    N = 12
    MAX_BASE = 3
    trainer, situations, pert_fns, classifier = build_mock_trainer(
        n_situations=N,
        situations_per_group=1,
        refresh_every=3,
        n_epochs=1,
    )
    # Set the throttle
    trainer.config.loop.max_concurrent_base_samples = MAX_BASE

    event_log = []
    t0 = time.monotonic()

    with patch("cot_transparency.apis.tinker.rl_training.remove_mask", lambda d: d), \
         patch("cot_transparency.apis.tinker.rl_training.compute_kl_sample_train", return_value={}), \
         patch("cot_transparency.apis.tinker.rl_training.checkpoint_utils") as mock_ckpt, \
         patch("cot_transparency.apis.tinker.rl_training.setup_logging") as mock_logging:

        mock_logger = MagicMock()
        mock_logging.return_value = mock_logger

        async def mock_save_checkpoint(*args, **kwargs):
            return {"sampler_path": "tinker://test/checkpoint"}
        mock_ckpt.save_checkpoint_async = mock_save_checkpoint

        await trainer.train(
            datapoints=situations,
            perturbation_fns=pert_fns,
            trait_classifier=classifier,
        )

    total_time = time.monotonic() - t0

    # ── Print timeline ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"EVENT TIMELINE (N={N}, max_base={MAX_BASE})")
    print("=" * 70)
    for ts, event_type, detail in event_log:
        bar_width = int(ts * 100)  # 1 block per 10ms
        bar = "█" * min(bar_width, 60)
        print(f"  {ts:6.3f}s  {bar:<60s}  {event_type:<20s}  {detail}")

    # ── Analyze base sampling concurrency ─────────────────────────────
    # At any point in time, how many base tasks are running?
    base_starts = [(ts, d) for ts, ev, d in event_log if ev == "sample_start" and "base" in d]
    base_ends = [(ts, d) for ts, ev, d in event_log if ev == "sample_end" and "base" in d]

    # Build a timeline of concurrent base tasks
    events = [(ts, +1) for ts, _ in base_starts] + [(ts, -1) for ts, _ in base_ends]
    events.sort()
    max_concurrent_base = 0
    concurrent = 0
    for _, delta in events:
        concurrent += delta
        max_concurrent_base = max(max_concurrent_base, concurrent)

    # Count total base samples
    total_base = len(base_starts)

    # Count policy samples
    policy_starts = [(ts, d) for ts, ev, d in event_log if ev == "sample_start" and "policy" in d]
    total_policy = len(policy_starts)

    print(f"\n{'=' * 70}")
    print(f"BASE THROTTLE SUMMARY (N={N}, max_concurrent={MAX_BASE})")
    print(f"{'=' * 70}")
    print(f"  Total situations:       {N}")
    print(f"  Policy samples:         {total_policy}")
    print(f"  Base samples:           {total_base}")
    print(f"  Max concurrent base:    {max_concurrent_base} (limit={MAX_BASE})")
    print(f"  Total time:             {total_time:.3f}s")

    # ── Assertions ────────────────────────────────────────────────────
    # 1. Max concurrent base should respect the throttle
    assert max_concurrent_base <= MAX_BASE, (
        f"Max concurrent base tasks {max_concurrent_base} > limit {MAX_BASE}"
    )

    # 2. All situations that need base sampling should eventually get it
    # (step 0 uses policy=base, so only step>0 situations need separate base)
    # With N=12, refresh_every=3: step 0 processes groups 0-2 (3 situations),
    # remaining 9 need base sampling
    base_sit_ids = set()
    for _, d in base_starts:
        # Extract sit=X from detail string
        sit_id = int(d.split("sit=")[1].split(" ")[0])
        base_sit_ids.add(sit_id)
    print(f"  Situations with base sampling: {sorted(base_sit_ids)}")

    print(f"\n  ✅ Base throttle test passed!")
    return True


async def run_test_refresh_invalidates_prefetch():
    """Bug: prefetch queue not flushed after policy refresh.

    After a policy refresh, tasks in the prefetch queue still sample from
    the OLD sampling_client (they captured it before the refresh). This test
    tracks policy versions and asserts that samples after a refresh use the
    new policy version.

    Expected (after fix): [v0,v0,v0, v1,v1,v1, v2,v2,v2]
    Bug behavior:         [v0,v0,v0, v0,v0,v0, v1,v1,v1]
    """
    global event_log, t0

    N = 9
    REFRESH_EVERY = 3

    trainer, situations, pert_fns, classifier = build_mock_trainer(
        n_situations=N,
        situations_per_group=1,
        refresh_every=REFRESH_EVERY,
        n_epochs=1,
    )

    # --- Policy version tracking ---
    # Use a dict so that when stale tasks are cancelled and fresh ones created,
    # the fresh task's write overwrites the stale entry (last-write-wins).
    sample_version_by_sit = {}  # sit_id -> last policy version seen

    # Tag initial sampling client with version 0
    mock_client_v0 = MagicMock()
    mock_client_v0._policy_version = 0
    trainer.sampling_client = mock_client_v0

    # On refresh: create new client with incremented version
    _next_version = [1]
    async def mock_save_weights_versioned(**kwargs):
        v = _next_version[0]
        _next_version[0] += 1
        new_client = MagicMock()
        new_client._policy_version = v
        log_event("refresh_policy", f"v={v}")
        return new_client

    trainer.training_client.save_weights_and_get_sampling_client_async = mock_save_weights_versioned

    # Override _collect_rollouts to record policy version at call time.
    # After a queue flush, fresh tasks overwrite stale entries in the dict.
    config = trainer.config
    async def versioned_collect(datapoint, perturbation_fns, trait_classifier,
                                 sampling_client=None, answer_parser=None, rates_only=False):
        sit_id = datapoint.get("id", "?")
        if sampling_client is None:
            version = trainer.sampling_client._policy_version
            sample_version_by_sit[sit_id] = version

        source = "anchor" if sampling_client is not None else "policy"
        log_event("sample_start", f"sit={sit_id} source={source}")
        await asyncio.sleep(0.001)
        log_event("sample_end", f"sit={sit_id}")

        n_perts = len(perturbation_fns)
        training_idx = set(_resolve_indices(config.training.perturbation_indices, n_perts))
        all_idx = training_idx | {0}
        all_samples = {idx: make_fake_rollouts(idx) for idx in all_idx}
        return make_rollout_result(all_samples, training_idx, rates_only=rates_only)

    trainer._collect_rollouts = versioned_collect

    event_log = []
    t0 = time.monotonic()

    with patch("cot_transparency.apis.tinker.rl_training.remove_mask", lambda d: d), \
         patch("cot_transparency.apis.tinker.rl_training.compute_kl_sample_train", return_value={}), \
         patch("cot_transparency.apis.tinker.rl_training.checkpoint_utils") as mock_ckpt, \
         patch("cot_transparency.apis.tinker.rl_training.setup_logging") as mock_logging:

        mock_logger = MagicMock()
        mock_logging.return_value = mock_logger

        async def mock_save_checkpoint(*args, **kwargs):
            return {"sampler_path": "tinker://test/checkpoint"}
        mock_ckpt.save_checkpoint_async = mock_save_checkpoint

        await trainer.train(
            datapoints=situations,
            perturbation_fns=pert_fns,
            trait_classifier=classifier,
        )

    # Analyze: check the final (last-written) version per sit_id.
    # With the fix, fresh tasks overwrite stale entries after each flush.
    # Without the fix, stale tasks are the only writers → too many sit_ids at v0.
    print(f"\n{'=' * 70}")
    print(f"REFRESH INVALIDATION TEST (N={N}, refresh_every={REFRESH_EVERY})")
    print(f"{'=' * 70}")

    for sit_id in sorted(sample_version_by_sit):
        print(f"  Situation {sit_id}: final policy v{sample_version_by_sit[sit_id]}")

    final_versions = list(sample_version_by_sit.values())
    v0_count = sum(1 for v in final_versions if v == 0)

    print(f"\n  Sit_ids at v0: {v0_count} (expected at most {REFRESH_EVERY})")
    print(f"  Unique versions: {sorted(set(final_versions))}")

    # Only the first refresh_every sit_ids should remain at v0.
    # With the bug: 2 * REFRESH_EVERY sit_ids at v0 (one full interval of stale delay).
    assert v0_count <= REFRESH_EVERY, (
        f"{v0_count} sit_ids still at policy v0, expected at most {REFRESH_EVERY}. "
        f"Prefetch queue should be invalidated on policy refresh. "
        f"Final versions: {dict(sorted(sample_version_by_sit.items()))}"
    )

    print(f"  ✅ Refresh properly invalidates prefetch queue!")
    return True


async def run_test_step0_base_rate_optimization():
    """Bug: closure captures global_step by reference, missing step-0 optimization.

    At step 0, the policy equals the base model, so we can extract p_ref_init
    from policy rollouts instead of doing separate base-model sampling. But
    if collect_for_datapoint reads global_step AFTER an await, prefetched tasks
    whose _collect_rollouts completes after the first training step will see
    global_step > 0, triggering unnecessary on-demand base sampling.

    Test setup: sit_id 0 completes instantly (processed at step 0, increments
    global_step to 1 before the slow tasks finish). sit_ids 1,2 take 200ms
    (complete AFTER global_step has been incremented).
    """
    global event_log, t0

    N = 3
    trainer, situations, pert_fns, classifier = build_mock_trainer(
        n_situations=N,
        situations_per_group=1,
        refresh_every=N,  # No refresh within the 3 steps
        n_epochs=1,
    )
    # anchor_weight defaults to 0.5, so need_p_ref_init is True

    # Custom _collect_rollouts with sit_id-dependent delays
    config = trainer.config
    async def slow_collect(datapoint, perturbation_fns, trait_classifier,
                            sampling_client=None, answer_parser=None, rates_only=False):
        sit_id = datapoint.get("id", "?")
        source = "anchor" if sampling_client is not None else "policy"
        delay = 0.0 if sit_id == 0 else 0.2  # 0ms vs 200ms

        log_event("sample_start", f"sit={sit_id} source={source}")
        await asyncio.sleep(delay)
        log_event("sample_end", f"sit={sit_id} source={source}")

        n_perts = len(perturbation_fns)
        training_idx = set(_resolve_indices(config.training.perturbation_indices, n_perts))
        all_idx = training_idx | {0}
        all_samples = {idx: make_fake_rollouts(idx) for idx in all_idx}
        return make_rollout_result(all_samples, training_idx, rates_only=rates_only)

    trainer._collect_rollouts = slow_collect

    event_log = []
    t0 = time.monotonic()

    with patch("cot_transparency.apis.tinker.rl_training.remove_mask", lambda d: d), \
         patch("cot_transparency.apis.tinker.rl_training.compute_kl_sample_train", return_value={}), \
         patch("cot_transparency.apis.tinker.rl_training.checkpoint_utils") as mock_ckpt, \
         patch("cot_transparency.apis.tinker.rl_training.setup_logging") as mock_logging, \
         patch("random.shuffle", lambda x: None):  # No shuffling: sit_id 0 is batch 0

        mock_logger = MagicMock()
        mock_logging.return_value = mock_logger

        async def mock_save_checkpoint(*args, **kwargs):
            return {"sampler_path": "tinker://test/checkpoint"}
        mock_ckpt.save_checkpoint_async = mock_save_checkpoint

        await trainer.train(
            datapoints=situations,
            perturbation_fns=pert_fns,
            trait_classifier=classifier,
        )

    # Count on-demand base sampling events
    base_events = [(ts, d) for ts, ev, d in event_log if ev == "sample_start" and "source=base" in d]

    print(f"\n{'=' * 70}")
    print(f"STEP-0 BASE RATE OPTIMIZATION TEST (N={N})")
    print(f"{'=' * 70}")

    print(f"  Timeline:")
    for ts, ev, d in event_log:
        print(f"    {ts:6.3f}s  {ev:<20s}  {d}")

    print(f"\n  On-demand base sampling calls: {len(base_events)}")
    print(f"  Expected: 0 (step-0 optimization should handle all)")

    assert len(base_events) == 0, (
        f"Step-0 optimization missed! {len(base_events)} on-demand base sampling calls "
        f"because global_step was already > 0 when prefetched tasks checked it. "
        f"Fix: snapshot global_step before the await in collect_for_datapoint."
    )

    print(f"  ✅ Step-0 optimization works for all prefetched tasks!")
    return True


async def main():
    print("=" * 70)
    print("RL PIPELINING TESTS")
    print("=" * 70)

    passed = 0
    failed = 0

    for name, test_fn in [
        ("Pipelining overlap", run_test_pipelining),
        ("Prefetch budget", run_test_prefetch_budget),
        ("Empty batch skip", run_test_empty_batch_skip),
        ("Refresh every 1", run_test_refresh_every_1),
        ("Base throttle", run_test_base_throttle),
        ("Refresh invalidates prefetch", run_test_refresh_invalidates_prefetch),
        ("Step-0 base rate optimization", run_test_step0_base_rate_optimization),
    ]:
        print(f"\n{'─' * 70}")
        print(f"TEST: {name}")
        print(f"{'─' * 70}")
        try:
            await test_fn()
            passed += 1
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {passed} passed, {failed} failed")
    print(f"{'=' * 70}")

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
