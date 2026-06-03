"""
Unit tests for RL consistency training module.

Tests the core functions, reward computation, rate estimation, and config
classes in cot_transparency/apis/tinker/rl_training.py.

No Tinker API connection needed — all external calls are mocked.

Run: python tests/test_rl_training.py
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent

from cot_transparency.apis.tinker.rl_training import (
    _resolve_indices,
    _select_rollouts,
    RLConfig,
    RLTrainer,
    RateEstimationConfig,
    TrainingSamplingConfig,
    TrainingLoopConfig,
    GenerationConfig,
    Rollout,
    RolloutResult,
    BatchItem,
    ConsistencyReward,
)
from cot_transparency.apis.tinker.common import CheckpointConfig, AdamConfig, LoRAConfig


# =============================================================================
# Helpers
# =============================================================================

def make_rollout(
    perturbation_idx: int = 0,
    trait_value: float = 1.0,
    parsed_successfully: bool = True,
    tokens: list[int] | None = None,
    logprobs: list[float] | None = None,
    text: str = "fake response",
) -> Rollout:
    """Create a Rollout with sensible defaults for testing."""
    return Rollout(
        tokens=tokens or [1, 2, 3],
        logprobs=logprobs or [-0.5, -0.3, -0.1],
        text=text,
        trait_value=trait_value,
        perturbation_idx=perturbation_idx,
        parsed_successfully=parsed_successfully,
    )


def make_minimal_trainer(**config_overrides) -> RLTrainer:
    """Create an RLTrainer with mocked tinker.ServiceClient."""
    config = RLConfig(**config_overrides)
    with patch("tinker.ServiceClient"):
        return RLTrainer(config=config, reward_function=ConsistencyReward())


# =============================================================================
# Pure Function Tests
# =============================================================================

class TestResolveIndices:
    def test_all_keyword(self):
        assert _resolve_indices("all", 5) == [0, 1, 2, 3, 4]

    def test_list_passthrough(self):
        assert _resolve_indices([1, 3], 5) == [1, 3]

    def test_empty_all(self):
        assert _resolve_indices("all", 0) == []

    def test_empty_list(self):
        assert _resolve_indices([], 5) == []


class TestAggregateRates:
    """Tests aggregation logic now inlined in _aggregate_ref_rates."""

    def _aggregate(self, rates_list: list[float], aggregation: str = "mean") -> float:
        """Helper: build a trainer with the given aggregation and call _aggregate_ref_rates."""
        trainer = make_minimal_trainer(
            reference_rate=RateEstimationConfig(perturbation_indices=[0], aggregation=aggregation)
        )
        rates = {i: v for i, v in enumerate(rates_list)}
        indices = list(range(len(rates_list)))
        return trainer._aggregate_ref_rates(rates, indices)

    def test_mean(self):
        assert self._aggregate([0.2, 0.4, 0.6], "mean") == pytest.approx(0.4)

    def test_min(self):
        assert self._aggregate([0.2, 0.4, 0.6], "min") == 0.2

    def test_max(self):
        assert self._aggregate([0.2, 0.4, 0.6], "max") == 0.6

    def test_single_value(self):
        assert self._aggregate([0.3], "mean") == pytest.approx(0.3)

    def test_invalid_raises(self):
        # aggregation is a Literal["mean","min","max"]; an invalid value is now rejected
        # by pydantic at config construction (ValidationError ⊂ ValueError), before
        # _aggregate_ref_rates' defensive `raise ValueError("Unknown aggregation")` is reached.
        with pytest.raises(ValueError):
            self._aggregate([0.5], "median")


class TestSelectRollouts:
    def test_filters_unparsed(self):
        rollouts = {
            0: [
                make_rollout(perturbation_idx=0, parsed_successfully=True),
                make_rollout(perturbation_idx=0, parsed_successfully=False),
                make_rollout(perturbation_idx=0, parsed_successfully=True),
            ]
        }
        result = _select_rollouts(rollouts, [0], n_gradient=None)
        assert len(result) == 2
        assert all(r.parsed_successfully for r in result)

    def test_n_gradient_limits(self):
        rollouts = {
            0: [make_rollout(perturbation_idx=0) for _ in range(10)]
        }
        result = _select_rollouts(rollouts, [0], n_gradient=3)
        assert len(result) == 3

    def test_n_gradient_none_returns_all_parsed(self):
        rollouts = {
            0: [make_rollout(perturbation_idx=0) for _ in range(5)]
        }
        result = _select_rollouts(rollouts, [0], n_gradient=None)
        assert len(result) == 5

    def test_missing_index_returns_empty(self):
        rollouts = {0: [make_rollout(perturbation_idx=0)]}
        result = _select_rollouts(rollouts, [99], n_gradient=None)
        assert len(result) == 0

    def test_multiple_indices(self):
        rollouts = {
            0: [make_rollout(perturbation_idx=0) for _ in range(3)],
            1: [make_rollout(perturbation_idx=1) for _ in range(4)],
        }
        result = _select_rollouts(rollouts, [0, 1], n_gradient=2)
        # 2 per index = 4 total
        assert len(result) == 4


class TestNormalizeAdvantages:
    """Test RLTrainer._normalize_advantages (stateless method)."""

    def test_basic_normalization(self):
        trainer = make_minimal_trainer()
        result = trainer._normalize_advantages([1.0, 3.0, 5.0])
        # mean=3, std=sqrt(8/3)≈1.633
        assert pytest.approx(sum(result) / len(result), abs=1e-6) == 0.0
        # Check std is ~1
        var = sum(r**2 for r in result) / len(result)
        assert pytest.approx(var, abs=1e-6) == 1.0

    def test_zero_std_returns_zeros(self):
        trainer = make_minimal_trainer()
        result = trainer._normalize_advantages([2.0, 2.0, 2.0])
        assert result == [0.0, 0.0, 0.0]

    def test_empty_returns_empty(self):
        trainer = make_minimal_trainer()
        result = trainer._normalize_advantages([])
        assert result == []

    def test_two_symmetric_values(self):
        trainer = make_minimal_trainer()
        result = trainer._normalize_advantages([-1.0, 1.0])
        assert pytest.approx(result[0], abs=1e-6) == -1.0
        assert pytest.approx(result[1], abs=1e-6) == 1.0


# =============================================================================
# ConsistencyReward Tests
# =============================================================================

class TestConsistencyReward:
    """Test reward computation logic.

    The consistency reward formula is:
        r = -(p_hat - p_ref) * (trait - p_hat)

    When p_hat > p_ref (model is more biased than reference):
    - trait=1 rollouts get NEGATIVE reward (discourage bias)
    - trait=0 rollouts get POSITIVE reward (encourage non-bias)
    """

    def test_consistency_reward_direction(self):
        reward_fn = ConsistencyReward()
        p_hat = {1: 0.7}
        p_ref = 0.3
        rollouts = [
            make_rollout(perturbation_idx=1, trait_value=1.0),
            make_rollout(perturbation_idx=1, trait_value=0.0),
        ]
        rewards = reward_fn.compute_rewards(rollouts, p_hat, p_ref)

        # trait=1: -(0.7-0.3)*(1.0-0.7) = -(0.4)*(0.3) = -0.12
        assert pytest.approx(rewards[0], abs=1e-6) == -0.12
        # trait=0: -(0.7-0.3)*(0.0-0.7) = -(0.4)*(-0.7) = +0.28
        assert pytest.approx(rewards[1], abs=1e-6) == 0.28

    def test_zero_gap_gives_zero_rewards(self):
        reward_fn = ConsistencyReward()
        p_hat = {1: 0.5}
        p_ref = 0.5
        rollouts = [
            make_rollout(perturbation_idx=1, trait_value=1.0),
            make_rollout(perturbation_idx=1, trait_value=0.0),
        ]
        rewards = reward_fn.compute_rewards(rollouts, p_hat, p_ref)
        assert all(r == 0.0 for r in rewards)

    def test_reversed_gap(self):
        """When p_hat < p_ref, trait=1 should get positive reward."""
        reward_fn = ConsistencyReward()
        p_hat = {1: 0.2}
        p_ref = 0.6
        rollouts = [make_rollout(perturbation_idx=1, trait_value=1.0)]
        rewards = reward_fn.compute_rewards(rollouts, p_hat, p_ref)
        # -(0.2-0.6)*(1.0-0.2) = -(-0.4)*(0.8) = +0.32
        assert pytest.approx(rewards[0], abs=1e-6) == 0.32

    def test_anchor_rewards_direction(self):
        """Anchor reward pushes p_ref back toward p_ref_initial."""
        reward_fn = ConsistencyReward()
        ref_rollouts = [
            make_rollout(perturbation_idx=0, trait_value=1.0),
            make_rollout(perturbation_idx=0, trait_value=0.0),
        ]
        # p_ref drifted up from 0.3 to 0.6
        rewards = reward_fn.compute_anchor_rewards(ref_rollouts, p_ref=0.6, p_ref_initial=0.3)

        # trait=1: -(0.6-0.3)*(1.0-0.6) = -(0.3)*(0.4) = -0.12
        assert pytest.approx(rewards[0], abs=1e-6) == -0.12
        # trait=0: -(0.6-0.3)*(0.0-0.6) = -(0.3)*(-0.6) = +0.18
        assert pytest.approx(rewards[1], abs=1e-6) == 0.18

    def test_anchor_zero_drift(self):
        reward_fn = ConsistencyReward()
        ref_rollouts = [
            make_rollout(perturbation_idx=0, trait_value=1.0),
            make_rollout(perturbation_idx=0, trait_value=0.0),
        ]
        rewards = reward_fn.compute_anchor_rewards(ref_rollouts, p_ref=0.4, p_ref_initial=0.4)
        assert all(r == 0.0 for r in rewards)

    def test_compute_rewards_no_p_ref_initial_param(self):
        """compute_rewards takes exactly 3 args (rollouts, p_hat, p_ref)."""
        reward_fn = ConsistencyReward()
        p_hat = {1: 0.7}
        rollouts = [make_rollout(perturbation_idx=1, trait_value=1.0)]
        rewards = reward_fn.compute_rewards(rollouts, p_hat, 0.3)
        assert len(rewards) == 1

    def test_multiple_perturbation_indices(self):
        """Rewards use per-perturbation p_hat values."""
        reward_fn = ConsistencyReward()
        p_hat = {1: 0.8, 2: 0.4}
        p_ref = 0.5
        rollouts = [
            make_rollout(perturbation_idx=1, trait_value=1.0),
            make_rollout(perturbation_idx=2, trait_value=1.0),
        ]
        rewards = reward_fn.compute_rewards(rollouts, p_hat, p_ref)

        # pert 1: -(0.8-0.5)*(1.0-0.8) = -0.06
        assert pytest.approx(rewards[0], abs=1e-6) == -0.06
        # pert 2: -(0.4-0.5)*(1.0-0.4) = -(-0.1)*(0.6) = +0.06
        assert pytest.approx(rewards[1], abs=1e-6) == 0.06

    def test_baseline_override_centres_on_control(self):
        """matched_pair passes baseline=p_ref so the score centres on the control,
        not the per-cue p_hat; gaps may also be supplied per perturbation."""
        reward_fn = ConsistencyReward()
        p_hat = {1: 0.8}
        p_ref = 0.3
        rollouts = [
            make_rollout(perturbation_idx=1, trait_value=1.0),
            make_rollout(perturbation_idx=1, trait_value=0.0),
        ]
        # gap fixed to 0.4, baseline = p_ref = 0.3
        rewards = reward_fn.compute_rewards(rollouts, p_hat, p_ref, gaps={1: 0.4}, baseline=0.3)
        # trait=1: -0.4*(1.0-0.3) = -0.28 ; trait=0: -0.4*(0.0-0.3) = +0.12
        assert pytest.approx(rewards[0], abs=1e-6) == -0.28
        assert pytest.approx(rewards[1], abs=1e-6) == 0.12
        # baseline=None preserves the original p_hat-centred behaviour
        default = reward_fn.compute_rewards(rollouts, p_hat, p_ref)
        assert pytest.approx(default[0], abs=1e-6) == -(0.8 - 0.3) * (1.0 - 0.8)


# =============================================================================
# RLTrainer Method Tests
# =============================================================================

class TestComputeRates:
    def test_basic_rates(self):
        trainer = make_minimal_trainer()
        rollouts = {
            0: [
                make_rollout(perturbation_idx=0, trait_value=1.0),
                make_rollout(perturbation_idx=0, trait_value=0.0),
                make_rollout(perturbation_idx=0, trait_value=1.0),
            ]
        }
        rates, counts = trainer._compute_rates(rollouts, [0])
        assert pytest.approx(rates[0]) == 2.0 / 3.0
        assert counts[0] == 3

    def test_unparsed_excluded(self):
        trainer = make_minimal_trainer()
        rollouts = {
            0: [
                make_rollout(perturbation_idx=0, trait_value=1.0, parsed_successfully=True),
                make_rollout(perturbation_idx=0, trait_value=1.0, parsed_successfully=False),
                make_rollout(perturbation_idx=0, trait_value=0.0, parsed_successfully=True),
            ]
        }
        rates, counts = trainer._compute_rates(rollouts, [0])
        # Only 2 parsed: traits [1.0, 0.0] → rate 0.5
        assert pytest.approx(rates[0]) == 0.5
        assert counts[0] == 2

    def test_all_unparsed_returns_none(self):
        trainer = make_minimal_trainer()
        rollouts = {
            0: [
                make_rollout(perturbation_idx=0, parsed_successfully=False),
                make_rollout(perturbation_idx=0, parsed_successfully=False),
            ]
        }
        rates, counts = trainer._compute_rates(rollouts, [0])
        assert rates[0] is None
        assert counts[0] == 0

    def test_empty_index_returns_none(self):
        trainer = make_minimal_trainer()
        rates, counts = trainer._compute_rates({}, [0])
        assert rates[0] is None
        assert counts[0] == 0


class TestAggregateRefRates:
    def test_single_index(self):
        trainer = make_minimal_trainer()
        rates = {0: 0.4}
        result = trainer._aggregate_ref_rates(rates, [0])
        assert pytest.approx(result) == 0.4

    def test_multiple_indices_mean(self):
        trainer = make_minimal_trainer()
        rates = {0: 0.2, 1: 0.6}
        result = trainer._aggregate_ref_rates(rates, [0, 1])
        assert pytest.approx(result) == 0.4

    def test_all_none_returns_none(self):
        trainer = make_minimal_trainer()
        rates = {0: None, 1: None}
        result = trainer._aggregate_ref_rates(rates, [0, 1])
        assert result is None

    def test_mixed_none_ignores_none(self):
        trainer = make_minimal_trainer()
        rates = {0: 0.6, 1: None}
        result = trainer._aggregate_ref_rates(rates, [0, 1])
        assert pytest.approx(result) == 0.6

    def test_custom_aggregation_min(self):
        """Uses config.reference_rate.aggregation for the agg method."""
        trainer = make_minimal_trainer(
            reference_rate=RateEstimationConfig(
                perturbation_indices=[0, 1],
                n_rollouts=64,
                aggregation="min",
            )
        )
        rates = {0: 0.3, 1: 0.7}
        result = trainer._aggregate_ref_rates(rates, [0, 1])
        assert pytest.approx(result) == 0.3


# =============================================================================
# Config Tests
# =============================================================================

class TestRLConfig:
    def test_defaults(self):
        config = RLConfig()
        assert config.kl_coef == 0.05
        assert config.loss_fn == "ppo"
        assert config.loop.batch_size == 1
        assert config.loop.n_epochs == 1
        assert config.generation.temperature == 0.7

    def test_nested_config_access(self):
        config = RLConfig(
            training=TrainingSamplingConfig(
                perturbation_indices=[1, 2],
                n_rollouts_for_rate=32,
                n_rollouts_for_consistency=16,
            )
        )
        assert config.training.perturbation_indices == [1, 2]
        assert config.training.n_rollouts_for_rate == 32
        assert config.training.n_rollouts_for_consistency == 16

    def test_model_dump_roundtrip(self):
        config = RLConfig(experiment_name="test", run_name="r1")
        dumped = config.model_dump()
        restored = RLConfig(**dumped)
        assert restored.experiment_name == "test"
        assert restored.run_name == "r1"
        assert restored.model_dump() == dumped


class TestBuildTrainingBatch:
    """Test the extracted _build_training_batch method."""

    def test_returns_datums_and_advantages(self):
        from tinker import types

        trainer = make_minimal_trainer(anchor_weight=0.5)
        trainer.reward_function = ConsistencyReward()

        prompt = types.ModelInput.from_ints(tokens=[1, 2, 3])
        train_rollouts = [
            Rollout(tokens=[4, 5], logprobs=[-0.2, -0.3], text="r1",
                    trait_value=1.0, perturbation_idx=1, prompt=prompt),
            Rollout(tokens=[6, 7], logprobs=[-0.4, -0.5], text="r2",
                    trait_value=0.0, perturbation_idx=1, prompt=prompt),
        ]
        anchor_rollouts = [
            Rollout(tokens=[8, 9], logprobs=[-0.1, -0.2], text="r3",
                    trait_value=1.0, perturbation_idx=0, prompt=prompt),
        ]

        batch_items = [BatchItem(
            datapoint_idx=0,
            datapoint={"id": 0},
            train_rollouts=train_rollouts,
            anchor_rollouts=anchor_rollouts,
            p_hat={1: 0.7},
            p_hat_counts={1: 2},
            p_ref=0.3,
            p_ref_init=0.3,
            n_total=3,
            n_parsed=3,
            n_ref_parsed=1,
            n_training_parsed=2,
        )]

        # Mock _create_rl_datum since it needs real tinker types
        trainer._create_rl_datum = lambda prompt_input, rollout, adv: MagicMock()

        result = trainer._build_training_batch(batch_items)
        assert result is not None
        datums, cons_rewards, anch_rewards, advantages, _ = result
        assert len(datums) == 3  # 2 train + 1 anchor
        assert len(cons_rewards) == 2
        assert len(anch_rewards) == 1
        assert len(advantages) == 3


class TestMatchedPairEstimator:
    """advantage_estimator='matched_pair': pool the cued rate across the cue family into
    ONE gap vs p_ref, and score every cued rollout against p_ref (the matched control)."""

    def _trainer(self, **overrides):
        cfg = dict(advantage_estimator="matched_pair", snr_z=0.0, anchor_weight=0.0)
        cfg.update(overrides)
        trainer = make_minimal_trainer(**cfg)
        trainer.reward_function = ConsistencyReward()
        trainer._create_rl_datum = lambda prompt_input, rollout, adv: MagicMock()
        return trainer

    def _item(self, p_hat, p_hat_counts, p_ref, rollouts):
        from tinker import types
        prompt = types.ModelInput.from_ints(tokens=[1, 2, 3])
        for r in rollouts:
            r.prompt = prompt
        return BatchItem(
            datapoint_idx=0, datapoint={"id": 0},
            train_rollouts=rollouts, anchor_rollouts=[],
            p_hat=p_hat, p_hat_counts=p_hat_counts, p_ref=p_ref, p_ref_init=None,
            n_total=sum(p_hat_counts.values()), n_parsed=sum(p_hat_counts.values()),
            n_ref_parsed=8, n_training_parsed=sum(p_hat_counts.values()),
        )

    def test_pooled_gap_and_control_baseline(self):
        # p_pool = (0.8*4 + 0.6*4)/8 = 0.7 ; gap = 0.7 - 0.3 = 0.4 ; baseline = p_ref = 0.3
        trainer = self._trainer(snr_normalizer="none")  # adv == raw reward
        item = self._item(
            p_hat={1: 0.8, 2: 0.6}, p_hat_counts={1: 4, 2: 4}, p_ref=0.3,
            rollouts=[
                make_rollout(perturbation_idx=1, trait_value=1.0),
                make_rollout(perturbation_idx=2, trait_value=0.0),
            ],
        )
        _datums, cons_rewards, _anch, advantages, _ = trainer._build_training_batch([item])
        # pert 1, trait=1: -0.4*(1.0-0.3) = -0.28 ; pert 2, trait=0: -0.4*(0.0-0.3) = +0.12
        # Note: the gap is SHARED across cues (pooled) and the baseline is p_ref for BOTH.
        assert pytest.approx(cons_rewards[0], abs=1e-6) == -0.28
        assert pytest.approx(cons_rewards[1], abs=1e-6) == 0.12
        assert advantages == pytest.approx(cons_rewards, abs=1e-6)  # anchor_weight=0, norm=none

    def test_pooled_gap_zero_skips_batch(self):
        # One cue above, one below the reference: pooling cancels (0.7) == p_ref (0.7) → gap 0.
        trainer = self._trainer()
        item = self._item(
            p_hat={1: 0.9, 2: 0.5}, p_hat_counts={1: 2, 2: 2}, p_ref=0.7,
            rollouts=[
                make_rollout(perturbation_idx=1, trait_value=1.0),
                make_rollout(perturbation_idx=2, trait_value=1.0),
            ],
        )
        assert trainer._build_training_batch([item]) is None

    def test_magnitude_preserving_not_grpo(self):
        # Default normalizer 'trait_std' keeps the gap magnitude (divides by per-rollout
        # Bernoulli std), unlike grpo_normalized which would standardise to mean 0 / std 1.
        trainer = self._trainer()  # snr_normalizer defaults to 'trait_std'
        item = self._item(
            p_hat={1: 0.8, 2: 0.6}, p_hat_counts={1: 4, 2: 4}, p_ref=0.3,
            rollouts=[
                make_rollout(perturbation_idx=1, trait_value=1.0),
                make_rollout(perturbation_idx=2, trait_value=0.0),
            ],
        )
        _d, _cr, _ar, advantages, _ = trainer._build_training_batch([item])
        div = (0.7 * 0.3 + 0.01) ** 0.5  # trait_std(p_pool=0.7)
        assert advantages[0] == pytest.approx(-0.28 / div, abs=1e-6)
        assert advantages[1] == pytest.approx(0.12 / div, abs=1e-6)
        # not grpo: the two advantages do NOT sum to ~0 (gap magnitude survives)
        assert abs(advantages[0] + advantages[1]) > 1e-3


class TestSEHelpers:
    """Statistics helpers: smoothed binomial variance + cluster-robust pooled SE (the
    matched_pair over-shrink fix)."""

    def test_binom_var_boundary_is_nonzero(self):
        # p=0 must NOT look like zero uncertainty (rule of three); smoothing recovers it.
        assert RLTrainer._binom_var(0.0, 10) > 0.0
        assert RLTrainer._binom_var(1.0, 10) > 0.0

    def test_gap_se_uses_binom_var(self):
        se = RLTrainer._gap_se(0.4, 50, 0.2, 50)
        expected = (RLTrainer._binom_var(0.4, 50) + RLTrainer._binom_var(0.2, 50)) ** 0.5
        assert pytest.approx(se) == expected

    def test_matched_pair_se_below_pooled_binomial_for_heterogeneous_cues(self):
        # Two cues at very different rates: the cluster-robust cued SE must be SMALLER than
        # the naive single-pooled-binomial SE (which counts between-cue spread as noise).
        p_hat, counts, p_ref, n_ref = {1: 0.9, 2: 0.1}, {1: 8, 2: 8}, 0.5, 64
        cluster = RLTrainer._matched_pair_gap_se(p_hat, counts, p_ref, n_ref)
        pooled = RLTrainer._gap_se(0.5, 16, p_ref, n_ref)  # old behaviour: p_pool=0.5, n_pool=16
        assert cluster < pooled

    def test_matched_pair_se_comparable_for_homogeneous_cues(self):
        # When cues share a rate, the stratified cued variance ≈ the pooled binomial.
        cluster = RLTrainer._matched_pair_gap_se({1: 0.5, 2: 0.5}, {1: 8, 2: 8}, 0.5, 64)
        pooled = RLTrainer._gap_se(0.5, 16, 0.5, 64)
        assert pytest.approx(cluster, rel=0.2) == pooled

    def test_matched_pair_se_handles_zero_counts(self):
        # No parsed counts: must not divide by zero.
        se = RLTrainer._matched_pair_gap_se({1: 0.8, 2: 0.4}, {1: 0, 2: 0}, 0.3, 0)
        assert se > 0.0


class TestMatchedPairShrinkFix:
    """End-to-end: with z>0 the cluster-robust SE shrinks heterogeneous-cue gaps LESS than
    the old pooled-binomial SE would have."""

    def test_cluster_se_shrinks_less_than_pooled(self):
        trainer = make_minimal_trainer(advantage_estimator="matched_pair", snr_z=2.0,
                                       anchor_weight=0.0, snr_normalizer="none")
        trainer.reward_function = ConsistencyReward()
        trainer._create_rl_datum = lambda p, r, a: MagicMock()
        from tinker import types
        prompt = types.ModelInput.from_ints(tokens=[1, 2, 3])
        rollouts = [make_rollout(perturbation_idx=1, trait_value=1.0),
                    make_rollout(perturbation_idx=2, trait_value=1.0)]
        for r in rollouts:
            r.prompt = prompt
        item = BatchItem(
            datapoint_idx=0, datapoint={}, train_rollouts=rollouts, anchor_rollouts=[],
            p_hat={1: 0.9, 2: 0.1}, p_hat_counts={1: 8, 2: 8}, p_ref=0.2, p_ref_init=None,
            n_total=16, n_parsed=16, n_ref_parsed=64, n_training_parsed=16,
        )
        trainer._build_training_batch([item])
        g_used = trainer._snr_metrics["train/gap_snr_scaled_abs_mean"]

        raw = 0.5 - 0.2  # pooled gap
        g_pooled = abs(trainer._snr_scale_gap(raw, RLTrainer._gap_se(0.5, 16, 0.2, 64)))
        g_cluster = abs(trainer._snr_scale_gap(
            raw, RLTrainer._matched_pair_gap_se({1: 0.9, 2: 0.1}, {1: 8, 2: 8}, 0.2, 64)))
        assert g_used == pytest.approx(g_cluster, abs=1e-9)   # the path uses the cluster SE
        assert g_used > g_pooled                              # ...and shrinks less


class TestLossFnContract:
    """The CLI no longer advertises an invalid loss_fn; the valid one is reachable."""

    def test_importance_sampling_valid(self):
        assert RLConfig(loss_fn="importance_sampling").loss_fn == "importance_sampling"

    def test_reinforce_rejected(self):
        with pytest.raises(Exception):
            RLConfig(loss_fn="reinforce")


class TestDecodeResponseRobustness:
    """decode_response guards the parse_response call itself (RendererError for gpt-oss)."""

    def test_falls_back_when_parse_raises(self):
        from cot_transparency.apis.tinker.inference import decode_response

        class Renderer:
            def parse_response(self, tokens):
                raise RuntimeError("boom")

        class Tok:
            def decode(self, tokens):
                return "decoded-fallback"

        assert decode_response(Renderer(), Tok(), [1, 2, 3]) == "decoded-fallback"


class TestUnparsedHandling:
    """discard (default) excludes unparsed rollouts from rate+gradient; resample re-samples until usable and
    logs retry amplification + give-up counts (so hedging stays visible)."""

    def _trainer(self, **ov):
        cfg = dict(
            reference_rate=RateEstimationConfig(perturbation_indices=[0], n_rollouts=4),
            training=TrainingSamplingConfig(perturbation_indices=[1], n_rollouts_for_rate=4,
                                            n_rollouts_for_consistency=4),
        )
        cfg.update(ov)
        t = make_minimal_trainer(**cfg)
        t.renderer = MagicMock()
        t.renderer.build_generation_prompt = lambda m: "PROMPT"
        return t

    @staticmethod
    def _fake_sampler():
        # each call returns n samples: the FIRST is usable "(A)", the rest unparseable
        async def fake(client, prompt, n):
            return [([1, 2], [-0.1, -0.2], "full",
                     "The best answer is: (A)." if i == 0 else "no commitment") for i in range(n)]
        return fake

    _parser = staticmethod(lambda a: "A" if "(A)" in a else None)
    _trait = staticmethod(lambda a, dp: 1.0 if "(A)" in a else 0.0)
    _perts = [lambda dp: {"messages": [{"role": "user", "content": "q"}]},
              lambda dp: {"messages": [{"role": "user", "content": "q"}]}]

    def test_discard_excludes_unparsed_no_resample(self):
        t = self._trainer(unparsed_handling="discard")
        t._sample_from_client = self._fake_sampler()
        r = asyncio.run(t._collect_rollouts({"x": 1}, self._perts, self._trait, answer_parser=self._parser))
        assert r.resample_stats == {}
        # one pass of 4; only the single usable rollout is retained for the gradient
        assert sum(rr.parsed_successfully for rr in r.train_rollouts) == 1

    def test_resample_refills_and_logs_retries(self):
        t = self._trainer(unparsed_handling="resample", max_resample_attempts=3)
        t._sample_from_client = self._fake_sampler()
        r = asyncio.run(t._collect_rollouts({"x": 1}, self._perts, self._trait, answer_parser=self._parser))
        s = r.resample_stats
        # 1 usable/call, want 4, 3 rounds -> drew 4+3+2=9, kept 3, gave up 1 (per side)
        assert (s["train_want"], s["train_drawn"], s["train_gave_up"]) == (4, 9, 1)
        assert (s["ref_want"], s["ref_drawn"], s["ref_gave_up"]) == (4, 9, 1)
        # every retained rollout is usable (parse_rate -> ~1 under resample)
        assert all(rr.parsed_successfully for rr in r.train_rollouts)


# =============================================================================
# Runner
# =============================================================================

async def main():
    """Run all tests with basic reporting."""
    import inspect

    test_classes = [
        TestResolveIndices,
        TestAggregateRates,
        TestSelectRollouts,
        TestNormalizeAdvantages,
        TestConsistencyReward,
        TestComputeRates,
        TestAggregateRefRates,
        TestRLConfig,
        TestBuildTrainingBatch,
        TestMatchedPairEstimator,
        TestSEHelpers,
        TestMatchedPairShrinkFix,
        TestLossFnContract,
        TestDecodeResponseRobustness,
        TestUnparsedHandling,
    ]

    passed = 0
    failed = 0

    for cls in test_classes:
        print(f"\n{'─' * 60}")
        print(f"  {cls.__name__}")
        print(f"{'─' * 60}")
        instance = cls()
        for name, method in inspect.getmembers(instance, predicate=inspect.ismethod):
            if not name.startswith("test_"):
                continue
            try:
                result = method()
                if asyncio.iscoroutine(result):
                    await result
                print(f"  ✅ {name}")
                passed += 1
            except Exception as e:
                print(f"  ❌ {name}: {e}")
                import traceback
                traceback.print_exc()
                failed += 1

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
