"""
Tests for GRPOTrainer — focused on the math, not the model.

We mock the model so these run on CPU without GPU or weights.
The goal: verify that advantage computation, loss clipping,
and KL penalty are implemented correctly before you ever
run a real training job.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch
from grpo.trainer import GRPOTrainer, GRPOConfig
from grpo.rollout import RolloutBuffer, Rollout


def make_mock_model():
    model = MagicMock()
    model.parameters.return_value = iter([torch.nn.Parameter(torch.zeros(1))])
    return model


def make_rollout_buffer(group_size=4, num_groups=2) -> RolloutBuffer:
    """Create a buffer with synthetic rollouts for testing."""
    buf = RolloutBuffer()
    rewards = [1.0, 0.0, 1.0, 0.0] * num_groups  # alternating correct/wrong
    for i in range(group_size * num_groups):
        buf.add(Rollout(
            prompt_ids=torch.zeros(5, dtype=torch.long),
            completion_ids=torch.zeros(8, dtype=torch.long),
            log_probs=torch.full((8,), -2.0),
            reward=rewards[i],
        ))
    return buf


class TestGRPOConfig:
    def test_default_config(self):
        config = GRPOConfig()
        assert config.group_size == 8
        assert config.epsilon == 0.2
        assert config.beta == 0.01


class TestComputeAdvantages:
    """
    Core GRPO math — test this before anything else.
    """

    def setup_method(self):
        config = GRPOConfig(group_size=4)
        model = make_mock_model()
        ref_model = make_mock_model()
        self.trainer = GRPOTrainer(model, ref_model, config, reward_fn=None)

    def test_advantages_zero_mean_per_group(self):
        # Within each group, advantages should sum to ~0
        rewards = torch.tensor([1.0, 0.0, 1.0, 0.0,   # group 1
                                 2.0, 1.0, 3.0, 2.0])  # group 2
        advantages = self.trainer.compute_advantages(rewards)
        assert advantages.shape == rewards.shape

        # Group 1 mean should be ~0
        g1 = advantages[:4]
        assert g1.mean().abs().item() < 1e-5

        # Group 2 mean should be ~0
        g2 = advantages[4:]
        assert g2.mean().abs().item() < 1e-5

    def test_advantages_unit_std_per_group(self):
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        advantages = self.trainer.compute_advantages(rewards)
        # std should be ~1 (normalized)
        assert advantages.std().item() == pytest.approx(1.0, abs=0.1)

    def test_all_same_reward_handled(self):
        # If all rewards in a group are equal, std=0 — should not NaN
        rewards = torch.tensor([1.0, 1.0, 1.0, 1.0])
        advantages = self.trainer.compute_advantages(rewards)
        assert not torch.isnan(advantages).any()


class TestPolicyLoss:
    def setup_method(self):
        config = GRPOConfig(group_size=4, epsilon=0.2)
        model = make_mock_model()
        ref_model = make_mock_model()
        self.trainer = GRPOTrainer(model, ref_model, config, reward_fn=None)

    def test_loss_shape(self):
        batch, seq = 4, 10
        log_probs = torch.full((batch, seq), -2.0)
        old_log_probs = torch.full((batch, seq), -2.0)
        advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])
        loss, clip_frac = self.trainer.compute_policy_loss(log_probs, old_log_probs, advantages)
        assert loss.shape == ()  # scalar

    def test_clip_fraction_zero_when_ratio_is_one(self):
        # If log_probs == old_log_probs, ratio=1, no clipping
        batch, seq = 4, 10
        lp = torch.full((batch, seq), -2.0)
        adv = torch.ones(batch)
        _, clip_frac = self.trainer.compute_policy_loss(lp, lp, adv)
        assert clip_frac == pytest.approx(0.0)

    def test_negative_advantage_positive_loss(self):
        # With all-negative advantages and ratio=1, loss should be positive
        batch, seq = 2, 5
        lp = torch.full((batch, seq), -2.0)
        adv = torch.tensor([-1.0, -1.0])
        loss, _ = self.trainer.compute_policy_loss(lp, lp, adv)
        assert loss.item() > 0


class TestKLPenalty:
    def setup_method(self):
        config = GRPOConfig()
        model = make_mock_model()
        ref_model = make_mock_model()
        self.trainer = GRPOTrainer(model, ref_model, config, reward_fn=None)

    def test_kl_zero_when_identical(self):
        lp = torch.full((4, 10), -2.0)
        kl = self.trainer.compute_kl_penalty(lp, lp)
        assert kl.item() == pytest.approx(0.0, abs=1e-5)

    def test_kl_positive_when_different(self):
        log_probs = torch.full((4, 10), -2.0)
        ref_log_probs = torch.full((4, 10), -3.0)
        kl = self.trainer.compute_kl_penalty(log_probs, ref_log_probs)
        assert kl.item() > 0


class TestRolloutBuffer:
    def test_iter_batches(self):
        buf = make_rollout_buffer(group_size=4, num_groups=2)
        batches = list(buf.iter_batches(batch_size=4))
        assert len(batches) == 2
        for b in batches:
            assert "prompts" in b
            assert "completions" in b
            assert "log_probs" in b
            assert "rewards" in b

    def test_mean_reward(self):
        buf = make_rollout_buffer(group_size=4, num_groups=1)
        assert buf.mean_reward == pytest.approx(0.5)

    def test_clear(self):
        buf = make_rollout_buffer()
        buf.clear()
        assert len(buf) == 0