"""
GRPOTrainer: Group Relative Policy Optimization for LLMs.

Implements GRPO from scratch as described in:
  DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
  arXiv:2402.03300

Key design decisions:
  - Separates rollout generation from policy updates (async-friendly)
  - Supports FSDP for multi-GPU training
  - Clean abstractions for swapping reward functions and environments
"""

import torch
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from dataclasses import dataclass, field
from typing import Optional
import logging

from grpo.rollout import RolloutBuffer
from grpo.model import PolicyModel

logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    # Training
    learning_rate: float = 1e-6
    num_iterations: int = 1000
    batch_size: int = 32
    gradient_accumulation_steps: int = 4

    # GRPO-specific
    group_size: int = 8           # G: number of outputs sampled per prompt
    epsilon: float = 0.2          # PPO clip ratio
    beta: float = 0.01            # KL penalty coefficient
    gamma: float = 1.0            # reward discount

    # Rollout
    max_new_tokens: int = 512
    temperature: float = 0.9
    rollout_batch_size: int = 16

    # Infrastructure
    use_fsdp: bool = False
    num_rollout_workers: int = 4
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 10


@dataclass
class GRPOStats:
    """Tracks training metrics per iteration."""
    policy_loss: float = 0.0
    kl_divergence: float = 0.0
    mean_reward: float = 0.0
    std_reward: float = 0.0
    clip_fraction: float = 0.0
    grad_norm: float = 0.0


class GRPOTrainer:
    """
    Trains a language model using Group Relative Policy Optimization.

    GRPO replaces the critic/value network from PPO with a group-based
    baseline: for each prompt, sample G completions, compute their rewards,
    and use the group mean/std to normalize advantages. This eliminates
    the need for a separate value model, making training cheaper.

    Training loop:
        1. Sample a batch of prompts from the environment
        2. Generate G completions per prompt (rollout workers)
        3. Score completions with the reward function
        4. Compute group-normalized advantages
        5. Update policy with clipped surrogate objective + KL penalty
        6. Repeat
    """

    def __init__(
        self,
        model: PolicyModel,
        ref_model: PolicyModel,
        config: GRPOConfig,
        reward_fn,
    ):
        self.model = model
        self.ref_model = ref_model  # frozen reference model for KL penalty
        self.config = config
        self.reward_fn = reward_fn

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
        )

        self.global_step = 0

        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False

        if config.use_fsdp:
            self.model = FSDP(self.model)
            logger.info("FSDP enabled for policy model")

    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute group-relative advantages.

        For each group of G completions from the same prompt:
          advantage_i = (reward_i - mean(rewards)) / (std(rewards) + eps)

        This is the core of GRPO — no value network needed.

        Args:
            rewards: (batch_size * group_size,) flat reward tensor

        Returns:
            advantages: same shape, group-normalized
        """
        G = self.config.group_size
        # Reshape to (batch_size, G)
        rewards_grouped = rewards.view(-1, G)

        mean = rewards_grouped.mean(dim=-1, keepdim=True)
        std = rewards_grouped.std(dim=-1, keepdim=True)

        advantages = (rewards_grouped - mean) / (std + 1e-8)
        return advantages.view(-1)  # flatten back

    def compute_policy_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """
        Clipped surrogate objective from PPO, applied per-token.

        L = -min(r * A, clip(r, 1-eps, 1+eps) * A)
        where r = exp(log_pi - log_pi_old)

        Args:
            log_probs:     (batch, seq_len) current policy log probs
            old_log_probs: (batch, seq_len) log probs at rollout time
            advantages:    (batch,) group-normalized advantages

        Returns:
            loss scalar, clip_fraction for monitoring
        """
        ratio = torch.exp(log_probs - old_log_probs)  # importance weight
        advantages = advantages.unsqueeze(-1)           # broadcast over tokens

        # Unclipped and clipped objectives
        obj_unclipped = ratio * advantages
        obj_clipped = torch.clamp(ratio, 1 - self.config.epsilon, 1 + self.config.epsilon) * advantages

        loss = -torch.min(obj_unclipped, obj_clipped).mean()

        # Track how often clipping fires (useful diagnostic)
        clip_fraction = ((ratio - 1).abs() > self.config.epsilon).float().mean().item()

        return loss, clip_fraction

    def compute_kl_penalty(
        self,
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Per-token KL divergence from reference model.
        Penalizes the policy from drifting too far from the base model.

        KL(pi || pi_ref) = exp(log_pi_ref - log_pi) - (log_pi_ref - log_pi) - 1
        (unbiased estimator, always >= 0)
        """
        log_ratio = ref_log_probs - log_probs
        kl = torch.exp(log_ratio) - log_ratio - 1
        return kl.mean()

    def train_step(self, rollout_buffer: RolloutBuffer) -> GRPOStats:
        """
        One GRPO update step from a filled rollout buffer.

        Returns GRPOStats for logging.
        """
        self.model.train()
        self.ref_model.eval()

        stats = GRPOStats()
        total_loss = torch.tensor(0.0)

        for acc_step, batch in enumerate(
            rollout_buffer.iter_batches(self.config.batch_size)
        ):
            prompts = batch["prompts"]
            completions = batch["completions"]
            old_log_probs = batch["log_probs"]
            rewards = batch["rewards"]

            # Compute group-normalized advantages
            advantages = self.compute_advantages(rewards)

            # Forward pass through current policy
            log_probs = self.model.get_log_probs(prompts, completions)

            # Forward pass through frozen reference model (for KL)
            with torch.no_grad():
                ref_log_probs = self.ref_model.get_log_probs(prompts, completions)

            # Losses
            policy_loss, clip_frac = self.compute_policy_loss(
                log_probs, old_log_probs, advantages
            )
            kl_penalty = self.compute_kl_penalty(log_probs, ref_log_probs)

            loss = policy_loss + self.config.beta * kl_penalty
            loss = loss / self.config.gradient_accumulation_steps

            loss.backward()
            total_loss += loss.detach()

            # Accumulate stats
            stats.policy_loss += policy_loss.item()
            stats.kl_divergence += kl_penalty.item()
            stats.clip_fraction += clip_frac
            stats.mean_reward += rewards.mean().item()
            stats.std_reward += rewards.std().item()

            # Gradient step after accumulation
            if (acc_step + 1) % self.config.gradient_accumulation_steps == 0:
                stats.grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                ).item()
                self.optimizer.step()
                self.optimizer.zero_grad()

        self.global_step += 1
        n = len(list(rollout_buffer.iter_batches(self.config.batch_size)))

        # Average stats over accumulation steps
        stats.policy_loss /= max(n, 1)
        stats.kl_divergence /= max(n, 1)
        stats.clip_fraction /= max(n, 1)
        stats.mean_reward /= max(n, 1)
        stats.std_reward /= max(n, 1)

        if self.global_step % self.config.log_interval == 0:
            logger.info(
                f"step={self.global_step} "
                f"loss={stats.policy_loss:.4f} "
                f"kl={stats.kl_divergence:.4f} "
                f"reward={stats.mean_reward:.4f} "
                f"clip={stats.clip_fraction:.3f}"
            )

        return stats

    def save_checkpoint(self, path: Optional[str] = None):
        path = path or f"{self.config.checkpoint_dir}/step_{self.global_step}.pt"
        torch.save({
            "step": self.global_step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": self.config,
        }, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.global_step = ckpt["step"]
        logger.info(f"Resumed from step {self.global_step}")