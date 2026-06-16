"""
Rollout generation for GRPO using Trio for async concurrency.

Design:
  - RolloutWorker: generates completions for a batch of prompts asynchronously
  - RolloutBuffer: stores (prompt, completion, log_prob, reward) tuples
  - Multiple workers run concurrently via Trio nurseries, keeping GPU saturated

Why Trio over asyncio?
  Trio's structured concurrency (nurseries) makes it impossible to silently
  lose exceptions from worker tasks — critical for long training runs where
  a silent failure would corrupt your reward buffer.
"""

import torch
import trio
from dataclasses import dataclass, field
from typing import Iterator
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class Rollout:
    """A single (prompt, completion) trajectory with its scored reward."""
    prompt_ids: torch.Tensor        # (prompt_len,)
    completion_ids: torch.Tensor    # (completion_len,)
    log_probs: torch.Tensor         # (completion_len,) — per-token log probs
    reward: float
    metadata: dict = field(default_factory=dict)  # e.g. {"correct": True, "ground_truth": "42"}


class RolloutBuffer:
    """
    Stores rollouts collected from workers before a policy update.

    Organized by group: for each prompt, config.group_size completions
    are stored adjacently so compute_advantages can reshape correctly.
    """

    def __init__(self):
        self._rollouts: list[Rollout] = []

    def add(self, rollout: Rollout):
        self._rollouts.append(rollout)

    def clear(self):
        self._rollouts.clear()

    def __len__(self):
        return len(self._rollouts)

    @property
    def mean_reward(self) -> float:
        if not self._rollouts:
            return 0.0
        return sum(r.reward for r in self._rollouts) / len(self._rollouts)

    def iter_batches(self, batch_size: int) -> Iterator[dict]:
        """
        Yield batches as dicts of padded tensors, ready for the trainer.
        Pads completions to the longest sequence in each batch.
        """
        for i in range(0, len(self._rollouts), batch_size):
            batch = self._rollouts[i:i + batch_size]

            max_prompt_len = max(r.prompt_ids.shape[0] for r in batch)
            max_comp_len = max(r.completion_ids.shape[0] for r in batch)

            prompts = torch.zeros(len(batch), max_prompt_len, dtype=torch.long)
            completions = torch.zeros(len(batch), max_comp_len, dtype=torch.long)
            log_probs = torch.zeros(len(batch), max_comp_len)
            rewards = torch.tensor([r.reward for r in batch])

            for j, rollout in enumerate(batch):
                pl = rollout.prompt_ids.shape[0]
                cl = rollout.completion_ids.shape[0]
                prompts[j, :pl] = rollout.prompt_ids
                completions[j, :cl] = rollout.completion_ids
                log_probs[j, :cl] = rollout.log_probs

            yield {
                "prompts": prompts,
                "completions": completions,
                "log_probs": log_probs,
                "rewards": rewards,
            }


class RolloutWorker:
    """
    Generates rollouts asynchronously using Trio.

    Each worker holds a reference to the policy model and generates
    completions for assigned prompts. Workers run concurrently in a
    Trio nursery — if any worker crashes, the nursery cancels all others
    and surfaces the exception immediately (no silent failures).

    Usage:
        async with trio.open_nursery() as nursery:
            for worker in workers:
                nursery.start_soon(worker.generate_batch, prompts, buffer)
    """

    def __init__(self, model, tokenizer, config, worker_id: int = 0):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.worker_id = worker_id
        self._total_generated = 0

    async def generate_single(
        self,
        prompt: str,
        reward_fn,
        send_channel: trio.MemorySendChannel,
    ):
        """
        Generate one group of completions for a single prompt.
        Sends Rollout objects into the channel as they complete.
        """
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").squeeze(0)

        # Sample G completions for this prompt (group_size)
        for _ in range(self.config.group_size):
            # Yield control to Trio scheduler between generations
            await trio.sleep(0)

            with torch.no_grad():
                output = self.model.generate(
                    prompt_ids.unsqueeze(0),
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    do_sample=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            completion_ids = output.sequences[0, prompt_ids.shape[0]:]
            completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)

            # Compute per-token log probs from generation scores
            log_probs = self._scores_to_log_probs(output.scores, completion_ids)

            # Score with reward function
            reward, metadata = reward_fn(prompt, completion_text)

            rollout = Rollout(
                prompt_ids=prompt_ids,
                completion_ids=completion_ids,
                log_probs=log_probs,
                reward=reward,
                metadata=metadata,
            )

            await send_channel.send(rollout)
            self._total_generated += 1

    async def generate_batch(
        self,
        prompts: list[str],
        reward_fn,
        buffer: RolloutBuffer,
        *,
        task_status=trio.TASK_STATUS_IGNORED,
    ):
        """
        Generate rollouts for a list of prompts, collecting into buffer.
        Reports ready via task_status once the first rollout is produced.
        """
        send_channel, recv_channel = trio.open_memory_channel(maxsize=64)
        started = False

        async def collect(recv_channel):
            nonlocal started
            async with recv_channel:
                async for rollout in recv_channel:
                    buffer.add(rollout)
                    if not started:
                        task_status.started(self.worker_id)
                        started = True

        t0 = time.perf_counter()
        async with trio.open_nursery() as nursery:
            nursery.start_soon(collect, recv_channel)
            async with send_channel:
                for prompt in prompts:
                    nursery.start_soon(self.generate_single, prompt, reward_fn, send_channel.clone())

        elapsed = time.perf_counter() - t0
        logger.debug(
            f"Worker {self.worker_id}: generated {len(prompts) * self.config.group_size} "
            f"rollouts in {elapsed:.2f}s"
        )

    @staticmethod
    def _scores_to_log_probs(
        scores: tuple[torch.Tensor],
        completion_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert generation scores (logits) to per-token log probs.

        scores: tuple of (vocab_size,) tensors, one per generated token
        completion_ids: (completion_len,) token ids
        """
        log_probs = []
        for i, score in enumerate(scores):
            token_id = completion_ids[i].item()
            lp = torch.log_softmax(score.squeeze(0), dim=-1)[token_id]
            log_probs.append(lp)
        return torch.stack(log_probs)


async def collect_rollouts(
    prompts: list[str],
    model,
    tokenizer,
    reward_fn,
    config,
    num_workers: int = 4,
) -> RolloutBuffer:
    """
    Top-level async function: distributes prompts across workers,
    collects all rollouts into a single buffer.

    Trio nursery guarantees: if any worker raises, all workers are
    cancelled and the exception propagates immediately.
    """
    buffer = RolloutBuffer()
    workers = [
        RolloutWorker(model, tokenizer, config, worker_id=i)
        for i in range(num_workers)
    ]

    # Partition prompts across workers
    chunks = [prompts[i::num_workers] for i in range(num_workers)]

    async with trio.open_nursery() as nursery:
        for worker, chunk in zip(workers, chunks):
            if chunk:
                nursery.start_soon(worker.generate_batch, chunk, reward_fn, buffer)

    logger.info(f"Collected {len(buffer)} rollouts, mean reward={buffer.mean_reward:.4f}")
    return buffer