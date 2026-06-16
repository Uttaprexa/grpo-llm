# grpo-llm

A from-scratch implementation of **Group Relative Policy Optimization (GRPO)** for training large language models on math reasoning — built with production-grade infrastructure.

Trains on GSM8K. No reward model required. Verifiable rewards only.

---

## Why this exists

Most open-source GRPO implementations wrap HuggingFace TRL and call it a day. This repo implements the full stack from scratch — the algorithm, the async rollout infrastructure, the sandboxed execution environment, and the distributed training layer — because understanding what breaks at scale requires building it yourself.

Key engineering decisions documented throughout:
- **Why Trio over asyncio** for rollout workers (structured concurrency, no silent failures)
- **Why FSDP over DDP** for multi-GPU (optimizer state sharding for >1B models)
- **Why verifiable rewards** over a reward model (un-gameable, zero overhead)
- **Why subprocess sandboxing** matters for code generation RL

---

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │           Training Loop              │
                    │                                      │
  GSM8K prompts ──► │  RolloutWorkers (Trio async)         │
                    │    ├── Worker 0: generate G rollouts │
                    │    ├── Worker 1: generate G rollouts │
                    │    └── Worker N: generate G rollouts │
                    │             │                        │
                    │    RolloutBuffer (rewards scored)    │
                    │             │                        │
                    │  GRPOTrainer.train_step()            │
                    │    ├── compute_advantages()  ← GRPO  │
                    │    ├── compute_policy_loss() ← PPO   │
                    │    └── compute_kl_penalty()  ← safe  │
                    │             │                        │
                    │    Policy update (FSDP multi-GPU)    │
                    └─────────────────────────────────────┘
```

---

## Project structure

```
grpo-llm/
├── grpo/
│   ├── trainer.py          # GRPOTrainer — group advantages, clipped loss, KL penalty
│   ├── rollout.py          # Async rollout workers (Trio nurseries)
│   ├── model.py            # PolicyModel — thin HuggingFace wrapper
│   └── reward/
│       ├── math_reward.py  # Verifiable math rewards (GSM8K, MATH)
│       └── code_reward.py  # Test-case-based code rewards
├── envs/
│   ├── math_env.py         # GSM8K environment + prompt formatting
│   └── code_exec.py        # Sandboxed subprocess executor
├── infra/k8s/              # Kubernetes deployment manifests
├── benchmarks/             # Nsight profiling + throughput benchmarks
├── tests/                  # 25 passing tests (no GPU required)
├── scripts/train.py        # Main training entry point
└── configs/grpo_gsm8k.yaml # Hyperparameters
```

---

## Quickstart

```bash
# Install
pip install -e ".[dev]"

# Run tests (no GPU needed)
pytest tests/test_reward.py tests/test_envs.py -v
# 25 passed

# Train on GSM8K (single GPU)
python scripts/train.py --config configs/grpo_gsm8k.yaml

# Resume from checkpoint
python scripts/train.py --config configs/grpo_gsm8k.yaml --resume checkpoints/step_100.pt

# Profile rollout throughput
python benchmarks/profile_rollout.py --model Qwen/Qwen2.5-1.5B-Instruct
```

---

## How GRPO works (and why it's cheaper than PPO)

Standard PPO requires a separate critic/value network to estimate baselines — doubling your memory and compute. GRPO replaces this with a **group-based baseline**:

1. For each prompt, sample G completions (default G=8)
2. Score each with the reward function
3. Normalize within the group: `advantage = (reward - mean) / std`
4. Use these normalized advantages in the clipped PPO objective

No value network. No critic training. The group *is* the baseline.

```python
def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
    rewards_grouped = rewards.view(-1, self.config.group_size)  # (batch, G)
    mean = rewards_grouped.mean(dim=-1, keepdim=True)
    std  = rewards_grouped.std(dim=-1, keepdim=True)
    return ((rewards_grouped - mean) / (std + 1e-8)).view(-1)
```

---

## Key design decisions

**Trio for async rollouts** — Trio's structured concurrency (nurseries) makes it impossible to silently lose exceptions from worker tasks. In a long training run, a silent worker failure corrupts your reward buffer without any error. `asyncio` requires manual exception plumbing to get the same guarantee; Trio gives it by default.

**Sandboxed code execution** — `envs/code_exec.py` runs LLM-generated code in isolated subprocesses with memory limits, timeouts, and output truncation. On macOS these are soft limits; on Linux, `RLIMIT_AS` enforces hard memory caps. Designed to swap for gVisor/Firecracker in production with no API changes.

**Verifiable rewards only** — GSM8K answers are exact numbers. Either the model's final answer matches or it doesn't. This gives a reward signal that's impossible to game and requires zero reward model overhead. The `extract_answer()` function handles 4 common output formats (GSM8K `####`, LaTeX `\boxed{}`, "the answer is X", last number fallback).

**FSDP for multi-GPU** — For models above 1B parameters, DDP requires each GPU to hold a full parameter copy. FSDP shards parameters, gradients, and optimizer states across GPUs. Enabled via `GRPOConfig(use_fsdp=True)` with no other code changes.

---

## Benchmarks

*Run on NVIDIA H100. Results from `benchmarks/profile_rollout.py`.*

| Workers | Rollouts/s | Tokens/s | GPU util |
|---------|-----------|----------|----------|
| 1 (sync) | — | — | — |
| 2 | — | — | — |
| 4 | — | — | — |
| 8 | — | — | — |

*Training in progress — benchmark numbers coming soon.*

---

## GSM8K Results

| Model | Method | GSM8K accuracy |
|-------|--------|---------------|
| Qwen2.5-1.5B | SFT baseline | — |
| Qwen2.5-1.5B | + GRPO (this repo) | — |

*Training in progress.*

---

## Tests

```
pytest tests/test_reward.py tests/test_envs.py -v

25 passed in 1.17s
```

Tests cover reward parsing, answer normalization, binary/format-aware rewards, sandboxed execution, timeout enforcement, and output isolation — all without a GPU.

---

## References

- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning](https://arxiv.org/abs/2402.03300) — original GRPO paper
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — PPO foundation
- [Trio: async concurrency for Python](https://trio.readthedocs.io)
- [PyTorch FSDP documentation](https://pytorch.org/docs/stable/fsdp.html)