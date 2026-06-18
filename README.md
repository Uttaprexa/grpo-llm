# GRPO-LLM

A from-scratch implementation of **Group Relative Policy Optimization (GRPO)** for training large language models on math reasoning — built with production-grade infrastructure.

Trains on GSM8K. No reward model required. Verifiable rewards only.

> **Uttapreksha Patel** · [GitHub](https://github.com/Uttaprexa/grpo-llm) · [LinkedIn](https://linkedin.com/in/uttaprexa)

---

## Why this exists

Most open-source GRPO implementations wrap HuggingFace TRL and call it a day. This repo implements the full stack from scratch — the algorithm, the async rollout infrastructure, the sandboxed execution environment, and the distributed training layer — because understanding what breaks at scale requires building it yourself.

Key engineering decisions documented throughout:
- **Why Trio over asyncio** for rollout workers (structured concurrency, no silent failures)
- **Why FSDP over DDP** for multi-GPU training (optimizer state sharding for >1B models)
- **Why verifiable rewards** over a reward model (un-gameable, zero overhead)
- **Why subprocess sandboxing** matters for code generation RL
- **Why C++ pybind11** for the reward function (1.7x speedup on normalize_answer)

---

## Results

### GSM8K Training (Qwen2.5-0.5B, AWS T4 GPU)

| Model | Method | GSM8K Accuracy |
|-------|--------|---------------|
| Qwen2.5-0.5B | Base (no training) | 8.7% |
| Qwen2.5-0.5B | + GRPO (200 iters, Colab T4) | 14.6% |
| Qwen2.5-0.5B | + GRPO (300 iters, AWS T4) | 28.5% |

### Algorithm Comparison (same model, same compute budget, same eval set)

| Algorithm | Final Accuracy | Training Time | Notes |
|-----------|---------------|---------------|-------|
| **DPO** | **29.5%** 🥇 | 120 min | Fastest — offline preference, no rollout overhead |
| GRPO | 28.5% 🥈 | 188 min | Group-relative baseline, no value network needed |
| PPO | 25.0% 🥉 | 191 min | Clipped surrogate + KL penalty |

**Key finding:** DPO outperformed both online RL methods at this scale. Sparse binary rewards limit the advantage of online exploration for a 0.5B model at 300 iterations — the simpler preference signal wins. This motivates future work on process reward models (step-level rewards).

### C++ Reward Extension

| Function | Python | C++ | Speedup |
|----------|--------|-----|---------|
| normalize_answer (100k calls) | 170.5ms | 100.8ms | **1.7x** |

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
│       ├── code_reward.py  # Test-case-based code rewards
│       ├── fast_reward.cpp # C++ pybind11 extension (1.7x speedup)
│       └── build_ext.sh    # Build script for C++ extension
├── envs/
│   ├── math_env.py         # GSM8K environment + prompt formatting
│   └── code_exec.py        # Sandboxed subprocess executor
├── experiments/
│   ├── compare_algorithms.py  # GRPO vs DPO vs PPO comparison
│   └── results.json           # Full experiment results
├── infra/k8s/              # Kubernetes deployment manifests
├── benchmarks/             # Throughput + profiling scripts
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
# 25 passed in 1.17s

# Build C++ reward extension
bash grpo/reward/build_ext.sh

# Train on GSM8K (single GPU)
python scripts/train.py --config configs/grpo_gsm8k.yaml

# Resume from checkpoint
python scripts/train.py --config configs/grpo_gsm8k.yaml --resume checkpoints/step_100.pt

# Run algorithm comparison
python experiments/compare_algorithms.py --algorithms grpo dpo ppo
```

---

## How GRPO works (and why it's cheaper than PPO)

Standard PPO requires a separate critic/value network to estimate baselines — doubling memory and compute. GRPO replaces this with a **group-based baseline**:

1. For each prompt, sample G completions (default G=8)
2. Score each with the reward function
3. Normalize within the group: `advantage = (reward - mean) / std`
4. Use normalized advantages in the clipped PPO objective

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

**C++ reward function** — The `normalize_answer` function is called millions of times during training. A pybind11 C++ extension achieves 1.7x speedup over pure Python. Built with `-O3` optimization and called from Python via a clean interface.

---

## Tests

```
pytest tests/test_reward.py tests/test_envs.py -v

25 passed in 1.17s
```

Tests cover reward parsing (4 answer formats), answer normalization, binary and format-aware rewards, sandboxed execution, timeout enforcement, output isolation, and process isolation — all without a GPU or model weights.

---

## Experiment: Why DPO beat GRPO at small scale

The comparison study reveals something counterintuitive: the offline algorithm (DPO) outperforms both online RL methods despite having access to less information per step.

The likely explanation: with binary rewards (correct/wrong), most rollouts in GRPO/PPO return reward=0 — giving zero gradient signal. DPO sidesteps this by directly contrasting a correct and incorrect completion, always producing a meaningful update even when both completions are wrong (since one is relatively better).

This suggests that at small model scale with sparse rewards, **preference signal quality matters more than exploration**. Process reward models (step-level rewards) are the natural next step to fix this for online RL.

---

## References

- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning](https://arxiv.org/abs/2402.03300) — original GRPO paper
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — PPO foundation
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — DPO paper
- [Trio: async concurrency for Python](https://trio.readthedocs.io)
- [PyTorch FSDP documentation](https://pytorch.org/docs/stable/fsdp.html)
