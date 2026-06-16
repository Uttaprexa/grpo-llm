"""
Reward functions for math reasoning (GSM8K, MATH datasets).

Design principle: rewards are verifiable — no reward model needed.
We parse the model's final answer and compare to ground truth exactly.

Two reward signals:
  1. Binary: 1.0 if correct, 0.0 if wrong  (sparse but unambiguous)
  2. Format-aware: partial credit for correct format, full for correct answer
"""

import re
from typing import Optional


def extract_answer(text: str) -> Optional[str]:
    """
    Extract the final numeric answer from model output.

    Handles common patterns:
      - "The answer is 42"
      - "#### 42"  (GSM8K format)
      - "= 42" at end of solution
      - boxed answers: \\boxed{42}
    """
    # GSM8K ground truth format
    gsm8k_match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if gsm8k_match:
        return gsm8k_match.group(1).replace(",", "")

    # LaTeX boxed (MATH dataset)
    boxed_match = re.search(r"\\boxed\{([^}]+)\}", text)
    if boxed_match:
        return boxed_match.group(1).strip()

    # "The answer is X" / "answer: X"
    answer_match = re.search(
        r"(?:the\s+)?answer\s+(?:is|:)\s*(-?[\d,]+\.?\d*)",
        text, re.IGNORECASE
    )
    if answer_match:
        return answer_match.group(1).replace(",", "")

    # Last number in the text as fallback
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def normalize_answer(answer: str) -> str:
    """Normalize numeric strings for comparison (strip trailing zeros, etc.)."""
    try:
        # Try parsing as float and round to avoid floating point noise
        val = float(answer.replace(",", ""))
        if val == int(val):
            return str(int(val))
        return f"{val:.6f}".rstrip("0")
    except ValueError:
        return answer.strip().lower()


def binary_math_reward(prompt: str, completion: str, ground_truth: str) -> tuple[float, dict]:
    """
    Binary reward: 1.0 if model answer matches ground truth, else 0.0.

    This is the standard GRPO reward for GSM8K — simple, unambiguous,
    and impossible for the model to game.

    Returns: (reward, metadata_dict)
    """
    predicted = extract_answer(completion)
    correct = extract_answer(ground_truth)

    metadata = {
        "predicted": predicted,
        "ground_truth": correct,
        "correct": False,
    }

    if predicted is None or correct is None:
        return 0.0, metadata

    is_correct = normalize_answer(predicted) == normalize_answer(correct)
    metadata["correct"] = is_correct

    return 1.0 if is_correct else 0.0, metadata


def format_aware_math_reward(
    prompt: str,
    completion: str,
    ground_truth: str,
    format_bonus: float = 0.1,
) -> tuple[float, dict]:
    """
    Reward with partial credit for good formatting.

    Rewards:
      1.0  — correct answer
      0.1  — wrong answer but has a clear final answer (good format)
      0.0  — no parseable answer found

    The format bonus encourages the model to structure its output
    even when it gets the math wrong.
    """
    predicted = extract_answer(completion)
    correct = extract_answer(ground_truth)

    metadata = {
        "predicted": predicted,
        "ground_truth": correct,
        "correct": False,
        "has_format": predicted is not None,
    }

    if predicted is None:
        return 0.0, metadata

    if correct and normalize_answer(predicted) == normalize_answer(correct):
        metadata["correct"] = True
        return 1.0, metadata

    # Partial credit for having a parseable answer
    return format_bonus, metadata


class GSM8KRewardFn:
    """
    Callable reward function for GSM8K, compatible with RolloutWorker.

    Holds the ground truth mapping so workers can score on the fly.

    Usage:
        reward_fn = GSM8KRewardFn(dataset)
        reward, meta = reward_fn(prompt, completion)
    """

    def __init__(self, dataset, use_format_reward: bool = False):
        # Map prompt -> ground truth answer
        self.ground_truths = {
            item["question"]: item["answer"]
            for item in dataset
        }
        self.use_format_reward = use_format_reward
        self._call_count = 0
        self._correct_count = 0

    def __call__(self, prompt: str, completion: str) -> tuple[float, dict]:
        ground_truth = self.ground_truths.get(prompt, "")
        self._call_count += 1

        if self.use_format_reward:
            reward, meta = format_aware_math_reward(prompt, completion, ground_truth)
        else:
            reward, meta = binary_math_reward(prompt, completion, ground_truth)

        if meta.get("correct"):
            self._correct_count += 1

        return reward, meta

    @property
    def accuracy(self) -> float:
        if self._call_count == 0:
            return 0.0
        return self._correct_count / self._call_count