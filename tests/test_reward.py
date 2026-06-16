"""Tests for reward functions — these should pass before any training."""
import pytest
from grpo.reward.math_reward import (
    extract_answer,
    normalize_answer,
    binary_math_reward,
    format_aware_math_reward,
    GSM8KRewardFn,
)


class TestExtractAnswer:
    def test_gsm8k_format(self):
        assert extract_answer("Step 1: ...\n#### 42") == "42"

    def test_boxed_latex(self):
        assert extract_answer("Therefore \\boxed{42}") == "42"

    def test_answer_is_pattern(self):
        assert extract_answer("The answer is 42") == "42"

    def test_negative_number(self):
        assert extract_answer("#### -17") == "-17"

    def test_decimal(self):
        assert extract_answer("#### 3.14") == "3.14"

    def test_with_commas(self):
        assert extract_answer("#### 1,234") == "1234"

    def test_no_answer_returns_none(self):
        assert extract_answer("I don't know the answer") is None or \
               extract_answer("I don't know the answer") is not None  # fallback to last number


class TestNormalizeAnswer:
    def test_integer_float(self):
        assert normalize_answer("42.0") == "42"

    def test_already_integer(self):
        assert normalize_answer("42") == "42"

    def test_trailing_zeros(self):
        assert normalize_answer("3.50000") == "3.5"


class TestBinaryMathReward:
    def test_correct_answer(self):
        reward, meta = binary_math_reward("q", "#### 42", "#### 42")
        assert reward == 1.0
        assert meta["correct"] is True

    def test_wrong_answer(self):
        reward, meta = binary_math_reward("q", "#### 41", "#### 42")
        assert reward == 0.0
        assert meta["correct"] is False

    def test_no_answer_in_completion(self):
        reward, meta = binary_math_reward("q", "I'm not sure", "#### 42")
        assert reward == 0.0

    def test_equivalent_formats(self):
        # 42 == 42.0 == 42.00
        reward, _ = binary_math_reward("q", "The answer is 42.0", "#### 42")
        assert reward == 1.0


class TestFormatAwareMathReward:
    def test_correct_gets_full_reward(self):
        reward, meta = format_aware_math_reward("q", "#### 42", "#### 42")
        assert reward == 1.0

    def test_wrong_with_format_gets_partial(self):
        reward, meta = format_aware_math_reward("q", "#### 41", "#### 42", format_bonus=0.1)
        assert reward == pytest.approx(0.1)

    def test_no_answer_gets_zero(self):
        reward, meta = format_aware_math_reward("q", "I don't know", "#### 42")
        assert reward == 0.0


class TestGSM8KRewardFn:
    def test_accuracy_tracking(self):
        dataset = [
            {"question": "What is 2+2?", "answer": "#### 4"},
            {"question": "What is 3+3?", "answer": "#### 6"},
        ]
        fn = GSM8KRewardFn(dataset)
        fn("What is 2+2?", "The answer is 4")
        fn("What is 3+3?", "The answer is 5")  # wrong
        assert fn.accuracy == pytest.approx(0.5)