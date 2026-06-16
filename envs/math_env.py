from datasets import load_dataset
from grpo.reward.math_reward import GSM8KRewardFn
import random
import logging

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a math reasoning assistant. "
    "Solve the problem step by step, showing your work clearly. "
    "End your response with '#### <answer>' where <answer> is the numeric result."
)


def format_gsm8k_prompt(question: str) -> str:
    return f"{SYSTEM_PROMPT}\n\nProblem: {question}\n\nSolution:"


class GSM8KEnv:
    def __init__(self, split: str = "train", seed: int = 42):
        logger.info(f"Loading GSM8K {split} split...")
        dataset = load_dataset("gsm8k", "main")[split]
        self.problems = list(dataset)
        self.reward_fn = GSM8KRewardFn(dataset)
        random.seed(seed)
        random.shuffle(self.problems)
        self._index = 0
        logger.info(f"Loaded {len(self.problems)} GSM8K problems")

    def sample_batch(self, batch_size: int) -> list[str]:
        batch = []
        for _ in range(batch_size):
            problem = self.problems[self._index % len(self.problems)]
            self._index += 1
            batch.append(format_gsm8k_prompt(problem["question"]))
        return batch

    def reward(self, prompt: str, completion: str) -> tuple[float, dict]:
        question = prompt.split("Problem:")[-1].split("Solution:")[0].strip()
        return self.reward_fn(question, completion)

    @property
    def accuracy(self) -> float:
        return self.reward_fn.accuracy

    def __len__(self):
        return len(self.problems)
