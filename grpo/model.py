"""
PolicyModel: thin wrapper around a HuggingFace causal LM.

Keeps model loading, log-prob computation, and generation in one place
so the trainer stays clean and model-agnostic.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class PolicyModel(torch.nn.Module):
    """
    Wraps a HuggingFace causal LM for GRPO training.

    Exposes:
      - get_log_probs(prompts, completions): used by trainer for loss
      - generate(...): used by rollout workers
      - from_pretrained(...): loads model + tokenizer together
    """

    def __init__(self, hf_model, tokenizer):
        super().__init__()
        self.model = hf_model
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, model_name: str, device: str = "cuda") -> "PolicyModel":
        logger.info(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,  # bfloat16 for training stability
            device_map=device,
        )
        return cls(model, tokenizer)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def get_log_probs(
        self,
        prompt_ids: torch.Tensor,   # (batch, prompt_len)
        completion_ids: torch.Tensor,  # (batch, completion_len)
    ) -> torch.Tensor:
        """
        Compute per-token log probabilities of completions given prompts.

        Concatenates prompt + completion, runs forward pass, then slices
        out only the completion token log probs.

        Returns: (batch, completion_len) log prob tensor
        """
        batch_size = prompt_ids.shape[0]
        full_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        attention_mask = (full_ids != self.tokenizer.pad_token_id).long()

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits = self.model(
                input_ids=full_ids,
                attention_mask=attention_mask,
            ).logits  # (batch, seq_len, vocab)

        # Shift: predict token t from tokens 0..t-1
        shift_logits = logits[:, :-1, :]       # (batch, seq_len-1, vocab)
        shift_ids = full_ids[:, 1:]             # (batch, seq_len-1)

        log_probs_all = F.log_softmax(shift_logits, dim=-1)
        # Gather log prob of the actual next token
        token_log_probs = log_probs_all.gather(
            2, shift_ids.unsqueeze(-1)
        ).squeeze(-1)  # (batch, seq_len-1)

        # Return only the completion portion
        prompt_len = prompt_ids.shape[1]
        completion_log_probs = token_log_probs[:, prompt_len - 1:]

        return completion_log_probs  # (batch, completion_len)

    def generate(self, input_ids: torch.Tensor, **kwargs):
        """Pass-through to HuggingFace generate."""
        return self.model.generate(input_ids=input_ids, **kwargs)

    def parameters(self):
        return self.model.parameters()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        return self.model.load_state_dict(state_dict)