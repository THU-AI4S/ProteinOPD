from __future__ import annotations

from typing import Any

import torch

from prollama_opd_config import PromptConfig


IGNORE_INDEX = -100


class ProLLaMAOpdDataCollator:
    """Build the fixed ProLLaMA prompt for every synthetic rollout step."""

    def __init__(self, tokenizer, prompt_config: PromptConfig) -> None:
        self.tokenizer = tokenizer
        self.prompt_config = prompt_config
        tokenized_prompt = tokenizer(
            prompt_config.prompt_text,
            return_attention_mask=False,
        )
        self.prompt_ids = list(tokenized_prompt["input_ids"])
        if len(self.prompt_ids) == 0:
            raise ValueError("Resolved ProLLaMA prompt token sequence is empty.")

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor | int]:
        if len(features) == 0:
            raise ValueError("Received an empty batch in ProLLaMA OPD data collator.")
        batch_size = len(features)
        prompt_length = len(self.prompt_ids)
        prompt_ids = torch.tensor([self.prompt_ids for _ in range(batch_size)], dtype=torch.long)
        prompt_attention_mask = torch.ones((batch_size, prompt_length), dtype=torch.long)
        return {
            "prompt_ids": prompt_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "prompt_length": prompt_length,
            "prompt_lengths_per_example": torch.full((batch_size,), prompt_length, dtype=torch.long),
        }
