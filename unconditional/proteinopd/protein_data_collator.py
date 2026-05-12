from __future__ import annotations

from typing import Any

import torch


class ProteinOPDDataCollator:
    """Build unconditional prompts for Protein-OPD training."""

    def __init__(
        self,
        tokenizer,
        prompt_mode: str = "unconditional",
        prompt_prefix_length: int = 32,
    ) -> None:
        if prompt_mode != "unconditional":
            raise ValueError("Protein-OPD currently supports unconditional prompts only.")
        if prompt_prefix_length <= 0:
            raise ValueError("prompt_prefix_length must be > 0.")

        self.tokenizer = tokenizer
        self.prompt_mode = prompt_mode
        self.prompt_prefix_length = prompt_prefix_length

        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id
        if self.pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id or eos_token_id.")

        self.default_prompt_ids = self._build_default_prompt_ids()

    def _build_default_prompt_ids(self) -> list[int]:
        if self.tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define eos_token_id for unconditional training.")
        # Keep unconditional prompt as a single eos token.
        return [int(self.tokenizer.eos_token_id)]

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor | int]:
        if len(features) == 0:
            raise ValueError("Received an empty batch in ProteinOPDDataCollator.")

        batch_size = len(features)
        prompt_length = len(self.default_prompt_ids)

        prompt_ids = torch.tensor(
            [self.default_prompt_ids for _ in range(batch_size)],
            dtype=torch.long,
        )
        prompt_attention_mask = torch.ones((batch_size, prompt_length), dtype=torch.long)

        return {
            "prompt_ids": prompt_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "prompt_length": prompt_length,
            "prompt_lengths_per_example": torch.full((batch_size,), prompt_length, dtype=torch.long),
        }
