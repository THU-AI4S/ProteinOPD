from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    from peft import PeftModel
except ImportError:  # pragma: no cover
    PeftModel = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
except ImportError:  # pragma: no cover
    AutoModelForCausalLM = None
    AutoTokenizer = None
    set_seed = None


@dataclass
class GenerateConfig:
    base_model_path: str
    adapter_path: str
    output_json_path: str
    num_sequences: int = 1000
    batch_size: int = 8
    max_new_tokens: int = 512
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 500
    repetition_penalty: float = 1.2
    seed: int = 42
    use_bf16_if_available: bool = True
    trust_remote_code: bool = True
    overwrite_output: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate unconditional ProtGPT2 protein sequences.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("generate.yaml")),
        help="YAML generation config path.",
    )
    bool_fields = {"do_sample", "use_bf16_if_available", "trust_remote_code", "overwrite_output"}
    for field in fields(GenerateConfig):
        arg_name = "--" + field.name
        if field.name in bool_fields:
            parser.add_argument(arg_name, default=None, action=argparse.BooleanOptionalAction)
        else:
            parser.add_argument(arg_name, default=None)
    return parser.parse_args()


def _cast_value(field_name: str, value: Any) -> Any:
    int_fields = {"num_sequences", "batch_size", "max_new_tokens", "top_k", "seed"}
    float_fields = {"temperature", "top_p", "repetition_penalty"}
    bool_fields = {"do_sample", "use_bf16_if_available", "trust_remote_code", "overwrite_output"}
    if field_name in int_fields:
        return int(value)
    if field_name in float_fields:
        return float(value)
    if field_name in bool_fields:
        return bool(value)
    return value


def _load_config(config_path: str, overrides: argparse.Namespace) -> GenerateConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise ValueError("Generation config must be a YAML mapping.")
    values: dict[str, Any] = dict(payload)
    for field in fields(GenerateConfig):
        override_value = getattr(overrides, field.name)
        if override_value is not None:
            values[field.name] = _cast_value(field.name, override_value)
    config = GenerateConfig(**values)
    if not config.base_model_path:
        raise ValueError("`base_model_path` must be set.")
    if not config.adapter_path:
        raise ValueError("`adapter_path` must be set.")
    if config.num_sequences <= 0:
        raise ValueError("`num_sequences` must be > 0.")
    if config.batch_size <= 0:
        raise ValueError("`batch_size` must be > 0.")
    if config.max_new_tokens <= 0:
        raise ValueError("`max_new_tokens` must be > 0.")
    return config


def _resolve_torch_dtype(config: GenerateConfig) -> torch.dtype:
    require_runtime_dependencies()
    if not torch.cuda.is_available():
        return torch.float32
    if config.use_bf16_if_available and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _postprocess_sequence(token_ids: list[int], tokenizer) -> str:
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    cleaned_ids = list(token_ids)
    if eos_token_id is not None and cleaned_ids and cleaned_ids[0] == eos_token_id:
        cleaned_ids = cleaned_ids[1:]
    while cleaned_ids and (
        (eos_token_id is not None and cleaned_ids[-1] == eos_token_id)
        or (pad_token_id is not None and cleaned_ids[-1] == pad_token_id)
    ):
        cleaned_ids.pop()
    seq = tokenizer.decode(cleaned_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return "".join(seq.split())


def _save_results(output_path: Path, records: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def require_runtime_dependencies() -> None:
    missing = []
    if torch is None:
        missing.append("torch")
    if PeftModel is None:
        missing.append("peft")
    if AutoModelForCausalLM is None or AutoTokenizer is None or set_seed is None:
        missing.append("transformers")
    if missing:
        raise ImportError("Missing runtime dependencies required for generation: " + ", ".join(missing))


def main() -> None:
    args = parse_args()
    config = _load_config(args.config, args)
    require_runtime_dependencies()
    output_path = Path(config.output_json_path)
    if output_path.suffix.lower() != ".json":
        raise ValueError("`output_json_path` must end with `.json`.")
    if output_path.exists() and not config.overwrite_output:
        raise FileExistsError(f"Output already exists: {output_path}")

    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_path,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer must have eos_token when pad_token is missing.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_path,
        torch_dtype=_resolve_torch_dtype(config),
        trust_remote_code=config.trust_remote_code,
    )
    model = PeftModel.from_pretrained(base_model, config.adapter_path, is_trainable=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id for unconditional generation.")

    records: list[dict[str, str]] = []
    while len(records) < config.num_sequences:
        current_batch = min(config.batch_size, config.num_sequences - len(records))
        prompt_ids = torch.full((current_batch, 1), eos_token_id, dtype=torch.long, device=device)
        prompt_attention_mask = torch.ones_like(prompt_ids)
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_attention_mask,
                max_new_tokens=config.max_new_tokens,
                do_sample=config.do_sample,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repetition_penalty=config.repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_token_id,
                use_cache=True,
            )
        for row in generated_ids:
            records.append(
                {
                    "instruction": "N/A",
                    "reference": "N/A",
                    "response#1": _postprocess_sequence(row.detach().cpu().tolist(), tokenizer),
                }
            )
        print(f"Generated {len(records)}/{config.num_sequences} sequences.")

    _save_results(output_path, records)
    print(f"Saved {len(records)} sequences to: {output_path}")


if __name__ == "__main__":
    main()
