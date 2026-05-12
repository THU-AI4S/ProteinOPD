from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Optional, Sequence

import yaml

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    from peft import PeftConfig, PeftModel
except ImportError:  # pragma: no cover
    PeftConfig = None
    PeftModel = None

try:
    from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
except ImportError:  # pragma: no cover
    GenerationConfig = None
    LlamaForCausalLM = None
    LlamaTokenizer = None


SEQUENCE_PATTERN = re.compile(r"Seq=<([^>]*)>")
STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


@dataclass
class GenerateConfig:
    load_mode: str = "lora"
    model: str = "/path/to/ProLLaMA_model"
    adapter_path: Optional[str] = "/path/to/prollama_lora_adapter"
    superfamily: str = "Lysozyme-like domain superfamily"
    seq_prefix: Optional[str] = None
    num_sequences: int = 256
    generation_batch_size: int = 8
    output_json: Optional[str] = "outputs/conditional_generate.json"
    temperature: float = 0.7
    top_k: int = 200
    top_p: float = 0.9
    repetition_penalty: float = 1.2
    max_new_tokens: int = 512
    do_sample: bool = True
    seed: int = 42


@dataclass(frozen=True)
class AdapterMetadata:
    adapter_type: str
    base_model_name_or_path: Optional[str]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ProLLaMA protein sequences for a target superfamily.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("generate.yaml")),
        help="YAML generation config path.",
    )
    parser.add_argument("--num_return_sequences", dest="num_sequences", default=None, type=int)
    bool_fields = {"do_sample"}
    for field in fields(GenerateConfig):
        if field.name == "num_sequences":
            parser.add_argument("--num_sequences", default=None, type=int)
        elif field.name in {"num_sequences"}:
            continue
        elif field.name in bool_fields:
            parser.add_argument("--" + field.name, default=None, action=argparse.BooleanOptionalAction)
        else:
            parser.add_argument("--" + field.name, default=None)
    return parser.parse_args(argv)


def _cast_value(field_name: str, value: Any) -> Any:
    if value is None:
        return None
    int_fields = {"num_sequences", "generation_batch_size", "top_k", "max_new_tokens", "seed"}
    float_fields = {"temperature", "top_p", "repetition_penalty"}
    bool_fields = {"do_sample"}
    if field_name in int_fields:
        return int(value)
    if field_name in float_fields:
        return float(value)
    if field_name in bool_fields:
        return bool(value)
    return value


def load_config(args: argparse.Namespace) -> GenerateConfig:
    with open(args.config, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise ValueError("Generation config must be a YAML mapping.")
    values: dict[str, Any] = dict(payload)
    for field in fields(GenerateConfig):
        override = getattr(args, field.name, None)
        if override is not None:
            values[field.name] = _cast_value(field.name, override)
    config = GenerateConfig(**values)
    validate_config(config)
    return config


def validate_config(config: GenerateConfig) -> None:
    if config.load_mode not in {"base", "lora", "prefix"}:
        raise ValueError("`load_mode` must be one of: base, lora, prefix.")
    if config.load_mode == "base":
        if not config.model:
            raise ValueError("`model` is required when `load_mode=base`.")
        if config.adapter_path is not None:
            raise ValueError("`adapter_path` is not allowed when `load_mode=base`.")
    elif not config.adapter_path:
        raise ValueError(f"`adapter_path` is required when `load_mode={config.load_mode}`.")
    if config.num_sequences <= 0:
        raise ValueError("`num_sequences` must be > 0.")
    if config.generation_batch_size <= 0:
        raise ValueError("`generation_batch_size` must be > 0.")
    if config.max_new_tokens <= 0:
        raise ValueError("`max_new_tokens` must be > 0.")


def build_prompt(superfamily: str, seq_prefix: Optional[str] = None) -> str:
    normalized_superfamily = (superfamily or "").strip()
    normalized_seq_prefix = (seq_prefix or "").strip()
    if not normalized_superfamily:
        return f"Seq=<{normalized_seq_prefix}"
    prompt = f"[Generate by superfamily] Superfamily=<{normalized_superfamily}>"
    if normalized_seq_prefix:
        prompt += f" Seq=<{normalized_seq_prefix}>"
    return prompt


def sanitize_sequence(text: str) -> str:
    return "".join(char for char in text.upper() if char in STANDARD_AMINO_ACIDS)


def extract_sequence_from_completion(completion_text: str, seq_prefix: Optional[str] = None) -> str:
    match = SEQUENCE_PATTERN.search(completion_text)
    sequence = sanitize_sequence(match.group(1)) if match else sanitize_sequence(completion_text)
    normalized_prefix = sanitize_sequence(seq_prefix or "")
    if normalized_prefix and not sequence.startswith(normalized_prefix):
        sequence = normalized_prefix + sequence
    if not sequence:
        raise ValueError("Unable to extract a valid protein sequence from the generated completion.")
    return sequence


def normalize_adapter_type(peft_type_value: Any) -> str:
    if peft_type_value is None:
        raise ValueError("Missing `peft_type` in adapter config.")
    normalized = str(peft_type_value).split(".")[-1].lower()
    if normalized == "lora":
        return "lora"
    if normalized in {"prefix_tuning", "prefix"}:
        return "prefix"
    raise ValueError(f"Unsupported adapter type: {peft_type_value}")


def load_adapter_metadata(adapter_path: str) -> AdapterMetadata:
    adapter_config_path = Path(adapter_path) / "adapter_config.json"
    if adapter_config_path.exists():
        config_dict = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    else:
        if PeftConfig is None:
            raise ImportError("peft is required to resolve adapter metadata when adapter_config.json is unavailable.")
        peft_config = PeftConfig.from_pretrained(adapter_path)
        config_dict = {
            "peft_type": getattr(peft_config, "peft_type", None),
            "base_model_name_or_path": getattr(peft_config, "base_model_name_or_path", None),
        }
    return AdapterMetadata(
        adapter_type=normalize_adapter_type(config_dict.get("peft_type")),
        base_model_name_or_path=config_dict.get("base_model_name_or_path"),
    )


def validate_adapter_mode(load_mode: str, adapter_path: str) -> AdapterMetadata:
    adapter_metadata = load_adapter_metadata(adapter_path)
    if adapter_metadata.adapter_type != load_mode:
        raise ValueError(f"`load_mode={load_mode}` does not match adapter type `{adapter_metadata.adapter_type}` under {adapter_path}.")
    return adapter_metadata


def resolve_model_path(model_path: Optional[str], adapter_path: Optional[str]) -> str:
    if model_path:
        return model_path
    if not adapter_path:
        raise ValueError("Unable to resolve a base model path. Pass `model` or adapter config with `base_model_name_or_path`.")
    adapter_metadata = load_adapter_metadata(adapter_path)
    if not adapter_metadata.base_model_name_or_path:
        raise ValueError(f"Unable to infer base model path from adapter_path={adapter_path}.")
    return adapter_metadata.base_model_name_or_path


def require_torch_and_transformers() -> None:
    missing = []
    if torch is None:
        missing.append("torch")
    if GenerationConfig is None or LlamaForCausalLM is None or LlamaTokenizer is None:
        missing.append("transformers")
    if missing:
        raise ImportError("Missing runtime dependencies required for generation: " + ", ".join(missing))


def require_peft() -> None:
    if PeftConfig is None or PeftModel is None:
        raise ImportError("Missing runtime dependency required for adapter loading: peft")


def resolve_load_dtype():
    require_torch_and_transformers()
    if not torch.cuda.is_available():
        raise ValueError("No GPU available.")
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def set_generation_seed(seed: int) -> None:
    require_torch_and_transformers()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_generation_config(config: GenerateConfig):
    require_torch_and_transformers()
    return GenerationConfig(
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        do_sample=config.do_sample,
        repetition_penalty=config.repetition_penalty,
        max_new_tokens=config.max_new_tokens,
    )


def load_model_and_tokenizer(config: GenerateConfig):
    require_torch_and_transformers()
    if config.load_mode in {"lora", "prefix"}:
        require_peft()
        validate_adapter_mode(config.load_mode, config.adapter_path)
    base_model_path = resolve_model_path(config.model, None if config.load_mode == "base" else config.adapter_path)
    model = LlamaForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=resolve_load_dtype(),
        low_cpu_mem_usage=True,
        device_map="auto",
        quantization_config=None,
    )
    if config.load_mode in {"lora", "prefix"}:
        model = PeftModel.from_pretrained(model, config.adapter_path)
    tokenizer = LlamaTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def generate_completions(model, tokenizer, prompt: str, generation_config, num_return_sequences: int) -> list[str]:
    require_torch_and_transformers()
    device = torch.device(0)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    generation_output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        generation_config=generation_config,
        num_return_sequences=num_return_sequences,
        output_attentions=False,
    )
    prompt_length = input_ids.shape[1]
    completion_ids = generation_output[:, prompt_length:]
    return tokenizer.batch_decode(completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def generate_sequences(
    model,
    tokenizer,
    prompt: str,
    generation_config,
    num_sequences: int,
    generation_batch_size: int,
    seq_prefix: Optional[str],
) -> list[str]:
    completions: list[str] = []
    remaining = num_sequences
    while remaining > 0:
        current_batch_size = min(generation_batch_size, remaining)
        completions.extend(generate_completions(model, tokenizer, prompt, generation_config, current_batch_size))
        remaining -= current_batch_size
    return [extract_sequence_from_completion(completion, seq_prefix=seq_prefix) for completion in completions]


def build_output_records(prompt: str, sequences: Sequence[str]) -> list[dict[str, str]]:
    return [
        {
            "instruction": prompt,
            "reference": "N/A",
            "response#1": sequence,
        }
        for sequence in sequences
    ]


def emit_output(records: Sequence[dict[str, str]], output_json: Optional[str]) -> None:
    payload = json.dumps(list(records), indent=2, ensure_ascii=False)
    if output_json:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
        return
    print(payload)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    config = load_config(args)
    set_generation_seed(config.seed)
    prompt = build_prompt(config.superfamily, config.seq_prefix)
    generation_config = build_generation_config(config)
    model, tokenizer = load_model_and_tokenizer(config)
    with torch.no_grad():
        sequences = generate_sequences(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            generation_config=generation_config,
            num_sequences=config.num_sequences,
            generation_batch_size=config.generation_batch_size,
            seq_prefix=config.seq_prefix,
        )
    emit_output(build_output_records(prompt, sequences), config.output_json)


if __name__ == "__main__":
    main()
