from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments, set_seed

from protein_opd_trainer import (
    STUDENT_PREFIX_INIT_FORMAT,
    STUDENT_PREFIX_INIT_METADATA_FILENAME,
    ProteinOPDTrainer,
    TeacherAdapterSpec,
)
from protein_data_collator import ProteinOPDDataCollator


logger = logging.getLogger(__name__)


class _StrictBoolSafeLoader(yaml.SafeLoader):
    pass


_StrictBoolSafeLoader.yaml_implicit_resolvers = {
    key: [(tag, regexp) for tag, regexp in resolvers if tag != "tag:yaml.org,2002:bool"]
    for key, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_StrictBoolSafeLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|false)$", re.IGNORECASE),
    list("tTfF"),
)


@dataclass
class TeacherEntryConfig:
    name: str
    adapter_path: str
    weight: float
    temperature: float


@dataclass
class TeacherConfig:
    teacher_backbone_path: str
    log_teacher_entropy: bool
    teachers: list[TeacherEntryConfig]


@dataclass
class ProteinOPDScriptArguments:
    num_train_samples: int = field(default=30000)
    student_model_name_or_path: str = field(default="")
    student_prefix_adapter_path: str = field(default="")
    teacher_config_path: str = field(default="")

    prompt_mode: str = field(default="unconditional")
    prompt_prefix_length: int = field(default=32)
    max_new_tokens: int = field(default=256)

    temperature: float = field(default=1.0)
    use_curriculum_tem: bool = field(default=False)
    curriculum_tem_start: float = field(default=0.7)
    repetition_penalty: float = field(default=1.2)
    top_p: float = field(default=1.0)
    top_k: int = field(default=0)

    beta: float = field(default=0.5)
    top_k_loss: int = field(default=0)
    useloss: str = field(default="jsd")
    useZ_t: bool = field(default=False)

    use_entropy_rule: bool = field(default=False)
    entropy_alpha: float = field(default=0.0)
    use_wi_entropy: bool = field(default=False)
    use_wi_distance: bool = field(default=False)
    wi_entropy_tau: float = field(default=1.0)
    wi_distance_beta: float = field(default=0.9)
    wi_distance_tau: float = field(default=1.0)
    debug_teacher_diff: bool = field(default=False)
    debug_teacher_diff_steps: int = field(default=5)

    save_generations: bool = field(default=True)
    generation_save_steps: int = field(default=5)

    student_tune_mode: str = field(default="lora")
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: list[str] = field(default_factory=lambda: ["c_attn", "c_proj", "c_fc"])


@dataclass
class ProteinOPDTrainingArguments(TrainingArguments):
    output_dir: str = field(default="")
    per_device_train_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=1)
    learning_rate: float = field(default=2e-5)
    num_train_epochs: float = field(default=1.0)
    remove_unused_columns: bool = field(default=False)
    report_to: str = field(default="none")


def _ensure_local_path(path_value: str, arg_name: str) -> None:
    if not path_value:
        raise ValueError(f"`{arg_name}` is required and must be a local path.")
    if not os.path.exists(path_value):
        raise ValueError(f"`{arg_name}` path does not exist: {path_value}")


def _require_dir(path_value: str, arg_name: str) -> None:
    _ensure_local_path(path_value, arg_name)
    if not os.path.isdir(path_value):
        raise ValueError(f"`{arg_name}` must be a directory: {path_value}")


def _normalize_lora_targets(target_modules: list[str]) -> list[str]:
    if len(target_modules) == 1 and "," in target_modules[0]:
        return [item.strip() for item in target_modules[0].split(",") if item.strip()]
    return target_modules


def _resolve_torch_dtype(training_args: ProteinOPDTrainingArguments) -> torch.dtype | None:
    if training_args.bf16:
        return torch.bfloat16
    if training_args.fp16:
        return torch.float16
    return None


def _normalize_cli_args(argv: list[str]) -> list[str]:
    option_aliases = {
        "--teacher-config-path": "--teacher_config_path",
        "--student-prefix-adapter-path": "--student_prefix_adapter_path",
        "--use-entropy-rule": "--use_entropy_rule",
        "--entropy-alpha": "--entropy_alpha",
        "--use-z-t": "--useZ_t",
        "--use-curriculum-tem": "--use_curriculum_tem",
        "--curriculum-tem-start": "--curriculum_tem_start",
        "--use-wi-entropy": "--use_wi_entropy",
        "--use-wi-distance": "--use_wi_distance",
        "--wi-entropy-tau": "--wi_entropy_tau",
        "--wi-distance-beta": "--wi_distance_beta",
        "--wi-distance-tau": "--wi_distance_tau",
        "--debug-teacher-diff": "--debug_teacher_diff",
        "--debug-teacher-diff-steps": "--debug_teacher_diff_steps",
    }
    normalized_args: list[str] = []
    for arg in argv:
        rewritten = arg
        for alias, canonical in option_aliases.items():
            if rewritten == alias:
                rewritten = canonical
                break
            if rewritten.startswith(alias + "="):
                rewritten = canonical + rewritten[len(alias) :]
                break
        normalized_args.append(rewritten)
    return normalized_args


def _validate_yaml_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"`{field_name}` in teacher config must be true/false, got: {value!r}")
    return value


def _load_student_prefix_init_metadata(model_path: str) -> dict[str, str] | None:
    metadata_path = Path(model_path) / STUDENT_PREFIX_INIT_METADATA_FILENAME
    if not metadata_path.is_file():
        return None
    with open(metadata_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Student prefix metadata must be a JSON object: {metadata_path}")
    if payload.get("format") != STUDENT_PREFIX_INIT_FORMAT:
        raise ValueError(f"Unsupported student prefix metadata format in {metadata_path}: {payload.get('format')!r}")
    adapter_relative_path = payload.get("adapter_relative_path")
    if not isinstance(adapter_relative_path, str) or not adapter_relative_path.strip():
        raise ValueError(f"Student prefix metadata must define `adapter_relative_path`: {metadata_path}")
    adapter_path = os.path.join(model_path, adapter_relative_path.strip())
    _require_dir(adapter_path, "student prefix adapter from checkpoint metadata")
    return {"metadata_path": str(metadata_path), "adapter_path": os.path.abspath(adapter_path)}


def _resolve_student_prefix_adapter_path(script_args: ProteinOPDScriptArguments) -> str | None:
    if script_args.student_prefix_adapter_path:
        return os.path.abspath(script_args.student_prefix_adapter_path.strip())
    metadata = _load_student_prefix_init_metadata(script_args.student_model_name_or_path)
    if metadata is None:
        return None
    return metadata["adapter_path"]


def _enable_backbone_training_for_prefix_initialized_student(student_model: PeftModel) -> None:
    student_model.requires_grad_(True)
    for name, param in student_model.named_parameters():
        if "prompt_encoder" in name or "prompt_embeddings" in name or "prompt_tokens" in name:
            param.requires_grad = False


def _load_teacher_config(config_path: str, default_teacher_temperature: float) -> TeacherConfig:
    _ensure_local_path(config_path, "teacher_config_path")
    if default_teacher_temperature <= 0:
        raise ValueError("`default_teacher_temperature` must be > 0.")
    with open(config_path, "r", encoding="utf-8") as f:
        payload = yaml.load(f, Loader=_StrictBoolSafeLoader)
    if not isinstance(payload, dict):
        raise ValueError("Teacher config must be a YAML mapping.")

    log_teacher_entropy = _validate_yaml_bool(payload.get("log_teacher_entropy", False), "log_teacher_entropy")
    teacher_backbone_path = payload.get("teacher_backbone_path")
    if not isinstance(teacher_backbone_path, str) or not teacher_backbone_path.strip():
        raise ValueError("Teacher config must define non-empty `teacher_backbone_path`.")
    teacher_backbone_path = teacher_backbone_path.strip()
    _ensure_local_path(teacher_backbone_path, "teacher_backbone_path")

    raw_teachers = payload.get("teachers")
    if not isinstance(raw_teachers, list) or len(raw_teachers) < 2:
        raise ValueError("Teacher config must define `teachers` as a list with at least 2 teachers.")

    teachers: list[TeacherEntryConfig] = []
    for idx, item in enumerate(raw_teachers):
        if not isinstance(item, dict):
            raise ValueError(f"Teacher entry at index {idx} must be a mapping.")
        name = str(item.get("name", f"teacher_{idx}")).strip()
        adapter_path = str(item.get("adapter_path", "")).strip()
        if not name:
            raise ValueError(f"Teacher entry at index {idx} has empty `name`.")
        _ensure_local_path(adapter_path, f"teachers[{idx}].adapter_path")
        try:
            weight = float(item.get("weight"))
            temperature = float(item.get("temperature", default_teacher_temperature))
        except (TypeError, ValueError) as error:
            raise ValueError(f"Teacher entry `{name}` must define numeric `weight` and `temperature`.") from error
        if weight < 0:
            raise ValueError(f"Teacher entry `{name}` has negative `weight`: {weight}")
        if temperature <= 0:
            raise ValueError(f"Teacher entry `{name}` has non-positive `temperature`: {temperature}")
        teachers.append(TeacherEntryConfig(name=name, adapter_path=adapter_path, weight=weight, temperature=temperature))

    total_weight = sum(t.weight for t in teachers)
    if total_weight <= 0:
        raise ValueError("Sum of teacher weights must be > 0.")
    normalized_teachers = [
        TeacherEntryConfig(name=t.name, adapter_path=t.adapter_path, weight=t.weight / total_weight, temperature=t.temperature)
        for t in teachers
    ]
    return TeacherConfig(
        teacher_backbone_path=teacher_backbone_path,
        log_teacher_entropy=log_teacher_entropy,
        teachers=normalized_teachers,
    )


def _validate_script_args(
    script_args: ProteinOPDScriptArguments,
    training_args: ProteinOPDTrainingArguments,
) -> TeacherConfig:
    script_args.student_prefix_adapter_path = script_args.student_prefix_adapter_path.strip()
    _ensure_local_path(script_args.student_model_name_or_path, "student_model_name_or_path")
    embedded_student_prefix = _load_student_prefix_init_metadata(script_args.student_model_name_or_path)

    if script_args.student_prefix_adapter_path:
        _require_dir(script_args.student_prefix_adapter_path, "student_prefix_adapter_path")
        if embedded_student_prefix is not None:
            raise ValueError("Do not pass `student_prefix_adapter_path` when `student_model_name_or_path` already contains prefix metadata.")
    if script_args.num_train_samples <= 0:
        raise ValueError("`num_train_samples` must be > 0")
    if script_args.prompt_mode != "unconditional":
        raise ValueError("Protein-OPD supports only `--prompt_mode unconditional`.")
    if script_args.prompt_prefix_length <= 0:
        raise ValueError("`prompt_prefix_length` must be > 0")
    if script_args.max_new_tokens <= 0:
        raise ValueError("`max_new_tokens` must be > 0")
    if script_args.temperature <= 0:
        raise ValueError("`temperature` must be > 0")
    if script_args.use_curriculum_tem and not (0 < script_args.curriculum_tem_start < script_args.temperature):
        raise ValueError("`curriculum_tem_start` must be > 0 and < `temperature` when curriculum is enabled.")
    if script_args.repetition_penalty <= 0:
        raise ValueError("`repetition_penalty` must be > 0")
    if not (0 < script_args.top_p <= 1.0):
        raise ValueError("`top_p` must be in (0, 1]")
    if not (0 <= script_args.beta <= 1):
        raise ValueError("`beta` must be in [0, 1]")
    if script_args.top_k_loss < 0:
        raise ValueError("`top_k_loss` must be >= 0")
    if script_args.generation_save_steps <= 0:
        raise ValueError("`generation_save_steps` must be > 0")
    if script_args.entropy_alpha < 0:
        raise ValueError("`entropy_alpha` must be >= 0")
    if script_args.wi_entropy_tau <= 0 or script_args.wi_distance_tau <= 0:
        raise ValueError("`wi_entropy_tau` and `wi_distance_tau` must be > 0")
    if not (0 <= script_args.wi_distance_beta < 1):
        raise ValueError("`wi_distance_beta` must be in [0, 1)")
    if script_args.debug_teacher_diff_steps < 0:
        raise ValueError("`debug_teacher_diff_steps` must be >= 0")
    if script_args.student_tune_mode not in {"lora", "full"}:
        raise ValueError("`student_tune_mode` must be one of: lora, full")
    if (script_args.student_prefix_adapter_path or embedded_student_prefix is not None) and script_args.student_tune_mode != "full":
        raise ValueError("Student prefix initialization supports only `student_tune_mode=full`.")
    useloss = script_args.useloss.strip().lower()
    if useloss == "pq":
        useloss = "pg"
    if useloss not in {"pg", "jsd"}:
        raise ValueError("`useloss` must be one of: pg, jsd")
    script_args.useloss = useloss
    if script_args.useloss == "jsd" and script_args.useZ_t:
        logger.warning("`useZ_t` is ignored when `useloss=jsd`.")
    if not script_args.use_entropy_rule and script_args.entropy_alpha > 0:
        logger.warning("`entropy_alpha` > 0 but `use_entropy_rule` is false; entropy term will be disabled.")
    if float(training_args.num_train_epochs) != 1.0:
        raise ValueError("`num_train_epochs` must be 1.0; use `--num_train_samples` to control training size.")
    return _load_teacher_config(script_args.teacher_config_path, default_teacher_temperature=float(script_args.temperature))


def _validate_training_args(training_args: ProteinOPDTrainingArguments) -> None:
    if not training_args.output_dir:
        raise ValueError("`output_dir` is required.")


def _load_tokenizer(student_model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(student_model_path)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has no pad_token and no eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def _load_student_model(
    script_args: ProteinOPDScriptArguments,
    training_args: ProteinOPDTrainingArguments,
) -> tuple[AutoModelForCausalLM | PeftModel, bool]:
    model_dtype = _resolve_torch_dtype(training_args)
    model_kwargs: dict[str, Any] = {}
    if model_dtype is not None:
        model_kwargs["torch_dtype"] = model_dtype
    student_model = AutoModelForCausalLM.from_pretrained(script_args.student_model_name_or_path, **model_kwargs)

    student_prefix_adapter_path = _resolve_student_prefix_adapter_path(script_args)
    if student_prefix_adapter_path is not None:
        logger.info("Loading student with frozen prefix adapter from: %s", student_prefix_adapter_path)
        student_model = PeftModel.from_pretrained(student_model, student_prefix_adapter_path, is_trainable=False)
        _enable_backbone_training_for_prefix_initialized_student(student_model)
        return student_model, True

    if script_args.student_tune_mode == "lora":
        target_modules = _normalize_lora_targets(script_args.lora_target_modules)
        if not target_modules:
            raise ValueError("`lora_target_modules` must not be empty when student_tune_mode=lora")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        student_model = get_peft_model(student_model, lora_config)
        student_model.print_trainable_parameters()
    return student_model, False


def _load_teacher_model_and_specs(
    teacher_config: TeacherConfig,
    training_args: ProteinOPDTrainingArguments,
) -> tuple[PeftModel, list[TeacherAdapterSpec]]:
    model_dtype = _resolve_torch_dtype(training_args)
    model_kwargs: dict[str, Any] = {}
    if model_dtype is not None:
        model_kwargs["torch_dtype"] = model_dtype
    teacher_base_model = AutoModelForCausalLM.from_pretrained(teacher_config.teacher_backbone_path, **model_kwargs)
    first_teacher = teacher_config.teachers[0]
    teacher_model = PeftModel.from_pretrained(teacher_base_model, first_teacher.adapter_path, is_trainable=False)
    specs = [
        TeacherAdapterSpec(
            name=first_teacher.name,
            adapter_name="default",
            weight=first_teacher.weight,
            temperature=first_teacher.temperature,
        )
    ]
    for idx, teacher in enumerate(teacher_config.teachers[1:], start=1):
        adapter_name = teacher.name if teacher.name != "default" else f"teacher_{idx}"
        try:
            teacher_model.load_adapter(teacher.adapter_path, adapter_name=adapter_name, is_trainable=False)
        except TypeError:
            teacher_model.load_adapter(teacher.adapter_path, adapter_name=adapter_name)
        specs.append(
            TeacherAdapterSpec(
                name=teacher.name,
                adapter_name=adapter_name,
                weight=teacher.weight,
                temperature=teacher.temperature,
            )
        )
    teacher_model.requires_grad_(False)
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.eval()
    return teacher_model, specs


class UnconditionalPromptDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples: int):
        self.num_samples = int(num_samples)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, int]:
        return {"sample_id": int(idx)}


class _TransformersRightPaddingWarningFilter(logging.Filter):
    _target_substring = "right-padding was detected"

    def filter(self, record: logging.LogRecord) -> bool:
        return self._target_substring not in record.getMessage()


def _suppress_right_padding_warning_log() -> None:
    logger_for_gen = logging.getLogger("transformers.generation.utils")
    if not any(isinstance(f, _TransformersRightPaddingWarningFilter) for f in logger_for_gen.filters):
        logger_for_gen.addFilter(_TransformersRightPaddingWarningFilter())


def main() -> None:
    _suppress_right_padding_warning_log()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = HfArgumentParser((ProteinOPDScriptArguments, ProteinOPDTrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses(args=_normalize_cli_args(sys.argv[1:]))
    teacher_config = _validate_script_args(script_args, training_args)
    _validate_training_args(training_args)

    if training_args.seed is not None:
        set_seed(training_args.seed)
    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = _load_tokenizer(script_args.student_model_name_or_path)
    train_dataset = UnconditionalPromptDataset(script_args.num_train_samples)
    student_model, student_prefix_init_enabled = _load_student_model(script_args, training_args)
    teacher_model, teacher_specs = _load_teacher_model_and_specs(teacher_config, training_args)

    if training_args.gradient_checkpointing:
        student_model.config.use_cache = False
        if hasattr(student_model, "enable_input_require_grads"):
            student_model.enable_input_require_grads()
    if student_model.get_input_embeddings().num_embeddings != len(tokenizer):
        student_model.resize_token_embeddings(len(tokenizer))
    if teacher_model.get_input_embeddings().num_embeddings != len(tokenizer):
        teacher_model.resize_token_embeddings(len(tokenizer))

    data_collator = ProteinOPDDataCollator(
        tokenizer=tokenizer,
        prompt_mode=script_args.prompt_mode,
        prompt_prefix_length=script_args.prompt_prefix_length,
    )
    trainer = ProteinOPDTrainer(
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        teacher_model=teacher_model,
        teacher_specs=teacher_specs,
        beta=script_args.beta,
        temperature=script_args.temperature,
        use_curriculum_tem=script_args.use_curriculum_tem,
        curriculum_tem_start=script_args.curriculum_tem_start,
        top_k_loss=script_args.top_k_loss,
        max_new_tokens=script_args.max_new_tokens,
        repetition_penalty=script_args.repetition_penalty,
        top_p=script_args.top_p,
        top_k=script_args.top_k,
        useloss=script_args.useloss,
        useZ_t=script_args.useZ_t,
        use_entropy_rule=script_args.use_entropy_rule,
        entropy_alpha=script_args.entropy_alpha,
        use_wi_entropy=script_args.use_wi_entropy,
        use_wi_distance=script_args.use_wi_distance,
        wi_entropy_tau=script_args.wi_entropy_tau,
        wi_distance_beta=script_args.wi_distance_beta,
        wi_distance_tau=script_args.wi_distance_tau,
        debug_teacher_diff=script_args.debug_teacher_diff,
        debug_teacher_diff_steps=script_args.debug_teacher_diff_steps,
        save_generations=script_args.save_generations,
        generation_save_steps=script_args.generation_save_steps,
        student_prefix_init_enabled=student_prefix_init_enabled,
        log_teacher_entropy=teacher_config.log_teacher_entropy,
    )
    trainer.train()
    if script_args.save_generations:
        trainer._save_generation_outputs(int(trainer.state.global_step))
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
