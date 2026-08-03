from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


ALLOWED_LOSS_TYPES = {"jsd", "pg"}
ALLOWED_TEACHER_ADAPTER_TYPES = {"lora", "prefix"}
DEFAULT_SEQUENCE_REGEX = r"Seq=<([^>]*)>"


@dataclass(frozen=True)
class StudentLoraConfig:
    r: int
    alpha: int
    dropout: float
    target_modules: list[str]


@dataclass(frozen=True)
class StudentConfig:
    backbone_path: str
    lora: StudentLoraConfig


@dataclass(frozen=True)
class TeacherAdapterConfig:
    name: str
    adapter_path: str
    weight: float
    temperature: float
    adapter_type: str


@dataclass(frozen=True)
class TeacherConfig:
    backbone_path: str
    adapters: list[TeacherAdapterConfig]
    adapter_type: str


@dataclass(frozen=True)
class PromptConfig:
    instruction: str
    input: str
    sequence_regex: str = DEFAULT_SEQUENCE_REGEX

    @property
    def prompt_text(self) -> str:
        return build_prompt_text(self.instruction, self.input)


@dataclass(frozen=True)
class ProteinOPDConfig:
    debug_teacher_diff: bool = False
    debug_teacher_diff_steps: int = 5


@dataclass(frozen=True)
class DistillConfig:
    loss_type: str = "jsd"
    beta: float = 0.5
    top_k_loss: int = 0
    use_z_t: bool = False
    use_entropy_rule: bool = False
    entropy_alpha: float = 0.0
    log_teacher_entropy: bool = False
    use_wi_entropy: bool = False
    use_wi_distance: bool = False
    wi_entropy_tau: float = 1.0
    wi_distance_beta: float = 0.9
    wi_distance_tau: float = 1.0
    protein_opd: ProteinOPDConfig = field(default_factory=ProteinOPDConfig)


@dataclass(frozen=True)
class GenerationConfig:
    num_train_samples: int
    max_new_tokens: int
    student_temperature: float = 1.0
    use_curriculum_tem: bool = False
    curriculum_tem_start: float = 0.7
    repetition_penalty: float = 1.2
    top_p: float = 1.0
    top_k: int = 0
    save_generations: bool = True
    generation_save_steps: int = 5
    sample: bool = False
    sample_steps: int = 100
    sample_num_sequences: int = 10
    sample_temperature: float = 1.0
    sample_batch_size: int = 1


@dataclass(frozen=True)
class ProLLaMAOpdConfig:
    student: StudentConfig
    teachers: TeacherConfig
    prompt: PromptConfig
    distill: DistillConfig
    generation: GenerationConfig


def build_prompt_text(instruction: str, input_text: str | None) -> str:
    instruction = "" if instruction is None else str(instruction)
    input_text = "" if input_text is None else str(input_text)
    if input_text != "":
        return instruction + " " + input_text
    return instruction


def _require_mapping(payload: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"`{field_name}` must be a mapping.")
    return payload


def _require_list(payload: Any, field_name: str) -> list[Any]:
    if not isinstance(payload, list):
        raise ValueError(f"`{field_name}` must be a list.")
    return payload


def _normalize_non_empty_string(value: Any, field_name: str) -> str:
    normalized = "" if value is None else str(value).strip()
    if normalized == "":
        raise ValueError(f"`{field_name}` must be a non-empty string.")
    return normalized


def _normalize_path(value: Any, field_name: str, expect_dir: bool = False) -> str:
    normalized = os.path.abspath(os.path.expanduser(_normalize_non_empty_string(value, field_name)))
    if not os.path.exists(normalized):
        raise ValueError(f"`{field_name}` path does not exist: {normalized}")
    if expect_dir and not os.path.isdir(normalized):
        raise ValueError(f"`{field_name}` must be a directory: {normalized}")
    if not expect_dir and not os.path.isfile(normalized) and not os.path.isdir(normalized):
        raise ValueError(f"`{field_name}` must be a file or directory path: {normalized}")
    return normalized


def _normalize_float(value: Any, field_name: str) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"`{field_name}` must be numeric.") from error
    if not math.isfinite(normalized):
        raise ValueError(f"`{field_name}` must be finite.")
    return normalized


def _normalize_int(value: Any, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"`{field_name}` must be an integer.") from error


def _normalize_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"`{field_name}` must be true or false.")


def _normalize_string_list(value: Any, field_name: str) -> list[str]:
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip() != ""]
    else:
        items = [str(item).strip() for item in _require_list(value, field_name) if str(item).strip() != ""]
    if not items:
        raise ValueError(f"`{field_name}` must not be empty.")
    return items


def _normalize_adapter_type(peft_type: str, field_name: str) -> str:
    normalized = peft_type.strip().lower()
    if normalized == "prefix_tuning":
        normalized = "prefix"
    if normalized not in ALLOWED_TEACHER_ADAPTER_TYPES:
        raise ValueError(f"`{field_name}` must resolve to one of {sorted(ALLOWED_TEACHER_ADAPTER_TYPES)}, got: {peft_type!r}")
    return normalized


def detect_adapter_type(adapter_path: str) -> str:
    adapter_config_path = Path(adapter_path) / "adapter_config.json"
    if not adapter_config_path.is_file():
        raise ValueError(f"Teacher adapter config not found: {adapter_config_path}")
    with open(adapter_config_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Adapter config must be a JSON object: {adapter_config_path}")
    peft_type = payload.get("peft_type")
    if peft_type is None:
        raise ValueError(f"`peft_type` is missing in {adapter_config_path}")
    return _normalize_adapter_type(str(peft_type), f"{adapter_config_path}.peft_type")


def load_opd_config(config_path: str) -> ProLLaMAOpdConfig:
    normalized_config_path = _normalize_path(config_path, "opd_config_path")
    with open(normalized_config_path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    root = _require_mapping(payload, "opd_config")

    student_payload = _require_mapping(root.get("student"), "student")
    student_lora_payload = _require_mapping(student_payload.get("lora"), "student.lora")
    student_lora = StudentLoraConfig(
        r=_normalize_int(student_lora_payload.get("r"), "student.lora.r"),
        alpha=_normalize_int(student_lora_payload.get("alpha"), "student.lora.alpha"),
        dropout=_normalize_float(student_lora_payload.get("dropout"), "student.lora.dropout"),
        target_modules=_normalize_string_list(student_lora_payload.get("target_modules"), "student.lora.target_modules"),
    )
    if student_lora.r <= 0:
        raise ValueError("`student.lora.r` must be > 0.")
    if student_lora.alpha <= 0:
        raise ValueError("`student.lora.alpha` must be > 0.")
    if not (0.0 <= student_lora.dropout < 1.0):
        raise ValueError("`student.lora.dropout` must be in [0, 1).")
    student = StudentConfig(
        backbone_path=_normalize_path(student_payload.get("backbone_path"), "student.backbone_path", expect_dir=True),
        lora=student_lora,
    )

    teachers_payload = _require_mapping(root.get("teachers"), "teachers")
    teacher_backbone_path = _normalize_path(teachers_payload.get("backbone_path"), "teachers.backbone_path", expect_dir=True)
    raw_adapters = _require_list(teachers_payload.get("adapters"), "teachers.adapters")
    if len(raw_adapters) == 0:
        raise ValueError("`teachers.adapters` must contain at least one teacher.")

    teacher_adapters: list[TeacherAdapterConfig] = []
    adapter_types: set[str] = set()
    total_weight = 0.0
    for index, raw_adapter in enumerate(raw_adapters):
        item = _require_mapping(raw_adapter, f"teachers.adapters[{index}]")
        adapter_path = _normalize_path(item.get("adapter_path"), f"teachers.adapters[{index}].adapter_path", expect_dir=True)
        adapter_type = detect_adapter_type(adapter_path)
        adapter_types.add(adapter_type)
        weight = _normalize_float(item.get("weight"), f"teachers.adapters[{index}].weight")
        if weight < 0:
            raise ValueError(f"`teachers.adapters[{index}].weight` must be >= 0.")
        temperature = _normalize_float(item.get("temperature", 1.0), f"teachers.adapters[{index}].temperature")
        if temperature <= 0:
            raise ValueError(f"`teachers.adapters[{index}].temperature` must be > 0.")
        teacher_adapters.append(
            TeacherAdapterConfig(
                name=_normalize_non_empty_string(item.get("name", f"teacher_{index}"), f"teachers.adapters[{index}].name"),
                adapter_path=adapter_path,
                weight=weight,
                temperature=temperature,
                adapter_type=adapter_type,
            )
        )
        total_weight += weight
    if len(adapter_types) != 1:
        raise ValueError(f"All teachers in one OPD run must use the same adapter type. Found: {sorted(adapter_types)}")
    if total_weight <= 0:
        raise ValueError("Sum of `teachers.adapters[*].weight` must be > 0.")
    teachers = TeacherConfig(
        backbone_path=teacher_backbone_path,
        adapters=[
            TeacherAdapterConfig(
                name=item.name,
                adapter_path=item.adapter_path,
                weight=item.weight / total_weight,
                temperature=item.temperature,
                adapter_type=item.adapter_type,
            )
            for item in teacher_adapters
        ],
        adapter_type=next(iter(adapter_types)),
    )

    prompt_payload = _require_mapping(root.get("prompt"), "prompt")
    prompt = PromptConfig(
        instruction=_normalize_non_empty_string(prompt_payload.get("instruction"), "prompt.instruction"),
        input="" if prompt_payload.get("input") is None else str(prompt_payload.get("input")),
        sequence_regex=str(prompt_payload.get("sequence_regex", DEFAULT_SEQUENCE_REGEX)),
    )
    if prompt.sequence_regex.strip() == "":
        raise ValueError("`prompt.sequence_regex` must be a non-empty string.")

    distill_payload = _require_mapping(root.get("distill"), "distill")
    loss_type = str(distill_payload.get("loss_type", "jsd")).strip().lower()
    if loss_type not in ALLOWED_LOSS_TYPES:
        raise ValueError(f"`distill.loss_type` must be one of {sorted(ALLOWED_LOSS_TYPES)}.")
    protein_opd_payload = _require_mapping(distill_payload.get("protein_opd", {}), "distill.protein_opd")
    protein_opd = ProteinOPDConfig(
        debug_teacher_diff=_normalize_bool(protein_opd_payload.get("debug_teacher_diff", False), "distill.protein_opd.debug_teacher_diff"),
        debug_teacher_diff_steps=_normalize_int(protein_opd_payload.get("debug_teacher_diff_steps", 5), "distill.protein_opd.debug_teacher_diff_steps"),
    )
    if protein_opd.debug_teacher_diff_steps < 0:
        raise ValueError("`distill.protein_opd.debug_teacher_diff_steps` must be >= 0.")
    distill = DistillConfig(
        loss_type=loss_type,
        beta=_normalize_float(distill_payload.get("beta", 0.5), "distill.beta"),
        top_k_loss=_normalize_int(distill_payload.get("top_k_loss", 0), "distill.top_k_loss"),
        use_z_t=_normalize_bool(distill_payload.get("use_z_t", False), "distill.use_z_t"),
        use_entropy_rule=_normalize_bool(distill_payload.get("use_entropy_rule", False), "distill.use_entropy_rule"),
        entropy_alpha=_normalize_float(distill_payload.get("entropy_alpha", 0.0), "distill.entropy_alpha"),
        log_teacher_entropy=_normalize_bool(distill_payload.get("log_teacher_entropy", False), "distill.log_teacher_entropy"),
        use_wi_entropy=_normalize_bool(distill_payload.get("use_wi_entropy", False), "distill.use_wi_entropy"),
        use_wi_distance=_normalize_bool(distill_payload.get("use_wi_distance", False), "distill.use_wi_distance"),
        wi_entropy_tau=_normalize_float(distill_payload.get("wi_entropy_tau", 1.0), "distill.wi_entropy_tau"),
        wi_distance_beta=_normalize_float(distill_payload.get("wi_distance_beta", 0.9), "distill.wi_distance_beta"),
        wi_distance_tau=_normalize_float(distill_payload.get("wi_distance_tau", 1.0), "distill.wi_distance_tau"),
        protein_opd=protein_opd,
    )
    if not (0.0 <= distill.beta <= 1.0):
        raise ValueError("`distill.beta` must be in [0, 1].")
    if distill.top_k_loss < 0:
        raise ValueError("`distill.top_k_loss` must be >= 0.")
    if distill.entropy_alpha < 0:
        raise ValueError("`distill.entropy_alpha` must be >= 0.")
    if distill.wi_entropy_tau <= 0 or distill.wi_distance_tau <= 0:
        raise ValueError("`distill.wi_entropy_tau` and `distill.wi_distance_tau` must be > 0.")
    if not (0.0 <= distill.wi_distance_beta < 1.0):
        raise ValueError("`distill.wi_distance_beta` must be in [0, 1).")

    generation_payload = _require_mapping(root.get("generation"), "generation")
    generation = GenerationConfig(
        num_train_samples=_normalize_int(generation_payload.get("num_train_samples"), "generation.num_train_samples"),
        max_new_tokens=_normalize_int(generation_payload.get("max_new_tokens"), "generation.max_new_tokens"),
        student_temperature=_normalize_float(generation_payload.get("student_temperature", 1.0), "generation.student_temperature"),
        use_curriculum_tem=_normalize_bool(generation_payload.get("use_curriculum_tem", False), "generation.use_curriculum_tem"),
        curriculum_tem_start=_normalize_float(generation_payload.get("curriculum_tem_start", 0.7), "generation.curriculum_tem_start"),
        repetition_penalty=_normalize_float(generation_payload.get("repetition_penalty", 1.2), "generation.repetition_penalty"),
        top_p=_normalize_float(generation_payload.get("top_p", 1.0), "generation.top_p"),
        top_k=_normalize_int(generation_payload.get("top_k", 0), "generation.top_k"),
        save_generations=_normalize_bool(generation_payload.get("save_generations", True), "generation.save_generations"),
        generation_save_steps=_normalize_int(generation_payload.get("generation_save_steps", 5), "generation.generation_save_steps"),
        sample=_normalize_bool(generation_payload.get("sample", False), "generation.sample"),
        sample_steps=_normalize_int(generation_payload.get("sample_steps", 100), "generation.sample_steps"),
        sample_num_sequences=_normalize_int(generation_payload.get("sample_num_sequences", 10), "generation.sample_num_sequences"),
        sample_temperature=_normalize_float(generation_payload.get("sample_temperature", 1.0), "generation.sample_temperature"),
        sample_batch_size=_normalize_int(generation_payload.get("sample_batch_size", 1), "generation.sample_batch_size"),
    )
    if generation.num_train_samples <= 0:
        raise ValueError("`generation.num_train_samples` must be > 0.")
    if generation.max_new_tokens <= 0:
        raise ValueError("`generation.max_new_tokens` must be > 0.")
    if generation.student_temperature <= 0:
        raise ValueError("`generation.student_temperature` must be > 0.")
    if generation.use_curriculum_tem and not (0 < generation.curriculum_tem_start < generation.student_temperature):
        raise ValueError("`generation.curriculum_tem_start` must be > 0 and < `generation.student_temperature`.")
    if generation.repetition_penalty <= 0:
        raise ValueError("`generation.repetition_penalty` must be > 0.")
    if not (0.0 < generation.top_p <= 1.0):
        raise ValueError("`generation.top_p` must be in (0, 1].")
    if generation.top_k < 0:
        raise ValueError("`generation.top_k` must be >= 0.")
    if generation.generation_save_steps <= 0:
        raise ValueError("`generation.generation_save_steps` must be > 0.")
    if generation.sample:
        if generation.sample_steps <= 0:
            raise ValueError("`generation.sample_steps` must be > 0 when sampling is enabled.")
        if generation.sample_num_sequences <= 0:
            raise ValueError("`generation.sample_num_sequences` must be > 0 when sampling is enabled.")
        if generation.sample_temperature <= 0:
            raise ValueError("`generation.sample_temperature` must be > 0 when sampling is enabled.")
        if generation.sample_batch_size <= 0:
            raise ValueError("`generation.sample_batch_size` must be > 0 when sampling is enabled.")

    return ProLLaMAOpdConfig(
        student=student,
        teachers=teachers,
        prompt=prompt,
        distill=distill,
        generation=generation,
    )
