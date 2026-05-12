from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from prollama_opd_config import ProLLaMAOpdConfig, load_opd_config
from prollama_opd_data import ProLLaMAOpdDataCollator
from prollama_opd_trainer import ProLLaMAProteinOPDTrainer, TeacherAdapterSpec


logger = logging.getLogger(__name__)


def _load_causal_lm_with_attention_backend(
    model_path: str,
    use_flash_attention_2: bool = False,
    **model_kwargs,
) -> AutoModelForCausalLM:
    if not use_flash_attention_2:
        return AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            attn_implementation="flash_attention_2",
            **model_kwargs,
        )
    except TypeError as error:
        if "attn_implementation" not in str(error):
            raise
        logger.warning("`attn_implementation` is unsupported; retrying with legacy `use_flash_attention_2=True`.")
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            use_flash_attention_2=True,
            **model_kwargs,
        )


def prepare_model_for_kbit_training(model, use_gradient_checkpointing: bool = True):
    loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.parameters():
        if ((param.dtype == torch.float16) or (param.dtype == torch.bfloat16)) and loaded_in_kbit:
            param.data = param.data.to(torch.float32)
    for name, module in model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)
    if loaded_in_kbit and use_gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(_module, _input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        model.gradient_checkpointing_enable()
    return model


@dataclass
class OpdScriptArguments:
    opd_config_path: str = field(default="")


@dataclass
class OpdTrainingArguments(TrainingArguments):
    output_dir: str = field(default="")
    load_in_kbits: int = field(default=16)
    double_quant: bool = field(default=True)
    quant_type: str = field(default="nf4")
    use_flash_attention_2: bool = field(default=False)
    remove_unused_columns: bool = field(default=False)
    report_to: str = field(default="none")


class FixedPromptDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples: int):
        self.num_samples = int(num_samples)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, int]:
        return {"sample_id": int(idx)}


def _resolve_torch_dtype(training_args: OpdTrainingArguments) -> torch.dtype | None:
    if training_args.bf16:
        return torch.bfloat16
    if training_args.fp16:
        return torch.float16
    return None


def _build_quantization_config(training_args: OpdTrainingArguments) -> tuple[BitsAndBytesConfig | None, dict[str, Any]]:
    if training_args.load_in_kbits not in {4, 8, 16}:
        raise ValueError("`load_in_kbits` must be one of: 4, 8, 16")
    if training_args.load_in_kbits == 16:
        return None, {}
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=training_args.load_in_kbits == 4,
        load_in_8bit=training_args.load_in_kbits == 8,
        llm_int8_threshold=6.0,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=training_args.double_quant,
        bnb_4bit_quant_type=training_args.quant_type,
    )
    return quantization_config, {"": int(os.environ.get("LOCAL_RANK") or 0)}


def _load_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has no pad_token and no eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def _load_model(model_path: str, training_args: OpdTrainingArguments) -> AutoModelForCausalLM:
    model_dtype = _resolve_torch_dtype(training_args)
    model_kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
    if model_dtype is not None:
        model_kwargs["torch_dtype"] = model_dtype
    quantization_config, device_map = _build_quantization_config(training_args)
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = device_map
        model_kwargs["load_in_4bit"] = training_args.load_in_kbits == 4
        model_kwargs["load_in_8bit"] = training_args.load_in_kbits == 8
    elif os.environ.get("LOCAL_RANK") is not None:
        model_kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    model = _load_causal_lm_with_attention_backend(
        model_path,
        use_flash_attention_2=training_args.use_flash_attention_2,
        **model_kwargs,
    )
    if training_args.load_in_kbits in {4, 8}:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
        )
    return model


def _enable_gradient_checkpointing_for_peft(model) -> None:
    model_modules_to_save = getattr(model, "modules_to_save", None)
    if model_modules_to_save and "embed_tokens" in model_modules_to_save:
        return
    hook_target = getattr(model, "base_model", model)
    if hasattr(hook_target, "enable_input_require_grads"):
        hook_target.enable_input_require_grads()
    elif hasattr(hook_target, "get_input_embeddings"):
        def make_inputs_require_grad(_module, _input, output):
            output.requires_grad_(True)

        hook_target.get_input_embeddings().register_forward_hook(make_inputs_require_grad)


def _load_student_model(
    config: ProLLaMAOpdConfig,
    training_args: OpdTrainingArguments,
) -> AutoModelForCausalLM:
    student_model = _load_model(config.student.backbone_path, training_args)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=config.student.lora.target_modules,
        inference_mode=False,
        r=config.student.lora.r,
        lora_alpha=config.student.lora.alpha,
        lora_dropout=config.student.lora.dropout,
        bias="none",
    )
    student_model = get_peft_model(student_model, lora_config)
    student_model.config.use_cache = False
    if training_args.gradient_checkpointing:
        _enable_gradient_checkpointing_for_peft(student_model)
    if hasattr(student_model, "print_trainable_parameters"):
        student_model.print_trainable_parameters()
    return student_model


def _load_teacher_model_and_specs(
    config: ProLLaMAOpdConfig,
    training_args: OpdTrainingArguments,
) -> tuple[PeftModel, list[TeacherAdapterSpec]]:
    teacher_base_model = _load_model(config.teachers.backbone_path, training_args)
    first_teacher = config.teachers.adapters[0]
    teacher_model = PeftModel.from_pretrained(
        teacher_base_model,
        first_teacher.adapter_path,
        is_trainable=False,
    )
    specs: list[TeacherAdapterSpec] = [
        TeacherAdapterSpec(
            name=first_teacher.name,
            adapter_name="default",
            weight=first_teacher.weight,
            temperature=first_teacher.temperature,
        )
    ]
    for idx, teacher in enumerate(config.teachers.adapters[1:], start=1):
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


def _validate_training_setup(
    config: ProLLaMAOpdConfig,
    training_args: OpdTrainingArguments,
) -> None:
    if not config.teachers.adapters:
        raise ValueError("OPD training requires configured teachers.")
    if not training_args.output_dir:
        raise ValueError("`output_dir` is required.")
    if float(training_args.num_train_epochs) != 1.0:
        raise ValueError("`num_train_epochs` must be 1.0; use `generation.num_train_samples` to control training size.")
    if not training_args.do_train:
        raise ValueError("ProLLaMA OPD training requires `--do_train`.")


def main() -> None:
    parser = HfArgumentParser((OpdScriptArguments, OpdTrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()
    config = load_opd_config(script_args.opd_config_path)
    _validate_training_setup(config, training_args)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(training_args.get_process_log_level())
    if training_args.seed is not None:
        set_seed(training_args.seed)
    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = _load_tokenizer(config.student.backbone_path)
    train_dataset = FixedPromptDataset(config.generation.num_train_samples)
    student_model = _load_student_model(config, training_args)
    teacher_model, teacher_specs = _load_teacher_model_and_specs(config, training_args)

    if student_model.get_input_embeddings().num_embeddings != len(tokenizer):
        student_model.resize_token_embeddings(len(tokenizer))
    if teacher_model.get_input_embeddings().num_embeddings != len(tokenizer):
        teacher_model.resize_token_embeddings(len(tokenizer))

    data_collator = ProLLaMAOpdDataCollator(tokenizer=tokenizer, prompt_config=config.prompt)
    trainer = ProLLaMAProteinOPDTrainer(
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        teacher_model=teacher_model,
        teacher_specs=teacher_specs,
        sequence_regex=config.prompt.sequence_regex,
        temperature=config.generation.student_temperature,
        use_curriculum_tem=config.generation.use_curriculum_tem,
        curriculum_tem_start=config.generation.curriculum_tem_start,
        max_new_tokens=config.generation.max_new_tokens,
        repetition_penalty=config.generation.repetition_penalty,
        top_p=config.generation.top_p,
        top_k=config.generation.top_k,
        save_generations=config.generation.save_generations,
        generation_save_steps=config.generation.generation_save_steps,
        beta=config.distill.beta,
        top_k_loss=config.distill.top_k_loss,
        useloss=config.distill.loss_type,
        useZ_t=config.distill.use_z_t,
        use_entropy_rule=config.distill.use_entropy_rule,
        entropy_alpha=config.distill.entropy_alpha,
        use_wi_entropy=config.distill.use_wi_entropy,
        use_wi_distance=config.distill.use_wi_distance,
        wi_entropy_tau=config.distill.wi_entropy_tau,
        wi_distance_beta=config.distill.wi_distance_beta,
        wi_distance_tau=config.distill.wi_distance_tau,
        debug_teacher_diff=config.distill.protein_opd.debug_teacher_diff,
        debug_teacher_diff_steps=config.distill.protein_opd.debug_teacher_diff_steps,
        log_teacher_entropy=config.distill.log_teacher_entropy,
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    if config.generation.save_generations:
        trainer._save_generation_outputs(int(trainer.state.global_step))
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
