#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
import datasets
import torch
from build_instruction_dataset import tokenize_instruction_dataset, DataCollatorForSupervisedDataset
import transformers
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    BitsAndBytesConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.versions import require_version

from peft import LoraConfig, TaskType, get_peft_model, PeftModel, PeftConfig
# from peft.tuners.lora import LoraLayer

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")


def prepare_model_for_kbit_training(model, use_gradient_checkpointing=True):
    r"""
    This method wraps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32

    Args:
        model, (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
    """
    loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)

    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

    # cast all non INT8/INT4 parameters to fp32
    for param in model.parameters():
        if ((param.dtype == torch.float16) or (param.dtype == torch.bfloat16)) and loaded_in_kbit:
            param.data = param.data.to(torch.float32)

    for name, module in model.named_modules():
        if 'norm' in name:
            module = module.to(torch.float32)

    if loaded_in_kbit and use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, _input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()

    return model


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    merge_when_finished : Optional[bool] = field(default=False,metadata={"help":"Merge the lora adapters into the original model when training finished."})
    
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The tokenizer for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )

    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_dir: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )

    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    data_cache_dir: Optional[str] = field(default=None, metadata={"help": "The datasets processed stored"})

    max_seq_length: Optional[int] = field(default=1024)


@dataclass
class MyTrainingArguments(TrainingArguments):
    trainable : Optional[str] = field(default="q_proj,v_proj")
    lora_rank : Optional[int] = field(default=8)
    lora_dropout : Optional[float] = field(default=0.1)
    lora_alpha : Optional[float] = field(default=32.)
    modules_to_save : Optional[str] = field(default=None)
    peft_path : Optional[str] = field(default=None)
    use_flash_attention_2 : Optional[bool] = field(default=False)
    double_quant: Optional[bool] = field(default=True)
    quant_type: Optional[str] = field(default="nf4")
    load_in_kbits: Optional[int] = field(default=16)


logger = logging.getLogger(__name__)


def _load_llama_with_attention_backend(model_name_or_path, use_flash_attention_2=False, **model_kwargs):
    if not use_flash_attention_2:
        return LlamaForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)

    try:
        return LlamaForCausalLM.from_pretrained(
            model_name_or_path,
            attn_implementation="flash_attention_2",
            **model_kwargs,
        )
    except TypeError as error:
        if "attn_implementation" not in str(error):
            raise
        logger.warning(
            "`attn_implementation` is unsupported in this transformers version; "
            "retrying with legacy `use_flash_attention_2=True`."
        )
        return LlamaForCausalLM.from_pretrained(
            model_name_or_path,
            use_flash_attention_2=True,
            **model_kwargs,
        )


def _ensure_existing_path(path_value: str, arg_name: str, expect_dir: bool = False) -> None:
    if not path_value:
        raise ValueError(f"`{arg_name}` is required.")
    if not os.path.exists(path_value):
        raise ValueError(f"`{arg_name}` path does not exist: {path_value}")
    if expect_dir and not os.path.isdir(path_value):
        raise ValueError(f"`{arg_name}` must be a directory: {path_value}")
    if (not expect_dir) and not os.path.isfile(path_value):
        raise ValueError(f"`{arg_name}` must be a file: {path_value}")


def _validate_local_model_path_if_applicable(path_value: Optional[str], arg_name: str) -> None:
    if not path_value:
        return
    expanded = os.path.expanduser(path_value)
    if os.path.isabs(expanded) or expanded.startswith("."):
        _ensure_existing_path(expanded, arg_name, expect_dir=True)


def _validate_output_parent(output_dir: str) -> None:
    output_parent = os.path.dirname(os.path.abspath(output_dir)) or "."
    if not os.path.isdir(output_parent):
        raise ValueError(f"Parent directory of `output_dir` does not exist: {output_parent}")


def _resolve_dataset_dir_files(dataset_dir: str) -> List[str]:
    dataset_root = Path(dataset_dir)
    files = sorted(
        {str(file.resolve()) for pattern in ("*.json", "*.jsonl") for file in dataset_root.glob(pattern)}
    )
    if len(files) == 0:
        raise ValueError(f"No .json or .jsonl files found under dataset_dir={dataset_dir}")
    return files


def _load_instruction_split(
    data_sources,
    tokenizer,
    max_seq_length: int,
    data_cache_dir: Optional[str],
    preprocessing_num_workers: Optional[int],
):
    return tokenize_instruction_dataset(
        data_path=data_sources,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        data_cache_dir=data_cache_dir,
        preprocessing_num_workers=preprocessing_num_workers,
    )


def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.merge_when_finished and training_args.peft_path:
        raise ValueError("`merge_when_finished=True` is only supported when training a new LoRA adapter.")
    if training_args.do_train and not (data_args.train_file or data_args.dataset_dir):
        raise ValueError("Please provide either `--train_file` or `--dataset_dir` when `--do_train` is enabled.")
    if training_args.do_eval and not (data_args.validation_file or data_args.train_file or data_args.dataset_dir):
        raise ValueError(
            "Please provide `--validation_file` or a training dataset source when `--do_eval` is enabled."
        )
    _validate_local_model_path_if_applicable(model_args.model_name_or_path, "model_name_or_path")
    _validate_local_model_path_if_applicable(model_args.tokenizer_name_or_path, "tokenizer_name_or_path")
    if data_args.train_file:
        _ensure_existing_path(data_args.train_file, "train_file")
    if data_args.validation_file:
        _ensure_existing_path(data_args.validation_file, "validation_file")
    if data_args.dataset_dir:
        _ensure_existing_path(data_args.dataset_dir, "dataset_dir", expect_dir=True)
    if training_args.peft_path:
        _ensure_existing_path(training_args.peft_path, "peft_path", expect_dir=True)
    _validate_output_parent(training_args.output_dir)



    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,  # if training_args.local_rank in [-1, 0] else logging.WARN,
        handlers=[logging.StreamHandler(sys.stdout)],)


    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16 or training_args.bf16}"
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.tokenizer_name_or_path:
        tokenizer = LlamaTokenizer.from_pretrained(model_args.tokenizer_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    #set pad_token to eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Add pad token: {}".format(tokenizer.pad_token))

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    eval_dataset = None
    train_dataset = None

    train_sources = None
    if data_args.train_file:
        train_sources = [os.path.abspath(data_args.train_file)]
        if data_args.dataset_dir:
            logger.warning("Both `train_file` and `dataset_dir` were provided. `train_file` will take precedence.")
    elif data_args.dataset_dir:
        train_sources = _resolve_dataset_dir_files(data_args.dataset_dir)

    validation_sources = None
    if data_args.validation_file:
        validation_sources = [os.path.abspath(data_args.validation_file)]

    if training_args.do_train or training_args.do_eval:
        with training_args.main_process_first(desc="loading and tokenization"):
            full_train_dataset = None
            if train_sources is not None:
                logger.info(f"Training files: {' '.join(train_sources)}")
                full_train_dataset = _load_instruction_split(
                    data_sources=train_sources,
                    tokenizer=tokenizer,
                    max_seq_length=data_args.max_seq_length,
                    data_cache_dir=data_args.data_cache_dir,
                    preprocessing_num_workers=data_args.preprocessing_num_workers,
                )

            if validation_sources is not None:
                logger.info(f"Validation files: {' '.join(validation_sources)}")
                eval_dataset = _load_instruction_split(
                    data_sources=validation_sources,
                    tokenizer=tokenizer,
                    max_seq_length=data_args.max_seq_length,
                    data_cache_dir=data_args.data_cache_dir,
                    preprocessing_num_workers=data_args.preprocessing_num_workers,
                )

            if full_train_dataset is not None:
                if training_args.do_eval and eval_dataset is None and data_args.validation_split_percentage > 0:
                    split_dataset = full_train_dataset.train_test_split(
                        test_size=data_args.validation_split_percentage,
                        seed=training_args.seed,
                    )
                    train_dataset = split_dataset["train"]
                    eval_dataset = split_dataset["test"]
                    logger.info(
                        "No validation_file provided. Split the training set with "
                        f"validation_split_percentage={data_args.validation_split_percentage}."
                    )
                else:
                    train_dataset = full_train_dataset

    if training_args.do_train:
        if train_dataset is None:
            raise ValueError("Training dataset could not be loaded. Please provide `train_file` or `dataset_dir`.")
        logger.info(f"Num train_samples {len(train_dataset)}")
        logger.info("Training example:")
        logger.info(tokenizer.decode(train_dataset[0]["input_ids"]))

    if training_args.do_eval:
        if eval_dataset is None:
            raise ValueError(
                "Evaluation was requested but no validation dataset is available. "
                "Provide `--validation_file` or allow a train split."
            )
        logger.info(f"Num eval_samples {len(eval_dataset)}")
        logger.info("Evaluation example:")
        logger.info(tokenizer.decode(eval_dataset[0]["input_ids"]))

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    if training_args.load_in_kbits in [4, 8]:
        load_in_4bit = training_args.load_in_kbits == 4
        load_in_8bit = training_args.load_in_kbits == 8
        if training_args.modules_to_save is not None:
            load_in_8bit_skip_modules = training_args.modules_to_save.split(',')
        else:
            load_in_8bit_skip_modules = None
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=training_args.load_in_kbits == 4,
            load_in_8bit=training_args.load_in_kbits == 8,
            llm_int8_threshold=6.0,
            load_in_8bit_skip_modules=load_in_8bit_skip_modules,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=training_args.double_quant,
            bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
        )
    else:
        load_in_4bit = False
        load_in_8bit = False
        quantization_config = None
    if quantization_config is not None:
        logger.info(f"quantization_config:{quantization_config.to_dict()}")
    device_map = {"":int(os.environ.get("LOCAL_RANK") or 0)}
    model = _load_llama_with_attention_backend(
        model_args.model_name_or_path,
        use_flash_attention_2=training_args.use_flash_attention_2,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        quantization_config=quantization_config,
    )
    if training_args.load_in_kbits in [4, 8]:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    model.config.use_cache = False
    model_vocab_size = model.get_input_embeddings().weight.shape[0]
    logger.info(f"Model vocab size: {model_vocab_size}")
    logger.info(f"len(tokenizer):{len(tokenizer)}")
    if model_vocab_size != len(tokenizer):
        logger.info(f"Resize model vocab size to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
    if training_args.peft_path is not None:
        logger.info("Loading LoRA adapter from %s", training_args.peft_path)
        peft_config = PeftConfig.from_pretrained(training_args.peft_path)
        peft_type = str(peft_config.peft_type).split(".")[-1].lower()
        if peft_type != "lora":
            raise ValueError(f"`peft_path` must point to a LoRA adapter, found `{peft_type}`.")
        try:
            model = PeftModel.from_pretrained(
                model,
                training_args.peft_path,
                device_map=device_map,
                is_trainable=True,
            )
        except TypeError:
            model = PeftModel.from_pretrained(model, training_args.peft_path, device_map=device_map)
    else:
        logger.info("Initializing new LoRA adapter")
        target_modules = [item.strip() for item in training_args.trainable.split(",") if item.strip()]
        modules_to_save = training_args.modules_to_save
        if modules_to_save is not None:
            modules_to_save = [item.strip() for item in modules_to_save.split(",") if item.strip()]
        logger.info(f"target_modules: {target_modules}")
        logger.info(f"lora_rank: {training_args.lora_rank}")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            inference_mode=False,
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            modules_to_save=modules_to_save,
        )
        model = get_peft_model(model, peft_config)

    model_modules_to_save = getattr(model, "modules_to_save", None)
    if (
        training_args.gradient_checkpointing
        and (not model_modules_to_save or "embed_tokens" not in model_modules_to_save)
    ):
        # enable requires_grad to avoid exception during backward pass when using gradient_checkpoint without tuning embed.
        hook_target = getattr(model, "base_model", model)
        if hasattr(hook_target, "enable_input_require_grads"):
            hook_target.enable_input_require_grads()
        elif hasattr(hook_target, "get_input_embeddings"):
            def make_inputs_require_grad(_module, _input, _output):
                _output.requires_grad_(True)
            hook_target.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    logger.info(f"model.modules_to_save: {model_modules_to_save}")

    #The following code will cause the adapter_model.bin saved during training to be empty, so it is commented out.
    # old_state_dict = model.state_dict
    # model.state_dict = (
    #     lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    # ).__get__(model, type(model))

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    # trainer.add_callback(SavePeftModelCallback)

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        trainer.save_model(output_dir=training_args.output_dir)

        if model_args.merge_when_finished:
            if os.environ.get("LOCAL_RANK") is None or os.environ.get("LOCAL_RANK")=='0':
                if training_args.output_dir.endswith('/'):
                    merged_model_path=training_args.output_dir[:-1]+'_merged'
                else:
                    merged_model_path=training_args.output_dir+'_merged'
                merged_model=model.merge_and_unload()
                os.makedirs(merged_model_path,exist_ok=True)
                merged_model.save_pretrained(merged_model_path)
                tokenizer.save_pretrained(merged_model_path)
                logging.info(f"Saved the merged model to {merged_model_path}")

    if training_args.do_eval:
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        metrics["eval_samples"] = len(eval_dataset)
        if "eval_loss" in metrics:
            try:
                metrics["perplexity"] = math.exp(metrics["eval_loss"])
            except OverflowError:
                metrics["perplexity"] = float("inf")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
