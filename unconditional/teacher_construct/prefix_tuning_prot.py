from __future__ import annotations

import argparse
import logging
import math
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover
    load_dataset = None

try:
    from peft import PrefixTuningConfig, TaskType, get_peft_model
except ImportError:  # pragma: no cover
    PrefixTuningConfig = None
    TaskType = None
    get_peft_model = None

try:
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover
    DataLoader = None

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        default_data_collator,
        get_polynomial_decay_schedule_with_warmup,
    )
except ImportError:  # pragma: no cover
    AutoModelForCausalLM = None
    AutoTokenizer = None
    default_data_collator = None
    get_polynomial_decay_schedule_with_warmup = None


logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    model_name_or_path: str
    tokenizer_name_or_path: str
    train_path: str
    test_path: str
    text_column: str = "sequence"
    dataset_name: str = "protein_prefix"
    output_dir: str = "outputs/protgpt2_prefix"
    log_dir: str = "logs/protgpt2_prefix"
    run_dir: str = "runs/protgpt2_prefix"
    num_virtual_tokens: int = 100
    max_length: int = 256
    learning_rate: float = 5e-3
    lr_end: float = 1e-7
    lr_power: float = 3.0
    num_epochs: int = 50
    batch_size: int = 8
    seed: int = 42
    device: str = "auto"
    early_stop: bool = True
    early_stop_patience: int = 5
    early_stop_min_delta: float = 1e-4
    wandb_project: str = "proteinopd-unconditional-teacher"
    wandb_run_name: str = ""
    wandb_entity: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prefix tune ProtGPT2 teacher adapters.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_suffix(".yaml")),
        help="YAML configuration path.",
    )
    return parser.parse_args()


def _require_mapping(payload: Any, name: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"`{name}` must be a mapping.")
    return payload


def _load_config(config_path: str) -> TrainConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        payload = _require_mapping(yaml.safe_load(f), "config")
    config = TrainConfig(**payload)
    if config.num_virtual_tokens <= 0:
        raise ValueError("`num_virtual_tokens` must be > 0.")
    if config.max_length <= 0:
        raise ValueError("`max_length` must be > 0.")
    if config.learning_rate <= 0:
        raise ValueError("`learning_rate` must be > 0.")
    if config.num_epochs <= 0:
        raise ValueError("`num_epochs` must be > 0.")
    if config.batch_size <= 0:
        raise ValueError("`batch_size` must be > 0.")
    if config.early_stop_patience <= 0:
        raise ValueError("`early_stop_patience` must be > 0.")
    for field_name in ("model_name_or_path", "tokenizer_name_or_path", "train_path", "test_path"):
        path_value = str(getattr(config, field_name))
        if not os.path.exists(path_value):
            raise ValueError(f"`{field_name}` path does not exist: {path_value}")
    return config


def _configure_logging(log_dir: str) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / "prefix_tuning_prot.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )


def set_seed(seed: int) -> None:
    require_runtime_dependencies()
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def _resolve_device(device_value: str) -> torch.device:
    require_runtime_dependencies()
    if device_value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_value)


def require_runtime_dependencies() -> None:
    missing = []
    if np is None:
        missing.append("numpy")
    if torch is None or DataLoader is None:
        missing.append("torch")
    if SummaryWriter is None:
        missing.append("tensorboard")
    if load_dataset is None:
        missing.append("datasets")
    if PrefixTuningConfig is None or TaskType is None or get_peft_model is None:
        missing.append("peft")
    if tqdm is None:
        missing.append("tqdm")
    if wandb is None:
        missing.append("wandb")
    if AutoModelForCausalLM is None or AutoTokenizer is None or default_data_collator is None or get_polynomial_decay_schedule_with_warmup is None:
        missing.append("transformers")
    if missing:
        raise ImportError("Missing runtime dependencies required for prefix tuning: " + ", ".join(missing))


def _get_split_max_token_length(split_dataset, split_name: str, tokenizer, text_column: str, scan_batch_size: int = 1024) -> int:
    max_token_len = 0
    max_raw_len = 0
    max_index = -1
    total_samples = len(split_dataset)
    for start_idx in range(0, total_samples, scan_batch_size):
        end_idx = min(start_idx + scan_batch_size, total_samples)
        batch_texts = split_dataset[start_idx:end_idx][text_column]
        tokenized = tokenizer(batch_texts, add_special_tokens=False)["input_ids"]
        for offset, token_ids in enumerate(tokenized):
            token_len = len(token_ids) + 2
            if token_len > max_token_len:
                max_token_len = token_len
                max_raw_len = len(batch_texts[offset])
                max_index = start_idx + offset
    logger.info(
        "%s max sequence length: tokens_with_eos=%s raw_chars=%s sample_index=%s",
        split_name,
        max_token_len,
        max_raw_len,
        max_index,
    )
    return max_token_len


def train(config: TrainConfig, config_path: str) -> None:
    require_runtime_dependencies()
    device = _resolve_device(config.device)
    writer = SummaryWriter(config.run_dir)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("csv", data_files={"train": config.train_path, "test": config.test_path})
    logger.info("Dataset: %s", dataset)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_max = _get_split_max_token_length(dataset["train"], "train", tokenizer, config.text_column)
    test_max = _get_split_max_token_length(dataset["test"], "test", tokenizer, config.text_column)
    longest = max(train_max, test_max)
    if longest > config.max_length:
        raise ValueError(
            f"max_length check failed: dataset longest token length with eos is {longest}, "
            f"but max_length={config.max_length}."
        )

    def preprocess_function(examples):
        batch_size = len(examples[config.text_column])
        inputs = [x for x in examples[config.text_column]]
        model_inputs = tokenizer(inputs)
        labels = tokenizer(inputs)
        for i in range(batch_size):
            sample_input_ids = [tokenizer.eos_token_id] + model_inputs["input_ids"][i] + [tokenizer.eos_token_id]
            label_input_ids = [tokenizer.eos_token_id] + labels["input_ids"][i] + [tokenizer.eos_token_id]
            pad_len = config.max_length - len(sample_input_ids)
            model_inputs["input_ids"][i] = torch.tensor((sample_input_ids + [tokenizer.pad_token_id] * pad_len)[: config.max_length])
            model_inputs["attention_mask"][i] = torch.tensor(([1] * len(sample_input_ids) + [0] * pad_len)[: config.max_length])
            labels["input_ids"][i] = torch.tensor((label_input_ids + [-100] * pad_len)[: config.max_length])
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    processed_datasets = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=1,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False,
        desc="Tokenizing protein sequences",
    )
    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["test"]

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=config.batch_size,
        pin_memory=True,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=default_data_collator,
        batch_size=config.batch_size,
        pin_memory=True,
    )

    peft_config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=config.num_virtual_tokens,
    )
    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_dataloader) * config.num_epochs,
        lr_end=config.lr_end,
        power=config.lr_power,
    )

    final_model_dir = (
        Path(config.output_dir)
        / f"prefix_{peft_config.peft_type}_{peft_config.task_type}_E{config.num_epochs}"
        f"_LR{config.learning_rate}_BS{config.batch_size}_ML{config.max_length}_VT{config.num_virtual_tokens}"
    )
    run_name = config.wandb_run_name or final_model_dir.name
    wandb_run = wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity or None,
        name=run_name,
        config=dict(config.__dict__),
    )
    wandb.define_metric("global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    wandb.define_metric("eval/*", step_metric="global_step")
    run_url = getattr(wandb_run, "url", None)
    print(f"Weights & Biases run: {run_url or 'URL unavailable (check WANDB_MODE/login)'}", flush=True)
    best_model_tmp_dir = Path(str(final_model_dir) + "_best_tmp")
    if config.early_stop and best_model_tmp_dir.exists():
        shutil.rmtree(best_model_tmp_dir)

    best_eval_loss = float("inf")
    best_epoch = -1
    no_improve_epochs = 0
    model = model.to(device)
    start_time = time.time()
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(tqdm(train_dataloader, ncols=80)):
            global_step = epoch * len(train_dataloader) + step + 1
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += float(loss.detach().item())
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            writer.add_scalar("train_step/loss", loss.item(), global_step)
            wandb.log(
                {
                    "train/step_loss": loss.item(),
                    "train/learning_rate": lr_scheduler.get_last_lr()[0],
                    "global_step": global_step,
                },
            )

        model.eval()
        eval_loss = 0.0
        for batch in tqdm(eval_dataloader, ncols=80):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            eval_loss += float(outputs.loss.detach().item())

        train_epoch_loss = total_loss / max(len(train_dataloader), 1)
        eval_epoch_loss = eval_loss / max(len(eval_dataloader), 1)
        train_ppl = math.exp(min(train_epoch_loss, 80.0))
        eval_ppl = math.exp(min(eval_epoch_loss, 80.0))
        writer.add_scalar("train_epoch/loss", train_epoch_loss, epoch)
        writer.add_scalar("train_epoch/ppl", train_ppl, epoch)
        writer.add_scalar("eval_epoch/loss", eval_epoch_loss, epoch)
        writer.add_scalar("eval_epoch/ppl", eval_ppl, epoch)
        logger.info(
            "epoch=%s train_loss=%.6f eval_loss=%.6f train_ppl=%.6f eval_ppl=%.6f",
            epoch + 1,
            train_epoch_loss,
            eval_epoch_loss,
            train_ppl,
            eval_ppl,
        )
        wandb.log(
            {
                "train/epoch_loss": train_epoch_loss,
                "train/epoch_ppl": train_ppl,
                "eval/loss": eval_epoch_loss,
                "eval/ppl": eval_ppl,
                "epoch": epoch + 1,
                "global_step": (epoch + 1) * len(train_dataloader),
            },
        )

        if config.early_stop:
            if eval_epoch_loss < best_eval_loss - config.early_stop_min_delta:
                best_eval_loss = eval_epoch_loss
                best_epoch = epoch + 1
                no_improve_epochs = 0
                model.save_pretrained(best_model_tmp_dir)
                tokenizer.save_pretrained(best_model_tmp_dir)
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= config.early_stop_patience:
                    logger.info("Early stopping at epoch=%s; best_epoch=%s best_eval_loss=%.6f", epoch + 1, best_epoch, best_eval_loss)
                    break

    writer.add_scalar("training/time_seconds", time.time() - start_time, 0)
    if config.early_stop and best_model_tmp_dir.exists():
        if final_model_dir.exists():
            shutil.rmtree(final_model_dir)
        shutil.copytree(best_model_tmp_dir, final_model_dir)
        logger.info("Best model saved to %s", final_model_dir)
    else:
        model.save_pretrained(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
        logger.info("Final model saved to %s", final_model_dir)
    shutil.copy2(config_path, final_model_dir / "training_config.yaml")
    logger.info("Training config saved to %s", final_model_dir / "training_config.yaml")
    writer.close()
    wandb.finish()


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    _configure_logging(config.log_dir)
    logger.info("Loaded config from %s", args.config)
    for key, value in config.__dict__.items():
        logger.info("%s = %s", key, value)
    set_seed(config.seed)
    train(config, args.config)


if __name__ == "__main__":
    main()
