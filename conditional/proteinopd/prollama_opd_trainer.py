from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, Trainer

from prollama_opd_data import IGNORE_INDEX


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TeacherAdapterSpec:
    name: str
    adapter_name: str
    weight: float
    temperature: float


class BaseProLLaMAOpdTrainer(Trainer):
    """Shared geometric OPD trainer core for ProLLaMA."""

    def __init__(
        self,
        *args,
        teacher_model: PreTrainedModel,
        teacher_specs: list[TeacherAdapterSpec],
        tokenizer,
        sequence_regex: str,
        beta: float = 0.5,
        temperature: float = 1.0,
        use_curriculum_tem: bool = False,
        curriculum_tem_start: float = 0.7,
        top_k_loss: int = 0,
        max_new_tokens: int = 256,
        repetition_penalty: float = 1.2,
        top_p: float = 1.0,
        top_k: int = 0,
        useloss: str = "jsd",
        useZ_t: bool = False,
        use_entropy_rule: bool = False,
        entropy_alpha: float = 0.0,
        use_wi_entropy: bool = False,
        use_wi_distance: bool = False,
        wi_entropy_tau: float = 1.0,
        wi_distance_beta: float = 0.9,
        wi_distance_tau: float = 1.0,
        debug_teacher_diff: bool = False,
        debug_teacher_diff_steps: int = 5,
        save_generations: bool = True,
        generation_save_steps: int = 5,
        log_teacher_entropy: bool = False,
        **kwargs,
    ) -> None:
        if useloss not in {"pg", "jsd"}:
            raise ValueError("`useloss` must be one of: pg, jsd")
        if len(teacher_specs) == 0:
            raise ValueError("`teacher_specs` must not be empty.")
        try:
            re.compile(sequence_regex)
        except re.error as error:
            raise ValueError(f"Invalid sequence regex: {sequence_regex!r}") from error

        self.teacher_model = teacher_model
        self.teacher_specs = teacher_specs
        self.teacher_weights = torch.tensor([spec.weight for spec in teacher_specs], dtype=torch.float32)
        for spec in teacher_specs:
            if float(spec.temperature) <= 0:
                raise ValueError(f"Teacher `{spec.name}` has non-positive temperature: {spec.temperature}")

        self.sequence_regex = sequence_regex
        self.distill_beta = float(beta)
        self.distill_temperature = float(temperature)
        self.use_curriculum_tem = bool(use_curriculum_tem)
        self.curriculum_tem_start = float(curriculum_tem_start)
        self.top_k_loss = top_k_loss if top_k_loss > 0 else None
        self.max_new_tokens = int(max_new_tokens)
        self.gen_repetition_penalty = float(repetition_penalty)
        self.gen_top_p = float(top_p)
        self.gen_top_k = int(top_k)
        if self.use_curriculum_tem:
            if self.curriculum_tem_start <= 0:
                raise ValueError("`curriculum_tem_start` must be > 0 when enabled.")
            if self.curriculum_tem_start >= self.distill_temperature:
                raise ValueError("`curriculum_tem_start` must be < `temperature` when enabled.")

        self.useloss = useloss
        self.useZ_t = bool(useZ_t)
        self.use_entropy_rule = bool(use_entropy_rule)
        self.entropy_alpha = float(entropy_alpha)
        self.use_wi_entropy = bool(use_wi_entropy)
        self.use_wi_distance = bool(use_wi_distance)
        self.wi_entropy_tau = float(wi_entropy_tau)
        self.wi_distance_beta = float(wi_distance_beta)
        self.wi_distance_tau = float(wi_distance_tau)
        self.log_teacher_entropy = bool(log_teacher_entropy)
        self.debug_teacher_diff = bool(debug_teacher_diff)
        self.debug_teacher_diff_steps = int(max(0, debug_teacher_diff_steps))
        if self.wi_entropy_tau <= 0:
            raise ValueError("`wi_entropy_tau` must be > 0.")
        if not (0 <= self.wi_distance_beta < 1):
            raise ValueError("`wi_distance_beta` must be in [0, 1).")
        if self.wi_distance_tau <= 0:
            raise ValueError("`wi_distance_tau` must be > 0.")

        self.save_generations = bool(save_generations)
        self.generation_save_steps = int(generation_save_steps)
        self._generation_outputs_buffer: list[dict[str, Any]] = []
        self._train_metric_buffer: dict[str, list[float]] = {}
        self.running_teacher_kl = torch.zeros(len(self.teacher_specs), dtype=torch.float32)
        self._teacher_metric_suffixes = [
            f"{idx}_{self._normalize_metric_name(spec.name)}"
            for idx, spec in enumerate(self.teacher_specs)
        ]
        self._tokenizer = tokenizer

        super().__init__(*args, **kwargs)

        self._assert_zero3_not_used()
        self._prepare_teacher_model()
        if self.useloss == "jsd" and self.useZ_t:
            logger.warning("`useZ_t` is ignored when `useloss=jsd`.")

    def _assert_zero3_not_used(self) -> None:
        deepspeed_plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        if getattr(deepspeed_plugin, "zero_stage", None) == 3:
            raise ValueError("ProLLaMA OPD does not support DeepSpeed ZeRO-3.")

    def _prepare_teacher_model(self) -> None:
        self.teacher_model.requires_grad_(False)
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()
        self.teacher_model.to(self.accelerator.device)

    def _get_effective_student_temperature(self) -> float:
        if not self.use_curriculum_tem:
            return self.distill_temperature
        max_steps = int(getattr(self.state, "max_steps", 0) or 0)
        if max_steps <= 1:
            return self.distill_temperature
        progress = min(float(max(0, self.state.global_step)) / float(max_steps - 1), 1.0)
        return self.curriculum_tem_start + (self.distill_temperature - self.curriculum_tem_start) * progress

    def _build_generation_kwargs(self, student_temperature: float) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": True,
            "temperature": float(student_temperature),
            "repetition_penalty": self.gen_repetition_penalty,
            "top_p": self.gen_top_p,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
            "use_cache": True,
        }
        if self.gen_top_k > 0:
            kwargs["top_k"] = self.gen_top_k
        return kwargs

    @staticmethod
    def _normalize_metric_name(name: str) -> str:
        normalized = "".join(ch if ch.isalnum() else "_" for ch in str(name).strip().lower()).strip("_")
        return normalized or "teacher"

    def _buffer_train_metrics(self, metrics: dict[str, float | torch.Tensor]) -> None:
        for key, value in metrics.items():
            scalar = float(value.detach().item()) if isinstance(value, torch.Tensor) and value.numel() == 1 else float(value)
            if math.isfinite(scalar):
                self._train_metric_buffer.setdefault(key, []).append(scalar)

    def _flush_buffered_train_metrics(self) -> dict[str, float]:
        aggregated = {
            key: float(sum(values) / len(values))
            for key, values in self._train_metric_buffer.items()
            if values
        }
        self._train_metric_buffer.clear()
        return aggregated

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        if ("loss" in logs or "grad_norm" in logs or "learning_rate" in logs) and self._train_metric_buffer:
            logs = dict(logs)
            logs.update(self._flush_buffered_train_metrics())
        try:
            super().log(logs, start_time=start_time)
        except TypeError:
            super().log(logs)

    def _generate_on_policy(
        self,
        model: torch.nn.Module,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        student_temperature: float,
    ) -> torch.Tensor:
        generation_model = self.accelerator.unwrap_model(model)
        generation_kwargs = self._build_generation_kwargs(student_temperature)
        was_training = generation_model.training
        original_use_cache = getattr(generation_model.config, "use_cache", True)
        generation_model.eval()
        generation_model.config.use_cache = True
        try:
            with torch.no_grad():
                return generation_model.generate(
                    input_ids=prompt_ids,
                    attention_mask=attention_mask,
                    **generation_kwargs,
                )
        finally:
            generation_model.config.use_cache = original_use_cache
            if was_training:
                generation_model.train()

    def _canonicalize_rollouts(
        self,
        generated_ids: torch.Tensor,
        prompt_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        eos_token_id = self._tokenizer.eos_token_id
        if eos_token_id is None:
            raise ValueError("Tokenizer must define eos_token_id.")
        pad_token_id = self._tokenizer.pad_token_id if self._tokenizer.pad_token_id is not None else eos_token_id
        prompt_length = int(prompt_ids.shape[1])
        rollout_tensors: list[torch.Tensor] = []
        for row in generated_ids:
            completion_ids = row[prompt_length:].tolist()
            if pad_token_id != eos_token_id:
                while completion_ids and completion_ids[-1] == pad_token_id:
                    completion_ids.pop()
            while completion_ids and completion_ids[-1] == eos_token_id:
                completion_ids.pop()
            rollout = row[:prompt_length].tolist() + completion_ids + [eos_token_id]
            rollout_tensors.append(torch.tensor(rollout, dtype=torch.long, device=generated_ids.device))

        max_len = max(t.numel() for t in rollout_tensors)
        rollout_batch = torch.full((len(rollout_tensors), max_len), pad_token_id, dtype=torch.long, device=generated_ids.device)
        attention_mask = torch.zeros_like(rollout_batch)
        labels = torch.full_like(rollout_batch, IGNORE_INDEX)
        for idx, rollout in enumerate(rollout_tensors):
            length = rollout.numel()
            rollout_batch[idx, :length] = rollout
            attention_mask[idx, :length] = 1
            labels[idx, prompt_length:length] = rollout[prompt_length:length]
        return rollout_batch, attention_mask, labels, rollout_tensors

    def _extract_sequence(self, output_text: str) -> str | None:
        matched = re.search(self.sequence_regex, output_text)
        if matched is None:
            return None
        sequence = "".join(str(matched.group(1)).split())
        return sequence or None

    def _record_generation_outputs(self, rollout_tensors: list[torch.Tensor], prompt_length: int) -> None:
        if not self.save_generations or not self.accelerator.is_main_process:
            return
        for rollout in rollout_tensors:
            token_ids = rollout.detach().cpu().tolist()
            output_ids = token_ids[prompt_length:]
            output_text = self._tokenizer.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            self._generation_outputs_buffer.append(
                {
                    "output_text": output_text,
                    "sequence": self._extract_sequence(output_text),
                }
            )

    def _save_generation_outputs(self, step: int) -> None:
        if not self.save_generations or not self.accelerator.is_main_process or not self._generation_outputs_buffer:
            return
        generations_dir = Path(self.args.output_dir) / "generations"
        generations_dir.mkdir(parents=True, exist_ok=True)
        output_file = generations_dir / f"generations_step_{step}.json"
        output_data = {
            "step": step,
            "num_samples": len(self._generation_outputs_buffer),
            "generations": self._generation_outputs_buffer,
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(self._generation_outputs_buffer)} generation outputs to: {output_file}")
        self._generation_outputs_buffer.clear()

    @staticmethod
    def _generalized_jsd_loss_from_log_probs(
        student_log_probs: torch.Tensor,
        target_log_probs: torch.Tensor,
        labels: torch.Tensor | None = None,
        beta: float = 0.5,
    ) -> torch.Tensor:
        if beta == 0:
            token_jsd = F.kl_div(student_log_probs, target_log_probs, reduction="none", log_target=True)
        elif beta == 1:
            token_jsd = F.kl_div(target_log_probs, student_log_probs, reduction="none", log_target=True)
        else:
            beta_tensor = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
            mixture_log_probs = torch.logsumexp(
                torch.stack([student_log_probs + torch.log1p(-beta_tensor), target_log_probs + torch.log(beta_tensor)]),
                dim=0,
            )
            token_jsd = beta_tensor * F.kl_div(mixture_log_probs, target_log_probs, reduction="none", log_target=True)
            token_jsd = token_jsd + (1.0 - beta_tensor) * F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
        token_jsd = token_jsd.sum(dim=-1)
        if labels is not None:
            valid_mask = labels != IGNORE_INDEX
            token_jsd = token_jsd[valid_mask]
            if token_jsd.numel() == 0:
                return student_log_probs.new_tensor(0.0)
        return token_jsd.mean()

    def _forward_teacher_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_length: int,
    ) -> list[torch.Tensor]:
        if not hasattr(self.teacher_model, "set_adapter"):
            raise ValueError("Teacher model does not support adapter switching via `set_adapter`.")
        teacher_logits_list: list[torch.Tensor] = []
        with torch.no_grad():
            for spec in self.teacher_specs:
                self.teacher_model.set_adapter(spec.adapter_name)
                outputs_teacher = self.teacher_model(input_ids=input_ids, attention_mask=attention_mask)
                teacher_logits_list.append(outputs_teacher.logits[:, prompt_length - 1 : -1, :])
        return teacher_logits_list

    def _build_teacher_log_probs(self, teacher_logits_list: list[torch.Tensor]) -> list[torch.Tensor]:
        return [
            F.log_softmax(teacher_logits / float(self.teacher_specs[idx].temperature), dim=-1)
            for idx, teacher_logits in enumerate(teacher_logits_list)
        ]

    @staticmethod
    def _stack_teacher_token_log_probs(
        teacher_log_probs_list: list[torch.Tensor],
        sampled_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stack(
            [
                torch.gather(log_probs, dim=-1, index=sampled_token_ids.unsqueeze(-1)).squeeze(-1)
                for log_probs in teacher_log_probs_list
            ],
            dim=-1,
        )

    def _compute_dynamic_teacher_weights(
        self,
        teacher_log_probs_list: list[torch.Tensor],
        student_log_probs_full: torch.Tensor,
        sampled_token_ids: torch.Tensor,
        shifted_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float | torch.Tensor]]:
        num_teachers = len(teacher_log_probs_list)
        batch_size, seq_len = sampled_token_ids.shape
        device = student_log_probs_full.device
        dtype = student_log_probs_full.dtype
        combined_weights = self.teacher_weights.to(device=device, dtype=dtype).view(1, 1, num_teachers).expand(batch_size, seq_len, num_teachers)
        valid_mask = shifted_labels != IGNORE_INDEX
        valid_float = valid_mask.to(dtype=dtype)
        valid_count = valid_float.sum()
        has_valid = bool(valid_count.item() > 0)
        denom = valid_count.clamp_min(1.0)
        metrics: dict[str, float | torch.Tensor] = {}

        with torch.no_grad():
            entropy_stack: torch.Tensor | None = None
            if self.use_wi_entropy or self.log_teacher_entropy:
                entropy_stack = torch.stack(
                    [-(log_probs.detach().exp() * log_probs.detach()).sum(dim=-1) for log_probs in teacher_log_probs_list],
                    dim=-1,
                )
            if self.log_teacher_entropy and entropy_stack is not None:
                teacher_entropy_mean = (entropy_stack * valid_float.unsqueeze(-1)).sum(dim=(0, 1)) / denom if has_valid else entropy_stack.mean(dim=(0, 1))
                for idx, suffix in enumerate(self._teacher_metric_suffixes):
                    metrics[f"train/teacher_entropy_{suffix}"] = teacher_entropy_mean[idx]
            if self.use_wi_entropy and entropy_stack is not None:
                w_conf = F.softmax(-entropy_stack / self.wi_entropy_tau, dim=-1)
                combined_weights = combined_weights * w_conf
                conf_mean = (w_conf * valid_float.unsqueeze(-1)).sum(dim=(0, 1)) / denom if has_valid else w_conf.mean(dim=(0, 1))
                for idx, suffix in enumerate(self._teacher_metric_suffixes):
                    metrics[f"train/wi_conf_{suffix}"] = conf_mean[idx]
            if self.use_wi_distance:
                teacher_sampled_stack = self._stack_teacher_token_log_probs(teacher_log_probs_list, sampled_token_ids).detach()
                student_sampled = torch.gather(student_log_probs_full.detach(), dim=-1, index=sampled_token_ids.unsqueeze(-1)).squeeze(-1)
                batch_kl = ((student_sampled.unsqueeze(-1) - teacher_sampled_stack) * valid_float.unsqueeze(-1)).sum(dim=(0, 1)) / denom if has_valid else torch.zeros((num_teachers,), dtype=dtype, device=device)
                if self.running_teacher_kl.device != device:
                    self.running_teacher_kl = self.running_teacher_kl.to(device)
                self.running_teacher_kl = self.wi_distance_beta * self.running_teacher_kl + (1.0 - self.wi_distance_beta) * batch_kl.to(self.running_teacher_kl.dtype)
                w_curr = F.softmax(self.running_teacher_kl.to(dtype=dtype) / self.wi_distance_tau, dim=-1)
                combined_weights = combined_weights * w_curr.view(1, 1, num_teachers)
                for idx, suffix in enumerate(self._teacher_metric_suffixes):
                    metrics[f"train/wi_curr_{suffix}"] = w_curr[idx]
                    metrics[f"train/kl_teacher_{suffix}"] = batch_kl[idx]

            combined_weights = combined_weights / combined_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            final_mean = (combined_weights * valid_float.unsqueeze(-1)).sum(dim=(0, 1)) / denom if has_valid else combined_weights.mean(dim=(0, 1))
            for idx, suffix in enumerate(self._teacher_metric_suffixes):
                metrics[f"train/wi_final_{suffix}"] = final_mean[idx]
        return combined_weights.detach(), metrics

    def _aggregate_teacher_log_probs(
        self,
        teacher_log_probs_list: list[torch.Tensor],
        teacher_weights_token: torch.Tensor,
    ) -> torch.Tensor:
        teacher_log_probs_tensor = torch.stack(teacher_log_probs_list, dim=2)
        return (teacher_weights_token.unsqueeze(-1) * teacher_log_probs_tensor).sum(dim=2)

    def _compute_pg_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits_list: list[torch.Tensor],
        sampled_token_ids: torch.Tensor,
        shifted_labels: torch.Tensor,
        student_temperature: float,
    ) -> tuple[torch.Tensor, dict[str, float | torch.Tensor]]:
        student_log_probs = F.log_softmax(student_logits / float(student_temperature), dim=-1)
        teacher_log_probs_list = self._build_teacher_log_probs(teacher_logits_list)
        teacher_weights_token, wi_metrics = self._compute_dynamic_teacher_weights(teacher_log_probs_list, student_log_probs, sampled_token_ids, shifted_labels)
        aggregated_teacher_log_probs = self._aggregate_teacher_log_probs(teacher_log_probs_list, teacher_weights_token)
        student_log_probs_sampled = torch.gather(student_log_probs, dim=-1, index=sampled_token_ids.unsqueeze(-1)).squeeze(-1)
        teacher_log_probs_sampled = torch.gather(aggregated_teacher_log_probs, dim=-1, index=sampled_token_ids.unsqueeze(-1)).squeeze(-1)
        if self.useZ_t:
            teacher_log_probs_sampled = teacher_log_probs_sampled - torch.logsumexp(aggregated_teacher_log_probs, dim=-1)
        advantage = (teacher_log_probs_sampled - student_log_probs_sampled).detach()
        if self.use_entropy_rule and self.entropy_alpha > 0:
            advantage = advantage + self.entropy_alpha * (-student_log_probs_sampled)
        valid_mask = shifted_labels != IGNORE_INDEX
        if valid_mask.any():
            return -(advantage[valid_mask] * student_log_probs_sampled[valid_mask]).mean(), wi_metrics
        return student_logits.new_tensor(0.0), wi_metrics

    def _compute_jsd_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits_list: list[torch.Tensor],
        shifted_labels: torch.Tensor,
        sampled_token_ids: torch.Tensor,
        student_temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, float | torch.Tensor]]:
        student_logits_scaled = student_logits / float(student_temperature)
        student_log_probs_full = F.log_softmax(student_logits_scaled, dim=-1)
        teacher_log_probs_list = self._build_teacher_log_probs(teacher_logits_list)
        teacher_weights_token, wi_metrics = self._compute_dynamic_teacher_weights(teacher_log_probs_list, student_log_probs_full, sampled_token_ids, shifted_labels)
        aggregated_teacher_log_probs = self._aggregate_teacher_log_probs(teacher_log_probs_list, teacher_weights_token)

        if self.top_k_loss is not None and self.top_k_loss > 0:
            k = min(self.top_k_loss, aggregated_teacher_log_probs.shape[-1])
            _, top_k_indices = torch.topk(aggregated_teacher_log_probs, k=k, dim=-1)
            student_log_probs = F.log_softmax(torch.gather(student_logits_scaled, dim=-1, index=top_k_indices), dim=-1)
            target_log_probs = F.log_softmax(torch.gather(aggregated_teacher_log_probs, dim=-1, index=top_k_indices), dim=-1)
        else:
            student_log_probs = student_log_probs_full
            target_log_probs = F.log_softmax(aggregated_teacher_log_probs, dim=-1)

        jsd_base_loss = self._generalized_jsd_loss_from_log_probs(student_log_probs, target_log_probs, labels=shifted_labels, beta=self.distill_beta)
        entropy_term = None
        loss = jsd_base_loss
        if self.use_entropy_rule and self.entropy_alpha > 0:
            token_entropy = -(student_log_probs_full.exp() * student_log_probs_full).sum(dim=-1)
            valid_mask = shifted_labels != IGNORE_INDEX
            if valid_mask.any():
                entropy_term = token_entropy[valid_mask].mean()
                loss = loss - self.entropy_alpha * entropy_term
        return loss, jsd_base_loss, entropy_term, wi_metrics

    def _maybe_log_teacher_pairwise_diff(self, teacher_logits_list: list[torch.Tensor]) -> None:
        if not self.debug_teacher_diff or not getattr(self.accelerator, "is_main_process", False):
            return
        if len(teacher_logits_list) < 2 or int(self.state.global_step) >= self.debug_teacher_diff_steps:
            return
        parts: list[str] = []
        for i in range(len(teacher_logits_list)):
            for j in range(i + 1, len(teacher_logits_list)):
                diff = (teacher_logits_list[i].detach().float() - teacher_logits_list[j].detach().float()).abs().mean()
                parts.append(f"{self.teacher_specs[i].name}-{self.teacher_specs[j].name}: {diff.item():.6e}")
        print(f"[TeacherDiff] step={int(self.state.global_step)} " + " | ".join(parts))


class ProLLaMAProteinOPDTrainer(BaseProLLaMAOpdTrainer):
    """Multi-teacher ProLLaMA OPD trainer with geometric teacher aggregation."""

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor | int],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del num_items_in_batch
        prompt_ids = inputs["prompt_ids"]
        prompt_attention_mask = inputs["prompt_attention_mask"]
        student_temperature = self._get_effective_student_temperature()

        generated_ids_raw = self._generate_on_policy(model, prompt_ids, prompt_attention_mask, student_temperature)
        generated_ids, generated_attention_mask, labels, rollout_tensors = self._canonicalize_rollouts(generated_ids_raw, prompt_ids)
        prompt_length = int(prompt_ids.shape[1])
        self._record_generation_outputs(rollout_tensors, prompt_length)

        outputs_student = model(input_ids=generated_ids, attention_mask=generated_attention_mask)
        student_logits = outputs_student.logits[:, prompt_length - 1 : -1, :]
        shifted_labels = labels[:, prompt_length:]
        sampled_token_ids = generated_ids[:, prompt_length:]
        teacher_logits_list = self._forward_teacher_logits(generated_ids, generated_attention_mask, prompt_length)
        self._maybe_log_teacher_pairwise_diff(teacher_logits_list)

        if self.useloss == "pg":
            loss, wi_metrics = self._compute_pg_loss(student_logits, teacher_logits_list, sampled_token_ids, shifted_labels, student_temperature)
            self._buffer_train_metrics(
                {
                    "train/pg_loss": loss.detach(),
                    "train/regularized_loss": loss.detach(),
                    "train/student_temperature": student_temperature,
                    **wi_metrics,
                }
            )
        else:
            loss, jsd_base_loss, entropy_term, wi_metrics = self._compute_jsd_loss(student_logits, teacher_logits_list, shifted_labels, sampled_token_ids, student_temperature)
            metric_payload: dict[str, float | torch.Tensor] = {
                "train/jsd_loss": jsd_base_loss.detach(),
                "train/regularized_loss": loss.detach(),
                "train/student_temperature": student_temperature,
                **wi_metrics,
            }
            if entropy_term is not None:
                metric_payload["train/entropy"] = entropy_term.detach()
                metric_payload["train/entropy_penalty"] = (self.entropy_alpha * entropy_term).detach()
            self._buffer_train_metrics(metric_payload)

        if self.save_generations and self.generation_save_steps > 0:
            current_step = int(self.state.global_step)
            if current_step > 0 and current_step % self.generation_save_steps == 0:
                self._save_generation_outputs(current_step)

        if return_outputs:
            return loss, {"loss": loss.detach()}
        return loss
