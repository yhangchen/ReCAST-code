"""Distributed SD3 LoRA training for ReCAST.

The implementation is deliberately explicit: every rank owns complete prompt
groups, rollout collection is separated from optimization, and DDP is used
only for the trainable forward pass. There is no retry or job-resubmission
logic in this module.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from safetensors.torch import load_file, save_file
from torch.nn.parallel import DistributedDataParallel

from .rewards import Reward, load_reward, score_reward
from .weighting import WeightTable


@dataclass(frozen=True)
class TrainingConfig:
    # Run and distributed topology.
    output_dir: str = "outputs/recast-h200"
    run_name: str = "recast-sd35m-stage1"
    seed: int = 42
    expected_world_size: int = 8

    # Model.
    model: str = "stabilityai/stable-diffusion-3.5-medium"
    revision: str = ""
    precision: str = "bf16"
    allow_tf32: bool = True
    gradient_checkpointing: bool = True
    lora_rank: int = 32
    lora_alpha: int = 64

    # Rollout collection, per rank.
    epochs: int = 121
    sampling_batches_per_epoch: int = 6
    prompts_per_rank: int = 1
    images_per_prompt: int = 24
    inference_steps: int = 25
    height: int = 512
    width: int = 512
    guidance_scale: float = 1.0
    max_sequence_length: int = 128

    # Optimization, per rank.
    train_batch_size: int = 24
    gradient_accumulation_batches: int = 6
    inner_epochs: int = 1
    train_timesteps_per_sample: int = 24
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    nft_beta: float = 0.1
    advantage_clip: float = 5.0
    kl_beta: float = 1e-4
    old_policy_decay: float = 0.5
    old_policy_warmup_updates: int = 500

    # Checkpointing and observability.
    checkpoint_every: int = 30
    initialize_from: str = ""
    resume_from: str = ""
    tracker: str = "jsonl"
    wandb_project: str = "recast"
    wandb_entity: str = ""

    def validate(self) -> None:
        positive_integers = {
            "expected_world_size": self.expected_world_size,
            "epochs": self.epochs,
            "sampling_batches_per_epoch": self.sampling_batches_per_epoch,
            "prompts_per_rank": self.prompts_per_rank,
            "images_per_prompt": self.images_per_prompt,
            "inference_steps": self.inference_steps,
            "height": self.height,
            "width": self.width,
            "max_sequence_length": self.max_sequence_length,
            "train_batch_size": self.train_batch_size,
            "gradient_accumulation_batches": self.gradient_accumulation_batches,
            "inner_epochs": self.inner_epochs,
            "train_timesteps_per_sample": self.train_timesteps_per_sample,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "checkpoint_every": self.checkpoint_every,
        }
        for name, value in positive_integers.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.images_per_prompt < 2:
            raise ValueError("images_per_prompt must be at least two")
        if self.train_timesteps_per_sample > self.inference_steps:
            raise ValueError("train_timesteps_per_sample cannot exceed inference_steps")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16, or bf16")
        if self.tracker not in {"none", "jsonl", "wandb"}:
            raise ValueError("tracker must be none, jsonl, or wandb")
        if self.nft_beta <= 0 or self.advantage_clip <= 0:
            raise ValueError("nft_beta and advantage_clip must be positive")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0:
            raise ValueError("learning_rate and max_grad_norm must be positive")
        if self.weight_decay < 0 or self.kl_beta < 0:
            raise ValueError("weight_decay and kl_beta cannot be negative")
        if not 0.0 <= self.adam_beta1 < 1.0 or not 0.0 <= self.adam_beta2 < 1.0:
            raise ValueError("Adam beta values must be in [0, 1)")
        if self.adam_epsilon <= 0:
            raise ValueError("adam_epsilon must be positive")
        if self.guidance_scale <= 0:
            raise ValueError("guidance_scale must be positive")
        if not 0.0 <= self.old_policy_decay < 1.0:
            raise ValueError("old_policy_decay must be in [0, 1)")
        if self.old_policy_warmup_updates < 0:
            raise ValueError("old_policy_warmup_updates cannot be negative")
        if self.height % 16 or self.width % 16:
            raise ValueError("height and width must be divisible by 16")
        if self.initialize_from and self.resume_from:
            raise ValueError("initialize_from and resume_from are mutually exclusive")


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    initialized: bool

    @property
    def is_main(self) -> bool:
        return self.rank == 0


@dataclass(frozen=True)
class PromptRecord:
    prompt: str
    metadata: dict[str, object]


@dataclass
class RolloutCollection:
    latents: torch.Tensor
    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor
    advantages: torch.Tensor
    raw_rewards: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.latents.shape[0])


class MetricTracker:
    def __init__(self, config: TrainingConfig, enabled: bool):
        self.kind = config.tracker if enabled else "none"
        self.output_dir = Path(config.output_dir)
        self.wandb_run = None
        if self.kind == "jsonl":
            self.path = self.output_dir / "metrics.jsonl"
        elif self.kind == "wandb":
            import wandb

            self.wandb_run = wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity or None,
                name=config.run_name,
                config=asdict(config),
                dir=str(self.output_dir),
            )

    def log(self, values: Mapping[str, float | int]) -> None:
        record = dict(values)
        print(json.dumps(record, sort_keys=True), flush=True)
        if self.kind == "jsonl":
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        elif self.kind == "wandb":
            self.wandb_run.log(record, step=int(record.get("updates", 0)))

    def finish(self) -> None:
        if self.wandb_run is not None:
            self.wandb_run.finish()


def setup_distributed(expected_world_size: int) -> DistributedContext:
    if not torch.cuda.is_available():
        raise RuntimeError("training requires CUDA")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != expected_world_size:
        raise RuntimeError(
            f"recipe expects {expected_world_size} processes, but torchrun provided "
            f"WORLD_SIZE={world_size}"
        )
    torch.cuda.set_device(local_rank)
    initialized = world_size > 1
    if initialized:
        dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=torch.device("cuda", local_rank),
        initialized=initialized,
    )


def cleanup_distributed(context: DistributedContext) -> None:
    if context.initialized and dist.is_initialized():
        dist.destroy_process_group()


def barrier(context: DistributedContext) -> None:
    if context.initialized:
        dist.barrier()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_prompt_records(path: str | Path) -> list[PromptRecord]:
    """Read plain-text prompts or JSONL records containing prompt metadata."""

    source = Path(path)
    lines = source.read_text(encoding="utf-8").splitlines()
    records: list[PromptRecord] = []
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if source.suffix.lower() == ".jsonl":
            value = json.loads(stripped)
            if not isinstance(value, dict) or not isinstance(value.get("prompt"), str):
                raise ValueError(
                    f"{source}:{line_number} must be an object with a string prompt"
                )
            prompt = value["prompt"].strip()
            metadata = value
        else:
            prompt = stripped
            metadata = {"prompt": prompt}
        if prompt:
            records.append(PromptRecord(prompt=prompt, metadata=metadata))
    if not records:
        raise ValueError(f"prompt file {path!s} contains no non-empty prompts")
    return records


def read_prompts(path: str | Path) -> list[str]:
    """Read only prompt strings; retained as a small public convenience API."""

    return [record.prompt for record in read_prompt_records(path)]


def select_rank_prompt_indices(
    dataset_size: int,
    prompts_per_rank: int,
    rank: int,
    world_size: int,
    seed: int,
) -> list[int]:
    """Select disjoint prompt groups for every rank from a shared seed."""

    required = prompts_per_rank * world_size
    if dataset_size < required:
        raise ValueError(
            f"prompt dataset has {dataset_size} rows but a distributed batch needs "
            f"at least {required}"
        )
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(dataset_size, generator=generator)[:required]
    start = rank * prompts_per_rank
    return permutation[start : start + prompts_per_rank].tolist()


def group_advantages(
    scores: torch.Tensor, group_size: int, epsilon: float = 1e-6
) -> torch.Tensor:
    """Standardize rewards independently inside each prompt group."""

    flat_scores = torch.as_tensor(scores, dtype=torch.float32)
    if flat_scores.ndim != 1 or flat_scores.numel() % group_size:
        raise ValueError("scores must be a flat tensor divisible by group_size")
    grouped = flat_scores.view(-1, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True, unbiased=False).clamp_min(epsilon)
    return ((grouped - mean) / std).reshape(-1)


def preaggregate_advantages(
    raw_rewards: torch.Tensor,
    effective_scheduler_weights: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Apply ReCAST weights before group normalization.

    ``raw_rewards`` has shape ``(rewards, samples)`` and the supplied ``T*W``
    coefficients have shape ``(rewards, timesteps)`` in scheduler order. The
    result has shape ``(samples, timesteps)``.
    """

    if raw_rewards.ndim != 2 or effective_scheduler_weights.ndim != 2:
        raise ValueError("raw_rewards and weights must both be matrices")
    if raw_rewards.shape[0] != effective_scheduler_weights.shape[0]:
        raise ValueError("reward dimensions do not match")
    mixed_rewards = effective_scheduler_weights.transpose(0, 1) @ raw_rewards
    normalized = [group_advantages(row, group_size) for row in mixed_rewards]
    return torch.stack(normalized, dim=1)


def _adapter_parameters(
    model: torch.nn.Module, adapter: str
) -> dict[str, torch.nn.Parameter]:
    marker = f".{adapter}."
    parameters = {
        name.replace(marker, ".<adapter>."): parameter
        for name, parameter in model.named_parameters()
        if marker in name
    }
    if not parameters:
        raise RuntimeError(f"PEFT model has no parameters for adapter {adapter!r}")
    return parameters


@torch.no_grad()
def update_old_adapter(model: torch.nn.Module, decay: float) -> None:
    current = _adapter_parameters(model, "default")
    old = _adapter_parameters(model, "old")
    if current.keys() != old.keys():
        raise RuntimeError(
            "current and old LoRA adapters have different parameter sets"
        )
    for key in current:
        old[key].mul_(decay).add_(current[key], alpha=1.0 - decay)


def old_policy_decay(config: TrainingConfig, updates: int) -> float:
    if config.old_policy_warmup_updates == 0:
        return config.old_policy_decay
    fraction = min(updates / config.old_policy_warmup_updates, 1.0)
    return config.old_policy_decay * fraction


def _autocast(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _decode_latents(pipeline, latents: torch.Tensor) -> list[Image.Image]:
    scaling = float(pipeline.vae.config.scaling_factor)
    shift = float(getattr(pipeline.vae.config, "shift_factor", 0.0) or 0.0)
    vae_input = latents.to(dtype=pipeline.vae.dtype) / scaling + shift
    with torch.inference_mode():
        decoded = pipeline.vae.decode(vae_input, return_dict=False)[0]
    return pipeline.image_processor.postprocess(decoded, output_type="pil")


def _sample_batch(
    pipeline,
    model,
    prompts: Sequence[str],
    config: TrainingConfig,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[Image.Image], list[str]]:
    model.set_adapter("old")
    model.eval()
    do_cfg = config.guidance_scale > 1.0
    with (
        torch.inference_mode(),
        _autocast(pipeline._execution_device, config.precision),
    ):
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipeline.encode_prompt(
            prompt=list(prompts),
            prompt_2=None,
            prompt_3=None,
            device=pipeline._execution_device,
            num_images_per_prompt=config.images_per_prompt,
            do_classifier_free_guidance=do_cfg,
            max_sequence_length=config.max_sequence_length,
        )
        output = pipeline(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_images_per_prompt=1,
            num_inference_steps=config.inference_steps,
            guidance_scale=config.guidance_scale,
            height=config.height,
            width=config.width,
            generator=generator,
            output_type="latent",
        )
        latents = output.images.detach()
        images = _decode_latents(pipeline, latents)
    repeated_prompts = [
        prompt for prompt in prompts for _ in range(config.images_per_prompt)
    ]
    model.set_adapter("default")
    return (
        latents,
        prompt_embeds.detach(),
        pooled_prompt_embeds.detach(),
        images,
        repeated_prompts,
    )


def recast_nft_loss(
    prediction: torch.Tensor,
    old_prediction: torch.Tensor,
    reference_prediction: torch.Tensor,
    x_t: torch.Tensor,
    x_0: torch.Tensor,
    sigma: torch.Tensor,
    advantages: torch.Tensor,
    step_indices: torch.Tensor,
    effective_scheduler_weights: torch.Tensor,
    *,
    aggregation: str = "loss",
    nft_beta: float,
    advantage_clip: float,
    kl_beta: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute ReCAST using ``T * W`` already aligned noise-to-clean."""

    spatial_dims = tuple(range(1, x_0.ndim))
    sigma_view = sigma.view(-1, *([1] * (x_0.ndim - 1)))
    positive_velocity = (
        nft_beta * prediction + (1.0 - nft_beta) * old_prediction.detach()
    )
    negative_velocity = (
        1.0 + nft_beta
    ) * old_prediction.detach() - nft_beta * prediction
    positive_x0 = x_t - sigma_view * positive_velocity
    negative_x0 = x_t - sigma_view * negative_velocity
    positive_scale = (
        (positive_x0.detach().float() - x_0.float())
        .abs()
        .mean(dim=spatial_dims)
        .clamp_min(1e-5)
    )
    negative_scale = (
        (negative_x0.detach().float() - x_0.float())
        .abs()
        .mean(dim=spatial_dims)
        .clamp_min(1e-5)
    )
    positive_loss = (positive_x0 - x_0).square().mean(dim=spatial_dims) / positive_scale
    negative_loss = (negative_x0 - x_0).square().mean(dim=spatial_dims) / negative_scale

    if aggregation == "pre":
        if advantages.shape != (x_0.shape[0],):
            raise ValueError("pre-aggregation advantages must have shape (batch,)")
        clipped = advantages.clamp(-advantage_clip, advantage_clip)
        mixing = (clipped / advantage_clip / 2.0 + 0.5).clamp(0.0, 1.0)
        policy_per_sample = (
            mixing * positive_loss + (1.0 - mixing) * negative_loss
        ) / nft_beta
    elif aggregation == "loss":
        if advantages.ndim != 2:
            raise ValueError("loss-stage advantages must have shape (rewards, batch)")
        if (
            advantages.shape[0] != effective_scheduler_weights.shape[0]
            or advantages.shape[1] != x_0.shape[0]
        ):
            raise ValueError("advantages do not match the rewards or batch")
        clipped = advantages.clamp(-advantage_clip, advantage_clip)
        mixing = (clipped / advantage_clip / 2.0 + 0.5).clamp(0.0, 1.0)
        per_reward_loss = (
            mixing * positive_loss.unsqueeze(0)
            + (1.0 - mixing) * negative_loss.unsqueeze(0)
        ) / nft_beta
        policy_per_sample = (
            effective_scheduler_weights[:, step_indices] * per_reward_loss
        ).sum(dim=0)
    else:
        raise ValueError(f"unknown aggregation mode {aggregation!r}")

    policy_loss = policy_per_sample.mean() * advantage_clip
    kl_loss = (prediction - reference_prediction).square().mean()
    total_loss = policy_loss + kl_beta * kl_loss
    return total_loss, {
        "loss": total_loss.detach(),
        "policy_loss": policy_loss.detach(),
        "kl_loss": kl_loss.detach(),
    }


def _collect_rollouts(
    pipeline,
    model,
    rewards: Mapping[str, Reward],
    prompts: Sequence[PromptRecord],
    table: WeightTable,
    config: TrainingConfig,
    context: DistributedContext,
    epoch: int,
    aggregation: str,
) -> RolloutCollection:
    latent_parts: list[torch.Tensor] = []
    prompt_parts: list[torch.Tensor] = []
    pooled_parts: list[torch.Tensor] = []
    advantage_parts: list[torch.Tensor] = []
    reward_parts: list[torch.Tensor] = []
    # WeightTable stores Figure 1's W (clean -> noise, column sum 1/T).  The
    # objective uses T*W and rollout advantages use scheduler order.
    effective_weight_cpu = torch.from_numpy(table.effective_scheduler_weights)

    for batch_index in range(config.sampling_batches_per_epoch):
        selection_seed = config.seed + epoch * 100_003 + batch_index
        indices = select_rank_prompt_indices(
            len(prompts),
            config.prompts_per_rank,
            context.rank,
            context.world_size,
            selection_seed,
        )
        records = [prompts[index] for index in indices]
        prompt_batch = [record.prompt for record in records]
        sample_seed = (
            config.seed
            + epoch * 1_000_003
            + batch_index * context.world_size
            + context.rank
        )
        generator = torch.Generator(device=context.device).manual_seed(sample_seed)
        latents, prompt_embeds, pooled_embeds, images, repeated_prompts = _sample_batch(
            pipeline, model, prompt_batch, config, generator
        )
        repeated_metadata = [
            record.metadata
            for record in records
            for _ in range(config.images_per_prompt)
        ]
        raw_rewards = torch.stack(
            [
                score_reward(
                    rewards[name], images, repeated_prompts, repeated_metadata
                ).cpu()
                for name in table.reward_names
            ]
        )
        if aggregation == "pre":
            advantages = preaggregate_advantages(
                raw_rewards, effective_weight_cpu, config.images_per_prompt
            )
        else:
            advantages = torch.stack(
                [group_advantages(row, config.images_per_prompt) for row in raw_rewards]
            )

        latent_parts.append(latents.cpu())
        prompt_parts.append(prompt_embeds.cpu())
        pooled_parts.append(pooled_embeds.cpu())
        advantage_parts.append(advantages)
        reward_parts.append(raw_rewards)

    advantage_dimension = 0 if aggregation == "pre" else 1
    return RolloutCollection(
        latents=torch.cat(latent_parts),
        prompt_embeds=torch.cat(prompt_parts),
        pooled_prompt_embeds=torch.cat(pooled_parts),
        advantages=torch.cat(advantage_parts, dim=advantage_dimension),
        raw_rewards=torch.cat(reward_parts, dim=1),
    )


def _reduce_metrics(
    metrics: Mapping[str, float], context: DistributedContext
) -> dict[str, float]:
    keys = sorted(metrics)
    values = torch.tensor(
        [metrics[key] for key in keys], device=context.device, dtype=torch.float64
    )
    if context.initialized:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= context.world_size
    return {key: float(values[index]) for index, key in enumerate(keys)}


def _train_collection(
    distributed_model,
    model,
    optimizer,
    scaler,
    trainable: Sequence[torch.nn.Parameter],
    collection: RolloutCollection,
    table: WeightTable,
    config: TrainingConfig,
    context: DistributedContext,
    epoch: int,
    aggregation: str,
) -> tuple[int, dict[str, float]]:
    model.train()
    # Training step zero is pure noise, the reverse of Figure 1, and Eq. (1)
    # applies T*W rather than the unscaled paper matrix W.
    effective_scheduler_weight_tensor = torch.from_numpy(
        table.effective_scheduler_weights
    ).to(device=context.device, dtype=torch.float32)
    sigma_tensor = torch.from_numpy(table.sigmas[:-1]).to(
        device=context.device, dtype=torch.float32
    )
    train_steps = config.train_timesteps_per_sample
    if train_steps > table.num_steps:
        raise ValueError("train_timesteps_per_sample exceeds the weight table")
    minibatches = math.ceil(collection.size / config.train_batch_size)
    microsteps_per_update = config.gradient_accumulation_batches * train_steps
    total_microsteps = config.inner_epochs * minibatches * train_steps
    optimizer.zero_grad(set_to_none=True)
    metric_sums = {"loss": 0.0, "policy_loss": 0.0, "kl_loss": 0.0}
    microstep = 0
    updates = 0

    train_generator = torch.Generator().manual_seed(
        config.seed + epoch * 10_000_019 + context.rank
    )
    for _ in range(config.inner_epochs):
        order = torch.randperm(collection.size, generator=train_generator)
        for start in range(0, collection.size, config.train_batch_size):
            indices = order[start : start + config.train_batch_size]
            x_0 = collection.latents[indices].to(context.device, non_blocking=True)
            prompt_embeds = collection.prompt_embeds[indices].to(
                context.device, non_blocking=True
            )
            pooled_embeds = collection.pooled_prompt_embeds[indices].to(
                context.device, non_blocking=True
            )
            timestep_order = torch.stack(
                [
                    torch.randperm(table.num_steps, generator=train_generator)[
                        :train_steps
                    ]
                    for _ in range(indices.numel())
                ]
            )
            for timestep_offset in range(train_steps):
                step_indices_cpu = timestep_order[:, timestep_offset]
                step_indices = step_indices_cpu.to(context.device)
                sigma = sigma_tensor[step_indices]
                sigma_view = sigma.view(-1, *([1] * (x_0.ndim - 1)))
                noise_generator = torch.Generator(device=context.device).manual_seed(
                    config.seed
                    + epoch * 100_000_007
                    + context.rank * 1_000_003
                    + microstep
                )
                noise = torch.randn(
                    x_0.shape,
                    generator=noise_generator,
                    device=context.device,
                    dtype=x_0.dtype,
                )
                x_t = (1.0 - sigma_view) * x_0 + sigma_view * noise
                timesteps = sigma * 1_000.0
                if aggregation == "pre":
                    advantages = collection.advantages[indices, step_indices_cpu].to(
                        context.device
                    )
                else:
                    advantages = collection.advantages[:, indices].to(context.device)

                window_start = (
                    microstep // microsteps_per_update
                ) * microsteps_per_update
                window_size = min(
                    microsteps_per_update,
                    total_microsteps - window_start,
                )
                boundary = (
                    microstep + 1
                ) % microsteps_per_update == 0 or microstep + 1 == total_microsteps
                sync_context = (
                    nullcontext()
                    if boundary or not context.initialized
                    else distributed_model.no_sync()
                )
                with sync_context:
                    with _autocast(context.device, config.precision):
                        model.set_adapter("old")
                        with torch.no_grad():
                            old_prediction = model(
                                hidden_states=x_t,
                                timestep=timesteps,
                                encoder_hidden_states=prompt_embeds,
                                pooled_projections=pooled_embeds,
                                return_dict=False,
                            )[0]
                        model.set_adapter("default")
                        prediction = distributed_model(
                            hidden_states=x_t,
                            timestep=timesteps,
                            encoder_hidden_states=prompt_embeds,
                            pooled_projections=pooled_embeds,
                            return_dict=False,
                        )[0]
                        with torch.no_grad(), model.disable_adapter():
                            reference_prediction = model(
                                hidden_states=x_t,
                                timestep=timesteps,
                                encoder_hidden_states=prompt_embeds,
                                pooled_projections=pooled_embeds,
                                return_dict=False,
                            )[0]
                        loss, batch_metrics = recast_nft_loss(
                            prediction,
                            old_prediction,
                            reference_prediction,
                            x_t,
                            x_0,
                            sigma,
                            advantages,
                            step_indices,
                            effective_scheduler_weight_tensor,
                            aggregation=aggregation,
                            nft_beta=config.nft_beta,
                            advantage_clip=config.advantage_clip,
                            kl_beta=config.kl_beta,
                        )
                        scaled_loss = loss / window_size
                    scaler.scale(scaled_loss).backward()

                for key, value in batch_metrics.items():
                    metric_sums[key] += float(value)
                microstep += 1
                if boundary:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    updates += 1

    local_metrics = {
        key: value / total_microsteps for key, value in metric_sums.items()
    }
    return updates, _reduce_metrics(local_metrics, context)


def _find_adapter_weights(directory: Path) -> Path:
    candidates = list(directory.rglob("adapter_model.safetensors"))
    if len(candidates) != 1:
        raise ValueError(
            f"expected one adapter_model.safetensors below {directory}, found "
            f"{len(candidates)}"
        )
    return candidates[0]


def _save_checkpoint(
    model,
    optimizer,
    scaler,
    output_dir: Path,
    epoch: int,
    updates: int,
    config: TrainingConfig,
    context: DistributedContext,
) -> Path:
    from peft import get_peft_model_state_dict

    destination = output_dir / f"checkpoint-{epoch:04d}"
    if context.is_main:
        destination.mkdir(parents=True, exist_ok=False)
        model.set_adapter("default")
        model.save_pretrained(
            destination / "lora",
            safe_serialization=True,
            selected_adapters=["default"],
        )
        old_state = get_peft_model_state_dict(model, adapter_name="old")
        save_file(
            {
                key: value.detach().cpu().contiguous()
                for key, value in old_state.items()
            },
            destination / "old_adapter.safetensors",
        )
        torch.save(optimizer.state_dict(), destination / "optimizer.pt")
        if config.precision == "fp16":
            torch.save(scaler.state_dict(), destination / "scaler.pt")
        state = {
            "completed_epoch": epoch,
            "updates": updates,
            "training_config": asdict(config),
        }
        (destination / "trainer_state.json").write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (destination / "COMPLETE").write_text("ok\n", encoding="utf-8")
    barrier(context)
    return destination


def _load_checkpoint(
    model,
    optimizer,
    scaler,
    checkpoint: str | Path,
    config: TrainingConfig,
) -> tuple[int, int]:
    from peft import set_peft_model_state_dict

    source = Path(checkpoint)
    if not (source / "COMPLETE").is_file():
        raise ValueError(f"checkpoint is incomplete: {source}")
    current_state = load_file(_find_adapter_weights(source / "lora"))
    set_peft_model_state_dict(model, current_state, adapter_name="default")
    old_state = load_file(source / "old_adapter.safetensors")
    set_peft_model_state_dict(model, old_state, adapter_name="old")
    optimizer.load_state_dict(
        torch.load(source / "optimizer.pt", map_location="cpu", weights_only=True)
    )
    scaler_path = source / "scaler.pt"
    if config.precision == "fp16" and scaler_path.is_file():
        scaler.load_state_dict(
            torch.load(scaler_path, map_location="cpu", weights_only=True)
        )
    state = json.loads((source / "trainer_state.json").read_text(encoding="utf-8"))
    return int(state["completed_epoch"]) + 1, int(state["updates"])


def _restore_training_state(model, optimizer, scaler, config: TrainingConfig):
    """Return the stage-local start epoch and cumulative optimizer updates."""

    if config.initialize_from:
        _, updates = _load_checkpoint(
            model, optimizer, scaler, config.initialize_from, config
        )
        return 1, updates
    if config.resume_from:
        return _load_checkpoint(model, optimizer, scaler, config.resume_from, config)
    return 1, 0


def _build_pipeline(config: TrainingConfig, context: DistributedContext):
    from diffusers import StableDiffusion3Pipeline
    from peft import LoraConfig, get_peft_model

    load_dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[config.precision]
    kwargs = {"torch_dtype": load_dtype}
    if config.revision:
        kwargs["revision"] = config.revision
    pipeline = StableDiffusion3Pipeline.from_pretrained(config.model, **kwargs)
    pipeline.to(context.device)
    pipeline.vae.to(dtype=torch.float32)
    pipeline.vae.requires_grad_(False)
    for encoder_name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
        getattr(pipeline, encoder_name).requires_grad_(False)
    pipeline.transformer.requires_grad_(False)
    if config.gradient_checkpointing:
        pipeline.transformer.enable_gradient_checkpointing()

    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=[
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_k",
            "attn.to_out.0",
            "attn.to_q",
            "attn.to_v",
        ],
    )
    model = get_peft_model(pipeline.transformer, lora_config)
    model.add_adapter("old", lora_config)
    update_old_adapter(model, decay=0.0)
    model.set_adapter("default")
    pipeline.transformer = model
    return pipeline, model


def _reward_metrics(
    collection: RolloutCollection,
    table: WeightTable,
    context: DistributedContext,
) -> dict[str, float]:
    local = {
        f"reward/{name}": float(collection.raw_rewards[index].mean())
        for index, name in enumerate(table.reward_names)
    }
    return _reduce_metrics(local, context)


def train(
    prompts: Sequence[str | PromptRecord],
    reward_budgets: Mapping[str, float],
    table: WeightTable,
    config: TrainingConfig,
    *,
    aggregation: str = "pre",
) -> Path:
    """Run the configured ReCAST training recipe under torchrun."""

    config.validate()
    table.validate()
    if aggregation not in {"pre", "loss"}:
        raise ValueError("aggregation must be pre or loss")
    if tuple(reward_budgets) != table.reward_names:
        raise ValueError("reward order must match the calculated weight table")
    if table.num_steps != config.inference_steps:
        raise ValueError("weight columns must match the configured inference steps")
    prompt_records = [
        prompt
        if isinstance(prompt, PromptRecord)
        else PromptRecord(prompt=prompt, metadata={"prompt": prompt})
        for prompt in prompts
    ]
    if len(prompt_records) < config.prompts_per_rank * config.expected_world_size:
        raise ValueError("not enough prompts for one distributed rollout batch")

    context = setup_distributed(config.expected_world_size)
    tracker = None
    try:
        seed_everything(config.seed + context.rank)
        if config.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        output_dir = Path(config.output_dir)
        if context.is_main:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "resolved_config.json").write_text(
                json.dumps(asdict(config), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        barrier(context)
        tracker = MetricTracker(config, context.is_main)

        pipeline, model = _build_pipeline(config, context)
        trainable = list(_adapter_parameters(model, "default").values())
        optimizer = torch.optim.AdamW(
            trainable,
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
            weight_decay=config.weight_decay,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=config.precision == "fp16")
        start_epoch, global_updates = _restore_training_state(
            model, optimizer, scaler, config
        )
        distributed_model = (
            DistributedDataParallel(
                model,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
            if context.initialized
            else model
        )
        rewards: dict[str, Reward] = {
            name: load_reward(name, context.device) for name in reward_budgets
        }

        for epoch in range(start_epoch, config.epochs + 1):
            epoch_start = time.monotonic()
            seed_everything(config.seed + epoch * 10_007 + context.rank)
            collection = _collect_rollouts(
                pipeline,
                model,
                rewards,
                prompt_records,
                table,
                config,
                context,
                epoch,
                aggregation,
            )
            update_delta, train_metrics = _train_collection(
                distributed_model,
                model,
                optimizer,
                scaler,
                trainable,
                collection,
                table,
                config,
                context,
                epoch,
                aggregation,
            )
            global_updates += update_delta
            decay = old_policy_decay(config, global_updates)
            update_old_adapter(model, decay)
            metrics = {
                "epoch": epoch,
                "updates": global_updates,
                "old_policy_decay": decay,
                "samples_per_rank": collection.size,
                "seconds": time.monotonic() - epoch_start,
                **train_metrics,
                **_reward_metrics(collection, table, context),
            }
            if context.is_main:
                tracker.log(metrics)
            if epoch % config.checkpoint_every == 0 or epoch == config.epochs:
                _save_checkpoint(
                    model,
                    optimizer,
                    scaler,
                    output_dir,
                    epoch,
                    global_updates,
                    config,
                    context,
                )
            del collection
            barrier(context)
        return output_dir
    finally:
        if tracker is not None:
            tracker.finish()
        cleanup_distributed(context)
