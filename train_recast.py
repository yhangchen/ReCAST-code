#!/usr/bin/env python3
"""Calculate ReCAST weights, then run one finite distributed training job."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path

from recast.config import Recipe, load_recipe
from recast.training import read_prompt_records
from recast.weighting import (
    WEIGHT_COLUMN_ORDER,
    calculate_weight_table,
    load_density_profiles,
    save_weight_table,
)


def _default_config_path() -> str:
    source_recipe = Path("configs/h200_8gpu.toml")
    if source_recipe.is_file():
        return str(source_recipe)
    return str(Path(sys.prefix) / "share/recast/configs/h200_8gpu.toml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate the constrained ReCAST reward-by-timestep matrix from "
            "density-ratio profiles and train an SD3.5 LoRA with torchrun."
        )
    )
    parser.add_argument(
        "--config",
        default=_default_config_path(),
        help="training recipe (defaults to the packaged 8xH200 recipe)",
    )
    parser.add_argument(
        "--profiles",
        required=True,
        help="NPZ file containing sigmas and ratio__REWARD arrays",
    )
    parser.add_argument(
        "--prompts",
        help="plain-text prompts or JSONL records with prompt metadata",
    )
    parser.add_argument("--output-dir", help="override run.output_dir")
    parser.add_argument(
        "--initialize-from",
        help="start a new stage from a completed parent-stage checkpoint",
    )
    parser.add_argument("--resume-from", help="override checkpointing.resume_from")
    parser.add_argument(
        "--tracker",
        choices=("none", "jsonl", "wandb"),
        help="override tracking.tracker",
    )
    parser.add_argument(
        "--weights-output",
        help="weight NPZ path (default: OUTPUT_DIR/recast_weights.npz)",
    )
    parser.add_argument(
        "--weights-only",
        action="store_true",
        help="calculate the weight table without loading training prompts or models",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and print the full training geometry without CUDA",
    )
    return parser


def _apply_overrides(recipe: Recipe, args: argparse.Namespace) -> Recipe:
    overrides: dict[str, object] = {}
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.initialize_from:
        overrides["initialize_from"] = args.initialize_from
    if args.resume_from:
        overrides["resume_from"] = args.resume_from
    if args.tracker:
        overrides["tracker"] = args.tracker
    if not overrides:
        return recipe
    updated = replace(recipe, training=replace(recipe.training, **overrides))
    updated.validate()
    return updated


def _geometry(recipe: Recipe, prompt_count: int | None) -> dict[str, object]:
    config = recipe.training
    samples_per_rank = (
        config.sampling_batches_per_epoch
        * config.prompts_per_rank
        * config.images_per_prompt
    )
    samples_per_epoch = samples_per_rank * config.expected_world_size
    prompt_groups_per_epoch = (
        config.sampling_batches_per_epoch
        * config.prompts_per_rank
        * config.expected_world_size
    )
    minibatches_per_rank = (
        samples_per_rank + config.train_batch_size - 1
    ) // config.train_batch_size
    microsteps_per_rank = (
        minibatches_per_rank * config.train_timesteps_per_sample * config.inner_epochs
    )
    accumulation_microsteps = (
        config.gradient_accumulation_batches * config.train_timesteps_per_sample
    )
    optimizer_updates_per_epoch = (
        microsteps_per_rank + accumulation_microsteps - 1
    ) // accumulation_microsteps
    return {
        "aggregation": recipe.weighting.aggregation,
        "expected_world_size": config.expected_world_size,
        "prompt_count": prompt_count,
        "prompt_groups_per_epoch": prompt_groups_per_epoch,
        "samples_per_rank": samples_per_rank,
        "samples_per_epoch": samples_per_epoch,
        "train_minibatches_per_rank": minibatches_per_rank,
        "train_timesteps_per_sample": config.train_timesteps_per_sample,
        "optimizer_updates_per_epoch": optimizer_updates_per_epoch,
        "effective_sample_batch": (
            config.expected_world_size
            * config.train_batch_size
            * config.gradient_accumulation_batches
        ),
    }


def _is_primary_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _validate_checkpoint_path(path: str, option: str) -> None:
    if path and not (Path(path) / "COMPLETE").is_file():
        raise ValueError(f"{option} is not a complete ReCAST checkpoint: {path}")


def main() -> None:
    args = build_parser().parse_args()
    recipe = _apply_overrides(load_recipe(args.config), args)
    config = recipe.training
    reward_names = tuple(recipe.reward_budgets)
    sigmas, profiles = load_density_profiles(args.profiles, reward_names)
    table = calculate_weight_table(
        profiles,
        sigmas,
        recipe.reward_budgets,
        recipe.reward_temperatures,
        alpha=recipe.weighting.alpha,
        sinkhorn_iterations=recipe.weighting.sinkhorn_iterations,
        sinkhorn_tolerance=recipe.weighting.sinkhorn_tolerance,
    )
    if table.num_steps != config.inference_steps:
        raise ValueError(
            f"profiles define {table.num_steps} steps but the recipe uses "
            f"{config.inference_steps} inference steps"
        )
    weights_output = Path(
        args.weights_output or Path(config.output_dir) / "recast_weights.npz"
    )
    if _is_primary_process():
        save_weight_table(table, weights_output)

    if args.weights_only:
        if _is_primary_process():
            print(
                json.dumps(
                    {
                        "reward_names": table.reward_names,
                        "paper_row_sums": table.weights.sum(axis=1).tolist(),
                        "row_sums": table.effective_weights.sum(axis=1).tolist(),
                        "training_weight_scale": table.num_steps,
                        "weight_column_order": WEIGHT_COLUMN_ORDER,
                        "weight_shape": list(table.weights.shape),
                        "weights": str(weights_output),
                    },
                    sort_keys=True,
                )
            )
        return
    if not args.prompts:
        raise ValueError("--prompts is required unless --weights-only is set")
    prompts = read_prompt_records(args.prompts)
    required_prompts = config.prompts_per_rank * config.expected_world_size
    if len(prompts) < required_prompts:
        raise ValueError(
            f"prompt file contains {len(prompts)} prompts; at least "
            f"{required_prompts} are required"
        )
    _validate_checkpoint_path(config.initialize_from, "--initialize-from")
    _validate_checkpoint_path(config.resume_from, "--resume-from")
    geometry = _geometry(recipe, len(prompts))
    if _is_primary_process():
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "resolved_recipe.json").write_text(
            json.dumps(
                {
                    "training": asdict(config),
                    "weighting": asdict(recipe.weighting),
                    "reward_budgets": recipe.reward_budgets,
                    "reward_temperatures": recipe.reward_temperatures,
                    "geometry": geometry,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "config": asdict(config),
                    "geometry": geometry,
                    "reward_budgets": recipe.reward_budgets,
                    "training_weight_scale": table.num_steps,
                    "weight_column_order": WEIGHT_COLUMN_ORDER,
                    "weight_shape": list(table.weights.shape),
                    "weights": str(weights_output),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if args.dry_run:
        return

    from recast.training import train

    output_dir = train(
        prompts,
        recipe.reward_budgets,
        table,
        config,
        aggregation=recipe.weighting.aggregation,
    )
    if _is_primary_process():
        print(json.dumps({"training_complete": str(output_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
