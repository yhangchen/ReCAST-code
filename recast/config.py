"""Strict TOML configuration for reproducible ReCAST runs."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from .training import TrainingConfig


@dataclass(frozen=True)
class WeightingConfig:
    alpha: float = 2.0
    aggregation: str = "pre"
    sinkhorn_iterations: int = 1_000
    sinkhorn_tolerance: float = 1e-9

    def validate(self) -> None:
        if self.alpha <= 1.0:
            raise ValueError("weighting.alpha must be greater than one")
        if self.aggregation not in {"pre", "loss"}:
            raise ValueError("weighting.aggregation must be 'pre' or 'loss'")
        if self.sinkhorn_iterations < 1 or self.sinkhorn_tolerance <= 0:
            raise ValueError("invalid Sinkhorn convergence settings")


@dataclass(frozen=True)
class Recipe:
    training: TrainingConfig
    weighting: WeightingConfig
    reward_budgets: dict[str, float]
    reward_temperatures: dict[str, float]

    def validate(self) -> None:
        self.training.validate()
        self.weighting.validate()
        if not self.reward_budgets:
            raise ValueError("the recipe must configure at least one reward")
        if any(value <= 0 for value in self.reward_budgets.values()):
            raise ValueError("reward budgets must be positive")
        if any(value <= 0 for value in self.reward_temperatures.values()):
            raise ValueError("reward temperatures must be positive")


_TRAINING_SECTIONS = {
    "run": {"output_dir", "run_name", "seed", "expected_world_size"},
    "model": {
        "model",
        "revision",
        "precision",
        "allow_tf32",
        "gradient_checkpointing",
        "lora_rank",
        "lora_alpha",
    },
    "sampling": {
        "sampling_batches_per_epoch",
        "prompts_per_rank",
        "images_per_prompt",
        "inference_steps",
        "height",
        "width",
        "guidance_scale",
        "max_sequence_length",
    },
    "training": {
        "epochs",
        "train_batch_size",
        "gradient_accumulation_batches",
        "inner_epochs",
        "train_timesteps_per_sample",
        "learning_rate",
        "weight_decay",
        "adam_beta1",
        "adam_beta2",
        "adam_epsilon",
        "max_grad_norm",
        "nft_beta",
        "advantage_clip",
        "kl_beta",
        "old_policy_decay",
        "old_policy_warmup_updates",
    },
    "checkpointing": {"checkpoint_every", "initialize_from", "resume_from"},
    "tracking": {"tracker", "wandb_project", "wandb_entity"},
}


def load_recipe(path: str | Path) -> Recipe:
    """Load a recipe and reject misspelled or unsupported fields."""

    recipe_path = Path(path)
    with recipe_path.open("rb") as handle:
        document = tomllib.load(handle)
    allowed_sections = set(_TRAINING_SECTIONS) | {"weighting", "rewards"}
    unknown_sections = document.keys() - allowed_sections
    if unknown_sections:
        raise ValueError(
            "unknown config sections: " + ", ".join(sorted(unknown_sections))
        )

    training_values: dict[str, object] = {}
    for section, allowed_fields in _TRAINING_SECTIONS.items():
        values = document.get(section, {})
        if not isinstance(values, dict):
            raise TypeError(f"[{section}] must be a table")
        misplaced_fields = values.keys() - allowed_fields
        if misplaced_fields:
            raise ValueError(
                f"unknown fields in [{section}]: " + ", ".join(sorted(misplaced_fields))
            )
        for key, value in values.items():
            if key in training_values:
                raise ValueError(f"duplicate training field {key!r}")
            training_values[key] = value
    training_fields = {field.name for field in fields(TrainingConfig)}
    unknown_training = training_values.keys() - training_fields
    if unknown_training:
        raise ValueError(
            "unknown training fields: " + ", ".join(sorted(unknown_training))
        )
    training = TrainingConfig(**training_values)

    weighting_values = document.get("weighting", {})
    if not isinstance(weighting_values, dict):
        raise TypeError("[weighting] must be a table")
    weighting_fields = {field.name for field in fields(WeightingConfig)}
    unknown_weighting = weighting_values.keys() - weighting_fields
    if unknown_weighting:
        raise ValueError(
            "unknown weighting fields: " + ", ".join(sorted(unknown_weighting))
        )
    weighting = WeightingConfig(**weighting_values)

    rewards = document.get("rewards", {})
    if not isinstance(rewards, dict):
        raise TypeError("[rewards] must be a table")
    budgets: dict[str, float] = {}
    temperatures: dict[str, float] = {}
    for name, values in rewards.items():
        if not isinstance(values, dict):
            raise TypeError(f"[rewards.{name}] must be a table")
        unknown_reward_fields = values.keys() - {"budget", "temperature"}
        if unknown_reward_fields:
            raise ValueError(
                f"unknown fields for reward {name!r}: "
                + ", ".join(sorted(unknown_reward_fields))
            )
        budgets[name] = float(values.get("budget", 1.0))
        temperatures[name] = float(values.get("temperature", 1.0))

    recipe = Recipe(training, weighting, budgets, temperatures)
    recipe.validate()
    return recipe
