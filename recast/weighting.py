"""Rényi discriminability gains and constrained ReCAST weights.

The input profile contains one non-negative density-ratio matrix per reward.
Each matrix has shape ``(observations, denoising_steps)`` and stores
``rho_i(x_t) = E[r_i | x_t] / E[r_i | prompt]``.  Input columns follow the
Diffusers scheduler destinations, from noise toward clean data: column ``j``
corresponds to ``sigmas[j + 1]``.  Exported weight-table columns follow the
paper/Figure 1 convention instead, from the clean-side transition to the
noise-side transition.  :attr:`WeightTable.effective_scheduler_weights`
performs the explicit reversal and ``T`` scaling required by training.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_EPS = 1e-30
WEIGHT_COLUMN_ORDER = "clean_to_noise"


@dataclass(frozen=True)
class WeightTable:
    """A validated ReCAST weight table in paper order (clean to noise)."""

    reward_names: tuple[str, ...]
    sigmas: np.ndarray
    weights: np.ndarray
    divergences: np.ndarray
    raw_gains: np.ndarray
    budgets: np.ndarray
    temperatures: np.ndarray
    alpha: float

    @property
    def num_steps(self) -> int:
        return int(self.weights.shape[1])

    @property
    def paper_source_sigmas(self) -> np.ndarray:
        """Source sigma for each paper-ordered weight column.

        The clean endpoint ``sigma=0`` is a state rather than a trainable
        transition.  Consequently column zero is the clean-side transition
        from ``sigmas[-2]`` into that endpoint, and the last column starts at
        pure noise ``sigmas[0]``.
        """

        return np.ascontiguousarray(self.sigmas[-2::-1])

    @property
    def paper_destination_sigmas(self) -> np.ndarray:
        """Destination sigma for each paper column, beginning at clean zero."""

        return np.ascontiguousarray(self.sigmas[:0:-1])

    @property
    def scheduler_to_paper_indices(self) -> np.ndarray:
        """Map scheduler step ``j`` to paper column ``T - 1 - j``."""

        return np.arange(self.num_steps - 1, -1, -1, dtype=np.int64)

    @property
    def scheduler_weights(self) -> np.ndarray:
        """Return paper ``W`` aligned with Diffusers' noise-to-clean order."""

        return np.ascontiguousarray(self.weights[:, self.scheduler_to_paper_indices])

    @property
    def effective_weights(self) -> np.ndarray:
        """Return paper-ordered ``T * W`` coefficients saved for training."""

        return np.ascontiguousarray(self.num_steps * self.weights)

    @property
    def effective_scheduler_weights(self) -> np.ndarray:
        """Return the ``T * W`` coefficients used by the training objective."""

        return np.ascontiguousarray(
            self.effective_weights[:, self.scheduler_to_paper_indices]
        )

    def validate(self, atol: float = 1e-5) -> None:
        rewards = len(self.reward_names)
        if rewards == 0:
            raise ValueError("at least one reward is required")
        if self.weights.shape != (rewards, self.num_steps):
            raise ValueError("weights must have shape (rewards, steps)")
        if self.divergences.shape != self.weights.shape:
            raise ValueError("divergences and weights must have the same shape")
        if self.raw_gains.shape != self.weights.shape:
            raise ValueError("raw_gains and weights must have the same shape")
        if self.sigmas.shape != (self.num_steps + 1,):
            raise ValueError("sigmas must contain one more entry than weights")
        if self.budgets.shape != (rewards,):
            raise ValueError("budgets must have one entry per reward")
        if self.temperatures.shape != (rewards,):
            raise ValueError("temperatures must have one entry per reward")
        for name, value in {
            "weights": self.weights,
            "divergences": self.divergences,
            "raw_gains": self.raw_gains,
            "sigmas": self.sigmas,
            "budgets": self.budgets,
            "temperatures": self.temperatures,
        }.items():
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains a non-finite value")
        if (self.weights < 0).any() or (self.raw_gains < 0).any():
            raise ValueError("weights and raw gains must be non-negative")
        if not np.all(np.diff(self.sigmas) <= 0):
            raise ValueError("sigmas must run from noise to clean data")
        if not np.isclose(self.sigmas[0], 1.0, atol=atol):
            raise ValueError("the noise endpoint sigmas[0] must be 1")
        if not np.isclose(self.sigmas[-1], 0.0, atol=atol):
            raise ValueError("the clean endpoint sigmas[-1] must be 0")
        expected_columns = np.full(self.num_steps, 1.0 / self.num_steps)
        if not np.allclose(self.weights.sum(axis=0), expected_columns, atol=atol):
            raise ValueError("each paper weight column must sum to 1 / steps")
        expected_rows = self.budgets / self.budgets.sum()
        if not np.allclose(self.weights.sum(axis=1), expected_rows, atol=atol):
            raise ValueError("weight rows do not match the requested reward budgets")


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(np.exp(values - maximum).sum(axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis)


def renyi_discriminability(
    density_ratios: np.ndarray,
    alpha: float = 2.0,
    epsilon: float = _EPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cumulative Rényi discriminability and clipped step gains.

    ``density_ratios`` must have shape ``(observations, steps)``.  Its columns
    are the destinations of scheduler transitions in noise-to-clean order.
    Pure-noise divergence is known to be zero and is prepended implicitly, so
    the returned gain at column ``j`` belongs to the scheduler transition that
    starts at ``sigmas[j]`` and ends at ``sigmas[j + 1]``.
    """

    ratios = np.asarray(density_ratios, dtype=np.float64)
    if alpha <= 1.0:
        raise ValueError("alpha must be greater than one")
    if ratios.ndim != 2 or 0 in ratios.shape:
        raise ValueError("density ratios must have shape (observations, steps)")
    if not np.isfinite(ratios).all():
        raise ValueError("density ratios contain a non-finite value")
    if (ratios < 0).any():
        raise ValueError("density ratios must be non-negative")

    moment = np.mean(np.power(ratios, alpha), axis=0)
    divergence = np.log(np.clip(moment, epsilon, None)) / (alpha - 1.0)
    gains = np.diff(np.concatenate(([0.0], divergence)))
    return divergence, np.maximum(gains, 0.0)


def sinkhorn_weights(
    raw_gains: np.ndarray,
    budgets: Sequence[float],
    temperatures: Sequence[float],
    *,
    max_iterations: int = 1_000,
    tolerance: float = 1e-9,
) -> np.ndarray:
    """Project gain preferences onto ReCAST row and column marginals.

    The returned paper matrix ``W`` sums to ``1 / num_steps`` in every column.
    Its row sums are the normalized reward budgets.  Training applies
    ``num_steps * W`` as specified by the ReCAST objective.
    """

    gains = np.asarray(raw_gains, dtype=np.float64)
    budget_array = np.asarray(budgets, dtype=np.float64)
    temperature_array = np.asarray(temperatures, dtype=np.float64)
    if gains.ndim != 2 or 0 in gains.shape:
        raise ValueError("raw_gains must have shape (rewards, steps)")
    rewards, steps = gains.shape
    if budget_array.shape != (rewards,) or temperature_array.shape != (rewards,):
        raise ValueError("budgets and temperatures need one entry per reward")
    if not np.isfinite(gains).all() or (gains < 0).any():
        raise ValueError("raw gains must be finite and non-negative")
    if not np.isfinite(budget_array).all() or (budget_array <= 0).any():
        raise ValueError("reward budgets must be finite and positive")
    if not np.isfinite(temperature_array).all() or (temperature_array <= 0).any():
        raise ValueError("temperatures must be finite and positive")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")

    normalized_budgets = budget_array / budget_array.sum()
    gain_means = gains.mean(axis=1, keepdims=True)
    normalized_gains = np.divide(
        gains,
        gain_means,
        out=np.ones_like(gains),
        where=gain_means > 0,
    )
    log_kernel = normalized_gains / temperature_array[:, None]
    log_row_target = np.log(normalized_budgets)
    log_column_target = np.full(steps, -np.log(steps), dtype=np.float64)
    log_u = np.zeros(rewards, dtype=np.float64)
    log_v = np.zeros(steps, dtype=np.float64)

    converged = False
    for _ in range(max_iterations):
        log_u = log_row_target - _logsumexp(log_kernel + log_v[None, :], axis=1)
        log_v = log_column_target - _logsumexp(log_kernel + log_u[:, None], axis=0)
        transport = np.exp(log_u[:, None] + log_kernel + log_v[None, :])
        row_error = np.max(np.abs(transport.sum(axis=1) - normalized_budgets))
        column_error = np.max(np.abs(transport.sum(axis=0) - (1.0 / steps)))
        if max(row_error, column_error) <= tolerance:
            converged = True
            break
    if not converged:
        raise RuntimeError(
            "Sinkhorn projection did not converge; increase --sinkhorn-iterations "
            "or use a larger --sinkhorn-tolerance"
        )

    return transport.astype(np.float32)


def load_density_profiles(
    path: str | Path,
    reward_names: Sequence[str],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load a safe NumPy archive containing sigmas and ``ratio__<reward>``."""

    archive_path = Path(path)
    with np.load(archive_path, allow_pickle=False) as archive:
        if "sigmas" not in archive:
            raise ValueError(f"{archive_path} does not contain 'sigmas'")
        sigmas = np.asarray(archive["sigmas"], dtype=np.float64)
        profiles: dict[str, np.ndarray] = {}
        for reward_name in reward_names:
            key = f"ratio__{reward_name}"
            if key not in archive:
                raise ValueError(f"{archive_path} does not contain {key!r}")
            profiles[reward_name] = np.asarray(archive[key], dtype=np.float64)
    return sigmas, profiles


def calculate_weight_table(
    profiles: Mapping[str, np.ndarray],
    sigmas: np.ndarray,
    budgets: Mapping[str, float],
    temperatures: Mapping[str, float] | None = None,
    *,
    alpha: float = 2.0,
    sinkhorn_iterations: int = 1_000,
    sinkhorn_tolerance: float = 1e-9,
) -> WeightTable:
    """Calculate a complete ReCAST table from empirical density ratios."""

    reward_names = tuple(budgets)
    if set(profiles) != set(reward_names):
        raise ValueError("profiles and budgets must name exactly the same rewards")
    temperatures = temperatures or {}

    divergence_rows: list[np.ndarray] = []
    gain_rows: list[np.ndarray] = []
    expected_steps: int | None = None
    for name in reward_names:
        divergence, gain = renyi_discriminability(profiles[name], alpha=alpha)
        expected_steps = expected_steps or int(gain.shape[0])
        if gain.shape != (expected_steps,):
            raise ValueError(
                "all density-ratio profiles must use the same number of steps"
            )
        divergence_rows.append(divergence)
        gain_rows.append(gain)

    sigmas_array = np.asarray(sigmas, dtype=np.float64)
    if sigmas_array.shape != (expected_steps + 1,):
        raise ValueError("sigmas must have one more entry than the profile steps")
    if not np.all(np.diff(sigmas_array) <= 0):
        raise ValueError("sigmas must be ordered from high noise to clean data")

    budget_array = np.asarray(
        [budgets[name] for name in reward_names], dtype=np.float64
    )
    temperature_array = np.asarray(
        [temperatures.get(name, 1.0) for name in reward_names], dtype=np.float64
    )
    # renyi_discriminability returns scheduler order (noise -> clean).  Figure
    # 1 and W_{i,t} use paper order (clean -> noise), so the persisted table is
    # reversed exactly once here.  Training reverses it back through
    # WeightTable.effective_scheduler_weights before indexing scheduler steps.
    divergences = np.ascontiguousarray(np.stack(divergence_rows)[:, ::-1])
    raw_gains = np.ascontiguousarray(np.stack(gain_rows)[:, ::-1])
    weights = sinkhorn_weights(
        raw_gains,
        budget_array,
        temperature_array,
        max_iterations=sinkhorn_iterations,
        tolerance=sinkhorn_tolerance,
    )
    table = WeightTable(
        reward_names=reward_names,
        sigmas=sigmas_array.astype(np.float32),
        weights=weights,
        divergences=divergences.astype(np.float32),
        raw_gains=raw_gains.astype(np.float32),
        budgets=budget_array.astype(np.float32),
        temperatures=temperature_array.astype(np.float32),
        alpha=float(alpha),
    )
    table.validate()
    return table


def save_weight_table(table: WeightTable, path: str | Path) -> None:
    """Save a table without pickle-backed objects."""

    table.validate()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps(
        {
            "alpha": table.alpha,
            "format_version": 2,
            "profile_column_order": "scheduler_destinations_noise_to_clean",
            "training_weight_scale": "num_steps",
            "weight_column_order": WEIGHT_COLUMN_ORDER,
            "weights_semantics": "num_steps_times_paper_weights",
        },
        sort_keys=True,
    )
    np.savez_compressed(
        destination,
        reward_names=np.asarray(table.reward_names, dtype=np.str_),
        sigmas=table.sigmas,
        paper_destination_sigmas=table.paper_destination_sigmas,
        paper_source_sigmas=table.paper_source_sigmas,
        weights=table.effective_weights,
        paper_weights=table.weights,
        divergences=table.divergences,
        raw_gains=table.raw_gains,
        budgets=table.budgets,
        temperatures=table.temperatures,
        metadata=np.asarray(metadata, dtype=np.str_),
    )


def load_weight_table(path: str | Path) -> WeightTable:
    """Load and validate a table produced by :func:`save_weight_table`."""

    with np.load(Path(path), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        if metadata.get("format_version") != 2:
            raise ValueError("unsupported weight-table format version")
        if metadata.get("weight_column_order") != WEIGHT_COLUMN_ORDER:
            raise ValueError(
                "weight table does not declare clean-to-noise paper column order"
            )
        if metadata.get("weights_semantics") != "num_steps_times_paper_weights":
            raise ValueError("weight table does not contain training-scale weights")
        paper_destination_sigmas = np.asarray(
            archive["paper_destination_sigmas"], dtype=np.float32
        )
        paper_source_sigmas = np.asarray(
            archive["paper_source_sigmas"], dtype=np.float32
        )
        effective_weights = np.asarray(archive["weights"], dtype=np.float32)
        table = WeightTable(
            reward_names=tuple(str(name) for name in archive["reward_names"].tolist()),
            sigmas=np.asarray(archive["sigmas"], dtype=np.float32),
            weights=np.asarray(archive["paper_weights"], dtype=np.float32),
            divergences=np.asarray(archive["divergences"], dtype=np.float32),
            raw_gains=np.asarray(archive["raw_gains"], dtype=np.float32),
            budgets=np.asarray(archive["budgets"], dtype=np.float32),
            temperatures=np.asarray(archive["temperatures"], dtype=np.float32),
            alpha=float(metadata["alpha"]),
        )
    table.validate()
    if not np.allclose(paper_destination_sigmas, table.paper_destination_sigmas):
        raise ValueError("paper_destination_sigmas do not match scheduler sigmas")
    if not np.allclose(paper_source_sigmas, table.paper_source_sigmas):
        raise ValueError("paper_source_sigmas do not match the scheduler sigmas")
    if not np.allclose(effective_weights, table.effective_weights):
        raise ValueError("saved weights do not equal num_steps * paper_weights")
    return table
