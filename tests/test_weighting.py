import tempfile
import unittest
from pathlib import Path

import numpy as np

from recast.weighting import (
    calculate_weight_table,
    load_weight_table,
    renyi_discriminability,
    save_weight_table,
)


class WeightingTest(unittest.TestCase):
    def test_renyi_discriminability_uses_clipped_forward_gains(self):
        ratios = np.array(
            [
                [1.0, np.exp(0.5), np.exp(1.0)],
                [1.0, np.exp(0.5), np.exp(1.0)],
            ]
        )
        divergence, gain = renyi_discriminability(ratios, alpha=2.0)
        np.testing.assert_allclose(divergence, [0.0, 1.0, 2.0])
        np.testing.assert_allclose(gain, [0.0, 1.0, 1.0])

    def test_weight_table_has_requested_marginals_and_round_trips(self):
        profiles = {
            "early": np.tile([1.0, np.exp(0.8), np.exp(0.9)], (8, 1)),
            "late": np.tile([1.0, np.exp(0.1), np.exp(1.0)], (8, 1)),
        }
        table = calculate_weight_table(
            profiles,
            sigmas=np.array([1.0, 0.7, 0.3, 0.0]),
            budgets={"early": 2.0, "late": 1.0},
            alpha=2.0,
        )
        np.testing.assert_allclose(table.weights.sum(axis=0), 1.0 / 3.0, atol=1e-6)
        np.testing.assert_allclose(
            table.weights.sum(axis=1), [2.0 / 3.0, 1.0 / 3.0], atol=1e-6
        )

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "weights.npz"
            save_weight_table(table, destination)
            with np.load(destination, allow_pickle=False) as archive:
                np.testing.assert_allclose(
                    archive["weights"], 3.0 * archive["paper_weights"]
                )
                np.testing.assert_allclose(
                    archive["weights"].sum(axis=0), 1.0, atol=1e-6
                )
            loaded = load_weight_table(destination)
        self.assertEqual(loaded.reward_names, table.reward_names)
        np.testing.assert_allclose(loaded.weights, table.weights)
        np.testing.assert_allclose(loaded.raw_gains, table.raw_gains)

    def test_25_step_table_matches_figure_and_scheduler_orders(self):
        steps = 25
        scheduler_gain_a = np.arange(1, steps + 1, dtype=np.float64) / 100.0
        scheduler_gain_b = scheduler_gain_a[::-1]
        divergence_a = np.cumsum(scheduler_gain_a)
        divergence_b = np.cumsum(scheduler_gain_b)
        profiles = {
            "a": np.tile(np.exp(divergence_a / 2.0), (8, 1)),
            "b": np.tile(np.exp(divergence_b / 2.0), (8, 1)),
        }
        sigmas = np.linspace(1.0, 0.0, steps + 1)

        table = calculate_weight_table(
            profiles,
            sigmas,
            budgets={"a": 1.0, "b": 1.0},
            alpha=2.0,
        )

        self.assertEqual(table.weights.shape, (2, 25))
        np.testing.assert_allclose(table.weights.sum(axis=0), 1.0 / 25.0)
        np.testing.assert_allclose(table.weights.sum(axis=1), [0.5, 0.5])
        # Persisted columns follow Figure 1: clean-side transition first.
        np.testing.assert_allclose(table.raw_gains[0], scheduler_gain_a[::-1])
        np.testing.assert_allclose(table.raw_gains[1], scheduler_gain_b[::-1])
        self.assertAlmostEqual(float(table.paper_destination_sigmas[0]), 0.0)
        self.assertAlmostEqual(float(table.paper_source_sigmas[0]), 0.04)
        self.assertAlmostEqual(float(table.paper_source_sigmas[-1]), 1.0)
        # Check all 25 transitions, not only the endpoints. Scheduler step j
        # starts at sigmas[j], ends at sigmas[j+1], and consumes paper column
        # T-1-j. This catches both reversal and one-column offset errors.
        scheduler_indices = np.arange(steps)
        paper_indices = table.scheduler_to_paper_indices
        np.testing.assert_array_equal(paper_indices, steps - 1 - scheduler_indices)
        np.testing.assert_allclose(
            table.paper_source_sigmas[paper_indices], sigmas[:-1]
        )
        np.testing.assert_allclose(
            table.paper_destination_sigmas[paper_indices], sigmas[1:]
        )
        np.testing.assert_allclose(table.raw_gains[0, paper_indices], scheduler_gain_a)
        np.testing.assert_allclose(
            table.effective_scheduler_weights,
            table.effective_weights[:, paper_indices],
        )
        np.testing.assert_allclose(table.effective_weights, 25.0 * table.weights)
        np.testing.assert_allclose(
            table.effective_scheduler_weights.sum(axis=0), np.ones(25)
        )

    def test_weight_table_requires_clean_zero_endpoint(self):
        ratios = np.ones((4, 2), dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "clean endpoint"):
            calculate_weight_table(
                {"reward": ratios},
                sigmas=np.array([1.0, 0.5, 0.1]),
                budgets={"reward": 1.0},
            )


if __name__ == "__main__":
    unittest.main()
