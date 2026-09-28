import tempfile
import textwrap
import unittest
from pathlib import Path

from recast.config import load_recipe
from recast.training import TrainingConfig
from train_recast import _geometry

VALID_RECIPE = """
[run]
expected_world_size = 8
output_dir = "outputs/test"

[sampling]
inference_steps = 25

[training]
epochs = 2
train_timesteps_per_sample = 24

[weighting]
aggregation = "pre"

[rewards.clipscore]
budget = 2.0

[rewards.pickscore]
budget = 1.0
temperature = 0.5
"""


class ConfigTest(unittest.TestCase):
    def _load(self, contents: str):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipe.toml"
            path.write_text(textwrap.dedent(contents), encoding="utf-8")
            return load_recipe(path)

    def test_recipe_loads_defaults_and_reward_order(self):
        recipe = self._load(VALID_RECIPE)
        self.assertEqual(recipe.training.expected_world_size, 8)
        self.assertEqual(recipe.training.epochs, 2)
        self.assertEqual(tuple(recipe.reward_budgets), ("clipscore", "pickscore"))
        self.assertEqual(recipe.reward_temperatures["clipscore"], 1.0)
        self.assertEqual(recipe.reward_temperatures["pickscore"], 0.5)

    def test_misspelled_or_misplaced_fields_are_rejected(self):
        invalid = VALID_RECIPE.replace(
            "expected_world_size = 8", "expected_world_size = 8\nepochs = 2"
        )
        with self.assertRaisesRegex(ValueError, r"unknown fields in \[run\]"):
            self._load(invalid)

    def test_checked_in_h200_recipe_has_documented_geometry(self):
        path = Path(__file__).parents[1] / "configs" / "h200_8gpu.toml"
        geometry = _geometry(load_recipe(path), prompt_count=2_048)
        self.assertEqual(geometry["prompt_groups_per_epoch"], 48)
        self.assertEqual(geometry["samples_per_rank"], 144)
        self.assertEqual(geometry["samples_per_epoch"], 1_152)
        self.assertEqual(geometry["train_minibatches_per_rank"], 6)
        self.assertEqual(geometry["optimizer_updates_per_epoch"], 1)
        self.assertEqual(geometry["effective_sample_batch"], 1_152)

    def test_stage2_recipe_adds_geneval_and_runs_for_61_epochs(self):
        path = Path(__file__).parents[1] / "configs" / "h200_8gpu_stage2_geneval.toml"
        recipe = load_recipe(path)
        self.assertEqual(recipe.training.epochs, 61)
        self.assertEqual(
            tuple(recipe.reward_budgets),
            ("clipscore", "hpsv2", "pickscore", "geneval"),
        )
        self.assertEqual(_geometry(recipe, 553)["samples_per_epoch"], 1_152)

    def test_initialize_and_resume_cannot_be_combined(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            TrainingConfig(
                initialize_from="stage1/checkpoint-0121",
                resume_from="stage2/checkpoint-0030",
            ).validate()


if __name__ == "__main__":
    unittest.main()
