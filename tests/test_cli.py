import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from train_recast import main


class CliTest(unittest.TestCase):
    def test_h200_dry_run_calculates_weights_and_reports_geometry(self):
        root = Path(__file__).parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            profile_path = temporary / "profiles.npz"
            prompts_path = temporary / "prompts.txt"
            output_dir = temporary / "output"
            ratios = np.tile(np.exp(np.linspace(0.0, 1.0, 25)), (8, 1))
            np.savez_compressed(
                profile_path,
                sigmas=np.linspace(1.0, 0.0, 26),
                ratio__clipscore=ratios,
                ratio__hpsv2=ratios,
                ratio__pickscore=ratios,
            )
            prompts_path.write_text(
                "\n".join(f"prompt {index}" for index in range(8)),
                encoding="utf-8",
            )
            arguments = [
                "train_recast.py",
                "--config",
                str(root / "configs" / "h200_8gpu.toml"),
                "--profiles",
                str(profile_path),
                "--prompts",
                str(prompts_path),
                "--output-dir",
                str(output_dir),
                "--dry-run",
            ]
            output = io.StringIO()
            with (
                patch.object(sys, "argv", arguments),
                patch.dict("os.environ", {"RANK": "0"}),
                redirect_stdout(output),
            ):
                main()

            summary = json.loads(output.getvalue())
            self.assertEqual(summary["geometry"]["samples_per_epoch"], 1_152)
            self.assertEqual(summary["geometry"]["optimizer_updates_per_epoch"], 1)
            self.assertEqual(summary["weight_shape"], [3, 25])
            self.assertEqual(summary["weight_column_order"], "clean_to_noise")
            self.assertEqual(summary["training_weight_scale"], 25)
            self.assertTrue((output_dir / "recast_weights.npz").is_file())
            self.assertTrue((output_dir / "resolved_recipe.json").is_file())

    def test_stage2_dry_run_accepts_metadata_and_parent_checkpoint(self):
        root = Path(__file__).parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            profile_path = temporary / "profiles.npz"
            prompts_path = temporary / "prompts.jsonl"
            parent = temporary / "stage1" / "checkpoint-0121"
            parent.mkdir(parents=True)
            (parent / "COMPLETE").write_text("ok\n", encoding="utf-8")
            ratios = np.tile(np.exp(np.linspace(0.0, 1.0, 25)), (8, 1))
            np.savez_compressed(
                profile_path,
                sigmas=np.linspace(1.0, 0.0, 26),
                ratio__clipscore=ratios,
                ratio__hpsv2=ratios,
                ratio__pickscore=ratios,
                ratio__geneval=ratios,
            )
            prompts_path.write_text(
                "\n".join(
                    '{"tag":"single_object","include":'
                    f'[{{"class":"cat","count":1}}],"prompt":"cat {index}"}}'
                    for index in range(8)
                ),
                encoding="utf-8",
            )
            arguments = [
                "train_recast.py",
                "--config",
                str(root / "configs" / "h200_8gpu_stage2_geneval.toml"),
                "--profiles",
                str(profile_path),
                "--prompts",
                str(prompts_path),
                "--output-dir",
                str(temporary / "stage2"),
                "--initialize-from",
                str(parent),
                "--dry-run",
            ]
            output = io.StringIO()
            with (
                patch.object(sys, "argv", arguments),
                patch.dict("os.environ", {"RANK": "0"}),
                redirect_stdout(output),
            ):
                main()

            summary = json.loads(output.getvalue())
            self.assertEqual(summary["config"]["epochs"], 61)
            self.assertEqual(summary["config"]["initialize_from"], str(parent))
            self.assertEqual(len(summary["reward_budgets"]), 4)


if __name__ == "__main__":
    unittest.main()
