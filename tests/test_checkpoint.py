import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from peft import LoraConfig, get_peft_model

from recast.training import (
    DistributedContext,
    TrainingConfig,
    _adapter_parameters,
    _load_checkpoint,
    _restore_training_state,
    _save_checkpoint,
    update_old_adapter,
)


class CheckpointTest(unittest.TestCase):
    def test_current_old_and_optimizer_state_round_trip(self):
        base = torch.nn.Sequential(torch.nn.Linear(4, 4))
        lora = LoraConfig(r=2, lora_alpha=4, target_modules=["0"])
        model = get_peft_model(base, lora)
        model.add_adapter("old", lora)
        update_old_adapter(model, decay=0.0)
        model.set_adapter("default")
        optimizer = torch.optim.AdamW(
            list(_adapter_parameters(model, "default").values()), lr=1e-3
        )
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        context = DistributedContext(
            rank=0,
            local_rank=0,
            world_size=1,
            device=torch.device("cpu"),
            initialized=False,
        )
        config = TrainingConfig()

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = _save_checkpoint(
                model,
                optimizer,
                scaler,
                Path(directory),
                epoch=3,
                updates=7,
                config=config,
                context=context,
            )
            start_epoch, updates = _load_checkpoint(
                model, optimizer, scaler, checkpoint, config
            )

        self.assertEqual(start_epoch, 4)
        self.assertEqual(updates, 7)

    @patch("recast.training._load_checkpoint", return_value=(122, 121))
    def test_parent_initialization_resets_stage_epoch_but_keeps_updates(self, load):
        config = TrainingConfig(initialize_from="stage1/checkpoint-0121")
        start_epoch, updates = _restore_training_state(None, None, None, config)
        self.assertEqual((start_epoch, updates), (1, 121))
        load.assert_called_once()

    @patch("recast.training._load_checkpoint", return_value=(31, 151))
    def test_same_stage_resume_keeps_next_epoch(self, load):
        config = TrainingConfig(resume_from="stage2/checkpoint-0030")
        start_epoch, updates = _restore_training_state(None, None, None, config)
        self.assertEqual((start_epoch, updates), (31, 151))
        load.assert_called_once()


if __name__ == "__main__":
    unittest.main()
