import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from recast.training import (
    TrainingConfig,
    _sample_batch,
    group_advantages,
    preaggregate_advantages,
    recast_nft_loss,
    select_rank_prompt_indices,
)


class _FakeModel:
    def __init__(self):
        self.adapter = None

    def set_adapter(self, adapter):
        self.adapter = adapter

    def eval(self):
        return self


class _FakePipeline:
    _execution_device = torch.device("cpu")

    def encode_prompt(
        self,
        prompt,
        prompt_2,
        prompt_3,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        max_sequence_length,
    ):
        del prompt_2, prompt_3, device, do_classifier_free_guidance, max_sequence_length
        batch = len(prompt) * num_images_per_prompt
        return (
            torch.zeros(batch, 2, 3),
            torch.zeros(batch, 2, 3),
            torch.zeros(batch, 4),
            torch.zeros(batch, 4),
        )

    def __call__(self, **kwargs):
        batch = kwargs["prompt_embeds"].shape[0]
        return SimpleNamespace(images=torch.zeros(batch, 4, 2, 2))


class TrainingMathTest(unittest.TestCase):
    @patch(
        "recast.training._decode_latents",
        side_effect=lambda pipeline, latents: [
            Image.new("RGB", (2, 2)) for _ in range(latents.shape[0])
        ],
    )
    def test_sampling_repeats_prompts_in_embedding_order(self, _decode):
        model = _FakeModel()
        config = TrainingConfig(precision="fp32", images_per_prompt=2)
        result = _sample_batch(
            _FakePipeline(),
            model,
            ["first", "second"],
            config,
            torch.Generator(),
        )
        latents, prompt_embeds, pooled_embeds, images, prompts = result
        self.assertEqual(latents.shape[0], 4)
        self.assertEqual(prompt_embeds.shape[0], 4)
        self.assertEqual(pooled_embeds.shape[0], 4)
        self.assertEqual(len(images), 4)
        self.assertEqual(prompts, ["first", "first", "second", "second"])
        self.assertEqual(model.adapter, "default")

    def test_group_advantages_are_normalized_per_prompt(self):
        advantages = group_advantages(
            torch.tensor([1.0, 3.0, 10.0, 14.0]), group_size=2
        )
        grouped = advantages.view(2, 2)
        torch.testing.assert_close(grouped.mean(dim=1), torch.zeros(2))
        torch.testing.assert_close(grouped.std(dim=1, unbiased=False), torch.ones(2))

    def test_preaggregation_mixes_raw_rewards_before_normalization(self):
        raw_rewards = torch.tensor([[1.0, 3.0, 10.0, 14.0], [4.0, 0.0, 2.0, 6.0]])
        weights = torch.eye(2)
        advantages = preaggregate_advantages(raw_rewards, weights, group_size=2)
        self.assertEqual(advantages.shape, (4, 2))
        grouped = advantages.T.reshape(2, 2, 2)
        torch.testing.assert_close(grouped.mean(dim=2), torch.zeros(2, 2))
        torch.testing.assert_close(grouped.std(dim=2, unbiased=False), torch.ones(2, 2))

    def test_rank_prompt_selections_are_disjoint_and_deterministic(self):
        selections = [
            select_rank_prompt_indices(32, 2, rank, 8, seed=123) for rank in range(8)
        ]
        flattened = [index for selection in selections for index in selection]
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(
            selections[3], select_rank_prompt_indices(32, 2, 3, 8, seed=123)
        )

    def test_recast_loss_is_finite_and_differentiable(self):
        prediction = torch.randn(2, 4, 2, 2, requires_grad=True)
        old = torch.randn_like(prediction)
        reference = torch.randn_like(prediction)
        x_t = torch.randn_like(prediction)
        x_0 = torch.randn_like(prediction)
        sigma = torch.tensor([0.8, 0.2])
        advantages = torch.tensor([[1.0, -1.0], [-0.5, 0.5]])
        step_indices = torch.tensor([0, 1])
        weights = torch.tensor([[0.7, 0.2], [0.3, 0.8]])

        loss, metrics = recast_nft_loss(
            prediction,
            old,
            reference,
            x_t,
            x_0,
            sigma,
            advantages,
            step_indices,
            weights,
            nft_beta=1.0,
            advantage_clip=5.0,
            kl_beta=0.01,
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertEqual(set(metrics), {"loss", "policy_loss", "kl_loss"})

    def test_preaggregated_loss_is_finite_and_differentiable(self):
        prediction = torch.randn(3, 4, 2, 2, requires_grad=True)
        inputs = [torch.randn_like(prediction) for _ in range(4)]
        loss, _ = recast_nft_loss(
            prediction,
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
            torch.tensor([0.9, 0.5, 0.1]),
            torch.tensor([1.0, 0.0, -1.0]),
            torch.tensor([0, 1, 0]),
            torch.tensor([[0.7, 0.2], [0.3, 0.8]]),
            aggregation="pre",
            nft_beta=0.1,
            advantage_clip=5.0,
            kl_beta=1e-4,
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(prediction.grad).all())


if __name__ == "__main__":
    unittest.main()
