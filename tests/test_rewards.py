import unittest

from PIL import Image

from recast.rewards import JpegIncompressibilityReward, score_reward


class RewardTest(unittest.TestCase):
    def test_jpeg_reward_returns_one_finite_score_per_image(self):
        images = [Image.new("RGB", (16, 16), color) for color in ("red", "blue")]
        scores = score_reward(
            JpegIncompressibilityReward(),
            images,
            ["red", "blue"],
            [{"prompt": "red"}, {"prompt": "blue"}],
        )
        self.assertEqual(tuple(scores.shape), (2,))
        self.assertTrue((scores > 0).all())


if __name__ == "__main__":
    unittest.main()
