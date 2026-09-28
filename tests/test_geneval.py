import unittest

import numpy as np
from PIL import Image

from recast.geneval import DetectedObject, GenEvalReward, relative_position


def _object(box):
    mask = np.zeros((32, 32), dtype=bool)
    left, top, right, bottom = box
    mask[top:bottom, left:right] = True
    return DetectedObject(box=box, mask=mask, score=0.99)


class GenEvalTest(unittest.TestCase):
    def test_relative_position(self):
        left = _object((1, 10, 5, 14))
        right = _object((20, 10, 24, 14))
        self.assertIn("left of", relative_position(left, right))
        self.assertIn("right of", relative_position(right, left))

    def test_structured_reward_scores_count_color_and_position(self):
        reward = object.__new__(GenEvalReward)
        reward._classify_colors = lambda image, objects, name: ["red"]
        objects = {
            "apple": [_object((2, 10, 6, 14))],
            "bowl": [_object((20, 10, 26, 16))],
        }
        metadata = {
            "tag": "position",
            "include": [
                {"class": "apple", "count": 1, "color": "red"},
                {"class": "bowl", "count": 1, "position": ["right of", 0]},
            ],
            "prompt": "a red apple left of a bowl",
        }
        score = reward._score_one(Image.new("RGB", (32, 32)), objects, metadata)
        self.assertEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
