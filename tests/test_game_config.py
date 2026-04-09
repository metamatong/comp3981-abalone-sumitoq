import unittest

from abalone.game.config import GameConfig, merge_config


class GameConfigTests(unittest.TestCase):
    def test_per_side_ai_depth_accepts_none(self):
        config = merge_config(GameConfig(), {"black_ai_depth": None, "white_ai_depth": None})

        self.assertIsNone(config.black_ai_depth)
        self.assertIsNone(config.white_ai_depth)

    def test_per_side_ai_depth_accepts_zero(self):
        config = merge_config(GameConfig(), {"black_ai_depth": 0, "white_ai_depth": "3"})

        self.assertEqual(config.black_ai_depth, 0)
        self.assertEqual(config.white_ai_depth, 3)

    def test_per_side_ai_depth_rejects_negative_values(self):
        with self.assertRaisesRegex(ValueError, "black_ai_depth must be non-negative."):
            merge_config(GameConfig(), {"black_ai_depth": -1})

    def test_per_side_ai_depth_rejects_decimal_values(self):
        with self.assertRaisesRegex(ValueError, "white_ai_depth must be an integer."):
            merge_config(GameConfig(), {"white_ai_depth": "1.5"})

        with self.assertRaisesRegex(ValueError, "white_ai_depth must be an integer."):
            merge_config(GameConfig(), {"white_ai_depth": 1.0})

    def test_per_side_ai_depth_rejects_non_numeric_text(self):
        with self.assertRaisesRegex(ValueError, "black_ai_depth must be an integer."):
            merge_config(GameConfig(), {"black_ai_depth": "abc"})


if __name__ == "__main__":
    unittest.main()
