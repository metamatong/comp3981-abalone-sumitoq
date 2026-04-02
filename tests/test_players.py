import unittest

from abalone.ai.agent import choose_move
from abalone.ai.types import AgentConfig
from abalone.game.board import BLACK, Board, Move


class MinimaxPlayerTests(unittest.TestCase):
    """Regression checks for minimax move selection."""

    def test_choose_move_is_legal_and_deterministic(self):
        """Agent should return a legal move and repeat deterministically."""
        board = Board()
        board.setup_standard()

        cfg = AgentConfig(depth=1)
        first = choose_move(board, BLACK, cfg)
        second = choose_move(board, BLACK, cfg)

        self.assertIsNotNone(first)
        self.assertTrue(board.is_legal_move(first, BLACK))
        self.assertEqual(first.to_notation(), second.to_notation())

    def test_move_normalizes_unsorted_marbles_without_affecting_equality(self):
        first = Move(marbles=((1, 2), (0, 1), (2, 3)), direction=(1, 1))
        second = Move(marbles=((2, 3), (1, 2), (0, 1)), direction=(1, 1))

        self.assertEqual(first.marbles, ((0, 1), (1, 2), (2, 3)))
        self.assertEqual(first, second)
        self.assertEqual(hash(first), hash(second))
        self.assertEqual(first.count, 3)
        self.assertTrue(first.is_inline)
        self.assertEqual(first.leading_trailing(), ((0, 1), (2, 3)))

    def test_move_notation_matches_inline_and_broadside_forms(self):
        inline = Move(marbles=((2, 3), (0, 1), (1, 2)), direction=(1, 1))
        broadside = Move(marbles=((1, 1), (2, 1)), direction=(0, 1))

        self.assertEqual(inline.to_notation(), "3:a1d4")
        self.assertEqual(inline.to_notation(pushed=True), "3:a1d4*")
        self.assertEqual(broadside.to_notation(), "2:b1-c1>E")

    def test_external_invalid_move_still_fails_full_validation(self):
        board = Board()
        board.setup_standard()
        move = Move(marbles=((8, 9),), direction=(0, 1))

        self.assertFalse(board.is_legal_move(move, BLACK))


if __name__ == "__main__":
    unittest.main()
