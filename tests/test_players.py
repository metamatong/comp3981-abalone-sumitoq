import unittest

from abalone.game.board import BLACK, Board
from abalone.players.agent import choose_move
from abalone.players.types import AgentConfig


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


if __name__ == "__main__":
    unittest.main()
