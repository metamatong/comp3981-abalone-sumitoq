import unittest

from abalone import native
import abalone.ai.minimax as minimax
from abalone.ai.agent import choose_move_with_info
from abalone.ai.types import AgentConfig
from abalone.game.board import BLACK, Board
from abalone.players.registry import get_agent
from abalone.state_space import generate_legal_moves


class TerminalSearchTests(unittest.TestCase):
    """Regression coverage for forced terminal-win move selection."""

    BOARD_TOKEN = "a5w,b1w,d1w,d6w,e5b,e6b,f2b,f6b,f8b,g4w,g5w,g7b,g9b,h5w,h7w,h8b,i7b,i9w|5-5"

    def test_immediate_sixth_capture_beats_nonterminal_heuristics(self):
        board = Board.from_compact_token(self.BOARD_TOKEN)
        agent = get_agent("tournament-belgian")
        winning_moves = []

        for move in generate_legal_moves(board, BLACK):
            child = board.copy()
            child.apply_move(move, BLACK)
            if child.score[BLACK] >= 6:
                winning_moves.append(move.to_notation())

        original_force_path = minimax._FORCE_WEIGHTED_SEARCH_PATH
        minimax._FORCE_WEIGHTED_SEARCH_PATH = "python"
        try:
            result = choose_move_with_info(
                board,
                BLACK,
                agent=agent,
                config=AgentConfig(depth=3),
            )
        finally:
            minimax._FORCE_WEIGHTED_SEARCH_PATH = original_force_path

        self.assertEqual(set(winning_moves), {"2:g7i9", "3:f6i9"})
        self.assertIsNotNone(result.move)
        self.assertIn(result.move.to_notation(), winning_moves)
        self.assertGreaterEqual(result.score, minimax.TERMINAL_SCORE)

    @unittest.skipUnless(native.is_available(), "native extension not built")
    def test_native_weighted_path_short_circuits_immediate_terminal_win(self):
        board = Board.from_compact_token(self.BOARD_TOKEN)
        agent = get_agent("tournament-belgian")

        result = choose_move_with_info(
            board,
            BLACK,
            agent=agent,
            config=AgentConfig(depth=3, root_candidate_limit=5),
        )

        self.assertIsNotNone(result.move)
        self.assertIn(result.move.to_notation(), {"2:g7i9", "3:f6i9"})
        self.assertEqual(result.score, minimax.TERMINAL_SCORE)
        self.assertEqual(result.completed_depth, 1)


if __name__ == "__main__":
    unittest.main()
