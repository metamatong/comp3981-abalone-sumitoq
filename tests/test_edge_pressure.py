import unittest

from abalone.ai.defaults import DEFAULT_AGENT
from abalone.ai.heuristics import DEFAULT_WEIGHTS, build_weighted_evaluator, edge_pressure, evaluate_with_weights
from abalone.game.board import BLACK, WHITE, Board, str_to_pos
from abalone.game.duel import _format_weights
from abalone.players.registry import get_agent


def _board_with_positions(black_positions, white_positions):
    board = Board()
    board.clear()
    for pos in black_positions:
        board.cells[str_to_pos(pos)] = BLACK
    for pos in white_positions:
        board.cells[str_to_pos(pos)] = WHITE
    board.recompute_zhash()
    return board


class EdgePressureTests(unittest.TestCase):
    def test_edge_pressure_scores_edge_near_and_safe_positions(self):
        board = _board_with_positions(["a1", "b2", "c3"], [])

        score = edge_pressure(board.get_marbles(BLACK), board.get_marbles(WHITE))

        self.assertEqual(score, -5.0)

    def test_safe_shapes_outscore_exposed_shapes(self):
        safe_board = _board_with_positions(["c3", "d4"], [])
        exposed_board = _board_with_positions(["a1", "b2"], [])

        safe_score = edge_pressure(safe_board.get_marbles(BLACK), safe_board.get_marbles(WHITE))
        exposed_score = edge_pressure(exposed_board.get_marbles(BLACK), exposed_board.get_marbles(WHITE))

        self.assertEqual(safe_score, 0.0)
        self.assertEqual(exposed_score, -5.0)
        self.assertGreater(safe_score, exposed_score)

    def test_opponent_rimmed_marbles_increase_edge_pressure(self):
        safe_opponent_board = _board_with_positions(["c3"], ["g5"])
        rimmed_opponent_board = _board_with_positions(["c3"], ["i9"])

        safe_score = edge_pressure(
            safe_opponent_board.get_marbles(BLACK),
            safe_opponent_board.get_marbles(WHITE),
        )
        rimmed_score = edge_pressure(
            rimmed_opponent_board.get_marbles(BLACK),
            rimmed_opponent_board.get_marbles(WHITE),
        )

        self.assertEqual(safe_score, 0.0)
        self.assertEqual(rimmed_score, 3.0)
        self.assertGreater(rimmed_score, safe_score)

    def test_evaluate_with_weights_uses_edge_pressure_key(self):
        board = _board_with_positions(["c3"], ["i9"])

        score = evaluate_with_weights(board, BLACK, {"edge_pressure": 1.0})

        self.assertEqual(score, 3.0)

    def test_build_weighted_evaluator_rejects_legacy_edge_key(self):
        with self.assertRaisesRegex(ValueError, "edge"):
            build_weighted_evaluator({"edge": 1.0})

    def test_build_weighted_evaluator_rejects_legacy_threat_key(self):
        with self.assertRaisesRegex(ValueError, "threat"):
            build_weighted_evaluator({"threat": 1.0})

    def test_default_weights_use_edge_pressure_only(self):
        self.assertIn("edge_pressure", DEFAULT_WEIGHTS)
        self.assertNotIn("edge", DEFAULT_WEIGHTS)
        self.assertNotIn("threat", DEFAULT_WEIGHTS)

    def test_all_agents_expose_edge_pressure_only(self):
        for agent_id in ("default", "abdullah", "cole", "jonah", "kyle"):
            agent = get_agent(agent_id)
            self.assertIn("edge_pressure", agent.evaluator.weights)
            self.assertNotIn("edge", agent.evaluator.weights)
            self.assertNotIn("threat", agent.evaluator.weights)

    def test_duel_weight_format_uses_edge_pressure_only(self):
        formatted = _format_weights(DEFAULT_AGENT)

        self.assertIn("edge_pressure=70.0", formatted)
        self.assertNotIn("edge=", formatted)
        self.assertNotIn("threat=", formatted)


if __name__ == "__main__":
    unittest.main()
