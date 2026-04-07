import unittest

from abalone import native
from abalone.ai.heuristics import DEFAULT_WEIGHTS, evaluate_with_weights
from abalone.ai.minimax import search_best_move
from abalone.ai.types import AgentConfig
from abalone.game.board import BLACK, WHITE, Board
from abalone.players.registry import get_agent
from abalone.state_space import generate_legal_moves


EXPECTED_MOVES = {
    ("standard", BLACK): [
        "3:a1d4", "2:a1c1", "3:a2d5", "2:a2c2", "3:a3d6", "3:a3d3",
        "2:a4c6", "3:a4d4", "2:a5c7", "3:a5d5", "1:b1c2", "1:b1c1",
        "2:b2d4", "1:b2c2", "2:b3d5", "2:b3d3", "2:b4d6", "2:b4d4",
        "1:b5c6", "2:b5d5", "1:b6c7", "1:b6c6", "1:c3d4", "3:c3c6",
        "1:c3c2", "1:c3d3", "1:c4d5", "2:c4c6", "2:c4c2", "1:c4d4",
        "1:c5d6", "1:c5c6", "3:c5c2", "1:c5d5", "2:b1-b2>NW", "2:b2-c3>NW",
        "2:b5-b6>NE", "2:b5-c5>NE", "2:c3-c4>NE", "2:c3-c4>NW", "3:c3-c5>NE",
        "3:c3-c5>NW", "2:c4-c5>NE", "2:c4-c5>NW",
    ],
    ("standard", WHITE): [
        "3:g5g8", "1:g5f5", "1:g5f4", "1:g5g4", "2:g6g8", "1:g6f6",
        "1:g6f5", "2:g6g4", "1:g7g8", "1:g7f7", "1:g7f6", "3:g7g4",
        "1:h4g4", "1:h4g3", "2:h5f5", "1:h5g4", "2:h6f6", "2:h6f4",
        "2:h7f7", "2:h7f5", "1:h8g8", "2:h8f6", "1:h9g9", "1:h9g8",
        "3:i5f5", "2:i5g3", "3:i6f6", "2:i6g4", "3:i7f7", "3:i7f4",
        "2:i8g8", "3:i8f5", "2:i9g9", "3:i9f6", "2:g5-g6>SE", "2:g5-g6>SW",
        "3:g5-g7>SE", "3:g5-g7>SW", "2:g5-h5>SW", "2:g6-g7>SE", "2:g6-g7>SW",
        "2:g7-h8>SE", "2:h4-h5>SW", "2:h8-h9>SE",
    ],
    ("belgian_daisy", BLACK): [
        "3:a1d4", "2:a1a3", "2:a1c1", "2:a2c4", "1:a2a3", "3:a2d2",
        "2:b1d3", "1:b1c1", "2:b2d4", "2:b2d2", "1:b3c4", "1:b3a3",
        "2:b3d3", "1:c2d3", "2:c2c4", "1:c2c1", "1:c2d2", "1:c3d4",
        "1:c3c4", "2:c3a3", "2:c3c1", "1:c3d3", "2:g7g9", "1:g7f7",
        "1:g7f6", "1:g7g6", "2:g7i7", "1:g8g9", "1:g8f8", "1:g8f7",
        "2:g8g6", "2:h7f7", "1:h7g6", "1:h7i7", "2:h8f8", "2:h8f6",
        "1:h9g9", "2:h9f7", "3:i8f8", "2:i8g6", "1:i8i7", "2:i9g9",
        "3:i9f6", "2:i9i7", "2:b1-c2>NW", "2:b3-c3>NE", "2:c2-c3>NE",
        "2:c2-c3>NW", "2:g7-g8>SE", "2:g7-g8>SW", "2:g7-h7>SW", "2:g8-h9>SE",
    ],
    ("belgian_daisy", WHITE): [
        "3:a4d7", "1:a4a3", "2:a4c4", "2:a5c7", "2:a5a3", "3:a5d5",
        "2:b4d6", "1:b4a3", "1:b4c4", "2:b5d7", "2:b5d5", "1:b6c7",
        "2:b6d6", "1:c5d6", "2:c5c7", "2:c5a3", "1:c5c4", "1:c5d5",
        "1:c6d7", "1:c6c7", "2:c6c4", "1:c6d6", "2:g4g6", "1:g4f4",
        "1:g4f3", "1:g4g3", "2:g5i7", "1:g5g6", "1:g5f5", "1:g5f4",
        "2:g5g3", "2:h4f4", "1:h4g3", "2:h5f5", "2:h5f3", "1:h6i7",
        "1:h6g6", "2:h6f4", "2:i5i7", "3:i5f5", "2:i5g3", "1:i6i7",
        "2:i6g6", "3:i6f3", "2:b4-c5>NW", "2:b6-c6>NE", "2:c5-c6>NE",
        "2:c5-c6>NW", "2:g4-g5>SE", "2:g4-g5>SW", "2:g4-h4>SW", "2:g5-h6>SE",
    ],
    ("german_daisy", BLACK): [
        "3:b1e4", "2:b1b3", "1:b1a1", "2:b1d1", "2:b2d4", "1:b2b3",
        "1:b2a2", "1:b2a1", "3:b2e2", "2:c1e3", "3:c1c4", "2:c1a1",
        "1:c1d1", "2:c2e4", "2:c2c4", "2:c2a2", "2:c2e2", "1:c3d4",
        "1:c3c4", "1:c3b3", "2:c3a1", "2:c3e3", "1:d2e3", "2:d2d4",
        "3:d2a2", "1:d2d1", "1:d2e2", "1:d3e4", "1:d3d4", "2:d3b3",
        "2:d3d1", "1:d3e3", "2:f7f9", "1:f7e7", "1:f7e6", "1:f7f6",
        "2:f7h7", "1:f8f9", "1:f8e8", "1:f8e7", "2:f8f6", "3:f8i8",
        "2:g7i9", "2:g7e7", "1:g7f6", "1:g7g6", "1:g7h7", "2:g8e8",
        "2:g8e6", "2:g8g6", "2:g8i8", "1:g9f9", "2:g9e7", "3:g9g6",
        "2:g9i9", "1:h8i9", "3:h8e8", "2:h8f6", "1:h8h7", "1:h8i8",
        "2:h9f9", "3:h9e6", "2:h9h7", "1:h9i9", "2:b1-b2>SE", "2:b2-c3>E",
        "2:b2-c3>SE", "2:c1-d2>NW", "2:c3-d3>NE", "2:c3-d3>E", "2:d2-d3>NE",
        "2:d2-d3>NW", "2:f7-f8>SE", "2:f7-f8>SW", "2:f7-g7>SW", "2:f7-g7>W",
        "2:f8-g9>SE", "2:g7-h8>W", "2:g7-h8>NW", "2:h8-h9>NW",
    ],
    ("german_daisy", WHITE): [
        "3:b5e8", "1:b5a5", "1:b5a4", "1:b5b4", "2:b5d5", "2:b6d8",
        "1:b6a5", "2:b6b4", "3:b6e6", "2:c5e7", "2:c5a5", "1:c5b4",
        "1:c5c4", "1:c5d5", "2:c6e8", "2:c6a4", "2:c6c4", "2:c6e6",
        "1:c7d8", "2:c7a5", "3:c7c4", "2:c7e7", "1:d6e7", "2:d6d8",
        "2:d6b4", "1:d6d5", "1:d6e6", "1:d7e8", "1:d7d8", "3:d7a4",
        "2:d7d5", "1:d7e7", "3:f3i6", "2:f3f5", "1:f3e3", "1:f3e2",
        "1:f3f2", "2:f4h6", "1:f4f5", "1:f4e4", "1:f4e3", "2:f4f2",
        "2:g3i5", "3:g3g6", "2:g3e3", "1:g3f2", "2:g4i6", "2:g4g6",
        "2:g4e4", "2:g4e2", "1:g5h6", "1:g5g6", "1:g5f5", "2:g5e3",
        "2:g5i5", "1:h4i5", "2:h4h6", "3:h4e4", "2:h4f2", "1:h5i6",
        "1:h5h6", "2:h5f5", "3:h5e2", "1:h5i5", "2:b5-b6>SW", "2:b5-c5>SW",
        "2:b5-c5>W", "2:c5-d6>W", "2:c5-d6>NW", "2:c7-d7>NE", "2:d6-d7>NE",
        "2:d6-d7>NW", "2:f3-f4>SE", "2:f3-f4>SW", "2:f3-g3>SW", "2:f4-g5>E",
        "2:f4-g5>SE", "2:g5-h5>NE", "2:g5-h5>E", "2:h4-h5>NE",
    ],
}

EXPECTED_EVALUATIONS = {
    ("belgian_daisy", BLACK): 280.0,
    ("belgian_daisy", WHITE): -280.0,
}

EXPECTED_SEARCHES = {
    ("standard", BLACK, "default", 2): {
        "move": "3:a1d4",
        "score": -535.0,
        "completed_depth": 2,
        "decision_source": "search",
    },
    ("belgian_daisy", WHITE, "kyle", 2): {
        "move": "3:a5d5",
        "score": -760.492,
        "completed_depth": 2,
        "decision_source": "search",
    },
    ("german_daisy", BLACK, "jonah", 2): {
        "move": "3:b1e4",
        "score": 420.0,
        "completed_depth": 2,
        "decision_source": "search",
    },
}


@unittest.skipUnless(native.is_available(), "native extension not built")
class NativeRegressionTests(unittest.TestCase):
    def test_native_move_generation_matches_regression_fixtures(self):
        for (layout, player), expected in EXPECTED_MOVES.items():
            board = Board()
            board.setup_layout(layout)
            actual = [move.to_notation() for move in generate_legal_moves(board, player)]
            self.assertEqual(actual, expected, msg=f"{layout}:{player}")

    def test_native_weighted_evaluation_matches_regression_fixtures(self):
        for (layout, player), expected in EXPECTED_EVALUATIONS.items():
            board = Board()
            board.setup_layout(layout)
            actual = evaluate_with_weights(board, player, DEFAULT_WEIGHTS)
            self.assertAlmostEqual(actual, expected, msg=f"{layout}:{player}")

    def test_native_search_matches_regression_fixtures(self):
        for (layout, player, agent_id, depth), expected in EXPECTED_SEARCHES.items():
            board = Board()
            board.setup_layout(layout)
            actual = search_best_move(board, player, get_agent(agent_id), AgentConfig(depth=depth))
            self.assertEqual(
                None if actual.move is None else actual.move.to_notation(),
                expected["move"],
                msg=f"{layout}:{player}:{agent_id}",
            )
            self.assertEqual(actual.completed_depth, expected["completed_depth"])
            self.assertEqual(actual.decision_source, expected["decision_source"])
            self.assertAlmostEqual(actual.score, expected["score"])


if __name__ == "__main__":
    unittest.main()
