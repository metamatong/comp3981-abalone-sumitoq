import contextlib
import io
import re
import time
import unittest

from abalone.ai.agent import choose_move, choose_move_with_info
from abalone.ai.heuristics import evaluate_board
from abalone.ai.types import AgentConfig, AgentDefinition
from abalone.game.board import BLACK, WHITE, Board
from abalone.game.config import GameConfig, MODE_AVA, MODE_HVA
from abalone.game.match import main as match_main
from abalone.game.session import GameSession
from abalone.players.registry import get_agent
from abalone.state_space import generate_legal_moves


class AgentRuntimeTests(unittest.TestCase):
    """Regression checks for opening randomness, time budgets, and AI selection."""

    def test_direct_choose_move_stays_deterministic_without_opening_context(self):
        board = Board()
        board.setup_standard()

        cfg = AgentConfig(depth=1)
        first = choose_move(board, BLACK, cfg)
        second = choose_move(board, BLACK, cfg)

        self.assertIsNotNone(first)
        self.assertEqual(first.to_notation(), second.to_notation())

    def test_ai_black_opening_move_uses_seeded_random_path(self):
        session_a = GameSession(config=GameConfig(mode=MODE_AVA, ai_depth=1), opening_seed=7)
        session_a.reset()
        first = session_a.apply_agent_move()

        session_b = GameSession(config=GameConfig(mode=MODE_AVA, ai_depth=1), opening_seed=7)
        session_b.reset()
        second = session_b.apply_agent_move()

        self.assertEqual(first["agent_id"], "default")
        self.assertEqual(first["search"]["decision_source"], "opening_random")
        self.assertTrue(first.get("ok"))
        self.assertEqual(first["notation"], second["notation"])

    def test_ai_black_opening_move_changes_with_different_seed(self):
        session_a = GameSession(config=GameConfig(mode=MODE_AVA, ai_depth=1), opening_seed=1)
        session_a.reset()
        first = session_a.apply_agent_move()

        session_b = GameSession(config=GameConfig(mode=MODE_AVA, ai_depth=1), opening_seed=2)
        session_b.reset()
        second = session_b.apply_agent_move()

        self.assertNotEqual(first["notation"], second["notation"])

    def test_opening_randomization_does_not_repeat_after_first_black_turn(self):
        session = GameSession(config=GameConfig(mode=MODE_AVA, ai_depth=1), opening_seed=11)
        session.reset()

        first = session.apply_agent_move()
        second = session.apply_agent_move()
        third = session.apply_agent_move()

        self.assertEqual(first["search"]["decision_source"], "opening_random")
        self.assertEqual(second["agent_id"], "default")
        self.assertNotEqual(third["search"]["decision_source"], "opening_random")

    def test_session_state_round_trips_selected_ai_ids(self):
        session = GameSession(
            config=GameConfig(
                mode=MODE_AVA,
                ai_depth=1,
                black_ai_id="kyle",
                white_ai_id="jonah",
            ),
            opening_seed=5,
        )
        session.reset()

        state = session.state_json()
        self.assertEqual(state["black_ai_id"], "kyle")
        self.assertEqual(state["white_ai_id"], "jonah")
        self.assertTrue(state["available_agents"])

        black_move = session.apply_agent_move()
        white_move = session.apply_agent_move()

        self.assertEqual(black_move["agent_id"], "kyle")
        self.assertEqual(white_move["agent_id"], "jonah")

    def test_session_uses_agent_default_depth_when_global_override_is_unset(self):
        board = Board()
        board.clear()
        board.cells[(4, 5)] = BLACK
        board.cells[(8, 9)] = WHITE
        board.recompute_zhash()

        session = GameSession(
            config=GameConfig(
                mode=MODE_AVA,
                ai_depth=None,
                black_ai_id="kyle",
                white_ai_id="abdullah",
            ),
            opening_seed=3,
        )
        session.board = board
        session.current_player = WHITE
        session.started = True

        result = session.apply_agent_move()
        self.assertTrue(result.get("ok"))
        self.assertEqual(result["agent_id"], "abdullah")
        self.assertEqual(result["search"]["depth"], get_agent("abdullah").default_depth)

    def test_hva_uses_only_ai_controlled_side_selection(self):
        session = GameSession(
            config=GameConfig(
                mode=MODE_HVA,
                human_side=BLACK,
                ai_depth=1,
                black_ai_id="kyle",
                white_ai_id="abdullah",
            ),
            opening_seed=3,
        )
        session.reset()

        state = session.state_json()
        human_move = state["legal_moves"][0]
        self.assertTrue(session.apply_human_move(human_move).get("ok"))

        ai_move = session.apply_agent_move()
        self.assertEqual(ai_move["agent_id"], "abdullah")

    def test_timeout_can_return_completed_depth_result(self):
        board = Board()
        board.clear()
        board.cells[(4, 5)] = BLACK
        board.cells[(8, 9)] = WHITE
        board.recompute_zhash()

        def slow_eval(current_board, player):
            time.sleep(0.003)
            return evaluate_board(current_board, player)

        agent = AgentDefinition(
            id="slow-depth",
            label="Slow Depth",
            owner="Test",
            evaluator=slow_eval,
            default_depth=3,
        )
        started = time.perf_counter()
        result = choose_move_with_info(
            board,
            BLACK,
            agent=agent,
            config=AgentConfig(depth=3, time_budget_ms=80),
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        self.assertTrue(board.is_legal_move(result.move, BLACK))
        self.assertTrue(result.timed_out)
        self.assertEqual(result.decision_source, "timeout_partial")
        self.assertGreaterEqual(result.completed_depth, 1)
        self.assertLess(elapsed_ms, 600.0)

    def test_timeout_fallback_returns_ordered_legal_move_when_no_depth_finishes(self):
        board = Board()
        board.setup_standard()

        def very_slow_eval(current_board, player):
            time.sleep(0.01)
            return evaluate_board(current_board, player)

        agent = AgentDefinition(
            id="slow-fallback",
            label="Slow Fallback",
            owner="Test",
            evaluator=very_slow_eval,
            default_depth=2,
        )
        result = choose_move_with_info(
            board,
            BLACK,
            agent=agent,
            config=AgentConfig(depth=2, time_budget_ms=1),
        )

        self.assertTrue(board.is_legal_move(result.move, BLACK))
        self.assertTrue(result.timed_out)
        self.assertEqual(result.decision_source, "timeout_fallback_partial")
        self.assertEqual(result.completed_depth, 0)

    def test_repetition_avoidance_breaks_loop(self):
        session = GameSession(config=GameConfig(mode=MODE_AVA, ai_depth=1), opening_seed=3)
        session.reset()

        board = Board()
        board.clear()
        board.cells[(4, 5)] = BLACK
        board.cells[(5, 6)] = WHITE
        board.recompute_zhash()
        session.board = board
        session.current_player = BLACK
        session.started = True

        black_moves = generate_legal_moves(board, BLACK)
        white_moves = generate_legal_moves(board, WHITE)
        self.assertGreaterEqual(len(black_moves), 2)
        self.assertGreaterEqual(len(white_moves), 1)

        baseline = choose_move_with_info(
            board,
            BLACK,
            agent=get_agent("default"),
            config=AgentConfig(depth=1, is_opening_turn=False),
        )
        move_a = baseline.move
        move_b = white_moves[0]
        snapshot = board.copy()
        session.move_history = [
            {"move": move_a, "player": BLACK, "snapshot": snapshot},
            {"move": move_b, "player": WHITE, "snapshot": board.copy()},
        ]

        result = session.apply_agent_move()
        self.assertTrue(result.get("ok"))
        self.assertNotEqual(result["notation"], move_a.to_notation())
        self.assertEqual(result["search"]["decision_source"], "repeat_avoidance")

    def test_two_move_cycle_avoidance_breaks_loop(self):
        session = GameSession(config=GameConfig(mode=MODE_AVA, ai_depth=1), opening_seed=3)
        session.reset()

        board = Board()
        board.clear()
        board.cells[(4, 5)] = BLACK
        board.cells[(5, 6)] = WHITE
        board.recompute_zhash()
        session.board = board
        session.current_player = BLACK
        session.started = True

        black_moves = generate_legal_moves(board, BLACK)
        white_moves = generate_legal_moves(board, WHITE)
        self.assertGreaterEqual(len(black_moves), 2)
        self.assertGreaterEqual(len(white_moves), 2)

        move_a1 = black_moves[0]
        move_a2 = black_moves[1]
        move_b1 = white_moves[0]
        move_b2 = white_moves[1]

        snapshot = board.copy()
        session.move_history = [
            {"move": move_a1, "player": BLACK, "snapshot": snapshot},
            {"move": move_b1, "player": WHITE, "snapshot": board.copy()},
            {"move": move_a2, "player": BLACK, "snapshot": board.copy()},
            {"move": move_b2, "player": WHITE, "snapshot": board.copy()},
        ]

        result = session.apply_agent_move()
        self.assertTrue(result.get("ok"))
        self.assertNotEqual(result["notation"], move_a1.to_notation())
        self.assertEqual(result["search"]["decision_source"], "repeat_avoidance")

    def test_match_mode_is_deterministic_with_fixed_seed(self):
        argv = [
            "--black-ai", "default",
            "--white-ai", "kyle",
            "--rounds", "1",
            "--depth", "1",
            "--move-time-s", "5",
            "--max-moves", "10",
            "--seed", "9",
        ]
        stdout_one = io.StringIO()
        with contextlib.redirect_stdout(stdout_one):
            match_main(argv)

        stdout_two = io.StringIO()
        with contextlib.redirect_stdout(stdout_two):
            match_main(argv)

        output_one = stdout_one.getvalue()
        output_two = stdout_two.getvalue()
        normalized_one = re.sub(r"avg_move_ms=\d+\.\d+", "avg_move_ms=<var>", output_one)
        normalized_two = re.sub(r"avg_move_ms=\d+\.\d+", "avg_move_ms=<var>", output_two)
        self.assertEqual(normalized_one, normalized_two)
        self.assertIn("Completed 1 round(s), 2 game(s).", output_one)
        self.assertIn("default:", output_one)
        self.assertIn("kyle:", output_one)


if __name__ == "__main__":
    unittest.main()
