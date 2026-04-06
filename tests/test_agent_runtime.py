from math import inf
import contextlib
import io
import re
import time
import unittest
from unittest import mock

from abalone import native
from abalone.ai.agent import choose_move, choose_move_with_info
from abalone.ai.heuristics import evaluate_board
from abalone.ai.minimax import TT_MODE_QUIESCENCE, _make_tt_key, _quiescence, SearchResult
from abalone.ai.types import AgentConfig, AgentDefinition
from abalone.game.board import BLACK, WHITE, Board
from abalone.game.config import GameConfig, MODE_AVA, MODE_HVA
from abalone.game.match import main as match_main
from abalone.game.session import GameSession
from abalone.players.registry import get_agent
from abalone.state_space import generate_legal_moves

if not native.is_available():
    raise unittest.SkipTest("native extension not built")


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

    def test_available_agent_metadata_includes_quiescence_depth(self):
        session = GameSession(config=GameConfig(mode=MODE_AVA, ai_depth=1), opening_seed=5)
        session.reset()

        state = session.state_json()
        default_agent = next(agent for agent in state["available_agents"] if agent["id"] == "default")

        self.assertIn("max_quiescence_depth", default_agent)
        self.assertEqual(default_agent["max_quiescence_depth"], 0)

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

    def test_session_uses_per_side_depth_overrides_before_global_depth(self):
        recorded_depths = []

        def fake_choose(board, player, config=None, agent=None):
            recorded_depths.append((player, config.depth))
            move = generate_legal_moves(board, player)[0]
            return SearchResult(
                move=move,
                score=0.0,
                nodes=0,
                elapsed_ms=0.0,
                depth=config.depth,
                completed_depth=0,
                decision_source="test",
                timed_out=False,
                time_budget_ms=config.time_budget_ms,
                agent_id=agent.id,
                agent_label=agent.label,
            )

        session = GameSession(
            config=GameConfig(
                mode=MODE_AVA,
                ai_depth=1,
                black_ai_id="kyle",
                white_ai_id="abdullah",
            ),
            opening_seed=3,
            agent_depth_overrides={BLACK: 2, WHITE: 6},
        )
        session.reset()

        with mock.patch("abalone.game.session.choose_move_with_info", side_effect=fake_choose):
            black_move = session.apply_agent_move()
            white_move = session.apply_agent_move()

        self.assertTrue(black_move.get("ok"))
        self.assertTrue(white_move.get("ok"))
        self.assertEqual(recorded_depths, [(BLACK, 2), (WHITE, 6)])
        self.assertEqual(black_move["search"]["depth"], 2)
        self.assertEqual(white_move["search"]["depth"], 6)

    def test_session_caps_search_depth_to_remaining_game_moves(self):
        board = Board()
        board.clear()
        board.cells[(4, 5)] = BLACK
        board.cells[(8, 9)] = WHITE
        board.recompute_zhash()

        session = GameSession(
            config=GameConfig(
                mode=MODE_AVA,
                ai_depth=6,
                max_moves=80,
                white_ai_id="default",
            ),
            opening_seed=3,
        )
        session.board = board
        session.current_player = WHITE
        session.move_history = [{} for _ in range(78)]
        session.started = True

        result = session.apply_agent_move()

        self.assertTrue(result.get("ok"))
        self.assertEqual(result["search"]["depth"], 2)
        self.assertEqual(result["search"]["completed_depth"], 2)

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

    def test_direct_search_caps_depth_to_remaining_game_moves(self):
        board = Board()
        board.clear()
        board.cells[(4, 5)] = BLACK
        board.cells[(8, 9)] = WHITE
        board.recompute_zhash()

        result = choose_move_with_info(
            board,
            WHITE,
            agent=get_agent("default"),
            config=AgentConfig(depth=5, remaining_game_moves=2, is_opening_turn=False),
        )

        self.assertTrue(board.is_legal_move(result.move, WHITE))
        self.assertEqual(result.depth, 2)
        self.assertEqual(result.completed_depth, 2)

    def test_quiescence_does_not_extend_beyond_remaining_game_moves(self):
        board = Board.from_compact_token(
            "a2b,a3b,a4b,a5b,b1b,b2b,b4b,b5b,b6b,c3b,c4b,c6b,d3b,d4b,"
            "e3w,e5w,f3w,g9w,h4w,h5w,h6w,h7w,h8w,i5w,i6w,i7w,i8w,i9w|0-0"
        )
        quiet_agent = AgentDefinition(
            id="quiet",
            label="Quiet",
            owner="Test",
            evaluator=evaluate_board,
            default_depth=1,
            max_quiescence_depth=4,
        )

        with mock.patch("abalone.ai.minimax._quiescence", side_effect=AssertionError("quiescence should not run")):
            result = choose_move_with_info(
                board,
                WHITE,
                agent=quiet_agent,
                config=AgentConfig(depth=1, is_opening_turn=False, remaining_game_moves=1),
            )

        self.assertTrue(board.is_legal_move(result.move, WHITE))
        self.assertEqual(result.depth, 1)
        self.assertEqual(result.completed_depth, 1)

    def test_quiescence_depth_from_agent_definition_can_be_overridden_to_zero(self):
        board = Board.from_compact_token(
            "a2b,a3b,a4b,a5b,b1b,b2b,b4b,b5b,b6b,c3b,c4b,c6b,d3b,d4b,"
            "e3w,e5w,f3w,g9w,h4w,h5w,h6w,h7w,h8w,i5w,i6w,i7w,i8w,i9w|0-0"
        )
        plain_agent = AgentDefinition(
            id="plain",
            label="Plain",
            owner="Test",
            evaluator=evaluate_board,
            default_depth=1,
            max_quiescence_depth=0,
        )
        quiet_agent = AgentDefinition(
            id="quiet",
            label="Quiet",
            owner="Test",
            evaluator=evaluate_board,
            default_depth=1,
            max_quiescence_depth=4,
        )

        plain = choose_move_with_info(
            board,
            WHITE,
            agent=plain_agent,
            config=AgentConfig(depth=1, is_opening_turn=False),
        )
        quiet = choose_move_with_info(
            board,
            WHITE,
            agent=quiet_agent,
            config=AgentConfig(depth=1, is_opening_turn=False),
        )
        override_disabled = choose_move_with_info(
            board,
            WHITE,
            agent=quiet_agent,
            config=AgentConfig(depth=1, is_opening_turn=False, max_quiescence_depth=0),
        )

        self.assertTrue(board.is_legal_move(plain.move, WHITE))
        self.assertTrue(board.is_legal_move(quiet.move, WHITE))
        self.assertGreater(plain.score, quiet.score)
        self.assertGreater(quiet.nodes, plain.nodes)
        self.assertTrue(board.is_legal_move(override_disabled.move, WHITE))
        self.assertEqual(override_disabled.move.to_notation(), plain.move.to_notation())
        self.assertEqual(override_disabled.score, plain.score)

    def test_quiescence_stores_and_reuses_distinct_tt_entries(self):
        board = Board.from_compact_token(
            "a1b,a2b,a3b,a5b,b1b,b2b,b3b,b4b,b5b,b6b,c3b,d4b,d5b,e5b,"
            "f5w,g4w,g5w,g7w,h4w,h5w,h6w,h7w,h8w,h9w,i5w,i7w,i8w,i9w|0-0"
        )
        tt = {}
        q_key = _make_tt_key(board, WHITE, TT_MODE_QUIESCENCE)
        full_key = _make_tt_key(board, WHITE)

        first_stats = {"nodes": 0}
        first_score, first_move = _quiescence(
            board,
            WHITE,
            WHITE,
            4,
            None,
            -inf,
            inf,
            evaluate_board,
            "lexicographic",
            None,
            first_stats,
            tt,
        )
        second_stats = {"nodes": 0}
        second_score, second_move = _quiescence(
            board,
            WHITE,
            WHITE,
            4,
            None,
            -inf,
            inf,
            evaluate_board,
            "lexicographic",
            None,
            second_stats,
            tt,
        )

        self.assertIn(q_key, tt)
        self.assertNotEqual(q_key, full_key)
        self.assertNotIn(full_key, tt)
        if first_move is None:
            self.assertIsNone(second_move)
        else:
            self.assertEqual(first_move.to_notation(), second_move.to_notation())
        self.assertEqual(first_score, second_score)
        self.assertGreater(first_stats["nodes"], 1)
        self.assertEqual(second_stats["nodes"], 1)

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
        stderr_one = io.StringIO()
        with contextlib.redirect_stdout(stdout_one), contextlib.redirect_stderr(stderr_one):
            match_main(argv)

        stdout_two = io.StringIO()
        stderr_two = io.StringIO()
        with contextlib.redirect_stdout(stdout_two), contextlib.redirect_stderr(stderr_two):
            match_main(argv)

        output_one = stdout_one.getvalue()
        output_two = stdout_two.getvalue()
        normalized_one = re.sub(r"avg_move_ms=\d+\.\d+", "avg_move_ms=<var>", output_one)
        normalized_two = re.sub(r"avg_move_ms=\d+\.\d+", "avg_move_ms=<var>", output_two)
        self.assertEqual(normalized_one, normalized_two)
        self.assertIn("Completed 1 round(s), 2 game(s).", output_one)
        self.assertIn("default:", output_one)
        self.assertIn("kyle:", output_one)

    def test_match_summary_counts_partial_search_sources(self):
        fake_game = {
            "winner": BLACK,
            "reason": "max_moves",
            "history": [
                {
                    "agent_id": "default",
                    "duration_ms": 10,
                    "search": {"completed_depth": 0, "decision_source": "timeout_fallback_partial"},
                },
                {
                    "agent_id": "kyle",
                    "duration_ms": 12,
                    "search": {"completed_depth": 1, "decision_source": "search"},
                },
                {
                    "agent_id": "default",
                    "duration_ms": 11,
                    "search": {"completed_depth": 1, "decision_source": "timeout_partial"},
                },
            ],
            "score": {BLACK: 2, WHITE: 1},
            "black_ai_id": "default",
            "white_ai_id": "kyle",
        }

        with mock.patch("abalone.game.match._run_game", side_effect=[fake_game, fake_game]):
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                match_main(
                    [
                        "--black-ai", "default",
                        "--white-ai", "kyle",
                        "--rounds", "1",
                        "--depth", "1",
                    ]
                )

        output = stdout.getvalue()
        self.assertIn("partial_searches=4", output)
        self.assertIn("kyle: W=0 L=2 D=0 captures=2 partial_searches=0", output)

    def test_match_mode_updates_progress_display(self):
        first_game = {
            "winner": BLACK,
            "reason": "max_moves",
            "history": [],
            "score": {BLACK: 2, WHITE: 1},
            "black_ai_id": "default",
            "white_ai_id": "kyle",
        }
        second_game = {
            "winner": WHITE,
            "reason": "max_moves",
            "history": [],
            "score": {BLACK: 1, WHITE: 3},
            "black_ai_id": "kyle",
            "white_ai_id": "default",
        }
        progress = mock.Mock()

        with mock.patch("abalone.game.match.resolve_worker_count", return_value=1):
            with mock.patch("abalone.game.match._run_game", side_effect=[first_game, second_game]):
                with mock.patch("abalone.game.match._MatchProgressDisplay", return_value=progress) as display_cls:
                    stdout = io.StringIO()
                    with contextlib.redirect_stdout(stdout):
                        match_main(
                            [
                                "--black-ai", "default",
                                "--white-ai", "kyle",
                                "--rounds", "1",
                                "--depth", "1",
                            ]
                        )

        display_cls.assert_called_once_with(game_count=2, worker_count=1)
        progress.start.assert_called_once_with()
        progress.finish.assert_called_once_with()
        self.assertEqual(progress.update.call_count, 2)
        self.assertEqual(progress.update.call_args_list[0].args, (1, first_game))
        self.assertEqual(progress.update.call_args_list[1].args, (2, second_game))

    def test_match_mode_passes_jobs_to_parallel_scheduler(self):
        games = [
            {
                "index": 0,
                "winner": BLACK,
                "reason": "max_moves",
                "history": [],
                "score": {BLACK: 2, WHITE: 1},
                "black_ai_id": "default",
                "white_ai_id": "kyle",
            },
            {
                "index": 1,
                "winner": WHITE,
                "reason": "max_moves",
                "history": [],
                "score": {BLACK: 1, WHITE: 3},
                "black_ai_id": "kyle",
                "white_ai_id": "default",
            },
        ]

        with mock.patch("abalone.game.match.resolve_worker_count", return_value=2) as resolve_count:
            with mock.patch("abalone.game.match._run_scheduled_matches", return_value=games) as run_matches:
                stdout = io.StringIO()
                stderr = io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    match_main(
                        [
                            "--black-ai", "default",
                            "--white-ai", "kyle",
                            "--rounds", "1",
                            "--depth", "1",
                            "--jobs", "4",
                        ]
                    )

        resolve_count.assert_called_once_with(4, 2)
        scheduled_games, worker_count, _progress = run_matches.call_args.args
        self.assertEqual(worker_count, 2)
        self.assertEqual(len(scheduled_games), 2)
        self.assertEqual(scheduled_games[0]["black_ai_id"], "default")
        self.assertEqual(scheduled_games[0]["white_ai_id"], "kyle")
        self.assertEqual(scheduled_games[1]["black_ai_id"], "kyle")
        self.assertEqual(scheduled_games[1]["white_ai_id"], "default")


if __name__ == "__main__":
    unittest.main()
