import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from abalone.ai.heuristics import DEFAULT_WEIGHTS, FEATURE_ORDER, WEIGHT_TUNING_RULES
from abalone.eval import gauntlet
from abalone.game import duel
from abalone.game.board import BLACK, WHITE


class _FakeFuture:
    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error

    def result(self):
        if self._error is not None:
            raise self._error
        return self._result

    def cancel(self):
        return True


class _FakeExecutor:
    def __init__(self, max_workers):
        self.max_workers = max_workers

    def submit(self, fn, *args):
        return _FakeFuture(result=fn(*args))

    def shutdown(self, wait=True, cancel_futures=False):
        del wait, cancel_futures


def _agents(*agent_ids):
    return [SimpleNamespace(id=agent_id) for agent_id in agent_ids]

def _workspace_temp_dir(name):
    path = Path('tests') / '_tmp' / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _feature_snapshot(**overrides):
    features = {
        "marble": -1.0,
        "center": -4.5,
        "cohesion": -1.0,
        "cluster": -1.0,
        "edge_pressure": -2.0,
        "formation": -1.0,
        "push": -1.0,
        "mobility": -1.5,
        "stability": -1.0,
    }
    features.update(overrides)
    return features


def _build_game(job, target_agent_id, outcome, features=None, decision_source="timeout_partial", completed_depth=1, depth=3):
    features = features or _feature_snapshot()
    target_is_black = job["black_ai_id"] == target_agent_id
    target_color = BLACK if target_is_black else WHITE
    opponent_color = WHITE if target_color == BLACK else BLACK

    if outcome == "win":
        winner_ai_id = target_agent_id
        target_score = 2
        opponent_score = 0
    elif outcome == "draw":
        winner_ai_id = None
        target_score = 1
        opponent_score = 1
    else:
        winner_ai_id = job["white_ai_id"] if target_is_black else job["black_ai_id"]
        target_score = 0
        opponent_score = 2

    score = {BLACK: 0, WHITE: 0}
    score[target_color] = target_score
    score[opponent_color] = opponent_score

    return {
        "index": job["index"],
        "black_ai_id": job["black_ai_id"],
        "white_ai_id": job["white_ai_id"],
        "winner_ai_id": winner_ai_id,
        "winner_tiebreak": None,
        "game_over_reason": "max_moves",
        "time_used_ms": {BLACK: 500, WHITE: 700},
        "score": score,
        "moves": 8,
        "duration_s": 0.05,
        "agent_color": job.get("agent_color"),
        "schedule_split": job.get("schedule_split", "train"),
        "history": [
            {
                "agent_id": target_agent_id,
                "duration_ms": 50,
                "notation": "1:a1a2",
                "search": {
                    "decision_source": decision_source,
                    "completed_depth": completed_depth,
                    "depth": depth,
                    "pre_move": {
                        "features": features,
                        "total": -10.0,
                    },
                    "post_move": {
                        "features": features,
                        "total": -10.0,
                    },
                    "board_token_before": "",
                    "board_token_after": "",
                },
            }
        ],
    }


def _deterministic_runner(target_agent_id, baseline_weights):
    center_threshold = baseline_weights["center"] * 1.05

    def runner(scheduled_games, worker_count, agent_weight_overrides, telemetry_agent_ids, on_game_complete, on_warning):
        del worker_count, telemetry_agent_ids, on_warning
        center_weight = agent_weight_overrides[target_agent_id]["center"]
        games = []
        for completed, job in enumerate(scheduled_games, start=1):
            if center_weight >= center_threshold:
                outcome = "win" if job["index"] % 2 == 0 else "draw"
            else:
                outcome = "loss" if job["index"] % 2 == 0 else "draw"
            game = _build_game(job, target_agent_id, outcome)
            games.append(game)
            on_game_complete(completed, len(scheduled_games), game)
        return games

    return runner


class DuelTuningCliTests(unittest.TestCase):
    def test_tune_requires_all_opponents(self):
        parser = duel._build_parser()
        args = parser.parse_args(["--tune", "--agent", "default", "--iterations", "2"])
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            duel._validate_args(parser, args)

    def test_resume_requires_tune(self):
        parser = duel._build_parser()
        args = parser.parse_args(["--resume-from", "checkpoint.json"])
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            duel._validate_args(parser, args)

    def test_main_forwards_tuning_arguments(self):
        agents = _agents("target", "opponent")
        stdout = io.StringIO()
        with mock.patch.object(duel, "list_agents", return_value=agents), mock.patch.object(
            duel,
            "get_agent",
            side_effect=lambda agent_id: SimpleNamespace(id=agent_id),
        ), mock.patch.object(
            duel,
            "get_agent_weights",
            return_value=DEFAULT_WEIGHTS,
        ), mock.patch.object(
            duel.eval_gauntlet,
            "run_tuning_loop",
            return_value=Path("abalone/eval_runs/target/checkpoint.json"),
        ) as run_tuning_loop, contextlib.redirect_stdout(stdout):
            duel.main(
                [
                    "--tune",
                    "--all-opponents",
                    "--agent",
                    "target",
                    "--iterations",
                    "3",
                    "--depth",
                    "2",
                    "--layout",
                    "german_daisy",
                    "--move-time-s",
                    "7",
                    "--max-moves",
                    "40",
                    "--seed",
                    "13",
                    "--jobs",
                    "1",
                ]
            )

        run_tuning_loop.assert_called_once()
        kwargs = run_tuning_loop.call_args.kwargs
        self.assertEqual(kwargs["agent_id"], "target")
        self.assertEqual(kwargs["iterations"], 3)
        self.assertEqual(kwargs["depth"], 2)
        self.assertEqual(kwargs["layout"], "german_daisy")
        self.assertEqual(kwargs["move_time_s"], 7)
        self.assertEqual(kwargs["max_moves"], 40)
        self.assertEqual(kwargs["seed"], 13)
        self.assertEqual(kwargs["jobs"], 1)
        self.assertIsNone(kwargs["resume_from"])
        self.assertEqual(stdout.getvalue(), "")

    def test_run_tuning_uses_iteration_prefix_for_live_progress(self):
        captured_prefixes = []

        class FakeProgress:
            def __init__(self, agent_id, game_count, worker_count, stream=None, refresh_s=0.12, prefix="duel"):
                del agent_id, game_count, worker_count, stream, refresh_s
                captured_prefixes.append(prefix)

            def start(self):
                return None

            def update(self, completed, game):
                del completed, game
                return None

            def warn_text(self, text):
                del text
                return None

            def restart_serial(self):
                return None

            def finish(self):
                return None

        def fake_run_tuning_loop(**kwargs):
            kwargs["run_gauntlet_iteration"](
                scheduled_games=[{"index": 0}],
                worker_count=2,
                agent_weight_overrides={"target": DEFAULT_WEIGHTS},
                telemetry_agent_ids=["target"],
                on_game_complete=lambda completed, total, game: None,
                on_warning=lambda message: None,
                iteration_index=5,
                total_iterations=20,
            )
            return Path("abalone/eval_runs/target/checkpoint.json")

        args = SimpleNamespace(
            agent="target",
            iterations=20,
            depth=3,
            layout="belgian_daisy",
            move_time_s=30,
            max_moves=80,
            seed=0,
            jobs=10,
            resume_from=None,
        )

        with mock.patch.object(duel, "_GauntletProgressDisplay", FakeProgress), mock.patch.object(
            duel.eval_gauntlet,
            "run_scheduled_games",
            return_value=[],
        ), mock.patch.object(
            duel.eval_gauntlet,
            "run_tuning_loop",
            side_effect=fake_run_tuning_loop,
        ):
            duel._run_tuning(args)

        self.assertEqual(captured_prefixes, ["Iteration 5/20"])


class AdaptiveGauntletTests(unittest.TestCase):
    def test_parallel_run_scheduled_games_sorts_results_and_forwards_overrides(self):
        scheduled_games = [
            {"index": 0},
            {"index": 1},
            {"index": 2},
        ]

        with mock.patch.object(gauntlet, "ProcessPoolExecutor", _FakeExecutor), mock.patch.object(
            gauntlet,
            "as_completed",
            side_effect=lambda futures: iter(reversed(futures)),
        ), mock.patch.object(
            gauntlet,
            "run_scheduled_game",
            side_effect=lambda job, agent_weight_overrides, telemetry_agent_ids: {
                "index": job["index"],
                "override_center": agent_weight_overrides["target"]["center"],
                "telemetry_ids": list(telemetry_agent_ids),
            },
        ):
            games = gauntlet.run_scheduled_games(
                scheduled_games,
                worker_count=2,
                agent_weight_overrides={"target": {"center": 99.0}},
                telemetry_agent_ids=["target"],
            )

        self.assertEqual([game["index"] for game in games], [0, 1, 2])
        self.assertEqual([game["override_center"] for game in games], [99.0, 99.0, 99.0])
        self.assertEqual(games[0]["telemetry_ids"], ["target"])

    def test_analyze_match_result_detects_expected_reasons(self):
        game = {
            "index": 0,
            "black_ai_id": "target",
            "white_ai_id": "opponent",
            "winner_ai_id": "opponent",
            "score": {BLACK: 0, WHITE: 2},
            "history": [
                {
                    "agent_id": "target",
                    "duration_ms": 75,
                    "search": {
                        "decision_source": "timeout_partial",
                        "completed_depth": 1,
                        "depth": 3,
                        "pre_move": {
                            "features": _feature_snapshot(),
                            "total": -10.0,
                        },
                        "post_move": {
                            "features": _feature_snapshot(),
                            "total": -10.0,
                        },
                    },
                }
            ],
        }

        analysis = gauntlet.analyze_match_result(game, "target")
        codes = [reason["code"] for reason in analysis["reasons"]]

        self.assertIn("edge_exposure", codes)
        self.assertIn("poor_center_control", codes)
        self.assertIn("material_loss", codes)
        self.assertIn("search_instability", codes)
        self.assertEqual(analysis["search_summary"]["partial_searches"], 1)
        self.assertEqual(analysis["outcome"], "loss")
        self.assertTrue(analysis["drives_tuning"])

    def test_analyze_match_result_ignores_wins_for_loss_attribution(self):
        job = {
            "index": 0,
            "black_ai_id": "target",
            "white_ai_id": "opponent",
            "schedule_split": "train",
            "agent_color": "black",
        }
        game = _build_game(job, "target", "win", features=_feature_snapshot(center=-20.0, mobility=-8.0))

        analysis = gauntlet.analyze_match_result(game, "target")

        self.assertEqual(analysis["outcome"], "win")
        self.assertFalse(analysis["drives_tuning"])
        self.assertEqual(analysis["reasons"], [])

    def test_analyze_match_result_marks_late_material_draw_as_weak(self):
        job = {
            "index": 0,
            "black_ai_id": "target",
            "white_ai_id": "opponent",
            "schedule_split": "train",
            "agent_color": "black",
        }
        game = _build_game(job, "target", "draw", features=_feature_snapshot(marble=-2.0, center=-3.0))

        analysis = gauntlet.analyze_match_result(game, "target")

        self.assertEqual(analysis["outcome"], "draw")
        self.assertTrue(analysis["weak_draw"])
        self.assertTrue(analysis["drives_tuning"])

    def test_propose_next_weights_biases_expected_features_and_clamps(self):
        proposal = gauntlet.propose_next_weights(
            baseline_weights=DEFAULT_WEIGHTS,
            current_weights=DEFAULT_WEIGHTS,
            best_weights=DEFAULT_WEIGHTS,
            top_reasons=[
                {"code": "edge_exposure", "weighted_score": 12.0, "occurrences": 2},
                {"code": "search_instability", "weighted_score": 12.0, "occurrences": 2},
            ],
            feature_delta_summary=None,
            iteration_index=4,
            total_iterations=8,
            stagnation_count=0,
            seed=7,
        )

        self.assertGreater(proposal["weights"]["edge_pressure"], DEFAULT_WEIGHTS["edge_pressure"])
        self.assertGreater(proposal["weights"]["stability"], DEFAULT_WEIGHTS["stability"])
        self.assertLess(proposal["weights"]["push"], DEFAULT_WEIGHTS["push"])

        extreme_weights = {key: DEFAULT_WEIGHTS[key] * 100.0 for key in FEATURE_ORDER}
        clamped = gauntlet.propose_next_weights(
            baseline_weights=DEFAULT_WEIGHTS,
            current_weights=extreme_weights,
            best_weights=extreme_weights,
            top_reasons=[],
            feature_delta_summary={"center": 999.0},
            iteration_index=1,
            total_iterations=2,
            stagnation_count=0,
            seed=11,
        )

        for key in FEATURE_ORDER:
            self.assertLessEqual(
                clamped["weights"][key],
                DEFAULT_WEIGHTS[key] * WEIGHT_TUNING_RULES[key]["max_multiplier"],
            )

    def test_is_better_score_requires_training_improvement_and_validation_non_regression(self):
        incumbent = {
            "training": {"match_points": 4},
            "validation": {
                "match_points": 2,
                "capture_diff": 1,
                "partial_searches": 0,
                "avg_completed_depth": 2.0,
                "avg_move_ms": 80.0,
            },
        }
        better = {
            "training": {"match_points": 5},
            "validation": {
                "match_points": 2,
                "capture_diff": 1,
                "partial_searches": 0,
                "avg_completed_depth": 2.0,
                "avg_move_ms": 80.0,
            },
        }
        worse_validation = {
            "training": {"match_points": 5},
            "validation": {
                "match_points": 1,
                "capture_diff": 1,
                "partial_searches": 0,
                "avg_completed_depth": 2.0,
                "avg_move_ms": 80.0,
            },
        }
        tie_training = {
            "training": {"match_points": 4},
            "validation": {
                "match_points": 3,
                "capture_diff": 2,
                "partial_searches": 0,
                "avg_completed_depth": 2.0,
                "avg_move_ms": 80.0,
            },
        }

        self.assertTrue(gauntlet.is_better_score(better, incumbent))
        self.assertFalse(gauntlet.is_better_score(worse_validation, incumbent))
        self.assertFalse(gauntlet.is_better_score(tie_training, incumbent))

    def test_run_tuning_loop_is_deterministic_with_fixed_seed(self):
        agents = _agents("target", "opponent")
        runner = _deterministic_runner("target", DEFAULT_WEIGHTS)
        announcements_one = []
        announcements_two = []

        temp_one = _workspace_temp_dir("deterministic-one")
        temp_two = _workspace_temp_dir("deterministic-two")
        try:
            with mock.patch.object(
                gauntlet,
                "list_agents",
                return_value=agents,
            ), mock.patch.object(
                gauntlet,
                "get_agent_weights",
                return_value=DEFAULT_WEIGHTS,
            ):
                checkpoint_one = gauntlet.run_tuning_loop(
                    agent_id="target",
                    iterations=2,
                    depth=2,
                    layout="standard",
                    move_time_s=5,
                    max_moves=20,
                    seed=17,
                    jobs=1,
                    runs_root=Path(temp_one),
                    run_gauntlet_iteration=runner,
                    announce=announcements_one.append,
                )
                checkpoint_two = gauntlet.run_tuning_loop(
                    agent_id="target",
                    iterations=2,
                    depth=2,
                    layout="standard",
                    move_time_s=5,
                    max_moves=20,
                    seed=17,
                    jobs=1,
                    runs_root=Path(temp_two),
                    run_gauntlet_iteration=runner,
                    announce=announcements_two.append,
                )

                state_one = json.loads(Path(checkpoint_one).read_text(encoding="utf-8"))
                state_two = json.loads(Path(checkpoint_two).read_text(encoding="utf-8"))
                iterations_one = Path(checkpoint_one).with_name("iterations.jsonl").read_text(encoding="utf-8")
                iterations_two = Path(checkpoint_two).with_name("iterations.jsonl").read_text(encoding="utf-8")
                self.assertIn("critical_positions", state_one["paths"])

            for state in (state_one, state_two):
                state.pop("created_at", None)
                state.pop("updated_at", None)
                state.pop("paths", None)
                state.pop("timing", None)
                final_summary = state.get("final_summary")
                if final_summary is not None:
                    final_summary.pop("total_time_s", None)
            self.assertEqual(state_one, state_two)
            self.assertEqual(iterations_one, iterations_two)
            self.assertEqual(state_one["status"], "completed")
            self.assertEqual(state_one["progress"]["completed_iterations"], 2)
            self.assertEqual(state_one["final_summary"]["best_win_rate_iteration"], 2)
            self.assertEqual(state_one["final_summary"]["best_win_rate_iterations"], [2])
            self.assertGreater(state_one["final_summary"]["average_win_rate"], 0.0)
            self.assertEqual(state_one["version"], 2)
            self.assertIn("analysis_version", state_one)
            self.assertIn("[Iteration 1/2] W=0 D=3 L=3 capture_diff=-6 partial_searches=6 accepted.", announcements_one)
            self.assertIn("[Iteration 2/2] W=3 D=3 L=0 capture_diff=6 partial_searches=6 accepted.", announcements_one)
            self.assertIn("Final summary:", announcements_one)
            self.assertTrue(any(message.startswith("Time elapsed: ") for message in announcements_one))
            self.assertIn("Iterations completed: 2", announcements_one)
            self.assertIn("Layout: standard", announcements_one)
            self.assertIn("Agent: target", announcements_one)
            self.assertIn("Max moves: 20", announcements_one)
            self.assertIn("Average win rate 25.00%", announcements_one)
            self.assertIn("Best win rate 50.00% (Iteration(s): 2)", announcements_one)
            self.assertIn("Best heuristic:", announcements_one)
            self.assertTrue(any(message.startswith("center ") for message in announcements_one))
            self.assertFalse(any("checkpoint" in message.lower() for message in announcements_one))
            self.assertFalse(any("prepared iteration" in message.lower() for message in announcements_one))
            self.assertEqual(len(announcements_one), len(announcements_two))
        finally:
            shutil.rmtree(temp_one, ignore_errors=True)
            shutil.rmtree(temp_two, ignore_errors=True)

    def test_interrupt_and_resume_preserve_checkpoint_state(self):
        agents = _agents("target", "opponent")
        seen_running_state = []

        def interrupting_runner(scheduled_games, worker_count, agent_weight_overrides, telemetry_agent_ids, on_game_complete, on_warning):
            del worker_count, agent_weight_overrides, telemetry_agent_ids, on_warning
            checkpoints = list(Path(temp_dir).glob("**/checkpoint.json"))
            self.assertEqual(len(checkpoints), 1)
            state = json.loads(checkpoints[0].read_text(encoding="utf-8"))
            seen_running_state.append(state)
            self.assertEqual(state["status"], "running")
            self.assertEqual(state["progress"]["current_iteration"], 1)
            self.assertEqual(state["active_iteration"]["status"], "running")
            game = _build_game(scheduled_games[0], "target", "loss")
            on_game_complete(1, len(scheduled_games), game)
            raise KeyboardInterrupt

        temp_dir = _workspace_temp_dir("interrupt-resume")
        try:
            with mock.patch.object(
                gauntlet,
                "list_agents",
                return_value=agents,
            ), mock.patch.object(
                gauntlet,
                "get_agent_weights",
                return_value=DEFAULT_WEIGHTS,
            ):
                checkpoint = gauntlet.run_tuning_loop(
                    agent_id="target",
                    iterations=2,
                    depth=1,
                    layout="standard",
                    move_time_s=5,
                    max_moves=20,
                    seed=3,
                    jobs=1,
                    runs_root=Path(temp_dir),
                    run_gauntlet_iteration=interrupting_runner,
                    announce=lambda *_args, **_kwargs: None,
                )

                interrupted_state = json.loads(Path(checkpoint).read_text(encoding="utf-8"))
                self.assertTrue(seen_running_state)
                self.assertEqual(interrupted_state["status"], "interrupted")
                self.assertEqual(interrupted_state["active_iteration"]["status"], "interrupted")
                self.assertEqual(interrupted_state["progress"]["current_iteration_matches"], 1)
                self.assertEqual(len(Path(checkpoint).with_name("matches.jsonl").read_text(encoding="utf-8").splitlines()), 1)

                resumed_checkpoint = gauntlet.run_tuning_loop(
                    agent_id=None,
                    iterations=2,
                    depth=1,
                    layout="standard",
                    move_time_s=5,
                    max_moves=20,
                    seed=3,
                    jobs=1,
                    resume_from=str(checkpoint),
                    runs_root=Path(temp_dir),
                    run_gauntlet_iteration=_deterministic_runner("target", DEFAULT_WEIGHTS),
                    announce=lambda *_args, **_kwargs: None,
                )

                resumed_state = json.loads(Path(resumed_checkpoint).read_text(encoding="utf-8"))

                self.assertEqual(Path(resumed_checkpoint), Path(checkpoint))
                self.assertEqual(resumed_state["status"], "completed")
                self.assertEqual(resumed_state["progress"]["completed_iterations"], 2)
                self.assertIsNone(resumed_state["active_iteration"])
                self.assertIn("final_summary", resumed_state)
                self.assertEqual(resumed_state["final_summary"]["best_win_rate_iteration"], 2)
                self.assertEqual(resumed_state["final_summary"]["best_win_rate_iterations"], [2])
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_resume_rejects_v1_checkpoint(self):
        temp_dir = _workspace_temp_dir("resume-v1")
        try:
            checkpoint = Path(temp_dir) / "checkpoint.json"
            checkpoint.write_text(json.dumps({"version": 1, "config": {"iterations": 1}}), encoding="utf-8")

            with self.assertRaises(ValueError):
                gauntlet.run_tuning_loop(
                    agent_id=None,
                    iterations=1,
                    depth=1,
                    layout="standard",
                    move_time_s=5,
                    max_moves=20,
                    seed=1,
                    jobs=1,
                    resume_from=str(checkpoint),
                    runs_root=Path(temp_dir),
                    announce=lambda *_args, **_kwargs: None,
                )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_tuning_run_does_not_mutate_source_heuristics_file(self):
        agents = _agents("target", "opponent")
        heuristics_path = Path("abalone/ai/heuristics.py")
        before = heuristics_path.read_text(encoding="utf-8")

        temp_dir = _workspace_temp_dir("no-mutate")
        try:
            with mock.patch.object(
                gauntlet,
                "list_agents",
                return_value=agents,
            ), mock.patch.object(
                gauntlet,
                "get_agent_weights",
                return_value=DEFAULT_WEIGHTS,
            ):
                gauntlet.run_tuning_loop(
                    agent_id="target",
                    iterations=1,
                    depth=1,
                    layout="standard",
                    move_time_s=5,
                    max_moves=20,
                    seed=5,
                    jobs=1,
                    runs_root=Path(temp_dir),
                    run_gauntlet_iteration=_deterministic_runner("target", DEFAULT_WEIGHTS),
                    announce=lambda *_args, **_kwargs: None,
                )

                after = heuristics_path.read_text(encoding="utf-8")
                self.assertEqual(before, after)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()


