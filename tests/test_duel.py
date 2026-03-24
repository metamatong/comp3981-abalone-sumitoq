import contextlib
import io
import unittest
from unittest import mock

from abalone.game import duel


class _FakeExecutor:
    def __init__(self, max_workers):
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, fn, job):
        return _FakeFuture(fn(job))


class _FakeFuture:
    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error

    def result(self):
        if self._error is not None:
            raise self._error
        return self._result


class _BrokenExecutor(_FakeExecutor):
    def submit(self, fn, job):
        return _FakeFuture(error=duel.BrokenProcessPool("pool broke"))


class DuelTests(unittest.TestCase):
    def test_resolve_worker_count_defaults_to_cpu_count_and_caps_to_game_count(self):
        with mock.patch("abalone.game.duel.os.cpu_count", return_value=8):
            self.assertEqual(duel._resolve_worker_count(None, 3), 3)
        self.assertEqual(duel._resolve_worker_count(1, 5), 1)

    def test_run_all_opponents_games_sorts_parallel_results_back_to_schedule_order(self):
        scheduled_jobs = [
            {"index": 0},
            {"index": 1},
            {"index": 2},
        ]

        with mock.patch.object(duel, "_build_all_opponents_jobs", return_value=scheduled_jobs), mock.patch.object(
            duel,
            "_run_all_opponents_game",
            side_effect=lambda job: {"index": job["index"]},
        ), mock.patch.object(duel, "ProcessPoolExecutor", _FakeExecutor), mock.patch.object(
            duel,
            "as_completed",
            side_effect=lambda futures: list(reversed(futures)),
        ), mock.patch.object(duel, "_print_gauntlet_start"), mock.patch.object(duel, "_print_gauntlet_progress"):
            games = duel._run_all_opponents_games(
                agent_id="default",
                depth=1,
                layout="standard",
                move_time_s=5,
                max_moves=10,
                seed=3,
                jobs=2,
            )

        self.assertEqual([game["index"] for game in games], [0, 1, 2])

    def test_run_all_opponents_games_falls_back_to_serial_if_process_pool_is_unavailable(self):
        scheduled_jobs = [
            {"index": 0},
            {"index": 1},
        ]

        stderr = io.StringIO()
        with mock.patch.object(duel, "_build_all_opponents_jobs", return_value=scheduled_jobs), mock.patch.object(
            duel,
            "_run_all_opponents_game",
            side_effect=lambda job: {"index": job["index"]},
        ), mock.patch.object(duel, "ProcessPoolExecutor", side_effect=PermissionError("blocked")), mock.patch.object(
            duel,
            "_print_gauntlet_progress",
        ), contextlib.redirect_stderr(
            stderr
        ):
            games = duel._run_all_opponents_games(
                agent_id="default",
                depth=1,
                layout="standard",
                move_time_s=5,
                max_moves=10,
                seed=3,
                jobs=2,
            )

        self.assertEqual([game["index"] for game in games], [0, 1])
        self.assertIn("Continuing serially.", stderr.getvalue())

    def test_run_all_opponents_games_warns_and_falls_back_if_process_pool_breaks_during_map(self):
        scheduled_jobs = [
            {"index": 0},
            {"index": 1},
        ]

        stderr = io.StringIO()
        with mock.patch.object(duel, "_build_all_opponents_jobs", return_value=scheduled_jobs), mock.patch.object(
            duel,
            "_run_all_opponents_game",
            side_effect=lambda job: {"index": job["index"]},
        ), mock.patch.object(duel, "ProcessPoolExecutor", _BrokenExecutor), mock.patch.object(
            duel,
            "as_completed",
            side_effect=lambda futures: futures,
        ), mock.patch.object(duel, "_print_gauntlet_progress"), contextlib.redirect_stderr(stderr):
            games = duel._run_all_opponents_games(
                agent_id="default",
                depth=1,
                layout="standard",
                move_time_s=5,
                max_moves=10,
                seed=3,
                jobs=2,
            )

        self.assertEqual([game["index"] for game in games], [0, 1])
        self.assertIn("BrokenProcessPool", stderr.getvalue())
        self.assertIn("Restarting gauntlet serially", stderr.getvalue())

    def test_run_all_opponents_games_prints_progress_in_serial_mode(self):
        scheduled_jobs = [
            {"index": 0, "black_ai_id": "default", "white_ai_id": "kyle"},
            {"index": 1, "black_ai_id": "kyle", "white_ai_id": "default"},
        ]

        stderr = io.StringIO()
        with mock.patch.object(duel, "_build_all_opponents_jobs", return_value=scheduled_jobs), mock.patch.object(
            duel,
            "_run_all_opponents_game",
            side_effect=lambda job: {
                "index": job["index"],
                "black_ai_id": job["black_ai_id"],
                "white_ai_id": job["white_ai_id"],
                "winner_ai_id": None,
                "moves": 6,
                "duration_s": 0.1,
                "agent_color": "black",
            },
        ), contextlib.redirect_stderr(stderr):
            duel._run_all_opponents_games(
                agent_id="default",
                depth=1,
                layout="standard",
                move_time_s=5,
                max_moves=10,
                seed=3,
                jobs=1,
            )

        output = stderr.getvalue()
        self.assertIn("Starting gauntlet for default", output)
        self.assertIn("1/2", output)
        self.assertIn("2/2", output)

    def test_main_forwards_jobs_for_all_opponents_mode(self):
        with mock.patch.object(duel, "_run_all_opponents_games", return_value=[]) as run_games, mock.patch.object(
            duel,
            "_print_all_opponents_report",
        ):
            duel.main(["--agent", "default", "--all-opponents", "--jobs", "3"])

        run_games.assert_called_once_with(
            agent_id="default",
            depth=None,
            layout="standard",
            move_time_s=30,
            max_moves=500,
            seed=0,
            jobs=3,
        )


if __name__ == "__main__":
    unittest.main()
