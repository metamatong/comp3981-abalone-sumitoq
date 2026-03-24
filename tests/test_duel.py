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

    def map(self, fn, jobs):
        scheduled = list(jobs)
        return [fn(job) for job in reversed(scheduled)]


class _BrokenMapExecutor(_FakeExecutor):
    def map(self, fn, jobs):
        raise duel.BrokenProcessPool("pool broke")


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
        ), mock.patch.object(duel, "ProcessPoolExecutor", _FakeExecutor):
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
        ), mock.patch.object(duel, "ProcessPoolExecutor", side_effect=PermissionError("blocked")), contextlib.redirect_stderr(
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
        ), mock.patch.object(duel, "ProcessPoolExecutor", _BrokenMapExecutor), contextlib.redirect_stderr(stderr):
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
