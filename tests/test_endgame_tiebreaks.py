import contextlib
import io
import unittest

from abalone.game.board import BLACK, WHITE
from abalone.game.cli import Game
from abalone.game.config import GameConfig
from abalone.game.session import GameSession


class EndgameTiebreakTests(unittest.TestCase):
    def test_timeout_uses_least_total_time_as_tiebreak(self):
        session = GameSession()
        session.started = True
        session.board.score[BLACK] = 2
        session.board.score[WHITE] = 2
        session.time_left_us = {BLACK: 0, WHITE: 0}
        session.time_used_us = {BLACK: 1_000, WHITE: 2_000}

        status = session.status()

        self.assertTrue(status["game_over"])
        self.assertEqual(status["game_over_reason"], "timeout")
        self.assertEqual(status["winner"], BLACK)
        self.assertEqual(status["winner_tiebreak"], "least_total_time")
        self.assertEqual(status["time_used_ms"][BLACK], 1)
        self.assertEqual(status["time_used_ms"][WHITE], 2)

    def test_max_moves_uses_least_total_time_as_tiebreak(self):
        session = GameSession(config=GameConfig(max_moves=3))
        session.started = True
        session.board.score[BLACK] = 1
        session.board.score[WHITE] = 1
        session.move_history = [{}, {}, {}]
        session.time_left_us = {BLACK: 5_000, WHITE: 5_000}
        session.time_used_us = {BLACK: 9_000, WHITE: 15_000}

        status = session.status()

        self.assertTrue(status["game_over"])
        self.assertEqual(status["game_over_reason"], "max_moves")
        self.assertEqual(status["winner"], BLACK)
        self.assertEqual(status["winner_tiebreak"], "least_total_time")

    def test_exact_total_time_tie_remains_draw(self):
        session = GameSession(config=GameConfig(max_moves=2))
        session.started = True
        session.board.score[BLACK] = 3
        session.board.score[WHITE] = 3
        session.move_history = [{}, {}]
        session.time_left_us = {BLACK: 2_000, WHITE: 2_000}
        session.time_used_us = {BLACK: 12_000, WHITE: 12_000}

        status = session.status()

        self.assertTrue(status["game_over"])
        self.assertIsNone(status["winner"])
        self.assertIsNone(status["winner_tiebreak"])

    def test_state_json_exposes_winner_tiebreak_and_time_used(self):
        session = GameSession()
        session.started = True
        session.board.score[BLACK] = 0
        session.board.score[WHITE] = 0
        session.time_left_us = {BLACK: 0, WHITE: 0}
        session.time_used_us = {BLACK: 4_000, WHITE: 8_000}

        state = session.state_json()

        self.assertEqual(state["winner"], BLACK)
        self.assertEqual(state["winner_tiebreak"], "least_total_time")
        self.assertEqual(state["time_used_ms"][BLACK], 4)
        self.assertEqual(state["time_used_ms"][WHITE], 8)

    def test_undo_restores_time_used_snapshot(self):
        session = GameSession(config=GameConfig(mode="hvh"))
        session.reset()
        payload = session.state_json()["legal_moves"][0]

        anchor = session._now_us() + 1_000_000
        session.time_left_us = {BLACK: 30_000_000, WHITE: 30_000_000}
        session.time_used_us = {BLACK: 123_000, WHITE: 456_000}
        session.last_clock_update_us = anchor
        session.turn_start_us = anchor

        result = session.apply_human_move(payload)

        self.assertTrue(result.get("ok"))
        snapshot = dict(session.move_history[-1]["time_used_snapshot"])

        session.time_used_us = {BLACK: 999_999, WHITE: 888_888}
        undo_result = session.undo()

        self.assertTrue(undo_result.get("ok"))
        self.assertEqual(session.time_used_us, snapshot)

    def test_cli_timeout_tiebreak_message_mentions_total_time(self):
        game = Game()
        game.board.score[BLACK] = 2
        game.board.score[WHITE] = 2
        status = {
            "winner": BLACK,
            "game_over_reason": "timeout",
            "winner_tiebreak": "least_total_time",
            "time_used_ms": {BLACK: 61_000, WHITE: 79_000},
        }

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            game._print_game_over(status)

        output = stdout.getvalue()
        self.assertIn("wins on total-time tiebreak", output)
        self.assertIn("Total time used", output)
        self.assertIn("01:01", output)
        self.assertIn("01:19", output)

    def test_cli_max_moves_tiebreak_message_mentions_total_time(self):
        game = Game()
        game.board.score[BLACK] = 1
        game.board.score[WHITE] = 1
        status = {
            "winner": WHITE,
            "game_over_reason": "max_moves",
            "winner_tiebreak": "least_total_time",
            "time_used_ms": {BLACK: 90_000, WHITE: 45_000},
        }

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            game._print_game_over(status)

        output = stdout.getvalue()
        self.assertIn("wins on total-time tiebreak", output)
        self.assertIn("max moves reached", output.lower())
        self.assertIn("Total time used", output)


if __name__ == "__main__":
    unittest.main()
