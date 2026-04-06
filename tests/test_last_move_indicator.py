import unittest

from abalone import native
from abalone.game.board import is_valid, neighbor, pos_to_str, str_to_pos
from abalone.game.session import GameSession

if not native.is_available():
    raise unittest.SkipTest("native extension not built")


class LastMoveIndicatorTests(unittest.TestCase):
    """Regression checks for `last_move_marbles` API serialization."""

    def test_last_move_marbles_empty_before_any_move(self):
        session = GameSession()
        state = session.state_json()
        self.assertEqual(state["last_move_marbles"], [])
        self.assertEqual(state["last_move_direction"], [])

    def test_last_move_marbles_match_changed_positions(self):
        session = GameSession()
        state = session.state_json()
        move_payload = state["legal_moves"][0]

        result = session.apply_human_move(move_payload)
        self.assertTrue(result.get("ok"))

        dr, dc = move_payload["direction"]
        direction = (dr, dc)
        expected = set()
        for marble in move_payload["marbles"]:
            dest = neighbor(str_to_pos(marble), direction)
            if is_valid(dest):
                expected.add(pos_to_str(dest))

        for pushed in result["result"]["pushed"]:
            dest = neighbor(str_to_pos(pushed), direction)
            if is_valid(dest):
                expected.add(pos_to_str(dest))

        next_state = session.state_json()
        self.assertEqual(set(next_state["last_move_marbles"]), expected)
        self.assertEqual(next_state["last_move_direction"], move_payload["direction"])

    def test_undo_clears_last_move_marbles_when_history_is_empty(self):
        session = GameSession()
        move_payload = session.state_json()["legal_moves"][0]
        self.assertTrue(session.apply_human_move(move_payload).get("ok"))

        undo_result = session.undo()
        self.assertTrue(undo_result.get("ok"))

        state = session.state_json()
        self.assertEqual(state["last_move_marbles"], [])
        self.assertEqual(state["last_move_direction"], [])


if __name__ == "__main__":
    unittest.main()
