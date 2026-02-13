import unittest

from abalone.game.config import GameConfig, MODE_HVA
from abalone.game.session import GameSession


class SessionModeTests(unittest.TestCase):
    def test_hva_turn_gating(self):
        session = GameSession(config=GameConfig(mode=MODE_HVA, human_side=1, ai_depth=1))

        state = session.state_json()
        self.assertEqual(state["current_controller"], "human")
        self.assertTrue(state["legal_moves"])

        human_move = state["legal_moves"][0]
        human_result = session.apply_human_move(human_move)
        self.assertTrue(human_result.get("ok"))

        blocked = session.apply_human_move(human_move)
        self.assertIn("AI-controlled", blocked.get("error", ""))

        ai_result = session.apply_agent_move()
        self.assertTrue(ai_result.get("ok"))
        self.assertEqual(ai_result.get("source"), "ai")


if __name__ == "__main__":
    unittest.main()
