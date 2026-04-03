import unittest

from abalone import native
from abalone.game.config import GameConfig, MODE_HVA
from abalone.game.session import GameSession

if not native.is_available():
    raise unittest.SkipTest("native extension not built")


class SessionModeTests(unittest.TestCase):
    """Regression checks for session controller turn-gating."""

    def test_hva_turn_gating(self):
        """Human-vs-AI mode should enforce controller ownership of each turn."""
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
