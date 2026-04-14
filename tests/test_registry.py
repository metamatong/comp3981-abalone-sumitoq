import unittest

from abalone.players.registry import get_agent, get_agent_weights, list_agents


class AgentRegistryTests(unittest.TestCase):
    def test_belgian_daisy_variants_are_registered(self):
        agent_ids = [agent.id for agent in list_agents()]

        self.assertIn("tournament-belgian-1", agent_ids)
        self.assertIn("tournament-belgian-2", agent_ids)
        self.assertIn("tournament-belgian-3", agent_ids)
        self.assertNotIn("tournament-belgian", agent_ids)

    def test_belgian_daisy_variants_share_same_weights(self):
        expected = get_agent_weights("tournament-belgian-1")

        self.assertEqual(get_agent_weights("tournament-belgian-2"), expected)
        self.assertEqual(get_agent_weights("tournament-belgian-3"), expected)

    def test_belgian_daisy_variants_have_distinct_evaluators(self):
        self.assertIsNot(get_agent("tournament-belgian-1").evaluator, get_agent("tournament-belgian-2").evaluator)
        self.assertIsNot(get_agent("tournament-belgian-1").evaluator, get_agent("tournament-belgian-3").evaluator)
        self.assertIsNot(get_agent("tournament-belgian-2").evaluator, get_agent("tournament-belgian-3").evaluator)

    def test_german_daisy_variants_are_registered(self):
        agent_ids = [agent.id for agent in list_agents()]

        self.assertIn("tournament-german-1", agent_ids)
        self.assertIn("tournament-german-2", agent_ids)
        self.assertIn("tournament-german-3", agent_ids)
        self.assertNotIn("tournament-german", agent_ids)

    def test_german_daisy_variants_share_same_weights(self):
        expected = get_agent_weights("tournament-german-1")

        self.assertEqual(get_agent_weights("tournament-german-2"), expected)
        self.assertEqual(get_agent_weights("tournament-german-3"), expected)

    def test_german_daisy_variants_have_distinct_evaluators(self):
        self.assertIsNot(get_agent("tournament-german-1").evaluator, get_agent("tournament-german-2").evaluator)
        self.assertIsNot(get_agent("tournament-german-1").evaluator, get_agent("tournament-german-3").evaluator)
        self.assertIsNot(get_agent("tournament-german-2").evaluator, get_agent("tournament-german-3").evaluator)


if __name__ == "__main__":
    unittest.main()
