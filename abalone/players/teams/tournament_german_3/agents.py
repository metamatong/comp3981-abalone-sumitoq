"""Tournament German Daisy 3 agent."""

from ...types import AgentDefinition
from .heuristic import evaluate_tournament_german_3

AGENTS = [
    AgentDefinition(
        id="tournament-german-3",
        label="Tournament: German Daisy 3",
        owner="Tournament",
        evaluator=evaluate_tournament_german_3,
        default_depth=7,
        tie_break="lexicographic",
        max_quiescence_depth=9,
    )
]
