"""Tournament German Daisy 1 agent."""

from ...types import AgentDefinition
from .heuristic import evaluate_tournament_german_1

AGENTS = [
    AgentDefinition(
        id="tournament-german-1",
        label="Tournament: German Daisy 1",
        owner="Tournament",
        evaluator=evaluate_tournament_german_1,
        default_depth=7,
        tie_break="lexicographic",
        max_quiescence_depth=9,
    )
]
