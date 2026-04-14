"""Tournament Belgian Daisy 2 agent."""

from ...types import AgentDefinition
from .heuristic import evaluate_tournament_belgian_2

AGENTS = [
    AgentDefinition(
        id="tournament-belgian-2",
        label="Tournament: Belgian Daisy 2",
        owner="Tournament",
        evaluator=evaluate_tournament_belgian_2,
        default_depth=7,
        tie_break="lexicographic",
        max_quiescence_depth=9,
    )
]
