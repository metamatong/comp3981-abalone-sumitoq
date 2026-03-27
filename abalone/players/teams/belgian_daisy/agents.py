"""Belgian Daisy tournament agent."""

from ...types import AgentDefinition
from .heuristic import evaluate_belgian_daisy

AGENTS = [
    AgentDefinition(
        id="tournament-belgian",
        label="Tournament: Belgian Daisy",
        owner="Tournament",
        evaluator=evaluate_belgian_daisy,
        default_depth=4,
        tie_break="lexicographic",
    )
]
