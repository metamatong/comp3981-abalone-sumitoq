"""Standard board tournament agent."""

from ...types import AgentDefinition
from .heuristic import evaluate_standard

AGENTS = [
    AgentDefinition(
        id="tournament-standard",
        label="Tournament: Standard",
        owner="Tournament",
        evaluator=evaluate_standard,
        default_depth=3,
        tie_break="lexicographic",
    )
]
