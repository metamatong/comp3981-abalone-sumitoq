"""Standard board tournament agent."""

from ...types import AgentDefinition
from .heuristic import evaluate_standard

AGENTS = [
    AgentDefinition(
        id="tournament-standard",
        label="Tournament: Standard",
        owner="Tournament",
        evaluator=evaluate_standard,
        default_depth=7,
        tie_break="lexicographic",
        max_quiescence_depth=9,
        forced_finish_enabled=False,
    )
]
