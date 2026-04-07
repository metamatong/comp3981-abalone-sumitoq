"""Kyle's heuristic presets."""

from ...types import AgentDefinition
from .heuristic import evaluate_kyle

AGENTS = [
    AgentDefinition(
        id="kyle",
        label="Kyle",
        owner="Kyle",
        evaluator=evaluate_kyle,
        default_depth=6,
        tie_break="lexicographic",
        max_quiescence_depth=10,
    )
]
