"""Kyle's heuristic presets."""

from ...types import AgentDefinition
from .heuristic import evaluate_kyle

AGENTS = [
    AgentDefinition(
        id="kyle",
        label="Kyle",
        owner="Kyle",
        evaluator=evaluate_kyle,
        default_depth=3,
        tie_break="lexicographic",
    )
]
