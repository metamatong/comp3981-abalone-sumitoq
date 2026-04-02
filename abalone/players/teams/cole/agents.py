"""Cole's heuristic presets."""

from ...types import AgentDefinition
from .heuristic import evaluate_cole

AGENTS = [
    AgentDefinition(
        id="cole",
        label="Cole",
        owner="Cole",
        evaluator=evaluate_cole,
        default_depth=4,
        tie_break="lexicographic",
    )
]
