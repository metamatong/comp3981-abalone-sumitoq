"""Abdullah's heuristic presets."""

from ...types import AgentDefinition
from .heuristic import evaluate_abdullah

AGENTS = [
    AgentDefinition(
        id="abdullah",
        label="Abdullah",
        owner="Abdullah",
        evaluator=evaluate_abdullah,
        default_depth=2,
        tie_break="lexicographic",
    )
]
