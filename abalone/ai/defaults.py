"""Default shared AI preset used when no specific preset is selected."""

from .heuristics import evaluate_board
from .types import AgentDefinition

DEFAULT_AGENT = AgentDefinition(
    id="default",
    label="Default",
    owner="Shared",
    evaluator=evaluate_board,
    default_depth=2,
    tie_break="lexicographic",
)
