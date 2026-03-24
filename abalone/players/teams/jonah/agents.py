"""Jonah's heuristic presets."""

from ...types import AgentDefinition
from .heuristic import evaluate_jonah

AGENTS = [
    AgentDefinition(
        id="jonah",
        label="Jonah",
        owner="Jonah",
        evaluator=evaluate_jonah,
        default_depth=3,
        tie_break="lexicographic",
    )
]
