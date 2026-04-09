"""Jonah's heuristic presets."""

from ...types import AgentDefinition
from .heuristic import evaluate_jonah

AGENTS = [
    AgentDefinition(
        id="jonah",
        label="Jonah",
        owner="Jonah",
        evaluator=evaluate_jonah,
        default_depth=6,
        tie_break="lexicographic",
        forced_finish_enabled=False,
    )
]
