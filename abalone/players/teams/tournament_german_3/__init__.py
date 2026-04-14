"""Tournament German Daisy 3 agent package."""

from .agents import AGENTS
from .heuristic import TOURNAMENT_GERMAN_3_WEIGHTS, evaluate_tournament_german_3

__all__ = ["AGENTS", "TOURNAMENT_GERMAN_3_WEIGHTS", "evaluate_tournament_german_3"]
