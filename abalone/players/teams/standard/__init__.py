"""Standard board tournament package."""

from .agents import AGENTS
from .heuristic import STANDARD_WEIGHTS, evaluate_standard

__all__ = ["AGENTS", "STANDARD_WEIGHTS", "evaluate_standard"]
