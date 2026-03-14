"""Cole's agent presets."""

from .agents import AGENTS
from .heuristic import COLE_WEIGHTS, evaluate_cole

__all__ = ["AGENTS", "COLE_WEIGHTS", "evaluate_cole"]
