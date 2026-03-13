"""Types used by bot players."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    """Search options for local minimax-based AI players."""

    depth: int = 5
    heuristic: str = "balanced"
    tie_break: str = "lexicographic"
