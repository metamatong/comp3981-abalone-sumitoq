"""Types used by bot players."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    depth: int = 2
    heuristic: str = "balanced"
    tie_break: str = "lexicographic"
