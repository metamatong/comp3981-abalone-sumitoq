"""Types used by the shared AI/search stack."""

from dataclasses import dataclass
from typing import Callable, Optional

from ..game.board import Board

Evaluator = Callable[[Board, int], float]


@dataclass(frozen=True)
class AgentDefinition:
    """A selectable AI preset backed by a shared search engine."""

    id: str
    label: str
    owner: str
    evaluator: Evaluator
    default_depth: int = 5
    tie_break: str = "lexicographic"


@dataclass(frozen=True)
class AgentConfig:
    """Runtime search configuration and per-move context for an agent."""

    depth: Optional[int] = None
    tie_break: Optional[str] = None
    time_budget_ms: Optional[int] = None
    opening_seed: Optional[int] = None
    is_opening_turn: bool = False


@dataclass(frozen=True)
class ResolvedAgentConfig:
    """Fully resolved search configuration after preset defaults are applied."""

    depth: int
    tie_break: str
    time_budget_ms: Optional[int]
    opening_seed: Optional[int]
    is_opening_turn: bool


def resolve_agent_config(agent: AgentDefinition, config: Optional[AgentConfig] = None) -> ResolvedAgentConfig:
    """Merge runtime overrides onto an agent preset."""
    config = config or AgentConfig()
    return ResolvedAgentConfig(
        depth=config.depth if config.depth is not None else agent.default_depth,
        tie_break=config.tie_break if config.tie_break is not None else agent.tie_break,
        time_budget_ms=config.time_budget_ms,
        opening_seed=config.opening_seed,
        is_opening_turn=config.is_opening_turn,
    )
