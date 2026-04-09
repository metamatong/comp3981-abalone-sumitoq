"""Types used by the shared AI/search stack."""

from dataclasses import dataclass
from typing import Callable, Optional

from ..game.board import Board, Move

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
    max_quiescence_depth: int = 0
    forced_finish_enabled: bool = False


@dataclass(frozen=True)
class AgentConfig:
    """Runtime search configuration and per-move context for an agent."""

    depth: Optional[int] = None
    tie_break: Optional[str] = None
    max_quiescence_depth: Optional[int] = None
    time_budget_ms: Optional[int] = None
    opening_seed: Optional[int] = None
    is_opening_turn: bool = False
    avoid_move: Optional[Move] = None
    root_candidate_limit: int = 0
    analysis_evaluator_id: Optional[str] = None
    board_token_before: Optional[str] = None
    remaining_game_moves: Optional[int] = None
    forced_finish_enabled: Optional[bool] = None


@dataclass(frozen=True)
class ResolvedAgentConfig:
    """Fully resolved search configuration after preset defaults are applied."""

    depth: int
    tie_break: str
    max_quiescence_depth: int
    time_budget_ms: Optional[int]
    opening_seed: Optional[int]
    is_opening_turn: bool
    avoid_move: Optional[Move]
    root_candidate_limit: int
    analysis_evaluator_id: Optional[str]
    board_token_before: Optional[str]
    remaining_game_moves: Optional[int]
    forced_finish_enabled: bool


def resolve_agent_config(agent: AgentDefinition, config: Optional[AgentConfig] = None) -> ResolvedAgentConfig:
    """Merge runtime overrides onto an agent preset."""
    config = config or AgentConfig()
    depth = config.depth if config.depth is not None else agent.default_depth
    remaining_game_moves = None
    if config.remaining_game_moves is not None:
        remaining_game_moves = max(0, int(config.remaining_game_moves))
        depth = min(depth, remaining_game_moves)
    return ResolvedAgentConfig(
        depth=depth,
        tie_break=config.tie_break if config.tie_break is not None else agent.tie_break,
        max_quiescence_depth=max(
            0,
            int(
                config.max_quiescence_depth
                if config.max_quiescence_depth is not None
                else agent.max_quiescence_depth
            ),
        ),
        time_budget_ms=config.time_budget_ms,
        opening_seed=config.opening_seed,
        is_opening_turn=config.is_opening_turn,
        avoid_move=config.avoid_move,
        root_candidate_limit=max(0, int(config.root_candidate_limit or 0)),
        analysis_evaluator_id=config.analysis_evaluator_id,
        board_token_before=config.board_token_before,
        remaining_game_moves=remaining_game_moves,
        forced_finish_enabled=(
            bool(config.forced_finish_enabled)
            if config.forced_finish_enabled is not None
            else bool(agent.forced_finish_enabled)
        ),
    )
