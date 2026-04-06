"""Public interface for choosing bot moves."""

from random import Random
from time import perf_counter
from typing import Optional

from ..game.board import BLACK
from ..state_space import generate_legal_moves
from .defaults import DEFAULT_AGENT
from .minimax import SearchResult, search_best_move
from .types import AgentConfig, AgentDefinition, resolve_agent_config


def choose_move(
    board,
    player: int,
    config: Optional[AgentConfig] = None,
    agent: Optional[AgentDefinition] = None,
):
    """Return the selected move for `player` using the selected agent preset."""
    result = choose_move_with_info(board, player, config=config, agent=agent)
    return result.move


def choose_move_with_info(
    board,
    player: int,
    config: Optional[AgentConfig] = None,
    agent: Optional[AgentDefinition] = None,
) -> SearchResult:
    """Return full move-selection metadata, including diagnostics."""
    resolved_agent = agent or DEFAULT_AGENT
    resolved_config = resolve_agent_config(resolved_agent, config)
    if player == BLACK and resolved_config.is_opening_turn:
        start = perf_counter()
        legal_moves = generate_legal_moves(board, player)
        if not legal_moves:
            return SearchResult(
                move=None,
                score=resolved_agent.evaluator(board, player),
                nodes=0,
                elapsed_ms=(perf_counter() - start) * 1000.0,
                depth=resolved_config.depth,
                completed_depth=0,
                decision_source="opening_random",
                timed_out=False,
                time_budget_ms=resolved_config.time_budget_ms,
                agent_id=resolved_agent.id,
                agent_label=resolved_agent.label,
            )
        rng = Random(resolved_config.opening_seed)
        move = legal_moves[rng.randrange(len(legal_moves))]
        return SearchResult(
            move=move,
            score=0.0,
            nodes=0,
            elapsed_ms=(perf_counter() - start) * 1000.0,
            depth=resolved_config.depth,
            completed_depth=0,
            decision_source="opening_random",
            timed_out=False,
            time_budget_ms=resolved_config.time_budget_ms,
            agent_id=resolved_agent.id,
            agent_label=resolved_agent.label,
        )
    return search_best_move(board, player, agent=resolved_agent, config=config)
