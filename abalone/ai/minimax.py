"""Deadline-aware deterministic minimax search with alpha-beta pruning."""

from dataclasses import dataclass
from math import inf
import time
from typing import Dict, List, Optional, Tuple

from ..game.board import BLACK, WHITE, Board, Move, is_valid, neighbor
from ..state_space import generate_legal_moves
from .defaults import DEFAULT_AGENT
from .types import AgentConfig, AgentDefinition, resolve_agent_config


@dataclass(frozen=True)
class SearchResult:
    """Search output including chosen move and diagnostic metadata."""

    move: Optional[Move]
    score: float
    nodes: int
    elapsed_ms: float
    depth: int
    completed_depth: int
    decision_source: str
    timed_out: bool
    time_budget_ms: Optional[int]
    agent_id: str
    agent_label: str

    def as_dict(self) -> Dict[str, object]:
        """Serialize search diagnostics for CLI/API responses."""
        notation = None if self.move is None else self.move.to_notation()
        return {
            "notation": notation,
            "score": self.score,
            "nodes": self.nodes,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "depth": self.depth,
            "completed_depth": self.completed_depth,
            "decision_source": self.decision_source,
            "timed_out": self.timed_out,
            "time_budget_ms": self.time_budget_ms,
            "agent_id": self.agent_id,
            "agent_label": self.agent_label,
        }


class _SearchTimeout(Exception):
    """Raised when a deadline-aware search exceeds its budget."""


def _opponent(player: int) -> int:
    """Return the opposing color constant."""
    return WHITE if player == BLACK else BLACK


def _is_terminal(board: Board) -> bool:
    """Return whether either player has reached the capture win condition."""
    return board.score[BLACK] >= 6 or board.score[WHITE] >= 6


def _is_push_move(board: Board, player: int, move: Move) -> bool:
    """Heuristic check used for move ordering: whether move contacts an opponent inline."""
    if not move.is_inline or move.count < 2:
        return False

    _, leading = move.leading_trailing()
    ahead = neighbor(leading, move.direction)
    opponent = _opponent(player)
    return is_valid(ahead) and board.cells.get(ahead) == opponent


def _ordered_moves(board: Board, player: int, moves: List[Move]) -> List[Move]:
    """Order moves to improve alpha-beta pruning deterministically."""

    def key(move: Move) -> Tuple[int, int, str]:
        return (
            0 if _is_push_move(board, player, move) else 1,
            -move.count,
            move.to_notation(),
        )

    return sorted(moves, key=key)


def _prefer_by_tie_break(tie_break: str, candidate: Move, incumbent: Optional[Move]) -> bool:
    """Resolve equal-valued moves using the configured deterministic tie-break."""
    if incumbent is None:
        return True
    if tie_break == "lexicographic":
        return candidate.to_notation() < incumbent.to_notation()
    return False


def _check_deadline(deadline_at: Optional[float]) -> None:
    """Raise when the active deadline has expired."""
    if deadline_at is not None and time.perf_counter() >= deadline_at:
        raise _SearchTimeout


def _minimax(
    board: Board,
    to_move: int,
    root_player: int,
    depth: int,
    alpha: float,
    beta: float,
    evaluator,
    tie_break: str,
    deadline_at: Optional[float],
    stats: Dict[str, int],
) -> Tuple[float, Optional[Move]]:
    """Run depth-limited minimax with alpha-beta pruning and deterministic ordering."""
    _check_deadline(deadline_at)
    stats["nodes"] += 1

    if depth == 0 or _is_terminal(board):
        return evaluator(board, root_player), None

    legal_moves = generate_legal_moves(board, to_move)
    if not legal_moves:
        return evaluator(board, root_player), None

    legal_moves = _ordered_moves(board, to_move, legal_moves)
    maximizing = to_move == root_player
    opponent = _opponent(to_move)
    best_move = None

    if maximizing:
        best_value = -inf
        for move in legal_moves:
            _check_deadline(deadline_at)
            child = board.copy()
            child.apply_move(move, to_move)
            value, _ = _minimax(
                child,
                opponent,
                root_player,
                depth - 1,
                alpha,
                beta,
                evaluator,
                tie_break,
                deadline_at,
                stats,
            )
            if value > best_value or (value == best_value and _prefer_by_tie_break(tie_break, move, best_move)):
                best_value = value
                best_move = move
            alpha = max(alpha, best_value)
            if beta <= alpha:
                break
        return best_value, best_move

    best_value = inf
    for move in legal_moves:
        _check_deadline(deadline_at)
        child = board.copy()
        child.apply_move(move, to_move)
        value, _ = _minimax(
            child,
            opponent,
            root_player,
            depth - 1,
            alpha,
            beta,
            evaluator,
            tie_break,
            deadline_at,
            stats,
        )
        if value < best_value or (value == best_value and _prefer_by_tie_break(tie_break, move, best_move)):
            best_value = value
            best_move = move
        beta = min(beta, best_value)
        if beta <= alpha:
            break

    return best_value, best_move


def search_best_move(
    board: Board,
    player: int,
    agent: Optional[AgentDefinition] = None,
    config: Optional[AgentConfig] = None,
) -> SearchResult:
    """Return best move and diagnostics for `player` from current board position."""
    resolved_agent = agent or DEFAULT_AGENT
    resolved_config = resolve_agent_config(resolved_agent, config)
    requested_depth = resolved_config.depth
    start = time.perf_counter()
    deadline_at = None
    if resolved_config.time_budget_ms and resolved_config.time_budget_ms > 0:
        deadline_at = start + (resolved_config.time_budget_ms / 1000.0)

    legal_moves = _ordered_moves(board, player, generate_legal_moves(board, player))
    if not legal_moves:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return SearchResult(
            move=None,
            score=resolved_agent.evaluator(board, player),
            nodes=0,
            elapsed_ms=elapsed_ms,
            depth=requested_depth,
            completed_depth=0,
            decision_source="search",
            timed_out=False,
            time_budget_ms=resolved_config.time_budget_ms,
            agent_id=resolved_agent.id,
            agent_label=resolved_agent.label,
        )

    total_nodes = 0
    best_move = None
    best_score = 0.0
    completed_depth = 0
    timed_out = False

    for depth in range(1, requested_depth + 1):
        stats = {"nodes": 0}
        try:
            score, move = _minimax(
                board,
                player,
                player,
                depth,
                -inf,
                inf,
                resolved_agent.evaluator,
                resolved_config.tie_break,
                deadline_at,
                stats,
            )
        except _SearchTimeout:
            total_nodes += stats["nodes"]
            timed_out = True
            break

        total_nodes += stats["nodes"]
        best_move = move
        best_score = score
        completed_depth = depth

    decision_source = "search"
    if completed_depth == 0:
        best_move = legal_moves[0]
        best_score = 0.0
        decision_source = "timeout_fallback"
        timed_out = True

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return SearchResult(
        move=best_move,
        score=best_score,
        nodes=total_nodes,
        elapsed_ms=elapsed_ms,
        depth=requested_depth,
        completed_depth=completed_depth,
        decision_source=decision_source,
        timed_out=timed_out,
        time_budget_ms=resolved_config.time_budget_ms,
        agent_id=resolved_agent.id,
        agent_label=resolved_agent.label,
    )
