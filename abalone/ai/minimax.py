"""Deadline-aware deterministic minimax search with alpha-beta pruning."""

from dataclasses import dataclass
from math import inf
import time
from typing import Dict, List, Optional, Tuple

from ..game.board import BLACK, WHITE, Board, Move, ZOBRIST, is_valid, neighbor
from ..state_space import generate_legal_moves
from .defaults import DEFAULT_AGENT
from .types import AgentConfig, AgentDefinition, resolve_agent_config


# Transposition Table flag constants
EXACT = 0
LOWERBOUND = 1
UPPERBOUND = 2


@dataclass
class TTEntry:
    """Entry stored in the transposition table."""

    depth: int
    value: float
    flag: int
    move: Optional[Move]


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


def _ordered_moves(
    board: Board,
    player: int,
    moves: List[Move],
    tt_move: Optional[Move] = None,
    killer_moves: Optional[List[Optional[Move]]] = None,
    depth: int = 0,
) -> List[Move]:
    """Order moves to improve alpha-beta pruning deterministically.

    Priority: TT move > killer moves > push moves > multi-marble > notation.
    """
    killer1 = killer_moves[depth] if killer_moves and depth < len(killer_moves) else None

    def key(move: Move) -> Tuple[int, int, int, int, str]:
        return (
            0 if move == tt_move else 1,
            0 if move == killer1 else 1,
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


def _make_tt_key(board: Board, to_move: int) -> int:
    """Create TT hash key using Zobrist hash XOR'd with side-to-move key."""
    return board.zhash ^ ZOBRIST[('side', to_move)]


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
    tt: Dict[int, TTEntry],
    killer_moves: List[Optional[Move]],
    root_legal_moves: Optional[List[Move]] = None,
) -> Tuple[float, Optional[Move]]:
    """Run depth-limited minimax with alpha-beta pruning, TT, killer moves, and undo/redo."""
    _check_deadline(deadline_at)
    stats["nodes"] += 1

    alpha_orig = alpha
    beta_orig = beta

    tt_key = _make_tt_key(board, to_move)
    tt_entry = tt.get(tt_key)

    if tt_entry is not None and tt_entry.depth >= depth:
        if tt_entry.flag == EXACT:
            return tt_entry.value, tt_entry.move
        elif tt_entry.flag == LOWERBOUND:
            alpha = max(alpha, tt_entry.value)
        elif tt_entry.flag == UPPERBOUND:
            beta = min(beta, tt_entry.value)
        if alpha >= beta:
            return tt_entry.value, tt_entry.move

    tt_move = tt_entry.move if tt_entry is not None else None

    if depth == 0 or _is_terminal(board):
        return evaluator(board, root_player), None

    if root_legal_moves is not None:
        legal_moves = root_legal_moves
    else:
        legal_moves = generate_legal_moves(board, to_move)

    if not legal_moves:
        return evaluator(board, root_player), None

    legal_moves = _ordered_moves(board, to_move, legal_moves, tt_move, killer_moves, depth)
    maximizing = to_move == root_player
    opponent = _opponent(to_move)
    best_move = None

    if maximizing:
        best_value = -inf
        for move in legal_moves:
            _check_deadline(deadline_at)
            undo_info = board.apply_move_undo(move, to_move)
            value, _ = _minimax(
                board,
                opponent,
                root_player,
                depth - 1,
                alpha,
                beta,
                evaluator,
                tie_break,
                deadline_at,
                stats,
                tt,
                killer_moves,
            )
            board.undo_move(undo_info)
            if value > best_value or (value == best_value and _prefer_by_tie_break(tie_break, move, best_move)):
                best_value = value
                best_move = move
            alpha = max(alpha, best_value)
            if beta <= alpha:
                # Record killer move on cutoff
                if depth < len(killer_moves):
                    killer_moves[depth] = move
                break
    else:
        best_value = inf
        for move in legal_moves:
            _check_deadline(deadline_at)
            undo_info = board.apply_move_undo(move, to_move)
            value, _ = _minimax(
                board,
                opponent,
                root_player,
                depth - 1,
                alpha,
                beta,
                evaluator,
                tie_break,
                deadline_at,
                stats,
                tt,
                killer_moves,
            )
            board.undo_move(undo_info)
            if value < best_value or (value == best_value and _prefer_by_tie_break(tie_break, move, best_move)):
                best_value = value
                best_move = move
            beta = min(beta, best_value)
            if beta <= alpha:
                if depth < len(killer_moves):
                    killer_moves[depth] = move
                break

    if best_value <= alpha_orig:
        flag = UPPERBOUND
    elif best_value >= beta_orig:
        flag = LOWERBOUND
    else:
        flag = EXACT

    tt[tt_key] = TTEntry(depth=depth, value=best_value, flag=flag, move=best_move)
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

    avoid_move = resolved_config.avoid_move
    avoidance_applied = False
    if avoid_move is not None and len(legal_moves) > 1:
        filtered = [move for move in legal_moves if move != avoid_move]
        if filtered:
            legal_moves = filtered
            avoidance_applied = True

    total_nodes = 0
    best_move = legal_moves[0]
    best_score = 0.0
    completed_depth = 0
    timed_out = False

    # Shared across iterative deepening iterations
    tt: Dict[int, TTEntry] = {}
    killer_moves: List[Optional[Move]] = [None] * (requested_depth + 1)
    opponent = _opponent(player)

    for depth in range(1, requested_depth + 1):
        stats = {"nodes": 0}
        # Reset killer moves each iteration (they're depth-relative)
        for i in range(len(killer_moves)):
            killer_moves[i] = None
            
        alpha = -inf
        beta = inf
        current_best_score = -inf
        current_best_move = best_move  # fallback to previous depth's best if timeout occurs early

        try:
            stats["nodes"] += 1
            tt_key = _make_tt_key(board, player)
            tt_entry = tt.get(tt_key)
            tt_move = tt_entry.move if tt_entry is not None else None
            ordered_moves = _ordered_moves(board, player, legal_moves, tt_move, killer_moves, depth)
            
            for move in ordered_moves:
                _check_deadline(deadline_at)
                undo_info = board.apply_move_undo(move, player)
                value, _ = _minimax(
                    board,
                    opponent,
                    player,
                    depth - 1,
                    alpha,
                    beta,
                    resolved_agent.evaluator,
                    resolved_config.tie_break,
                    deadline_at,
                    stats,
                    tt,
                    killer_moves,
                )
                board.undo_move(undo_info)
                if value > current_best_score or (value == current_best_score and _prefer_by_tie_break(resolved_config.tie_break, move, current_best_move)):
                    current_best_score = value
                    current_best_move = move
                alpha = max(alpha, current_best_score)
                # alpha-beta at root: root beta is inf so beta <= alpha is never true.
                
            tt[tt_key] = TTEntry(depth=depth, value=current_best_score, flag=EXACT, move=current_best_move)
            
            total_nodes += stats["nodes"]
            best_move = current_best_move
            best_score = current_best_score
            completed_depth = depth

        except _SearchTimeout:
            total_nodes += stats["nodes"]
            best_move = current_best_move  # best partial move from this depth (or previous if early)
            if current_best_score != -inf:
                best_score = current_best_score # use score if we searched at least one move
            timed_out = True
            break

    decision_source = "search"
    if completed_depth == 0 and not timed_out:
        # Fallback only if we didn't search at all
        best_move = legal_moves[0]
        best_score = 0.0
        decision_source = "fallback"
    elif timed_out and completed_depth == 0:
        decision_source = "timeout_fallback_partial"
    elif timed_out:
        decision_source = "timeout_partial"
    if avoidance_applied and decision_source == "search":
        decision_source = "repeat_avoidance"

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
