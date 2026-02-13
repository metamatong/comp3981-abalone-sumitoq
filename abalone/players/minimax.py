"""Basic deterministic minimax search with alpha-beta pruning."""

from dataclasses import dataclass
from math import inf
import time
from typing import Dict, List, Optional, Tuple

from ..game.board import BLACK, WHITE, Board, Move, is_valid, neighbor
from ..state_space import generate_legal_moves
from .heuristics import evaluate_board
from .types import AgentConfig


@dataclass(frozen=True)
class SearchResult:
    move: Optional[Move]
    score: float
    nodes: int
    elapsed_ms: float
    depth: int

    def as_dict(self) -> Dict[str, object]:
        if self.move is None:
            notation = None
        else:
            notation = self.move.to_notation()

        return {
            "notation": notation,
            "score": self.score,
            "nodes": self.nodes,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "depth": self.depth,
        }


def _opponent(player: int) -> int:
    return WHITE if player == BLACK else BLACK


def _is_terminal(board: Board) -> bool:
    return board.score[BLACK] >= 6 or board.score[WHITE] >= 6


def _is_push_move(board: Board, player: int, move: Move) -> bool:
    if not move.is_inline or move.count < 2:
        return False

    _, leading = move._leading_trailing()
    ahead = neighbor(leading, move.direction)
    opponent = _opponent(player)
    return is_valid(ahead) and board.cells.get(ahead) == opponent


def _ordered_moves(board: Board, player: int, moves: List[Move]) -> List[Move]:
    def key(move: Move) -> Tuple[int, int, str]:
        return (
            0 if _is_push_move(board, player, move) else 1,
            -move.count,
            move.to_notation(),
        )

    return sorted(moves, key=key)


def _prefer_by_tie_break(config: AgentConfig, candidate: Move, incumbent: Optional[Move]) -> bool:
    if incumbent is None:
        return True
    if config.tie_break == "lexicographic":
        return candidate.to_notation() < incumbent.to_notation()
    return False


def _minimax(
    board: Board,
    to_move: int,
    root_player: int,
    depth: int,
    alpha: float,
    beta: float,
    config: AgentConfig,
    stats: Dict[str, int],
) -> Tuple[float, Optional[Move]]:
    stats["nodes"] += 1

    if depth == 0 or _is_terminal(board):
        return evaluate_board(board, root_player, preset=config.heuristic), None

    legal_moves = generate_legal_moves(board, to_move)
    if not legal_moves:
        return evaluate_board(board, root_player, preset=config.heuristic), None

    legal_moves = _ordered_moves(board, to_move, legal_moves)
    maximizing = (to_move == root_player)
    opponent = _opponent(to_move)

    best_move = None

    if maximizing:
        best_value = -inf
        for move in legal_moves:
            child = board.copy()
            child.apply_move(move, to_move)

            value, _ = _minimax(
                child,
                opponent,
                root_player,
                depth - 1,
                alpha,
                beta,
                config,
                stats,
            )

            if value > best_value or (value == best_value and _prefer_by_tie_break(config, move, best_move)):
                best_value = value
                best_move = move

            alpha = max(alpha, best_value)
            if beta <= alpha:
                break

        return best_value, best_move

    best_value = inf
    for move in legal_moves:
        child = board.copy()
        child.apply_move(move, to_move)

        value, _ = _minimax(
            child,
            opponent,
            root_player,
            depth - 1,
            alpha,
            beta,
            config,
            stats,
        )

        if value < best_value or (value == best_value and _prefer_by_tie_break(config, move, best_move)):
            best_value = value
            best_move = move

        beta = min(beta, best_value)
        if beta <= alpha:
            break

    return best_value, best_move


def search_best_move(board: Board, player: int, config: Optional[AgentConfig] = None) -> SearchResult:
    resolved_config = config or AgentConfig()
    stats = {"nodes": 0}
    start = time.perf_counter()

    score, move = _minimax(
        board,
        player,
        player,
        resolved_config.depth,
        -inf,
        inf,
        resolved_config,
        stats,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return SearchResult(
        move=move,
        score=score,
        nodes=stats["nodes"],
        elapsed_ms=elapsed_ms,
        depth=resolved_config.depth,
    )
