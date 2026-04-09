"""Native-backed deterministic minimax search."""

from dataclasses import dataclass
from math import inf
import time
from typing import Dict, List, Optional, Tuple

from ..game.board import BLACK, WHITE, Board, DIRECTION_INDEX, Move, NEIGHBOR_TABLE, ZOBRIST
from ..native import search_weighted as native_search_weighted
from ..state_space import generate_legal_moves
from .defaults import DEFAULT_AGENT
from .heuristics import FEATURE_ORDER
from .types import AgentConfig, AgentDefinition, resolve_agent_config


# Transposition Table flag constants
EXACT = 0
LOWERBOUND = 1
UPPERBOUND = 2

# Search-mode keys for shared TT storage.
TT_MODE_FULL = 0
TT_MODE_QUIESCENCE = 1

_FORCE_WEIGHTED_SEARCH_PATH: Optional[str] = None


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
    root_candidates: Optional[List[Dict[str, object]]] = None
    analysis_evaluator_id: Optional[str] = None
    board_token_before: Optional[str] = None

    def as_dict(self) -> Dict[str, object]:
        """Serialize search diagnostics for CLI/API responses."""
        notation = None if self.move is None else self.move.to_notation()
        payload = {
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
        if self.root_candidates:
            payload["root_candidates"] = list(self.root_candidates)
        if self.analysis_evaluator_id:
            payload["analysis_evaluator_id"] = self.analysis_evaluator_id
        if self.board_token_before:
            payload["board_token_before"] = self.board_token_before
        return payload


class _SearchTimeout(Exception):
    """Raised when a deadline-aware search exceeds its budget."""


def _resolve_decision_source(
    *,
    move: Optional[Move],
    completed_depth: int,
    timed_out: bool,
    avoidance_applied: bool,
) -> str:
    """Map native search metadata onto the public decision-source strings."""
    decision_source = "search"
    if move is None:
        decision_source = "search"
    elif completed_depth == 0 and not timed_out:
        decision_source = "fallback"
    elif timed_out and completed_depth == 0:
        decision_source = "timeout_fallback_partial"
    elif timed_out:
        decision_source = "timeout_partial"
    if avoidance_applied and decision_source == "search":
        decision_source = "repeat_avoidance"
    return decision_source


def _normalize_root_candidates(raw_candidates, limit: int):
    """Normalize native root-candidate payloads to the public response shape."""
    if limit <= 0 or not raw_candidates:
        return None
    root_candidates = [
        {
            "notation": candidate["move"].to_notation(),
            "score": round(float(candidate["score"]), 6),
            "depth": int(candidate["depth"]),
        }
        for candidate in raw_candidates
        if candidate.get("move") is not None
    ]
    if not root_candidates:
        return None
    return sorted(
        root_candidates,
        key=lambda item: (-float(item["score"]), str(item["notation"])),
    )[:limit]
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

    ahead = NEIGHBOR_TABLE[move._leading][DIRECTION_INDEX[move.direction]]
    opponent = _opponent(player)
    return ahead is not None and board.cells[ahead] == opponent


def _is_quiescence_move(board: Board, player: int, move: Move) -> bool:
    """Return whether a legal move is tactical enough to extend beyond the nominal leaf."""
    return _is_push_move(board, player, move)


def _quiescence(
    board: Board,
    to_move: int,
    root_player: int,
    remaining_depth: int,
    remaining_game_moves: Optional[int],
    alpha: float,
    beta: float,
    evaluator,
    tie_break: str,
    deadline_at: Optional[float],
    stats: Dict[str, int],
    tt: Dict[tuple[int, int, int], TTEntry],
) -> tuple[float, Optional[Move]]:
    """Extend only tactical leaf moves to reduce horizon effects in noisy positions."""
    _check_deadline(deadline_at)
    stats["nodes"] += 1
    alpha_orig = alpha
    beta_orig = beta

    tt_key = _make_tt_key(board, to_move, TT_MODE_QUIESCENCE, remaining_game_moves)
    tt_entry = tt.get(tt_key)
    if tt_entry is not None and tt_entry.depth >= remaining_depth:
        if tt_entry.flag == EXACT:
            return tt_entry.value, tt_entry.move
        if tt_entry.flag == LOWERBOUND:
            alpha = max(alpha, tt_entry.value)
        elif tt_entry.flag == UPPERBOUND:
            beta = min(beta, tt_entry.value)
        if alpha >= beta:
            return tt_entry.value, tt_entry.move

    tt_move = tt_entry.move if tt_entry is not None else None

    stand_pat = evaluator(board, root_player)
    if remaining_depth <= 0 or _is_terminal(board) or (remaining_game_moves is not None and remaining_game_moves <= 0):
        tt[tt_key] = TTEntry(depth=remaining_depth, value=stand_pat, flag=EXACT, move=None)
        return stand_pat, None

    maximizing = to_move == root_player
    if maximizing:
        if stand_pat >= beta:
            tt[tt_key] = TTEntry(depth=remaining_depth, value=stand_pat, flag=LOWERBOUND, move=None)
            return stand_pat, None
        alpha = max(alpha, stand_pat)
        best_value = stand_pat
    else:
        if stand_pat <= alpha:
            tt[tt_key] = TTEntry(depth=remaining_depth, value=stand_pat, flag=UPPERBOUND, move=None)
            return stand_pat, None
        beta = min(beta, stand_pat)
        best_value = stand_pat

    tactical_moves = [
        move
        for move in generate_legal_moves(board, to_move)
        if _is_quiescence_move(board, to_move, move)
    ]
    if not tactical_moves:
        tt[tt_key] = TTEntry(depth=remaining_depth, value=stand_pat, flag=EXACT, move=None)
        return stand_pat, None

    tactical_moves = _ordered_moves(board, to_move, tactical_moves, tt_move)
    best_move = None
    opponent = _opponent(to_move)
    next_remaining_game_moves = None if remaining_game_moves is None else remaining_game_moves - 1

    if maximizing:
        for move in tactical_moves:
            _check_deadline(deadline_at)
            undo_info = board.apply_move_undo(move, to_move)
            try:
                value, _ = _quiescence(
                    board,
                    opponent,
                    root_player,
                    remaining_depth - 1,
                    next_remaining_game_moves,
                    alpha,
                    beta,
                    evaluator,
                    tie_break,
                    deadline_at,
                    stats,
                    tt,
                )
            finally:
                board.undo_move(undo_info)

            if value > best_value or (value == best_value and _prefer_by_tie_break(tie_break, move, best_move)):
                best_value = value
                best_move = move
            alpha = max(alpha, best_value)
            if beta <= alpha:
                break
    else:
        for move in tactical_moves:
            _check_deadline(deadline_at)
            undo_info = board.apply_move_undo(move, to_move)
            try:
                value, _ = _quiescence(
                    board,
                    opponent,
                    root_player,
                    remaining_depth - 1,
                    next_remaining_game_moves,
                    alpha,
                    beta,
                    evaluator,
                    tie_break,
                    deadline_at,
                    stats,
                    tt,
                )
            finally:
                board.undo_move(undo_info)

            if value < best_value or (value == best_value and _prefer_by_tie_break(tie_break, move, best_move)):
                best_value = value
                best_move = move
            beta = min(beta, best_value)
            if beta <= alpha:
                break

    if best_value <= alpha_orig:
        flag = UPPERBOUND
    elif best_value >= beta_orig:
        flag = LOWERBOUND
    else:
        flag = EXACT

    tt[tt_key] = TTEntry(depth=remaining_depth, value=best_value, flag=flag, move=best_move)
    return best_value, best_move


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

    def key(move: Move) -> Tuple[int, int, int, int, tuple]:
        return (
            0 if move == tt_move else 1,
            0 if move == killer1 else 1,
            0 if _is_push_move(board, player, move) else 1,
            -move.count,
            move.ordering_key,
        )

    return sorted(moves, key=key)


def _prefer_by_tie_break(tie_break: str, candidate: Move, incumbent: Optional[Move]) -> bool:
    """Resolve equal-valued moves using the configured deterministic tie-break."""
    if incumbent is None:
        return True
    if tie_break == "lexicographic":
        return candidate.ordering_key < incumbent.ordering_key
    return False


def _check_deadline(deadline_at: Optional[float]) -> None:
    """Raise when the active deadline has expired."""
    if deadline_at is not None and time.perf_counter() >= deadline_at:
        raise _SearchTimeout


def _make_tt_key(
    board: Board,
    to_move: int,
    mode: int = TT_MODE_FULL,
    remaining_game_moves: Optional[int] = None,
) -> tuple[int, int, int]:
    """Create a TT key from the board hash, side to move, search mode, and move budget."""
    remaining_key = -1 if remaining_game_moves is None else int(remaining_game_moves)
    return board.zhash ^ ZOBRIST[('side', to_move)], mode, remaining_key


def _minimax(
    board: Board,
    to_move: int,
    root_player: int,
    depth: int,
    remaining_game_moves: Optional[int],
    alpha: float,
    beta: float,
    evaluator,
    tie_break: str,
    deadline_at: Optional[float],
    stats: Dict[str, int],
    tt: Dict[tuple[int, int, int], TTEntry],
    killer_moves: List[Optional[Move]],
    max_quiescence_depth: int,
    root_legal_moves: Optional[List[Move]] = None,
) -> tuple[float, Optional[Move]]:
    """Run depth-limited minimax with alpha-beta pruning, TT, killer moves, and undo/redo."""
    _check_deadline(deadline_at)
    if remaining_game_moves is not None and remaining_game_moves <= 0:
        return evaluator(board, root_player), None
    if depth == 0 and not _is_terminal(board) and max_quiescence_depth > 0:
        return _quiescence(
            board,
            to_move,
            root_player,
            max_quiescence_depth,
            remaining_game_moves,
            alpha,
            beta,
            evaluator,
            tie_break,
            deadline_at,
            stats,
            tt,
        )

    stats["nodes"] += 1

    alpha_orig = alpha
    beta_orig = beta

    tt_key = _make_tt_key(board, to_move, TT_MODE_FULL, remaining_game_moves)
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
    next_remaining_game_moves = None if remaining_game_moves is None else remaining_game_moves - 1

    if maximizing:
        best_value = -inf
        for move in legal_moves:
            _check_deadline(deadline_at)
            undo_info = board.apply_move_undo(move, to_move)
            try:
                value, _ = _minimax(
                    board,
                    opponent,
                    root_player,
                    depth - 1,
                    next_remaining_game_moves,
                    alpha,
                    beta,
                    evaluator,
                    tie_break,
                    deadline_at,
                    stats,
                    tt,
                    killer_moves,
                    max_quiescence_depth,
                )
            finally:
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
            try:
                value, _ = _minimax(
                    board,
                    opponent,
                    root_player,
                    depth - 1,
                    next_remaining_game_moves,
                    alpha,
                    beta,
                    evaluator,
                    tie_break,
                    deadline_at,
                    stats,
                    tt,
                    killer_moves,
                    max_quiescence_depth,
                )
            finally:
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
    """Return best move and diagnostics for `player`."""
    resolved_agent = agent or DEFAULT_AGENT
    resolved_config = resolve_agent_config(resolved_agent, config)
    if resolved_config.depth == 0:
        return _search_best_move_depth_zero(board, player, resolved_agent, resolved_config)
    agent_weights = getattr(resolved_agent.evaluator, "weights", None)
    force_path = _FORCE_WEIGHTED_SEARCH_PATH
    use_native = bool(agent_weights) and force_path != "python"
    if use_native:
        return _search_best_move_native(board, player, resolved_agent, resolved_config, agent_weights)
    return _search_best_move_python(board, player, resolved_agent, resolved_config)


def _search_best_move_depth_zero(
    board: Board,
    player: int,
    resolved_agent: AgentDefinition,
    resolved_config,
) -> SearchResult:
    """Resolve an explicit depth-0 request without depending on backend search-path quirks."""
    start = time.perf_counter()
    legal_moves = _ordered_moves(board, player, generate_legal_moves(board, player))
    if not legal_moves:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return SearchResult(
            move=None,
            score=resolved_agent.evaluator(board, player),
            nodes=0,
            elapsed_ms=elapsed_ms,
            depth=0,
            completed_depth=0,
            decision_source="search",
            timed_out=False,
            time_budget_ms=resolved_config.time_budget_ms,
            agent_id=resolved_agent.id,
            agent_label=resolved_agent.label,
            analysis_evaluator_id=resolved_config.analysis_evaluator_id,
            board_token_before=resolved_config.board_token_before,
        )

    avoid_move = resolved_config.avoid_move
    avoidance_applied = False
    if avoid_move is not None and len(legal_moves) > 1:
        filtered = [move for move in legal_moves if move != avoid_move]
        if filtered:
            legal_moves = filtered
            avoidance_applied = True

    chosen_move = legal_moves[0]
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return SearchResult(
        move=chosen_move,
        score=resolved_agent.evaluator(board, player),
        nodes=0,
        elapsed_ms=elapsed_ms,
        depth=0,
        completed_depth=0,
        decision_source=_resolve_decision_source(
            move=chosen_move,
            completed_depth=0,
            timed_out=False,
            avoidance_applied=avoidance_applied,
        ),
        timed_out=False,
        time_budget_ms=resolved_config.time_budget_ms,
        agent_id=resolved_agent.id,
        agent_label=resolved_agent.label,
        analysis_evaluator_id=resolved_config.analysis_evaluator_id,
        board_token_before=resolved_config.board_token_before,
    )


def _search_best_move_native(
    board: Board,
    player: int,
    resolved_agent: AgentDefinition,
    resolved_config,
    agent_weights,
) -> SearchResult:
    """Run the native search path for weighted evaluators."""
    ordered_weights = tuple(float(agent_weights[key]) for key in FEATURE_ORDER)
    native_start = time.perf_counter()
    native_result = native_search_weighted(
        board,
        player,
        ordered_weights,
        depth=resolved_config.depth,
        max_quiescence_depth=resolved_config.max_quiescence_depth,
        time_budget_ms=resolved_config.time_budget_ms,
        remaining_game_moves=resolved_config.remaining_game_moves,
        tie_break=resolved_config.tie_break,
        avoid_move=resolved_config.avoid_move,
        root_candidate_limit=resolved_config.root_candidate_limit,
    )
    elapsed_ms = (time.perf_counter() - native_start) * 1000.0

    move = native_result["move"]
    completed_depth = int(native_result["completed_depth"])
    timed_out = bool(native_result["timed_out"])
    avoidance_applied = bool(native_result.get("avoidance_applied"))
    return SearchResult(
        move=move,
        score=float(native_result["score"]),
        nodes=int(native_result["nodes"]),
        elapsed_ms=elapsed_ms,
        depth=resolved_config.depth,
        completed_depth=completed_depth,
        decision_source=_resolve_decision_source(
            move=move,
            completed_depth=completed_depth,
            timed_out=timed_out,
            avoidance_applied=avoidance_applied,
        ),
        timed_out=timed_out,
        time_budget_ms=resolved_config.time_budget_ms,
        agent_id=resolved_agent.id,
        agent_label=resolved_agent.label,
        root_candidates=_normalize_root_candidates(
            native_result.get("root_candidates"),
            resolved_config.root_candidate_limit,
        ),
        analysis_evaluator_id=resolved_config.analysis_evaluator_id,
        board_token_before=resolved_config.board_token_before,
    )


def _search_best_move_python(
    board: Board,
    player: int,
    resolved_agent: AgentDefinition,
    resolved_config,
) -> SearchResult:
    """Run the Python search path, including quiescence when configured."""
    requested_depth = resolved_config.depth
    remaining_game_moves = resolved_config.remaining_game_moves
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
            analysis_evaluator_id=resolved_config.analysis_evaluator_id,
            board_token_before=resolved_config.board_token_before,
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
    root_candidates: List[Dict[str, object]] = []

    tt: Dict[tuple[int, int, int], TTEntry] = {}
    killer_moves: List[Optional[Move]] = [None] * (requested_depth + 1)
    opponent = _opponent(player)
    child_remaining_game_moves = None if remaining_game_moves is None else max(0, remaining_game_moves - 1)

    for depth in range(1, requested_depth + 1):
        stats = {"nodes": 0}
        for i in range(len(killer_moves)):
            killer_moves[i] = None

        alpha = -inf
        beta = inf
        current_best_score = -inf
        current_best_move = best_move
        depth_candidates: List[Dict[str, object]] = []

        try:
            tt_key = _make_tt_key(board, player, TT_MODE_FULL, remaining_game_moves)
            tt_entry = tt.get(tt_key)
            tt_move = tt_entry.move if tt_entry is not None else None
            ordered_moves = _ordered_moves(board, player, legal_moves, tt_move, killer_moves, depth)

            for move in ordered_moves:
                _check_deadline(deadline_at)
                undo_info = board.apply_move_undo(move, player)
                try:
                    stats["nodes"] += 1
                    value, _ = _minimax(
                        board,
                        opponent,
                        player,
                        depth - 1,
                        child_remaining_game_moves,
                        alpha,
                        beta,
                        resolved_agent.evaluator,
                        resolved_config.tie_break,
                        deadline_at,
                        stats,
                        tt,
                        killer_moves,
                        resolved_config.max_quiescence_depth,
                    )
                finally:
                    board.undo_move(undo_info)

                if value > current_best_score or (
                    value == current_best_score
                    and _prefer_by_tie_break(resolved_config.tie_break, move, current_best_move)
                ):
                    current_best_score = value
                    current_best_move = move
                if resolved_config.root_candidate_limit > 0:
                    depth_candidates.append(
                        {
                            "notation": move.to_notation(),
                            "score": round(value, 6),
                            "depth": depth,
                        }
                    )
                alpha = max(alpha, current_best_score)

            tt[tt_key] = TTEntry(depth=depth, value=current_best_score, flag=EXACT, move=current_best_move)

            total_nodes += stats["nodes"]
            best_move = current_best_move
            best_score = current_best_score
            completed_depth = depth
            if resolved_config.root_candidate_limit > 0:
                root_candidates = sorted(
                    depth_candidates,
                    key=lambda item: (-float(item["score"]), str(item["notation"])),
                )[: resolved_config.root_candidate_limit]

        except _SearchTimeout:
            total_nodes += stats["nodes"]
            best_move = current_best_move
            if current_best_score != -inf:
                best_score = current_best_score
            timed_out = True
            if resolved_config.root_candidate_limit > 0 and depth_candidates:
                root_candidates = sorted(
                    depth_candidates,
                    key=lambda item: (-float(item["score"]), str(item["notation"])),
                )[: resolved_config.root_candidate_limit]
            break

    if completed_depth == 0 and not timed_out:
        best_move = legal_moves[0]
        best_score = 0.0

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return SearchResult(
        move=best_move,
        score=best_score,
        nodes=total_nodes,
        elapsed_ms=elapsed_ms,
        depth=requested_depth,
        completed_depth=completed_depth,
        decision_source=_resolve_decision_source(
            move=best_move,
            completed_depth=completed_depth,
            timed_out=timed_out,
            avoidance_applied=avoidance_applied,
        ),
        timed_out=timed_out,
        time_budget_ms=resolved_config.time_budget_ms,
        agent_id=resolved_agent.id,
        agent_label=resolved_agent.label,
        root_candidates=root_candidates or None,
        analysis_evaluator_id=resolved_config.analysis_evaluator_id,
        board_token_before=resolved_config.board_token_before,
    )
