"""Helpers for the required native search extension on this branch."""

from __future__ import annotations

from typing import Iterable, Optional

try:
    from . import _native as _native_ext
except ImportError:
    _native_ext = None


def is_available() -> bool:
    """Return whether the compiled native extension is importable."""
    return _native_ext is not None


def missing_extension_message(build_command: Optional[str] = None) -> str:
    """Return a short user-facing message for the missing native engine."""
    if build_command is None:
        build_command = "python3 setup.py build_ext --inplace"
    return (
        "Native engine not built for this branch.\n"
        f"Build it first: `{build_command}`\n"
        "If that fails on Windows, install MSVC Build Tools and retry."
    )


def require_available():
    """Raise a helpful runtime error when the compiled extension is missing."""
    if _native_ext is None:
        raise RuntimeError(missing_extension_message())
    return _native_ext


def preflight_or_exit(build_command: Optional[str] = None) -> None:
    """Exit early with short build instructions when the native engine is missing."""
    if is_available():
        return
    raise SystemExit(missing_extension_message(build_command))


def _board_api():
    from .game.board import (
        BLACK,
        DIRECTION_INDEX,
        DIRECTIONS,
        Move,
        ORDERED_VALID_POSITIONS,
        POSITION_INDEX,
        WHITE,
    )

    return {
        "BLACK": BLACK,
        "DIRECTION_INDEX": DIRECTION_INDEX,
        "DIRECTIONS": DIRECTIONS,
        "Move": Move,
        "ORDERED_VALID_POSITIONS": ORDERED_VALID_POSITIONS,
        "POSITION_INDEX": POSITION_INDEX,
        "WHITE": WHITE,
    }


def _encode_board_cells(board) -> bytes:
    api = _board_api()
    ordered_valid_positions = api["ORDERED_VALID_POSITIONS"]
    return bytes(int(board.cells[pos]) for pos in ordered_valid_positions)


def _decode_move(payload) -> Optional[object]:
    api = _board_api()
    if payload is None:
        return None
    marble_indexes, direction_index = payload
    ordered_valid_positions = api["ORDERED_VALID_POSITIONS"]
    move_type = api["Move"]
    directions = api["DIRECTIONS"]
    marbles = tuple(ordered_valid_positions[int(index)] for index in marble_indexes)
    return move_type.from_canonical(marbles, directions[int(direction_index)])


def _encode_move(move: Optional[object]):
    api = _board_api()
    if move is None:
        return None
    position_index = api["POSITION_INDEX"]
    direction_index = api["DIRECTION_INDEX"]
    marble_indexes = tuple(position_index[pos] for pos in move.marbles)
    return marble_indexes, direction_index[move.direction]


def generate_legal_moves(board, player: int):
    """Return native-generated legal moves."""
    native_ext = require_available()
    return [
        _decode_move(payload)
        for payload in native_ext.generate_legal_moves(_encode_board_cells(board), int(player))
    ]


def evaluate_weighted(board, player: int, ordered_weights: Iterable[float]):
    """Return a native weighted evaluation."""
    native_ext = require_available()
    api = _board_api()
    return float(
        native_ext.evaluate_weighted(
            _encode_board_cells(board),
            int(board.score[api["BLACK"]]),
            int(board.score[api["WHITE"]]),
            int(player),
            tuple(float(weight) for weight in ordered_weights),
        )
    )


def search_weighted(
    board,
    player: int,
    ordered_weights: Iterable[float],
    *,
    depth: int,
    time_budget_ms,
    tie_break: str,
    avoid_move: Optional[object],
    root_candidate_limit: int,
):
    """Run the native weighted search."""
    native_ext = require_available()
    api = _board_api()

    raw_result = native_ext.search_weighted(
        _encode_board_cells(board),
        int(board.score[api["BLACK"]]),
        int(board.score[api["WHITE"]]),
        int(player),
        tuple(float(weight) for weight in ordered_weights),
        int(depth),
        time_budget_ms,
        str(tie_break),
        _encode_move(avoid_move),
        int(root_candidate_limit),
    )
    result = dict(raw_result)
    result["move"] = _decode_move(result.get("move"))
    raw_candidates = result.get("root_candidates") or []
    result["root_candidates"] = [
        {
            "move": _decode_move(candidate.get("move")),
            "score": float(candidate.get("score", 0.0)),
            "depth": int(candidate.get("depth", 0)),
        }
        for candidate in raw_candidates
    ]
    return result
