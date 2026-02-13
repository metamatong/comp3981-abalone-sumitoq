"""Move validation helpers shared by humans and bot players."""

from collections.abc import Mapping
from typing import Optional, Tuple

from ..game.board import Board, Move, str_to_pos


def build_move_from_payload(payload: Optional[Mapping]) -> Tuple[Optional[Move], Optional[str]]:
    if not isinstance(payload, Mapping):
        return None, "Move payload must be an object."

    marbles_raw = payload.get("marbles")
    direction_raw = payload.get("direction")

    if not isinstance(marbles_raw, list):
        return None, "'marbles' must be a list of positions."
    if not 1 <= len(marbles_raw) <= 3:
        return None, "Move must contain between 1 and 3 marbles."
    if len(set(marbles_raw)) != len(marbles_raw):
        return None, "Move contains duplicate marbles."

    if (
        not isinstance(direction_raw, list)
        or len(direction_raw) != 2
    ):
        return None, "'direction' must be a two-item list."

    try:
        marbles = tuple(str_to_pos(str(pos).strip().lower()) for pos in marbles_raw)
        direction = (int(direction_raw[0]), int(direction_raw[1]))
    except (TypeError, ValueError, IndexError):
        return None, "Malformed move payload."

    return Move(marbles=marbles, direction=direction), None


def validate_move(board: Board, player: int, move: Optional[Move]) -> Tuple[bool, Optional[str]]:
    if move is None:
        return False, "Move is missing."
    if not 1 <= move.count <= 3:
        return False, "Move must contain between 1 and 3 marbles."
    if not board.is_legal_move(move, player):
        return False, "Illegal move."
    return True, None


def validate_payload_move(
    board: Board,
    player: int,
    payload: Optional[Mapping],
) -> Tuple[Optional[Move], Optional[str]]:
    move, error = build_move_from_payload(payload)
    if error:
        return None, error

    ok, error = validate_move(board, player, move)
    if not ok:
        return None, error

    return move, None
