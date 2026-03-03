"""Configuration helpers for game mode and controller wiring."""

from dataclasses import dataclass
from typing import Dict, Optional

from .board import BLACK, WHITE

MODE_HVH = "hvh"
MODE_HVA = "hva"
MODE_AVA = "ava"

CONTROLLER_HUMAN = "human"
CONTROLLER_AI = "ai"

VALID_MODES = {MODE_HVH, MODE_HVA, MODE_AVA}
VALID_LAYOUTS = {"standard", "belgian_daisy", "german_daisy"}


@dataclass(frozen=True)
class GameConfig:
    """Runtime settings that determine controllers, layout, and timing behavior."""

    mode: str = MODE_HVH
    human_side: int = BLACK
    ai_depth: int = 2
    board_layout: str = "standard"
    game_time_ms: int = 30 * 60 * 1000  # 0 = unlimited
    max_moves: int = 500
    player1_time_per_turn_s: int = 30
    player2_time_per_turn_s: int = 30

    def controllers(self) -> Dict[int, str]:
        """Return the per-color controller mapping derived from the selected mode."""
        if self.mode == MODE_HVH:
            return {BLACK: CONTROLLER_HUMAN, WHITE: CONTROLLER_HUMAN}
        if self.mode == MODE_AVA:
            return {BLACK: CONTROLLER_AI, WHITE: CONTROLLER_AI}

        ai_side = WHITE if self.human_side == BLACK else BLACK
        return {
            self.human_side: CONTROLLER_HUMAN,
            ai_side: CONTROLLER_AI,
        }


def normalize_mode(value: object) -> str:
    """Normalize and validate game mode input."""
    mode = str(value).strip().lower()
    if mode not in VALID_MODES:
        raise ValueError(f"Unsupported mode '{value}'.")
    return mode


def normalize_human_side(value: object) -> int:
    """Normalize and validate the side selected for a human player."""
    if isinstance(value, int):
        if value in (BLACK, WHITE):
            return value
        raise ValueError("human_side must be 1 (black) or 2 (white).")

    side = str(value).strip().lower()
    if side in ("black", "b", str(BLACK)):
        return BLACK
    if side in ("white", "w", str(WHITE)):
        return WHITE
    raise ValueError(f"Unsupported human side '{value}'.")


def normalize_depth(value: object) -> int:
    """Normalize and validate search depth for the AI agent."""
    try:
        depth = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("ai_depth must be an integer.") from exc

    if depth < 1 or depth > 5:
        raise ValueError("ai_depth must be between 1 and 5.")
    return depth


def normalize_board_layout(value: object) -> str:
    """Normalize and validate board layout identifier."""
    layout = str(value).strip().lower()
    if layout not in VALID_LAYOUTS:
        raise ValueError(f"Unsupported board layout '{value}'.")
    return layout


def normalize_non_negative_int(value: object, name: str) -> int:
    """Convert a value to a non-negative integer with field-specific error text."""
    try:
        v = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if v < 0:
        raise ValueError(f"{name} must be non-negative.")
    return v


def merge_config(current: GameConfig, payload: Optional[dict]) -> GameConfig:
    """Merge a partial config payload into an existing immutable game config."""
    if payload is None:
        return current

    mode = current.mode
    human_side = current.human_side
    ai_depth = current.ai_depth
    board_layout = current.board_layout
    game_time_ms = current.game_time_ms
    max_moves = current.max_moves
    player1_time_per_turn_s = current.player1_time_per_turn_s
    player2_time_per_turn_s = current.player2_time_per_turn_s

    if "mode" in payload:
        mode = normalize_mode(payload["mode"])
    if "human_side" in payload:
        human_side = normalize_human_side(payload["human_side"])
    if "ai_depth" in payload:
        ai_depth = normalize_depth(payload["ai_depth"])
    if "board_layout" in payload:
        board_layout = normalize_board_layout(payload["board_layout"])
    if "game_time_ms" in payload:
        game_time_ms = normalize_non_negative_int(payload["game_time_ms"], "game_time_ms")
    if "max_moves" in payload:
        max_moves = normalize_non_negative_int(payload["max_moves"], "max_moves")
    if "player1_time_per_turn_s" in payload:
        player1_time_per_turn_s = normalize_non_negative_int(payload["player1_time_per_turn_s"], "player1_time_per_turn_s")
    if "player2_time_per_turn_s" in payload:
        player2_time_per_turn_s = normalize_non_negative_int(payload["player2_time_per_turn_s"], "player2_time_per_turn_s")

    return GameConfig(
        mode=mode,
        human_side=human_side,
        ai_depth=ai_depth,
        board_layout=board_layout,
        game_time_ms=game_time_ms,
        max_moves=max_moves,
        player1_time_per_turn_s=player1_time_per_turn_s,
        player2_time_per_turn_s=player2_time_per_turn_s,
    )
