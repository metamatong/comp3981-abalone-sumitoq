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


@dataclass(frozen=True)
class GameConfig:
    mode: str = MODE_HVH
    human_side: int = BLACK
    ai_depth: int = 2

    def controllers(self) -> Dict[int, str]:
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
    mode = str(value).strip().lower()
    if mode not in VALID_MODES:
        raise ValueError(f"Unsupported mode '{value}'.")
    return mode


def normalize_human_side(value: object) -> int:
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
    try:
        depth = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("ai_depth must be an integer.") from exc

    if depth < 1 or depth > 5:
        raise ValueError("ai_depth must be between 1 and 5.")
    return depth


def merge_config(current: GameConfig, payload: Optional[dict]) -> GameConfig:
    if payload is None:
        return current

    mode = current.mode
    human_side = current.human_side
    ai_depth = current.ai_depth

    if "mode" in payload:
        mode = normalize_mode(payload["mode"])
    if "human_side" in payload:
        human_side = normalize_human_side(payload["human_side"])
    if "ai_depth" in payload:
        ai_depth = normalize_depth(payload["ai_depth"])

    return GameConfig(mode=mode, human_side=human_side, ai_depth=ai_depth)
