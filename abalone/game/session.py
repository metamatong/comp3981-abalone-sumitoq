"""Shared game session state used by CLI and HTTP server."""

import time
from collections.abc import Mapping
from typing import Dict, List, Optional

from ..players.agent import choose_move_with_info
from ..players.types import AgentConfig
from ..players.validator import validate_move, validate_payload_move
from ..state_space import generate_legal_moves
from .board import BLACK, WHITE, Board, pos_to_str
from .config import CONTROLLER_AI, CONTROLLER_HUMAN, GameConfig, merge_config


class GameSession:
    def __init__(self, config: Optional[GameConfig] = None, initial_time_ms: int = 30 * 60 * 1000):
        self.initial_time_ms = initial_time_ms
        self.config = config or GameConfig()

        self.board = Board()
        self.current_player = BLACK
        self.move_history: List[dict] = []
        self.time_left_ms = {BLACK: self.initial_time_ms, WHITE: self.initial_time_ms}
        self.last_clock_update_ms = self._now_ms()
        self.turn_start_ms = self._now_ms()
        self.pause_start_ms: Optional[int] = None
        self.paused = False

        self.board.setup_standard()

    @property
    def controllers(self) -> Dict[int, str]:
        return self.config.controllers()

    @property
    def current_controller(self) -> str:
        return self.controllers[self.current_player]

    def _now_ms(self) -> int:
        return int(time.time() * 1000)

    def _status(self) -> dict:
        if self.board.score[BLACK] >= 6:
            return {
                "game_over": True,
                "winner": BLACK,
                "game_over_reason": "score",
                "timeout_player": None,
            }

        if self.board.score[WHITE] >= 6:
            return {
                "game_over": True,
                "winner": WHITE,
                "game_over_reason": "score",
                "timeout_player": None,
            }

        if self.time_left_ms[BLACK] <= 0:
            return {
                "game_over": True,
                "winner": WHITE,
                "game_over_reason": "timeout",
                "timeout_player": BLACK,
            }

        if self.time_left_ms[WHITE] <= 0:
            return {
                "game_over": True,
                "winner": BLACK,
                "game_over_reason": "timeout",
                "timeout_player": WHITE,
            }

        return {
            "game_over": False,
            "winner": None,
            "game_over_reason": None,
            "timeout_player": None,
        }

    def _tick_clock(self):
        now = self._now_ms()
        if self._status()["game_over"] or self.paused:
            self.last_clock_update_ms = now
            return

        elapsed = max(0, now - self.last_clock_update_ms)
        if elapsed > 0:
            self.time_left_ms[self.current_player] = max(0, self.time_left_ms[self.current_player] - elapsed)

        self.last_clock_update_ms = now

    def _before_turn_action(self) -> Optional[str]:
        self._tick_clock()

        if self._status()["game_over"]:
            return "Game is over"
        if self.paused:
            return "Game is paused"
        return None

    def configure(self, payload: Optional[Mapping]) -> dict:
        if payload is None:
            payload = {}

        try:
            self.config = merge_config(self.config, dict(payload))
        except (TypeError, ValueError) as exc:
            return {"error": str(exc)}

        return {
            "ok": True,
            "config": {
                "mode": self.config.mode,
                "human_side": self.config.human_side,
                "ai_depth": self.config.ai_depth,
            },
        }

    def status(self) -> dict:
        self._tick_clock()
        return self._status()

    def _apply_move(self, move, source: str, search: Optional[dict] = None) -> dict:
        player = self.current_player
        snapshot = self.board.copy()
        clock_snapshot = dict(self.time_left_ms)

        now = self._now_ms()
        duration_ms = max(0, now - self.turn_start_ms)

        result = self.board.apply_move(move, player)
        self.move_history.append(
            {
                "move": move,
                "result": result,
                "snapshot": snapshot,
                "clock_snapshot": clock_snapshot,
                "player": player,
                "source": source,
                "search": search,
                "duration_ms": duration_ms,
            }
        )

        self.current_player = WHITE if self.current_player == BLACK else BLACK
        self.last_clock_update_ms = now
        self.turn_start_ms = now

        return {
            "ok": True,
            "player": player,
            "notation": move.to_notation(pushed=bool(result["pushed"])),
            "result": result,
            "source": source,
            "search": search,
        }

    def apply_human_move(self, payload: Optional[Mapping]) -> dict:
        error = self._before_turn_action()
        if error:
            return {"error": error}

        if self.current_controller != CONTROLLER_HUMAN:
            return {"error": "It is an AI-controlled turn."}

        move, error = validate_payload_move(self.board, self.current_player, payload)
        if error:
            return {"error": error}

        return self._apply_move(move, source=CONTROLLER_HUMAN)

    def apply_agent_move(self) -> dict:
        error = self._before_turn_action()
        if error:
            return {"error": error}

        if self.current_controller != CONTROLLER_AI:
            return {"error": "It is a human-controlled turn."}

        search_result = choose_move_with_info(
            self.board,
            self.current_player,
            config=AgentConfig(depth=self.config.ai_depth),
        )

        move = search_result.move
        ok, error = validate_move(self.board, self.current_player, move)
        if not ok:
            return {"error": f"Agent produced invalid move: {error}"}

        return self._apply_move(move, source=CONTROLLER_AI, search=search_result.as_dict())

    def undo(self) -> dict:
        if not self.move_history:
            return {"error": "Nothing to undo"}

        entry = self.move_history.pop()
        self.board = entry["snapshot"]
        self.current_player = entry["player"]
        self.time_left_ms = dict(entry["clock_snapshot"])
        now = self._now_ms()
        self.last_clock_update_ms = now
        self.turn_start_ms = now
        return {"ok": True}

    def reset(self) -> dict:
        self.board = Board()
        self.board.setup_standard()
        self.current_player = BLACK
        self.move_history = []
        self.time_left_ms = {BLACK: self.initial_time_ms, WHITE: self.initial_time_ms}
        now = self._now_ms()
        self.last_clock_update_ms = now
        self.turn_start_ms = now
        self.pause_start_ms = None
        self.paused = False
        return {"ok": True}

    def toggle_pause(self) -> dict:
        self._tick_clock()
        now = self._now_ms()
        if not self.paused:
            # Pausing: record when we paused
            self.paused = True
            self.pause_start_ms = now
        else:
            # Resuming: shift turn_start_ms forward so paused time isn't counted
            self.paused = False
            if self.pause_start_ms is not None:
                paused_duration = now - self.pause_start_ms
                self.turn_start_ms += paused_duration
                self.pause_start_ms = None
        self.last_clock_update_ms = now
        return {"ok": True, "paused": self.paused}

    def state_json(self) -> dict:
        self._tick_clock()
        status = self._status()

        cells = {}
        for pos, val in self.board.cells.items():
            cells[pos_to_str(pos)] = val

        legal_list = []
        if not status["game_over"] and self.current_controller == CONTROLLER_HUMAN and not self.paused:
            for move in generate_legal_moves(self.board, self.current_player):
                dr, dc = move.direction
                legal_list.append(
                    {
                        "marbles": [pos_to_str(p) for p in move.marbles],
                        "direction": [dr, dc],
                        "notation": move.to_notation(),
                        "is_inline": move.is_inline,
                    }
                )

        history = []
        for entry in self.move_history:
            history.append(
                {
                    "notation": entry["move"].to_notation(pushed=bool(entry["result"]["pushed"])),
                    "player": entry["player"],
                    "pushoff": entry["result"]["pushoff"],
                    "source": entry["source"],
                    "search": entry["search"],
                    "duration_ms": entry.get("duration_ms", 0),
                }
            )

        controllers = {str(player): ctrl for player, ctrl in self.controllers.items()}
        return {
            "cells": cells,
            "current_player": self.current_player,
            "current_controller": self.current_controller,
            "controllers": controllers,
            "mode": self.config.mode,
            "human_side": self.config.human_side,
            "ai_depth": self.config.ai_depth,
            "score": self.board.score,
            "game_over": status["game_over"],
            "winner": status["winner"],
            "game_over_reason": status["game_over_reason"],
            "timeout_player": status["timeout_player"],
            "legal_moves": legal_list,
            "history": history,
            "marble_counts": {
                BLACK: self.board.marble_count(BLACK),
                WHITE: self.board.marble_count(WHITE),
            },
            "time_left_ms": {
                BLACK: self.time_left_ms[BLACK],
                WHITE: self.time_left_ms[WHITE],
            },
            "paused": self.paused,
            "initial_time_ms": self.initial_time_ms,
        }
