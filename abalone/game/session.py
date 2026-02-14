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
    def __init__(self, config: Optional[GameConfig] = None, initial_time_ms: int = 10 * 60 * 1000):
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
        self.started = False

        # Resign state
        self._resigned = False
        self._resign_winner: Optional[int] = None

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
        # Check resign first
        if self._resigned:
            return {
                "game_over": True,
                "winner": self._resign_winner,
                "game_over_reason": "resign",
                "timeout_player": None,
            }

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

        # Max moves check
        if self.config.max_moves > 0 and len(self.move_history) >= self.config.max_moves:
            # Winner is the player who captured more marbles
            if self.board.score[BLACK] > self.board.score[WHITE]:
                winner = BLACK
            elif self.board.score[WHITE] > self.board.score[BLACK]:
                winner = WHITE
            else:
                winner = None  # draw
            return {
                "game_over": True,
                "winner": winner,
                "game_over_reason": "max_moves",
                "timeout_player": None,
            }

        return {
            "game_over": False,
            "winner": None,
            "game_over_reason": None,
            "timeout_player": None,
        }

    def _tick_clock(self):
        now = self._now_ms()
        if not self.started or self._status()["game_over"] or self.paused:
            self.last_clock_update_ms = now
            return

        elapsed = max(0, now - self.last_clock_update_ms)
        if elapsed > 0:
            self.time_left_ms[self.current_player] = max(0, self.time_left_ms[self.current_player] - elapsed)

        self.last_clock_update_ms = now

    def _check_move_time_limit(self):
        """If per-move time limit is exceeded, auto-switch to next player."""
        if not self.started:
            return
        if self.config.time_limit_per_move_s <= 0:
            return
        if self._status()["game_over"] or self.paused:
            return

        now = self._now_ms()
        elapsed_ms = max(0, now - self.turn_start_ms)
        limit_ms = self.config.time_limit_per_move_s * 1000

        if elapsed_ms >= limit_ms:
            # Auto-switch turn (skip current player's move)
            self.current_player = WHITE if self.current_player == BLACK else BLACK
            self.last_clock_update_ms = now
            self.turn_start_ms = now

    def _before_turn_action(self) -> Optional[str]:
        self._tick_clock()
        self._check_move_time_limit()

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
                "board_layout": self.config.board_layout,
                "player1_time_ms": self.config.player1_time_ms,
                "player2_time_ms": self.config.player2_time_ms,
                "max_moves": self.config.max_moves,
                "time_limit_per_move_s": self.config.time_limit_per_move_s,
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
        self.board.setup_layout(self.config.board_layout)
        self.current_player = BLACK
        self.move_history = []

        # Use per-player time from config, fallback to initial_time_ms
        p1_time = self.config.player1_time_ms if self.config.player1_time_ms > 0 else self.initial_time_ms
        p2_time = self.config.player2_time_ms if self.config.player2_time_ms > 0 else self.initial_time_ms
        self.time_left_ms = {BLACK: p1_time, WHITE: p2_time}

        now = self._now_ms()
        self.last_clock_update_ms = now
        self.turn_start_ms = now
        self.pause_start_ms = None
        self.paused = False
        self.started = True
        self._resigned = False
        self._resign_winner = None
        return {"ok": True}

    def resign(self, payload: Optional[Mapping] = None) -> dict:
        """Current player resigns. Opponent wins."""
        if self._status()["game_over"]:
            return {"error": "Game is already over"}

        self._resigned = True
        self._resign_winner = WHITE if self.current_player == BLACK else BLACK
        return {"ok": True, "winner": self._resign_winner}

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
        self._check_move_time_limit()
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
            "board_layout": self.config.board_layout,
            "max_moves": self.config.max_moves,
            "time_limit_per_move_s": self.config.time_limit_per_move_s,
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
            "turn_start_ms": self.turn_start_ms,
            "started": self.started,
        }
