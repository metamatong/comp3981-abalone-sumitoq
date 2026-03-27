"""Shared game session state used by CLI and HTTP server."""

import time
from collections.abc import Mapping
from typing import Dict, List, Optional, Set

from ..ai.agent import choose_move_with_info
from ..ai.types import AgentConfig
from ..players.registry import get_agent, list_agent_metadata
from ..players.validator import validate_move, validate_payload_move
from ..state_space import generate_legal_moves
from .board import BLACK, WHITE, Board, EMPTY, Move, is_valid, neighbor, pos_to_str, str_to_pos
from .config import CONTROLLER_AI, CONTROLLER_HUMAN, GameConfig, merge_config


class GameSession:
    """Own the mutable game runtime state shared by CLI and HTTP layers."""

    def __init__(
        self,
        config: Optional[GameConfig] = None,
        initial_time_ms: int = 10 * 60 * 1000,
        opening_seed: Optional[int] = None,
    ):
        """Initialize session state, timers, and a standard starting board."""
        self.initial_time_ms = initial_time_ms
        self.config = config or GameConfig()
        self.opening_seed = opening_seed

        self.board = Board()
        self.current_player = BLACK
        self.move_history: List[dict] = []
        self.time_left_us = {BLACK: self.initial_time_ms * 1000, WHITE: self.initial_time_ms * 1000}
        self.time_used_us = {BLACK: 0, WHITE: 0}
        self.last_clock_update_us = self._now_us()
        self.turn_start_us = self._now_us()
        self.turn_start_epoch_ms = int(time.time() * 1000)
        self.pause_start_us: Optional[int] = None
        self.paused = False
        self.started = False

        # Resign state
        self._resigned = False
        self._resign_winner: Optional[int] = None

        self.board.setup_standard()

    @property
    def controllers(self) -> Dict[int, str]:
        """Return per-player controller mapping (`human` or `ai`)."""
        return self.config.controllers()

    @property
    def current_controller(self) -> str:
        """Return controller type for the side currently on turn."""
        return self.controllers[self.current_player]

    def ai_id_for_player(self, player: int) -> str:
        """Return the configured AI preset ID for a player color."""
        return self.config.black_ai_id if player == BLACK else self.config.white_ai_id

    def _now_us(self) -> int:
        """Return current monotonic time in microseconds."""
        return time.perf_counter_ns() // 1000

    def _time_used_ms(self) -> Dict[int, int]:
        """Return total cumulative time used by each player in milliseconds."""
        return {
            BLACK: self.time_used_us[BLACK] // 1000,
            WHITE: self.time_used_us[WHITE] // 1000,
        }

    def _winner_by_score_then_time(self) -> tuple[Optional[int], Optional[str]]:
        """Resolve terminal winner by score first, then least total time used."""
        if self.board.score[BLACK] > self.board.score[WHITE]:
            return BLACK, None
        if self.board.score[WHITE] > self.board.score[BLACK]:
            return WHITE, None

        if self.time_used_us[BLACK] < self.time_used_us[WHITE]:
            return BLACK, "least_total_time"
        if self.time_used_us[WHITE] < self.time_used_us[BLACK]:
            return WHITE, "least_total_time"

        return None, None

    def _status_payload(
        self,
        game_over: bool,
        winner: Optional[int],
        game_over_reason: Optional[str],
        winner_tiebreak: Optional[str] = None,
    ) -> dict:
        """Build a consistent status payload including additive timing fields."""
        return {
            "game_over": game_over,
            "winner": winner,
            "game_over_reason": game_over_reason,
            "winner_tiebreak": winner_tiebreak,
            "time_used_ms": self._time_used_ms(),
        }

    def _status(self) -> dict:
        """Compute terminal status metadata without mutating session state."""
        # Check resign first
        if self._resigned:
            return self._status_payload(True, self._resign_winner, "resign")

        if self.board.score[BLACK] >= 6:
            return self._status_payload(True, BLACK, "score")

        if self.board.score[WHITE] >= 6:
            return self._status_payload(True, WHITE, "score")

        # Timeout: shared game time expired (sum of both clocks)
        total_time_left = self.time_left_us[BLACK] + self.time_left_us[WHITE]
        if total_time_left <= 0:
            winner, winner_tiebreak = self._winner_by_score_then_time()
            return self._status_payload(True, winner, "timeout", winner_tiebreak)

        # Max moves check
        if self.config.max_moves > 0 and len(self.move_history) >= self.config.max_moves:
            winner, winner_tiebreak = self._winner_by_score_then_time()
            return self._status_payload(True, winner, "max_moves", winner_tiebreak)

        return self._status_payload(False, None, None)

    def _tick_clock(self):
        """Update active player's remaining time based on elapsed monotonic time."""
        now = self._now_us()
        if not self.started or self._status()["game_over"] or self.paused:
            self.last_clock_update_us = now
            return

        elapsed = max(0, now - self.last_clock_update_us)
        if elapsed > 0:
            self.time_used_us[self.current_player] += elapsed
            self.time_left_us[self.current_player] = max(0, self.time_left_us[self.current_player] - elapsed)

        self.last_clock_update_us = now

    def _check_move_time_limit(self):
        """If per-move time limit is exceeded, auto-switch to next player."""
        if not self.started:
            return

        # Use per-player turn time limit
        if self.current_player == BLACK:
            turn_limit_s = self.config.player1_time_per_turn_s
        else:
            turn_limit_s = self.config.player2_time_per_turn_s

        if turn_limit_s <= 0:
            return
        if self._status()["game_over"] or self.paused:
            return

        now = self._now_us()
        elapsed_us = max(0, now - self.turn_start_us)
        limit_us = turn_limit_s * 1000000

        if elapsed_us >= limit_us:
            # Auto-switch turn (skip current player's move)
            self.current_player = WHITE if self.current_player == BLACK else BLACK
            self.last_clock_update_us = now
            self.turn_start_us = now
            self.turn_start_epoch_ms = int(time.time() * 1000)

    def _before_turn_action(self) -> Optional[str]:
        """Run shared pre-action checks and return an error message when blocked."""
        self._tick_clock()
        self._check_move_time_limit()

        if self._status()["game_over"]:
            return "Game is over"
        if self.paused:
            return "Game is paused"
        return None

    def configure(self, payload: Optional[Mapping]) -> dict:
        """Apply a partial runtime config update and return normalized config values."""
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
                "black_ai_id": self.config.black_ai_id,
                "white_ai_id": self.config.white_ai_id,
                "board_layout": self.config.board_layout,
                "game_time_ms": self.config.game_time_ms,
                "max_moves": self.config.max_moves,
                "player1_time_per_turn_s": self.config.player1_time_per_turn_s,
                "player2_time_per_turn_s": self.config.player2_time_per_turn_s,
            },
            "available_agents": list_agent_metadata(),
        }

    def status(self) -> dict:
        """Return current game status after advancing session clocks."""
        self._tick_clock()
        return self._status()

    def _moved_positions(self, move, result: Mapping) -> List[str]:
        """Return board coordinates occupied by marbles that moved in this turn."""
        moved: Set[str] = set()

        for marble in move.marbles:
            dest = neighbor(marble, move.direction)
            if is_valid(dest):
                moved.add(pos_to_str(dest))

        for pushed in result.get("pushed", []):
            pushed_pos = str_to_pos(pushed)
            dest = neighbor(pushed_pos, move.direction)
            if is_valid(dest):
                moved.add(pos_to_str(dest))

        return sorted(moved)

    def _apply_move(
        self,
        move,
        source: str,
        search: Optional[dict] = None,
        agent_id: Optional[str] = None,
        agent_label: Optional[str] = None,
    ) -> dict:
        """Apply a validated move, record history entry, and advance the turn."""
        self._tick_clock()
        player = self.current_player
        snapshot = self.board.copy()
        clock_snapshot = dict(self.time_left_us)
        time_used_snapshot = dict(self.time_used_us)

        now = self._now_us()
        duration_ms = max(0, now - self.turn_start_us) // 1000

        result = self.board.apply_move(move, player)
        moved_to = self._moved_positions(move, result)
        self.move_history.append(
            {
                "move": move,
                "result": result,
                "moved_to": moved_to,
                "snapshot": snapshot,
                "clock_snapshot": clock_snapshot,
                "time_used_snapshot": time_used_snapshot,
                "player": player,
                "source": source,
                "search": search,
                "agent_id": agent_id,
                "agent_label": agent_label,
                "duration_ms": duration_ms,
            }
        )

        self.current_player = WHITE if self.current_player == BLACK else BLACK
        self.last_clock_update_us = now
        self.turn_start_us = now
        self.turn_start_epoch_ms = int(time.time() * 1000)

        return {
            "ok": True,
            "player": player,
            "notation": move.to_notation(pushed=bool(result["pushed"])),
            "result": result,
            "source": source,
            "search": search,
            "agent_id": agent_id,
            "agent_label": agent_label,
        }

    def apply_human_move(self, payload: Optional[Mapping]) -> dict:
        """Validate and apply a user-provided move payload for human-controlled turns."""
        error = self._before_turn_action()
        if error:
            return {"error": error}

        if self.current_controller != CONTROLLER_HUMAN:
            return {"error": "It is an AI-controlled turn."}

        move, error = validate_payload_move(self.board, self.current_player, payload)
        if error:
            return {"error": error}

        return self._apply_move(move, source=CONTROLLER_HUMAN)

    def _current_turn_budget_ms(self) -> Optional[int]:
        """Return remaining search budget for the active turn, including safety margin."""
        if self.current_player == BLACK:
            turn_limit_s = self.config.player1_time_per_turn_s
        else:
            turn_limit_s = self.config.player2_time_per_turn_s
        if turn_limit_s <= 0:
            return None

        elapsed_ms = max(0, (self._now_us() - self.turn_start_us) // 1000)
        remaining_ms = (turn_limit_s * 1000) - elapsed_ms - 100
        return max(0, remaining_ms)

    @staticmethod
    def _board_signature(board: Board) -> tuple:
        """Serialize board occupancy and score for cycle detection."""
        occupied = tuple(sorted((pos, val) for pos, val in board.cells.items() if val != EMPTY))
        return (occupied, board.score[BLACK], board.score[WHITE])

    def _repeat_move_to_avoid(self) -> Optional[Move]:
        """Detect repeated board states and return the last move from that state to avoid."""
        if len(self.move_history) < 2:
            return None

        current_sig = self._board_signature(self.board)
        # Look back at least one full ply so we only match prior states.
        for index in range(len(self.move_history) - 2, -1, -1):
            entry = self.move_history[index]
            if entry.get("player") != self.current_player:
                continue
            snapshot = entry.get("snapshot")
            move = entry.get("move")
            if snapshot is None or move is None:
                continue
            if self._board_signature(snapshot) == current_sig:
                return move
        return None

    def apply_agent_move(self) -> dict:
        """Ask the minimax agent for a move and apply it on AI-controlled turns."""
        error = self._before_turn_action()
        if error:
            return {"error": error}

        if self.current_controller != CONTROLLER_AI:
            return {"error": "It is a human-controlled turn."}

        agent_id = self.ai_id_for_player(self.current_player)
        agent = get_agent(agent_id)
        time_budget_ms = self._current_turn_budget_ms()
        avoid_move = self._repeat_move_to_avoid()
        search_result = choose_move_with_info(
            self.board,
            self.current_player,
            agent=agent,
            config=AgentConfig(
                depth=self.config.ai_depth,
                time_budget_ms=time_budget_ms,
                opening_seed=self.opening_seed,
                is_opening_turn=(self.current_player == BLACK and not self.move_history),
                avoid_move=avoid_move,
            ),
        )

        move = search_result.move
        ok, error = validate_move(self.board, self.current_player, move)
        if not ok:
            return {"error": f"Agent produced invalid move: {error}"}

        return self._apply_move(
            move,
            source=CONTROLLER_AI,
            search=search_result.as_dict(),
            agent_id=agent.id,
            agent_label=agent.label,
        )

    def undo(self) -> dict:
        """Revert the last move, including board and clock snapshots."""
        if not self.move_history:
            return {"error": "Nothing to undo"}

        entry = self.move_history.pop()
        self.board = entry["snapshot"]
        self.current_player = entry["player"]
        self.time_left_us = dict(entry["clock_snapshot"])
        self.time_used_us = dict(entry.get("time_used_snapshot", {BLACK: 0, WHITE: 0}))
        now = self._now_us()
        self.last_clock_update_us = now
        self.turn_start_us = now
        self.turn_start_epoch_ms = int(time.time() * 1000)
        return {"ok": True}

    def reset(self) -> dict:
        """Reset board, timers, and session flags using current config values."""
        self.board = Board()
        self.board.setup_layout(self.config.board_layout)
        self.current_player = BLACK
        self.move_history = []

        # Use shared game time from config, fallback to initial_time_ms
        # Each player gets half the total game time
        game_time = self.config.game_time_ms if self.config.game_time_ms > 0 else self.initial_time_ms
        per_player_time_us = (game_time // 2) * 1000
        self.time_left_us = {BLACK: per_player_time_us, WHITE: per_player_time_us}
        self.time_used_us = {BLACK: 0, WHITE: 0}

        now = self._now_us()
        self.last_clock_update_us = now
        self.turn_start_us = now
        self.turn_start_epoch_ms = int(time.time() * 1000)
        self.pause_start_us = None
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
        """Toggle pause state while preserving accurate per-turn timing."""
        self._tick_clock()
        now = self._now_us()
        if not self.paused:
            # Pausing: record when we paused
            self.paused = True
            self.pause_start_us = now
        else:
            # Resuming: shift turn_start_us forward so paused time isn't counted
            self.paused = False
            if self.pause_start_us is not None:
                paused_duration = now - self.pause_start_us
                self.turn_start_us += paused_duration
                self.pause_start_us = None
        self.last_clock_update_us = now
        return {"ok": True, "paused": self.paused}

    def state_json(self) -> dict:
        """Serialize full session state for the web client API contract."""
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
                    "agent_id": entry.get("agent_id"),
                    "agent_label": entry.get("agent_label"),
                    "duration_ms": entry.get("duration_ms", 0),
                }
            )

        last_move_marbles = []
        last_move_direction = []
        if self.move_history:
            last_move_marbles = list(self.move_history[-1].get("moved_to", []))
            dr, dc = self.move_history[-1]["move"].direction
            last_move_direction = [dr, dc]

        controllers = {str(player): ctrl for player, ctrl in self.controllers.items()}
        return {
            "cells": cells,
            "current_player": self.current_player,
            "current_controller": self.current_controller,
            "controllers": controllers,
            "mode": self.config.mode,
            "human_side": self.config.human_side,
            "ai_depth": self.config.ai_depth,
            "black_ai_id": self.config.black_ai_id,
            "white_ai_id": self.config.white_ai_id,
            "board_layout": self.config.board_layout,
            "game_time_ms": self.config.game_time_ms,
            "max_moves": self.config.max_moves,
            "player1_time_per_turn_s": self.config.player1_time_per_turn_s,
            "player2_time_per_turn_s": self.config.player2_time_per_turn_s,
            "available_agents": list_agent_metadata(),
            "score": self.board.score,
            "game_over": status["game_over"],
            "winner": status["winner"],
            "game_over_reason": status["game_over_reason"],
            "winner_tiebreak": status["winner_tiebreak"],
            "legal_moves": legal_list,
            "history": history,
            "last_move_marbles": last_move_marbles,
            "last_move_direction": last_move_direction,
            "marble_counts": {
                BLACK: self.board.marble_count(BLACK),
                WHITE: self.board.marble_count(WHITE),
            },
            "time_left_ms": {
                BLACK: self.time_left_us[BLACK] // 1000,
                WHITE: self.time_left_us[WHITE] // 1000,
            },
            "time_used_ms": status["time_used_ms"],
            "paused": self.paused,
            "initial_time_ms": self.initial_time_ms,
            "turn_start_ms": self.turn_start_epoch_ms,
            "started": self.started,
        }
