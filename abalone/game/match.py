"""Lightweight AI-vs-AI match runner for local benchmarking."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
import sys
import threading
import time
from typing import Dict, List, Optional

from ..eval.gauntlet import PARTIAL_SEARCH_SOURCES, resolve_worker_count, run_game_session
from .board import BLACK, WHITE
from ..players.registry import get_agent


@dataclass
class _AgentStats:
    wins: int = 0
    losses: int = 0
    draws: int = 0
    captures: int = 0
    partial_searches: int = 0
    total_move_time_ms: float = 0.0
    completed_depth_sum: int = 0
    moves: int = 0


class _MatchProgressDisplay:
    """Lightweight live progress display for serial side-swapped match runs."""

    _FRAMES = ("|", "/", "-", "\\")

    def __init__(self, game_count: int, worker_count: int, stream=None, refresh_s: float = 0.12):
        self.game_count = game_count
        self.worker_count = worker_count
        self.stream = stream or sys.stderr
        self.refresh_s = refresh_s
        self.started_at = time.perf_counter()
        self.completed = 0
        self.latest_game = None
        self._frame_index = 0
        self._last_width = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._live = bool(getattr(self.stream, "isatty", lambda: False)())

    def start(self) -> None:
        if not self._live:
            return
        self._render_live_line()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def update(self, completed: int, game: dict) -> None:
        if not self._live:
            return
        with self._lock:
            self.completed = completed
            self.latest_game = game

    def finish(self) -> None:
        if not self._live:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def warn_text(self, text: str) -> None:
        print(text, file=self.stream, flush=True)

    def restart_serial(self) -> None:
        self.warn_text(f"[match] Restarting benchmark serially for {self.game_count} game(s).")

    def _run_loop(self) -> None:
        while not self._stop.wait(self.refresh_s):
            self._render_live_line()
        self._render_live_line()
        print(file=self.stream, flush=True)

    def _render_live_line(self) -> None:
        with self._lock:
            line = self._build_live_line()
            padded = line.ljust(self._last_width)
            print(f"\r{padded}", end="", file=self.stream, flush=True)
            self._last_width = max(self._last_width, len(line))

    def _build_live_line(self) -> str:
        frame = self._FRAMES[self._frame_index]
        self._frame_index = (self._frame_index + 1) % len(self._FRAMES)
        elapsed_s = max(0.0, time.perf_counter() - self.started_at)
        remaining = max(0, self.game_count - self.completed)
        avg_s = elapsed_s / self.completed if self.completed else 0.0
        eta_s = avg_s * remaining
        if self.latest_game is None:
            last_summary = "last=starting"
        else:
            winner = _winner_label(self.latest_game)
            last_summary = (
                f"last={self.latest_game.get('black_ai_id', '?')} vs "
                f"{self.latest_game.get('white_ai_id', '?')} -> {winner}"
            )
        return (
            f"[match] {frame} {_progress_bar(self.completed, self.game_count)} "
            f"{self.completed}/{self.game_count} elapsed={_format_duration(elapsed_s)} "
            f"eta={_format_duration(eta_s)} mode={self._mode_label()} {last_summary}"
        )

    def _mode_label(self) -> str:
        return "serial" if self.worker_count <= 1 else f"{self.worker_count} workers"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local AI-vs-AI benchmark matches.")
    parser.add_argument("--black-ai", required=True, help="AI preset ID assigned to black in odd games.")
    parser.add_argument("--white-ai", required=True, help="AI preset ID assigned to white in odd games.")
    parser.add_argument("--rounds", type=int, default=1, help="Number of side-swapped rounds to play.")
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Optional shared AI search depth override (1-5). Omit to use each preset's default depth.",
    )
    parser.add_argument(
        "--layout",
        default="standard",
        choices=["standard", "belgian_daisy", "german_daisy"],
        help="Opening board layout.",
    )
    parser.add_argument("--move-time-s", type=int, default=30, help="Per-turn time limit for both players.")
    parser.add_argument("--max-moves", type=int, default=500, help="Maximum moves per game before draw/tiebreak.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed for reproducible random black openings.")
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Worker process count for parallel match mode. Defaults to CPU count; use 1 for serial execution.",
    )
    return parser


def _run_game(
    black_ai_id: str,
    white_ai_id: str,
    depth: Optional[int],
    layout: str,
    move_time_s: int,
    max_moves: int,
    opening_seed: int,
) -> dict:
    session = run_game_session(
        black_ai_id=black_ai_id,
        white_ai_id=white_ai_id,
        depth=depth,
        layout=layout,
        move_time_s=move_time_s,
        max_moves=max_moves,
        opening_seed=opening_seed,
    )
    status = session.status()
    return {
        "winner": status["winner"],
        "reason": status["game_over_reason"],
        "history": session.move_history,
        "score": dict(session.board.score),
        "black_ai_id": black_ai_id,
        "white_ai_id": white_ai_id,
    }


def _run_scheduled_match_game(job: dict) -> dict:
    game = _run_game(
        black_ai_id=job["black_ai_id"],
        white_ai_id=job["white_ai_id"],
        depth=job["depth"],
        layout=job["layout"],
        move_time_s=job["move_time_s"],
        max_moves=job["max_moves"],
        opening_seed=job["opening_seed"],
    )
    game["index"] = job["index"]
    return game


def _shutdown_executor(executor) -> None:
    try:
        executor.shutdown(cancel_futures=True)
    except TypeError:
        executor.shutdown()


def _apply_game_stats(stats: Dict[str, _AgentStats], game: dict) -> None:
    black_id = game["black_ai_id"]
    white_id = game["white_ai_id"]

    stats[black_id].captures += game["score"][BLACK]
    stats[white_id].captures += game["score"][WHITE]

    if game["winner"] == BLACK:
        stats[black_id].wins += 1
        stats[white_id].losses += 1
    elif game["winner"] == WHITE:
        stats[white_id].wins += 1
        stats[black_id].losses += 1
    else:
        stats[black_id].draws += 1
        stats[white_id].draws += 1

    for entry in game["history"]:
        agent_id = entry.get("agent_id")
        if not agent_id:
            continue
        agent_stats = stats[agent_id]
        agent_stats.moves += 1
        agent_stats.total_move_time_ms += entry.get("duration_ms", 0)
        search = entry.get("search") or {}
        agent_stats.completed_depth_sum += int(search.get("completed_depth", 0) or 0)
        if search.get("decision_source") in PARTIAL_SEARCH_SOURCES:
            agent_stats.partial_searches += 1


def _winner_label(game: dict) -> str:
    if game["winner"] == BLACK:
        return game["black_ai_id"]
    if game["winner"] == WHITE:
        return game["white_ai_id"]
    return "draw"


def _format_duration(seconds: float) -> str:
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    total_seconds = int(round(seconds))
    minutes, secs = divmod(total_seconds, 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _progress_bar(completed: int, total: int, width: int = 20) -> str:
    if total <= 0:
        return "[" + ("-" * width) + "]"
    filled = min(width, int((completed * width) / total))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _run_serial_matches(
    scheduled_games: List[dict],
    progress: _MatchProgressDisplay,
    start_completed: int = 0,
    existing_games: Optional[List[dict]] = None,
) -> List[dict]:
    games = list(existing_games or [])
    completed = start_completed
    for job in scheduled_games:
        game = _run_scheduled_match_game(job)
        games.append(game)
        completed += 1
        progress.update(completed, game)
    return games


def _run_scheduled_matches(
    scheduled_games: List[dict],
    worker_count: int,
    progress: _MatchProgressDisplay,
) -> List[dict]:
    if worker_count <= 1:
        return _run_serial_matches(scheduled_games, progress)

    try:
        executor = ProcessPoolExecutor(max_workers=worker_count)
    except (BrokenProcessPool, OSError, PermissionError) as exc:
        progress.warn_text(
            "Warning: parallel match execution is unavailable "
            f"({exc.__class__.__name__}: {exc}). Continuing serially."
        )
        progress.restart_serial()
        return _run_serial_matches(scheduled_games, progress)

    futures = {}
    games: List[dict] = []
    try:
        for job in scheduled_games:
            futures[executor.submit(_run_scheduled_match_game, job)] = job

        completed_futures = set()
        for future in as_completed(list(futures.keys())):
            try:
                game = future.result()
            except (BrokenProcessPool, OSError, PermissionError) as exc:
                progress.warn_text(
                    "Warning: parallel match execution is unavailable "
                    f"({exc.__class__.__name__}: {exc}). Continuing serially."
                )
                progress.restart_serial()
                remaining_jobs = [
                    job
                    for pending, job in futures.items()
                    if pending not in completed_futures
                ]
                for pending in futures:
                    pending.cancel()
                _shutdown_executor(executor)
                games = _run_serial_matches(
                    remaining_jobs,
                    progress,
                    start_completed=len(games),
                    existing_games=games,
                )
                return sorted(games, key=lambda item: item["index"])

            completed_futures.add(future)
            games.append(game)
            progress.update(len(games), game)
    finally:
        _shutdown_executor(executor)

    return sorted(games, key=lambda item: item["index"])


def _print_summary(stats: Dict[str, _AgentStats], ordered_agent_ids: List[str], rounds: int) -> None:
    print(f"Completed {rounds} round(s), {rounds * 2} game(s).")
    for agent_id in ordered_agent_ids:
        agent_stats = stats[agent_id]
        avg_time = agent_stats.total_move_time_ms / agent_stats.moves if agent_stats.moves else 0.0
        avg_depth = agent_stats.completed_depth_sum / agent_stats.moves if agent_stats.moves else 0.0
        print(
            f"{agent_id}: "
            f"W={agent_stats.wins} L={agent_stats.losses} D={agent_stats.draws} "
            f"captures={agent_stats.captures} "
            f"partial_searches={agent_stats.partial_searches} "
            f"avg_move_ms={avg_time:.1f} avg_completed_depth={avg_depth:.2f}"
        )


def main(argv: Optional[List[str]] = None) -> None:
    """Run side-swapped AI-vs-AI benchmark rounds."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.rounds < 1:
        parser.error("--rounds must be >= 1")
    if args.depth is not None and (args.depth < 1 or args.depth > 5):
        parser.error("--depth must be between 1 and 5")
    try:
        get_agent(args.black_ai)
        get_agent(args.white_ai)
    except ValueError as exc:
        parser.error(str(exc))

    ordered_ids = [args.black_ai, args.white_ai]
    stats = {agent_id: _AgentStats() for agent_id in ordered_ids}
    scheduled_games = []
    for round_index in range(args.rounds):
        opening_seed = args.seed + (round_index * 2)
        scheduled_games.append(
            {
                "index": len(scheduled_games),
                "black_ai_id": args.black_ai,
                "white_ai_id": args.white_ai,
                "depth": args.depth,
                "layout": args.layout,
                "move_time_s": args.move_time_s,
                "max_moves": args.max_moves,
                "opening_seed": opening_seed,
            }
        )
        scheduled_games.append(
            {
                "index": len(scheduled_games),
                "black_ai_id": args.white_ai,
                "white_ai_id": args.black_ai,
                "depth": args.depth,
                "layout": args.layout,
                "move_time_s": args.move_time_s,
                "max_moves": args.max_moves,
                "opening_seed": opening_seed + 1,
            }
        )

    worker_count = resolve_worker_count(args.jobs, len(scheduled_games))
    progress = _MatchProgressDisplay(game_count=len(scheduled_games), worker_count=worker_count)

    progress.start()
    try:
        games = _run_scheduled_matches(scheduled_games, worker_count, progress)
    finally:
        progress.finish()

    for game in games:
        _apply_game_stats(stats, game)

    _print_summary(stats, ordered_ids, args.rounds)
