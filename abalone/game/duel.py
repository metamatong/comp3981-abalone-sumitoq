"""AI-vs-AI duel runners with single and one-vs-all reporting modes."""

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from typing import Dict, List, Optional

from .board import BLACK, WHITE
from .config import GameConfig
from .session import GameSession
from ..ai.heuristics import DEFAULT_WEIGHTS
from ..players.registry import get_agent, list_agents


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AI-vs-AI duel simulations and print a report.")
    parser.add_argument("--black-ai", help="AI preset ID assigned to black (single-game mode).")
    parser.add_argument("--white-ai", help="AI preset ID assigned to white (single-game mode).")
    parser.add_argument("--agent", help="Target AI preset ID for one-vs-all gauntlet mode.")
    parser.add_argument(
        "--all-opponents",
        action="store_true",
        help="Run one agent against every other registered agent in two color-swapped games.",
    )
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
    parser.add_argument("--max-moves", type=int, default=500, help="Maximum moves before draw/tiebreak.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for the opening random black move (defaults to random).")
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Worker process count for --all-opponents. Defaults to CPU count; use 1 for serial execution.",
    )
    return parser


def _run_game(
    black_ai_id: str,
    white_ai_id: str,
    depth: Optional[int],
    layout: str,
    move_time_s: int,
    max_moves: int,
    opening_seed: Optional[int],
) -> GameSession:
    config = GameConfig(
        mode="ava",
        ai_depth=depth,
        black_ai_id=black_ai_id,
        white_ai_id=white_ai_id,
        board_layout=layout,
        max_moves=max_moves,
        player1_time_per_turn_s=move_time_s,
        player2_time_per_turn_s=move_time_s,
    )
    session = GameSession(config=config, opening_seed=opening_seed)
    session.reset()

    while not session.status()["game_over"]:
        result = session.apply_agent_move()
        if "error" in result:
            raise RuntimeError(result["error"])

    return session


def _total_time_by_player(history: List[dict]) -> Dict[int, int]:
    totals = {BLACK: 0, WHITE: 0}
    for entry in history:
        player = entry.get("player")
        if player in totals:
            totals[player] += int(entry.get("duration_ms", 0) or 0)
    return totals


def _format_weights(agent) -> str:
    weights = getattr(agent.evaluator, "weights", None)
    if not weights:
        return "unknown"

    ordered_keys = list(DEFAULT_WEIGHTS.keys())
    seen = set()
    parts = []
    for key in ordered_keys:
        if key in weights:
            parts.append(f"{key}={weights[key]:.1f}")
            seen.add(key)
    for key in sorted(k for k in weights.keys() if k not in seen):
        parts.append(f"{key}={weights[key]:.1f}")
    return ", ".join(parts)


def _winner_label(winner: Optional[int]) -> str:
    if winner == BLACK:
        return "black"
    if winner == WHITE:
        return "white"
    return "draw"


def _winner_agent_id(winner: Optional[int], black_ai_id: str, white_ai_id: str) -> Optional[str]:
    if winner == BLACK:
        return black_ai_id
    if winner == WHITE:
        return white_ai_id
    return None


def _print_single_game_report(session: GameSession, black_agent, white_agent) -> None:
    status = session.status()
    score = dict(session.board.score)
    totals = status.get("time_used_ms", {BLACK: 0, WHITE: 0})
    total_moves = len(session.move_history)

    winner = _winner_label(status.get("winner"))
    print("__________________________________")
    print("AI vs AI simulation complete.")
    print(f"Winner: {winner} (black {score[BLACK]} - white {score[WHITE]})")
    print(f"Total moves: {total_moves}")
    print(
        "Time spent: "
        f"black {totals[BLACK] / 1000.0:.2f}s, "
        f"white {totals[WHITE] / 1000.0:.2f}s"
    )
    if status.get("winner_tiebreak") == "least_total_time":
        print("Tiebreak: lower total time used")
    print("Heuristic weights:")
    print(f"Black {black_agent.id} ({black_agent.label}): {_format_weights(black_agent)}")
    print(f"White {white_agent.id} ({white_agent.label}): {_format_weights(white_agent)}")


def _run_all_opponents_games(
    agent_id: str,
    depth: Optional[int],
    layout: str,
    move_time_s: int,
    max_moves: int,
    seed: Optional[int],
    jobs: Optional[int],
) -> List[dict]:
    scheduled_games = _build_all_opponents_jobs(
        agent_id=agent_id,
        depth=depth,
        layout=layout,
        move_time_s=move_time_s,
        max_moves=max_moves,
        seed=seed,
    )
    worker_count = _resolve_worker_count(jobs, len(scheduled_games))
    progress = _GauntletProgressDisplay(
        agent_id=agent_id,
        game_count=len(scheduled_games),
        worker_count=worker_count,
    )
    progress.start()
    if worker_count <= 1:
        try:
            return _run_scheduled_games_serial(scheduled_games, progress)
        finally:
            progress.finish()

    try:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_run_all_opponents_game, job) for job in scheduled_games]
            games = []
            for completed_count, future in enumerate(as_completed(futures), start=1):
                game = future.result()
                games.append(game)
                progress.update(completed_count, game)
    except (BrokenProcessPool, OSError) as exc:
        progress.warn_parallel_unavailable(exc)
        progress.restart_serial()
        try:
            return _run_scheduled_games_serial(scheduled_games, progress, reset=True)
        finally:
            progress.finish()
    finally:
        progress.finish()
    return sorted(games, key=lambda game: game["index"])


def _build_all_opponents_jobs(
    agent_id: str,
    depth: Optional[int],
    layout: str,
    move_time_s: int,
    max_moves: int,
    seed: Optional[int],
) -> List[dict]:
    opponents = [agent.id for agent in list_agents() if agent.id != agent_id]
    jobs: List[dict] = []
    game_offset = 0

    for opponent_id in opponents:
        for black_ai_id, white_ai_id in ((agent_id, opponent_id), (opponent_id, agent_id)):
            jobs.append(
                {
                    "index": len(jobs),
                    "black_ai_id": black_ai_id,
                    "white_ai_id": white_ai_id,
                    "depth": depth,
                    "layout": layout,
                    "move_time_s": move_time_s,
                    "max_moves": max_moves,
                    "opening_seed": seed + game_offset if seed is not None else None,
                    "agent_color": "black" if black_ai_id == agent_id else "white",
                }
            )
            game_offset += 1

    return jobs


def _resolve_worker_count(jobs: Optional[int], game_count: int) -> int:
    if game_count <= 0:
        return 0
    requested = jobs if jobs is not None else (os.cpu_count() or 1)
    return max(1, min(requested, game_count))


def _run_scheduled_games_serial(
    scheduled_games: List[dict],
    progress: "_GauntletProgressDisplay",
    reset: bool = False,
) -> List[dict]:
    games = []
    if reset:
        progress.reset()
    for completed_count, job in enumerate(scheduled_games, start=1):
        game = _run_all_opponents_game(job)
        games.append(game)
        progress.update(completed_count, game)
    return games


class _GauntletProgressDisplay:
    _FRAMES = (
        "B>------<W",
        "-B>----<W-",
        "--B>--<W--",
        "---B><W---",
        "--B>--<W--",
        "-B>----<W-",
    )

    def __init__(
        self,
        agent_id: str,
        game_count: int,
        worker_count: int,
        stream=None,
        refresh_s: float = 0.12,
    ):
        self.agent_id = agent_id
        self.game_count = game_count
        self.worker_count = worker_count
        self.stream = stream or sys.stderr
        self.refresh_s = refresh_s
        self.started_at = time.perf_counter()
        self.completed = 0
        self.latest_game: Optional[dict] = None
        self._frame_index = 0
        self._last_width = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._live = bool(getattr(self.stream, "isatty", lambda: False)())

    def start(self) -> None:
        if not self._live:
            _print_gauntlet_start(self.agent_id, self.game_count, self.worker_count)
            return

        self._render_live_line()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def update(self, completed: int, game: dict) -> None:
        if not self._live:
            self.completed = completed
            self.latest_game = game
            _print_gauntlet_progress(self.completed, self.game_count, self.started_at, game)
            return

        with self._lock:
            self.completed = completed
            self.latest_game = game

    def reset(self) -> None:
        with self._lock:
            self.started_at = time.perf_counter()
            self.completed = 0
            self.latest_game = None

    def warn_parallel_unavailable(self, exc: BaseException) -> None:
        self._message(
            "Warning: parallel duel execution is unavailable "
            f"({exc.__class__.__name__}: {exc}). Continuing serially."
        )

    def restart_serial(self) -> None:
        self._message(f"[duel] Restarting gauntlet serially for {self.game_count} game(s).")

    def finish(self) -> None:
        if not self._live:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _run_loop(self) -> None:
        while not self._stop.wait(self.refresh_s):
            self._render_live_line()
        self._render_live_line()
        print(file=self.stream, flush=True)

    def _message(self, text: str) -> None:
        if not self._live:
            print(text, file=self.stream)
            return

        with self._lock:
            self._clear_live_line()
            print(text, file=self.stream, flush=True)

    def _render_live_line(self) -> None:
        with self._lock:
            line = self._build_live_line()
            padded = line.ljust(self._last_width)
            print(f"\r{padded}", end="", file=self.stream, flush=True)
            self._last_width = max(self._last_width, len(line))

    def _clear_live_line(self) -> None:
        if self._last_width <= 0:
            return
        print(f"\r{' ' * self._last_width}\r", end="", file=self.stream, flush=True)
        self._last_width = 0

    def _build_live_line(self) -> str:
        frame = self._FRAMES[self._frame_index]
        self._frame_index = (self._frame_index + 1) % len(self._FRAMES)
        elapsed_s = max(0.0, time.perf_counter() - self.started_at)
        remaining = max(0, self.game_count - self.completed)
        avg_s = elapsed_s / self.completed if self.completed else 0.0
        eta_s = avg_s * remaining
        mode = "serial" if self.worker_count <= 1 else f"{self.worker_count} workers"
        latest = "warming up..."
        if self.latest_game is not None:
            winner = self.latest_game["winner_ai_id"] or "draw"
            latest = (
                f"last={self.latest_game['black_ai_id']} vs {self.latest_game['white_ai_id']} "
                f"-> {winner} ({self.latest_game['moves']}m)"
            )
        return (
            f"[duel] {frame} {_progress_bar(self.completed, self.game_count)} "
            f"{self.completed}/{self.game_count} elapsed={_format_duration(elapsed_s)} "
            f"eta={_format_duration(eta_s)} mode={mode} {latest}"
        )


def _print_gauntlet_start(agent_id: str, game_count: int, worker_count: int) -> None:
    mode = "serial" if worker_count <= 1 else f"{worker_count} workers"
    print(
        f"[duel] Starting gauntlet for {agent_id}: {game_count} game(s), {mode}.",
        file=sys.stderr,
    )


def _print_gauntlet_restart(game_count: int) -> None:
    print(
        f"[duel] Restarting gauntlet serially for {game_count} game(s).",
        file=sys.stderr,
    )


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


def _print_gauntlet_progress(completed: int, total: int, started_at: float, game: dict) -> None:
    elapsed_s = max(0.0, time.perf_counter() - started_at)
    remaining = max(0, total - completed)
    avg_s = elapsed_s / completed if completed else 0.0
    eta_s = avg_s * remaining
    winner = game["winner_ai_id"] or "draw"
    print(
        f"[duel] {_progress_bar(completed, total)} {completed}/{total} "
        f"elapsed={_format_duration(elapsed_s)} eta={_format_duration(eta_s)} "
        f"last={game['black_ai_id']} vs {game['white_ai_id']} -> {winner} "
        f"({game['moves']} moves, {game['duration_s']:.1f}s)",
        file=sys.stderr,
    )


def _print_parallel_fallback_warning(exc: BaseException) -> None:
    print(
        "Warning: parallel duel execution is unavailable "
        f"({exc.__class__.__name__}: {exc}). Continuing serially.",
        file=sys.stderr,
    )


def _run_all_opponents_game(job: dict) -> dict:
    start_time = time.perf_counter()
    session = _run_game(
        black_ai_id=job["black_ai_id"],
        white_ai_id=job["white_ai_id"],
        depth=job["depth"],
        layout=job["layout"],
        move_time_s=job["move_time_s"],
        max_moves=job["max_moves"],
        opening_seed=job["opening_seed"],
    )
    elapsed_s = time.perf_counter() - start_time
    status = session.status()
    return {
        "index": job["index"],
        "black_ai_id": job["black_ai_id"],
        "white_ai_id": job["white_ai_id"],
        "winner_ai_id": _winner_agent_id(status.get("winner"), job["black_ai_id"], job["white_ai_id"]),
        "winner_tiebreak": status.get("winner_tiebreak"),
        "game_over_reason": status.get("game_over_reason"),
        "time_used_ms": status.get("time_used_ms", {BLACK: 0, WHITE: 0}),
        "score": dict(session.board.score),
        "moves": len(session.move_history),
        "duration_s": elapsed_s,
        "agent_color": job["agent_color"],
    }


def _format_rate(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0%"
    return f"{(numerator * 100.0 / denominator):.0f}%"


def _print_all_opponents_report(agent_id: str, games: List[dict]) -> None:
    wins = 0
    losses = 0
    draws = 0
    black_games = 0
    black_wins = 0
    white_games = 0
    white_wins = 0

    print("_________________________________________________________")
    print(f"Agent: {agent_id}")

    for index, game in enumerate(games, start=1):
        black_ai_id = game["black_ai_id"]
        white_ai_id = game["white_ai_id"]
        winner_ai_id = game["winner_ai_id"]
        black_score = game["score"][BLACK]
        white_score = game["score"][WHITE]
        duration_s = game["duration_s"]
        moves = game["moves"]
        agent_color = game["agent_color"]

        if agent_color == "black":
            black_games += 1
        else:
            white_games += 1

        if winner_ai_id == agent_id:
            if agent_color == "black":
                black_wins += 1
            else:
                white_wins += 1
            wins += 1
            winner_label = winner_ai_id
        elif winner_ai_id is None:
            draws += 1
            winner_label = "draw"
        else:
            losses += 1
            winner_label = winner_ai_id

        time_used_ms = game.get("time_used_ms", {BLACK: 0, WHITE: 0})
        print(
            f"Game {index} ({duration_s:.1f}s, {moves} moves): "
            f"{black_ai_id} (black) vs {white_ai_id} (white) -> "
            f"{winner_label} ({black_score}-{white_score}, "
            f"{time_used_ms[BLACK] / 1000.0:.2f}s-{time_used_ms[WHITE] / 1000.0:.2f}s)"
        )

    print()
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Draws: {draws}")
    print(f"Winrate: {_format_rate(wins, len(games))}")
    print(f"Winrate as black: {_format_rate(black_wins, black_games)}")
    print(f"Winrate as white: {_format_rate(white_wins, white_games)}")


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.depth is not None and (args.depth < 1 or args.depth > 5):
        parser.error("--depth must be between 1 and 5")
    if args.jobs is not None and args.jobs < 1:
        parser.error("--jobs must be >= 1")

    if args.all_opponents:
        if not args.agent:
            parser.error("--agent is required when --all-opponents is used")
        if args.black_ai or args.white_ai:
            parser.error("--black-ai/--white-ai cannot be combined with --all-opponents")
        if len(list_agents()) < 2:
            parser.error("--all-opponents requires at least two registered agents")
        try:
            get_agent(args.agent)
        except ValueError as exc:
            parser.error(str(exc))
        return

    if args.jobs is not None:
        parser.error("--jobs can only be used with --all-opponents")
    if args.agent:
        parser.error("--agent can only be used with --all-opponents")
    if not args.black_ai or not args.white_ai:
        parser.error("--black-ai and --white-ai are required unless --all-opponents is used")
    try:
        get_agent(args.black_ai)
        get_agent(args.white_ai)
    except ValueError as exc:
        parser.error(str(exc))


def main(argv: Optional[List[str]] = None) -> None:
    """Run AI-vs-AI duel simulations and print reports."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)

    if args.all_opponents:
        games = _run_all_opponents_games(
            agent_id=args.agent,
            depth=args.depth,
            layout=args.layout,
            move_time_s=args.move_time_s,
            max_moves=args.max_moves,
            seed=args.seed,
            jobs=args.jobs,
        )
        _print_all_opponents_report(args.agent, games)
        return

    black_agent = get_agent(args.black_ai)
    white_agent = get_agent(args.white_ai)
    session = _run_game(
        black_ai_id=args.black_ai,
        white_ai_id=args.white_ai,
        depth=args.depth,
        layout=args.layout,
        move_time_s=args.move_time_s,
        max_moves=args.max_moves,
        opening_seed=args.seed,
    )
    _print_single_game_report(session, black_agent, white_agent)
