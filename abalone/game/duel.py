"""AI-vs-AI duel runners with single, gauntlet, and tuning modes."""

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from typing import List, Optional

from .. import native
from ..ai.heuristics import FEATURE_ORDER
from ..eval import gauntlet as eval_gauntlet
from ..game.board import BLACK, WHITE
from ..players.registry import get_agent, get_agent_weights, list_agents


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
    parser.add_argument("--tune", action="store_true", help="Run adaptive heuristic tuning in one-vs-all mode.")
    parser.add_argument("--iterations", type=int, help="Number of tuning iterations to run when --tune is used.")
    parser.add_argument("--resume-from", help="Resume a prior tuning run from checkpoint.json.")
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
    parser.add_argument("--seed", type=int, default=0, help="Base seed for reproducible opening move selection.")
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Worker process count for gauntlet and tuning modes. Defaults to CPU count; use 1 for serial execution.",
    )
    return parser


def _preflight_native_runtime() -> None:
    native.preflight_or_exit()


def _run_game(
    black_ai_id: str,
    white_ai_id: str,
    depth: Optional[int],
    layout: str,
    move_time_s: int,
    max_moves: int,
    opening_seed: Optional[int],
    agent_weight_overrides=None,
    telemetry_agent_ids=None,
):
    return eval_gauntlet.run_game_session(
        black_ai_id=black_ai_id,
        white_ai_id=white_ai_id,
        depth=depth,
        layout=layout,
        move_time_s=move_time_s,
        max_moves=max_moves,
        opening_seed=opening_seed,
        agent_weight_overrides=agent_weight_overrides,
        telemetry_agent_ids=telemetry_agent_ids,
    )


def _build_all_opponents_jobs(
    agent_id: str,
    depth: Optional[int],
    layout: str,
    move_time_s: int,
    max_moves: int,
    seed: Optional[int],
) -> List[dict]:
    return eval_gauntlet.build_all_opponents_jobs(agent_id, depth, layout, move_time_s, max_moves, seed)


def _resolve_worker_count(jobs: Optional[int], game_count: int) -> int:
    return eval_gauntlet.resolve_worker_count(jobs, game_count)


def _run_all_opponents_game(job: dict) -> dict:
    return eval_gauntlet.run_scheduled_game(job)


def _shutdown_executor(executor) -> None:
    shutdown = getattr(executor, "shutdown", None)
    if shutdown is None:
        return
    try:
        shutdown(wait=True, cancel_futures=True)
    except TypeError:
        shutdown()


def _cancel_future(future) -> None:
    cancel = getattr(future, "cancel", None)
    if cancel is not None:
        cancel()


def _format_weights(agent) -> str:
    weights = getattr(agent.evaluator, "weights", None)
    if not weights:
        return "unknown"

    parts = []
    for key in FEATURE_ORDER:
        if key in weights:
            parts.append(f"{key}={weights[key]:.1f}")
    for key in sorted(k for k in weights.keys() if k not in FEATURE_ORDER):
        parts.append(f"{key}={weights[key]:.1f}")
    return ", ".join(parts)


def _winner_label(winner: Optional[int]) -> str:
    if winner == BLACK:
        return "black"
    if winner == WHITE:
        return "white"
    return "draw"


def _winner_with_tiebreak(base_label: str, winner_tiebreak: Optional[str]) -> str:
    if winner_tiebreak == "least_total_time":
        return f"{base_label}  (time)"
    return base_label


def _print_single_game_report(session, black_agent, white_agent) -> None:
    status = session.status()
    score = dict(session.board.score)
    totals = status.get("time_used_ms", {BLACK: 0, WHITE: 0})
    total_moves = len(session.move_history)

    winner = _winner_with_tiebreak(_winner_label(status.get("winner")), status.get("winner_tiebreak"))
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
        prefix: str = "duel",
    ):
        self.agent_id = agent_id
        self.game_count = game_count
        self.worker_count = worker_count
        self.stream = stream or sys.stderr
        self.refresh_s = refresh_s
        self.prefix = prefix
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
            self._message(f"[{self.prefix}] Starting gauntlet for {self.agent_id}: {self.game_count} game(s), {self._mode_label()}.")
            return
        self._render_live_line()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def update(self, completed: int, game: dict) -> None:
        if not self._live:
            self.completed = completed
            self.latest_game = game
            self._message(self._build_progress_line(completed, game))
            return
        with self._lock:
            self.completed = completed
            self.latest_game = game

    def warn_text(self, text: str) -> None:
        self._message(text)

    def restart_serial(self) -> None:
        self._message(f"[{self.prefix}] Restarting gauntlet serially for {self.game_count} game(s).")

    def finish(self) -> None:
        if not self._live:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _mode_label(self) -> str:
        return "serial" if self.worker_count <= 1 else f"{self.worker_count} workers"

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
        return (
            f"[{self.prefix}] {frame} {_progress_bar(self.completed, self.game_count)} "
            f"{self.completed}/{self.game_count} elapsed={_format_duration(elapsed_s)} "
            f"eta={_format_duration(eta_s)} mode={self._mode_label()}"
        )

    def _build_progress_line(self, completed: int, game: dict) -> str:
        elapsed_s = max(0.0, time.perf_counter() - self.started_at)
        remaining = max(0, self.game_count - completed)
        avg_s = elapsed_s / completed if completed else 0.0
        eta_s = avg_s * remaining
        winner = game.get("winner_ai_id") or "draw"
        return (
            f"[{self.prefix}] {_progress_bar(completed, self.game_count)} {completed}/{self.game_count} "
            f"elapsed={_format_duration(elapsed_s)} eta={_format_duration(eta_s)} "
            f"last={game.get('black_ai_id', '?')} vs {game.get('white_ai_id', '?')} -> {winner} "
            f"({game.get('moves', '?')} moves, {float(game.get('duration_s', 0.0) or 0.0):.1f}s)"
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


def _run_all_opponents_games(
    agent_id: str,
    depth: Optional[int],
    layout: str,
    move_time_s: int,
    max_moves: int,
    seed: Optional[int],
    jobs: Optional[int],
    agent_weight_overrides=None,
    telemetry_agent_ids=None,
    prefix: str = "duel",
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
        prefix=prefix,
    )
    if agent_weight_overrides or telemetry_agent_ids:
        warned_parallel = False

        def on_game_complete(completed: int, total: int, game: dict) -> None:
            del total
            progress.update(completed, game)

        def on_warning(message: str) -> None:
            nonlocal warned_parallel
            progress.warn_text(message)
            if not warned_parallel and worker_count > 1:
                warned_parallel = True
                progress.restart_serial()

        progress.start()
        try:
            return eval_gauntlet.run_scheduled_games(
                scheduled_games,
                worker_count,
                agent_weight_overrides=agent_weight_overrides,
                telemetry_agent_ids=telemetry_agent_ids,
                on_game_complete=on_game_complete,
                on_warning=on_warning,
            )
        finally:
            progress.finish()

    def run_serial(serial_jobs, start_completed: int = 0, existing_games: Optional[List[dict]] = None) -> List[dict]:
        games = list(existing_games or [])
        completed = start_completed
        for job in serial_jobs:
            game = _run_all_opponents_game(job)
            games.append(game)
            completed += 1
            progress.update(completed, game)
        return games

    progress.start()
    try:
        if worker_count <= 1:
            return run_serial(scheduled_games)

        try:
            executor = ProcessPoolExecutor(max_workers=worker_count)
        except (BrokenProcessPool, OSError, PermissionError) as exc:
            progress.warn_text(
                "Warning: parallel duel execution is unavailable "
                f"({exc.__class__.__name__}: {exc}). Continuing serially."
            )
            progress.restart_serial()
            return run_serial(scheduled_games)

        futures = {}
        games = []
        completed_futures = set()
        try:
            for job in scheduled_games:
                futures[executor.submit(_run_all_opponents_game, job)] = job

            for future in as_completed(list(futures.keys())):
                try:
                    game = future.result()
                except (BrokenProcessPool, OSError, PermissionError) as exc:
                    progress.warn_text(
                        "Warning: parallel duel execution is unavailable "
                        f"({exc.__class__.__name__}: {exc}). Continuing serially."
                    )
                    progress.restart_serial()
                    remaining_jobs = [
                        job
                        for pending, job in futures.items()
                        if pending not in completed_futures
                    ]
                    for pending in futures:
                        _cancel_future(pending)
                    _shutdown_executor(executor)
                    games = run_serial(remaining_jobs, start_completed=len(games), existing_games=games)
                    return sorted(games, key=lambda item: item["index"])

                completed_futures.add(future)
                games.append(game)
                progress.update(len(games), game)
        finally:
            _shutdown_executor(executor)

        return sorted(games, key=lambda item: item["index"])
    finally:
        progress.finish()


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

        winner_label = _winner_with_tiebreak(winner_label, game.get("winner_tiebreak"))
        time_used_ms = game.get("time_used_ms", {BLACK: 0, WHITE: 0})
        print(
            f"Game {index} ({duration_s:.1f}s, {moves} moves): "
            f"{black_ai_id} (black) vs {white_ai_id} (white) -> "
            f"{winner_label} ({black_score}-{white_score}, "
            f"time {time_used_ms[BLACK] / 1000.0:.2f}s-{time_used_ms[WHITE] / 1000.0:.2f}s)"
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
    if args.iterations is not None and args.iterations < 1:
        parser.error("--iterations must be >= 1")

    if args.resume_from and not args.tune:
        parser.error("--resume-from can only be used with --tune")

    if args.tune:
        if not args.all_opponents:
            parser.error("--tune requires --all-opponents")
        if args.black_ai or args.white_ai:
            parser.error("--black-ai/--white-ai cannot be combined with --tune")
        if args.iterations is None:
            parser.error("--iterations is required when --tune is used")
        if not args.resume_from:
            if not args.agent:
                parser.error("--agent is required when starting a fresh tuning run")
            if len(list_agents()) < 2:
                parser.error("--tune requires at least two registered agents")
            try:
                get_agent(args.agent)
                get_agent_weights(args.agent)
            except ValueError as exc:
                parser.error(str(exc))
        return

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
        parser.error("--jobs can only be used with --all-opponents or --tune")
    if args.agent:
        parser.error("--agent can only be used with --all-opponents")
    if args.iterations is not None:
        parser.error("--iterations can only be used with --tune")
    if not args.black_ai or not args.white_ai:
        parser.error("--black-ai and --white-ai are required unless --all-opponents is used")
    try:
        get_agent(args.black_ai)
        get_agent(args.white_ai)
    except ValueError as exc:
        parser.error(str(exc))


def _run_tuning(args: argparse.Namespace) -> None:
    target_agent_id = args.agent or "resumed-agent"

    def run_gauntlet_iteration(
        scheduled_games,
        worker_count,
        agent_weight_overrides,
        telemetry_agent_ids,
        on_game_complete,
        on_warning,
        iteration_index=None,
        total_iterations=None,
    ):
        if iteration_index is not None and total_iterations is not None:
            progress_prefix = f"Iteration {iteration_index}/{total_iterations}"
        else:
            progress_prefix = "tune"
        progress = _GauntletProgressDisplay(
            agent_id=target_agent_id,
            game_count=len(scheduled_games),
            worker_count=worker_count,
            prefix=progress_prefix,
        )
        warned_parallel = False

        def wrapped_complete(completed, total, game):
            del total
            progress.update(completed, game)
            on_game_complete(completed, len(scheduled_games), game)

        def wrapped_warning(message):
            nonlocal warned_parallel
            progress.warn_text(message)
            if not warned_parallel and worker_count > 1:
                warned_parallel = True
                progress.restart_serial()
            on_warning(message)

        progress.start()
        try:
            return eval_gauntlet.run_scheduled_games(
                scheduled_games,
                worker_count,
                agent_weight_overrides=agent_weight_overrides,
                telemetry_agent_ids=telemetry_agent_ids,
                on_game_complete=wrapped_complete,
                on_warning=wrapped_warning,
            )
        finally:
            progress.finish()

    eval_gauntlet.run_tuning_loop(
        agent_id=args.agent,
        iterations=args.iterations,
        depth=args.depth,
        layout=args.layout,
        move_time_s=args.move_time_s,
        max_moves=args.max_moves,
        seed=args.seed,
        jobs=args.jobs,
        resume_from=args.resume_from,
        run_gauntlet_iteration=run_gauntlet_iteration,
        announce=print,
    )


def main(argv: Optional[List[str]] = None) -> None:
    """Run AI-vs-AI duel simulations and print reports."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    _preflight_native_runtime()
    _validate_args(parser, args)

    if args.tune:
        _run_tuning(args)
        return

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

