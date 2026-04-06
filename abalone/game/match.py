"""Lightweight AI-vs-AI match runner for local benchmarking."""

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional

from ..eval.gauntlet import PARTIAL_SEARCH_SOURCES, run_game_session
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
    if args.depth is not None and (args.depth < 1 or args.depth > 10):
        parser.error("--depth must be between 1 and 10")
    try:
        get_agent(args.black_ai)
        get_agent(args.white_ai)
    except ValueError as exc:
        parser.error(str(exc))

    ordered_ids = [args.black_ai, args.white_ai]
    stats = {agent_id: _AgentStats() for agent_id in ordered_ids}

    for round_index in range(args.rounds):
        opening_seed = args.seed + (round_index * 2)
        first_game = _run_game(
            black_ai_id=args.black_ai,
            white_ai_id=args.white_ai,
            depth=args.depth,
            layout=args.layout,
            move_time_s=args.move_time_s,
            max_moves=args.max_moves,
            opening_seed=opening_seed,
        )
        _apply_game_stats(stats, first_game)

        second_game = _run_game(
            black_ai_id=args.white_ai,
            white_ai_id=args.black_ai,
            depth=args.depth,
            layout=args.layout,
            move_time_s=args.move_time_s,
            max_moves=args.max_moves,
            opening_seed=opening_seed + 1,
        )
        _apply_game_stats(stats, second_game)

    _print_summary(stats, ordered_ids, args.rounds)
