"""Single-game AI-vs-AI simulation with a final report."""

import argparse
from typing import Dict, List, Optional

from .board import BLACK, WHITE
from .config import GameConfig
from .session import GameSession
from ..ai.heuristics import DEFAULT_WEIGHTS
from ..players.registry import get_agent


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a single AI-vs-AI simulation and print a report.")
    parser.add_argument("--black-ai", required=True, help="AI preset ID assigned to black.")
    parser.add_argument("--white-ai", required=True, help="AI preset ID assigned to white.")
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
    parser.add_argument("--seed", type=int, default=0, help="Seed for the opening random black move.")
    return parser


def _run_game(
    black_ai_id: str,
    white_ai_id: str,
    depth: Optional[int],
    layout: str,
    move_time_s: int,
    max_moves: int,
    opening_seed: int,
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


def _print_report(session: GameSession, black_agent, white_agent) -> None:
    status = session.status()
    score = dict(session.board.score)
    totals = _total_time_by_player(session.move_history)
    total_moves = len(session.move_history)

    winner = _winner_label(status.get("winner"))
    print("AI vs AI simulation complete.")
    print(f"Winner: {winner} (black {score[BLACK]} - white {score[WHITE]})")
    print(f"Total moves: {total_moves}")
    print(
        "Time spent: "
        f"black {totals[BLACK] / 1000.0:.2f}s, "
        f"white {totals[WHITE] / 1000.0:.2f}s"
    )
    print("Heuristic weights:")
    print(f"Black {black_agent.id} ({black_agent.label}): {_format_weights(black_agent)}")
    print(f"White {white_agent.id} ({white_agent.label}): {_format_weights(white_agent)}")


def main(argv: Optional[List[str]] = None) -> None:
    """Run a single AI-vs-AI simulation and print a report."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.depth is not None and (args.depth < 1 or args.depth > 5):
        parser.error("--depth must be between 1 and 5")

    try:
        black_agent = get_agent(args.black_ai)
        white_agent = get_agent(args.white_ai)
    except ValueError as exc:
        parser.error(str(exc))

    session = _run_game(
        black_ai_id=args.black_ai,
        white_ai_id=args.white_ai,
        depth=args.depth,
        layout=args.layout,
        move_time_s=args.move_time_s,
        max_moves=args.max_moves,
        opening_seed=args.seed,
    )
    _print_report(session, black_agent, white_agent)
