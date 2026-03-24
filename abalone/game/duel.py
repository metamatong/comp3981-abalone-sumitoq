"""AI-vs-AI duel runners with single and one-vs-all reporting modes."""

import argparse
import time
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


def _winner_agent_id(winner: Optional[int], black_ai_id: str, white_ai_id: str) -> Optional[str]:
    if winner == BLACK:
        return black_ai_id
    if winner == WHITE:
        return white_ai_id
    return None


def _print_single_game_report(session: GameSession, black_agent, white_agent) -> None:
    status = session.status()
    score = dict(session.board.score)
    totals = _total_time_by_player(session.move_history)
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
    print("Heuristic weights:")
    print(f"Black {black_agent.id} ({black_agent.label}): {_format_weights(black_agent)}")
    print(f"White {white_agent.id} ({white_agent.label}): {_format_weights(white_agent)}")


def _run_all_opponents_games(
    agent_id: str,
    depth: Optional[int],
    layout: str,
    move_time_s: int,
    max_moves: int,
    seed: int,
) -> List[dict]:
    opponents = [agent.id for agent in list_agents() if agent.id != agent_id]
    games: List[dict] = []
    game_offset = 0

    for opponent_id in opponents:
        for black_ai_id, white_ai_id in ((agent_id, opponent_id), (opponent_id, agent_id)):
            start_time = time.perf_counter()
            session = _run_game(
                black_ai_id=black_ai_id,
                white_ai_id=white_ai_id,
                depth=depth,
                layout=layout,
                move_time_s=move_time_s,
                max_moves=max_moves,
                opening_seed=seed + game_offset,
            )
            elapsed_s = time.perf_counter() - start_time
            status = session.status()
            games.append(
                {
                    "black_ai_id": black_ai_id,
                    "white_ai_id": white_ai_id,
                    "winner_ai_id": _winner_agent_id(status.get("winner"), black_ai_id, white_ai_id),
                    "score": dict(session.board.score),
                    "moves": len(session.move_history),
                    "duration_s": elapsed_s,
                    "agent_color": "black" if black_ai_id == agent_id else "white",
                }
            )
            game_offset += 1

    return games


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

        print(
            f"Game {index} ({duration_s:.1f}s, {moves} moves): "
            f"{black_ai_id} (black) vs {white_ai_id} (white) -> "
            f"{winner_label} ({black_score}-{white_score})"
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
