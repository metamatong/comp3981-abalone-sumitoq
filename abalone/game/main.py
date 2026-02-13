#!/usr/bin/env python3
"""CLI entry point for Abalone."""

import argparse
from typing import List, Optional

from ..state_space import generate_legal_moves, print_state_space_summary
from .board import BLACK, WHITE, Board
from .cli import Game
from .config import MODE_AVA, MODE_HVA, MODE_HVH


def _print_state_space(state_depth_one: bool, state_player: str, state_child_index: int):
    board = Board()
    board.setup_standard()
    print(board.display())

    if not state_depth_one:
        for player in [BLACK, WHITE]:
            print_state_space_summary(board, player)
        return

    root_player = BLACK if state_player == "black" else WHITE
    root_name = "Black" if root_player == BLACK else "White"
    next_player = WHITE if root_player == BLACK else BLACK
    next_name = "Black" if next_player == BLACK else "White"

    print(f"=== Root Node ({root_name} to move) ===")
    print_state_space_summary(board, root_player)

    root_moves = generate_legal_moves(board, root_player)
    if not root_moves:
        print("No legal root moves to expand.")
        return

    if state_child_index >= len(root_moves):
        print(
            f"Requested child index {state_child_index} is out of range. "
            f"Valid range: 0..{len(root_moves) - 1}."
        )
        return

    root_move = root_moves[state_child_index]
    child = board.copy()
    result = child.apply_move(root_move, root_player)
    notation = root_move.to_notation(pushed=bool(result["pushed"]))

    print(f"=== Depth 1 Child ({next_name} to move) ===")
    print(f"Expanded root move [{state_child_index}]: {notation}")
    print(child.display())
    print_state_space_summary(child, next_player)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play Abalone in the terminal.")
    parser.add_argument("--state-space", action="store_true", help="Print state-space analysis and exit.")
    parser.add_argument(
        "--state-depth-one",
        action="store_true",
        help="With --state-space, print root state plus one depth-1 child state.",
    )
    parser.add_argument(
        "--state-player",
        default="black",
        choices=["black", "white"],
        help="Root player when using --state-depth-one.",
    )
    parser.add_argument(
        "--state-child-index",
        type=int,
        default=0,
        help="Legal move index to expand at depth 1 when using --state-depth-one.",
    )
    parser.add_argument(
        "--mode",
        default=MODE_HVH,
        choices=[MODE_HVH, MODE_HVA, MODE_AVA],
        help="Game mode: hvh (human-human), hva (human-ai), ava (ai-ai).",
    )
    parser.add_argument(
        "--human-side",
        default="black",
        choices=["black", "white"],
        help="Human side for hva mode.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        help="AI search depth (1-5).",
    )
    return parser


def main(argv: Optional[List[str]] = None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.state_space:
        if args.state_child_index < 0:
            parser.error("--state-child-index must be >= 0")
        _print_state_space(
            state_depth_one=args.state_depth_one,
            state_player=args.state_player,
            state_child_index=args.state_child_index,
        )
        return

    if args.depth < 1 or args.depth > 5:
        parser.error("--depth must be between 1 and 5")

    human_side = BLACK if args.human_side == "black" else WHITE
    game = Game(mode=args.mode, human_side=human_side, ai_depth=args.depth)
    game.play()


if __name__ == "__main__":
    main()
