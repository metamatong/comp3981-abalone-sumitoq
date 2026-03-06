#!/usr/bin/env python3
"""CLI entry point for Abalone."""

import argparse
from typing import List, Optional

from ..file_handler import (
    compare_and_save_position_list_files,
    compare_position_list_files,
    export_position_list_states,
)
from ..state_space import (
    format_position_list_comparison,
    generate_legal_moves,
    print_state_space_summary,
)
from .board import BLACK, WHITE, Board
from .cli import Game
from .config import MODE_AVA, MODE_HVA, MODE_HVH


def _print_state_space(state_depth_one: bool, state_player: str, state_child_index: int):
    """Print legal-move summary for the initial board and optional depth-1 child."""
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
    """Build the CLI argument parser for game and state-space modes."""
    parser = argparse.ArgumentParser(description="Play Abalone in the terminal.")
    parser.add_argument("--state-space", action="store_true", help="Print state-space analysis and exit.")
    parser.add_argument(
        "--state-input-file",
        help="Path to a compact input file (`b|w` + comma-separated `C5b` tokens) to expand one ply.",
    )
    parser.add_argument(
        "--state-output-file",
        help="Optional output path for one-ply child states generated from --state-input-file.",
    )
    parser.add_argument(
        "--state-verify",
        action="store_true",
        help="Compare generated child states against an expected `.board` file.",
    )
    parser.add_argument(
        "--state-expected-file",
        help=(
            "Expected `.board` path used with --state-verify. "
            "If omitted, infer `state_space_outputs/<name>.board` from the input path."
        ),
    )
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
    """Parse CLI arguments and dispatch into play mode or state-space reporting."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.state_space:
        if args.state_output_file and not args.state_input_file:
            parser.error("--state-output-file requires --state-input-file")
        if args.state_expected_file and not args.state_input_file:
            parser.error("--state-expected-file requires --state-input-file")
        if args.state_verify and not args.state_input_file:
            parser.error("--state-verify requires --state-input-file")
        if args.state_input_file and args.state_depth_one:
            parser.error("--state-input-file cannot be combined with --state-depth-one")
        if args.state_verify and args.state_output_file:
            parser.error("--state-verify cannot be combined with --state-output-file")
        if args.state_input_file:
            try:
                if args.state_verify or args.state_expected_file:
                    if args.state_verify:
                        comparison, expected_path, generated_path = compare_and_save_position_list_files(
                            input_path=args.state_input_file,
                            expected_path=args.state_expected_file,
                        )
                    else:
                        comparison, expected_path = compare_position_list_files(
                            input_path=args.state_input_file,
                            expected_path=args.state_expected_file,
                        )
                        generated_path = None
                    print(
                        format_position_list_comparison(
                            comparison,
                            input_path=args.state_input_file,
                            expected_path=str(expected_path),
                            generated_path=str(generated_path) if generated_path is not None else None,
                        )
                    )
                    return
                move_out = None
                if args.state_output_file:
                    # Derive .move path next to the .board output
                    out = args.state_output_file
                    if out.endswith(".board"):
                        move_out = out[: -len(".board")] + ".move"
                    else:
                        move_out = out + ".move"
                count = export_position_list_states(
                    input_path=args.state_input_file,
                    output_path=args.state_output_file,
                    move_output_path=move_out,
                )
            except (OSError, ValueError) as exc:
                parser.error(str(exc))
            if args.state_output_file:
                print(f"Wrote {count} states to {args.state_output_file}")
                print(f"Wrote {count} moves  to {move_out}")
            return
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
