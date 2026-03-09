#!/usr/bin/env python3
"""Part 2 state-space submission entry point."""

import argparse
from pathlib import Path
from typing import Optional

from abalone.file_handler import (
    compare_and_save_position_list_files,
    default_board_output_path,
    export_position_list_states,
)
from abalone.state_space import format_position_list_comparison


def _derive_move_output_path(board_output_path: str) -> str:
    """Return the sibling .move path for a requested .board output path."""
    if board_output_path.endswith(".board"):
        return board_output_path[: -len(".board")] + ".move"
    return board_output_path + ".move"


def _build_parser() -> argparse.ArgumentParser:
    """Build the minimal Part 2 CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate legal Abalone next states from a Test#.input file. "
            "Writes both .board and .move outputs."
        )
    )
    parser.add_argument(
        "--state-input-file",
        required=True,
        help="Path to a compact input file (first line b|w, remaining lines comma-separated C5b tokens).",
    )
    parser.add_argument(
        "--state-output-file",
        help=(
            "Optional .board output path. If omitted, defaults to "
            "abalone/state_space_outputs/<input-stem>.board."
        ),
    )
    parser.add_argument(
        "--state-verify",
        action="store_true",
        help=(
            "Compare generated states against the expected .board file in "
            "abalone/state_space_outputs/<input-stem>.board and save a .generated.board file."
        ),
    )
    parser.add_argument(
        "--state-expected-file",
        help="Optional expected .board path used together with --state-verify.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Run the state-space generator or verification flow."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.state_verify and args.state_output_file:
        parser.error("--state-verify cannot be combined with --state-output-file")

    input_path = args.state_input_file

    try:
        if args.state_verify:
            comparison, expected_path, generated_path = compare_and_save_position_list_files(
                input_path=input_path,
                expected_path=args.state_expected_file,
            )
            print(
                format_position_list_comparison(
                    comparison,
                    input_path=input_path,
                    expected_path=str(expected_path),
                    generated_path=str(generated_path),
                )
            )
            return 0

        board_output_path = args.state_output_file or str(default_board_output_path(input_path))
        move_output_path = _derive_move_output_path(board_output_path)

        Path(board_output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(move_output_path).parent.mkdir(parents=True, exist_ok=True)

        count = export_position_list_states(
            input_path=input_path,
            output_path=board_output_path,
            move_output_path=move_output_path,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Wrote {count} states to {board_output_path}")
    print(f"Wrote {count} moves  to {move_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
