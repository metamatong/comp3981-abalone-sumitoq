"""State-space generator for legal Abalone moves.

This module provides legal, deduplicated move expansion:
  - generate_legal_moves(): all unique legal moves for a position
  - generate_next_states(): all depth-1 child boards for a position
  - compact position-list parsing/serialization helpers for file-based expansion
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple

from .board import (
    BLACK,
    WHITE,
    Board,
    Direction,
    DIRECTIONS,
    Move,
    Position,
    is_valid,
    neighbor,
    pos_to_str,
    str_to_pos,
)

# 3 "positive" directions (one per axis, to avoid counting pairs/triples twice)
POSITIVE_DIRS: List[Direction] = [(0, 1), (1, 0), (1, 1)]
INITIAL_MARBLES_PER_PLAYER = 14

# Direction order used by the reference implementation for move enumeration.
REFERENCE_DIRECTIONS: List[Direction] = [
    (1, 1),    # NE
    (0, 1),    # E
    (-1, 0),   # SE
    (-1, -1),  # SW
    (0, -1),   # W
    (1, 0),    # NW
]


@dataclass(frozen=True)
class PositionListComparison:
    """Comparison result for generated child states versus an expected output file."""

    actual_lines: int
    expected_lines: int
    exact_match: bool
    same_set: bool
    mismatch_count: int
    first_mismatch_line: Optional[int]
    first_actual: Optional[str]
    first_expected: Optional[str]
    extra_actual_lines: int
    extra_expected_lines: int


def generate_legal_moves(board: Board, player: int) -> List[Move]:
    """Generate all unique legal moves for the given player.

    Moves are enumerated using trailing-marble-based iteration: each marble
    is treated as the rearmost (trailing) marble and inline groups of 1, 2,
    and 3 marbles are extended forward in the move direction.  The direction
    order follows the reference implementation (NE, E, SE, SW, W, NW).

    :param board: Current board state to evaluate.
    :param player: Color constant (BLACK or WHITE) whose moves to generate.
    :return: Deduplicated list of legal Move objects.
    """
    moves: List[Move] = []
    seen: Set[Tuple[Tuple[Position, ...], Direction]] = set()
    marbles = board.get_marbles(player)
    marble_set = set(marbles)

    def _add(marbles_list, direction):
        """Insert a move candidate once, then keep only legal candidates.

        :param marbles_list: List of (row, col) positions for the marbles in the move.
        :param direction: (dr, dc) direction tuple for the move.
        """
        key = (tuple(sorted(marbles_list)), direction)
        if key in seen:
            return

        seen.add(key)
        move = Move(marbles=tuple(sorted(marbles_list)), direction=direction)
        if board.is_legal_move(move, player):
            moves.append(move)

    # --- Pass 1: inline moves (trailing-marble enumeration) ---
    # This determines the canonical ordering for inline moves.
    for marble in marbles:
        for direction in REFERENCE_DIRECTIONS:
            # 1-stone move
            _add([marble], direction)

            # 2-stone inline: extend forward from trailing marble
            fwd1 = neighbor(marble, direction)
            if fwd1 in marble_set:
                _add([marble, fwd1], direction)

                # 3-stone inline: extend one more step forward
                fwd2 = neighbor(fwd1, direction)
                if fwd2 in marble_set:
                    _add([marble, fwd1, fwd2], direction)

    # --- Pass 2: broadside moves ---
    # Find pairs/triples along each line axis and try all directions.
    # The ``seen`` set ensures inline moves from pass 1 are not duplicated.
    for marble in marbles:
        for line_dir in POSITIVE_DIRS:
            second = neighbor(marble, line_dir)
            if second in marble_set:
                pair = [marble, second]
                for direction in REFERENCE_DIRECTIONS:
                    _add(pair, direction)

                third = neighbor(second, line_dir)
                if third in marble_set:
                    triple = [marble, second, third]
                    for direction in REFERENCE_DIRECTIONS:
                        _add(triple, direction)

    return moves


def generate_next_states(board: Board, player: int) -> List[Board]:
    """Generate every depth-1 child board reachable by one legal move."""
    next_states: List[Board] = []
    for move in generate_legal_moves(board, player):
        child = board.copy()
        child.apply_move(move, player)
        next_states.append(child)
    return next_states


def load_position_list_state(text: str) -> Tuple[Board, int]:
    """Parse a compact board-state text block into a board and side to move.

    Expected format:
      1. First non-empty line: `b` or `w`
      2. Remaining comma-separated tokens: `C5b`, `H9w`, ...

    Scores are inferred from how many marbles are missing from the standard
    14-marble starting count for each side.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("State input is empty.")

    turn = lines[0].lower()
    if turn not in {"b", "w"}:
        raise ValueError("First line must be `b` or `w` to indicate the side to move.")

    board = Board()
    board.clear()
    seen: Set[Position] = set()
    tokens: List[str] = []
    for line in lines[1:]:
        tokens.extend(token.strip() for token in line.split(",") if token.strip())

    for token in tokens:
        if len(token) != 3:
            raise ValueError(f"Invalid board token: {token!r}. Expected format like `C5b`.")

        coord_text = token[:2].lower()
        color_text = token[2].lower()
        if color_text not in {"b", "w"}:
            raise ValueError(f"Invalid marble color in token: {token!r}.")

        try:
            pos = str_to_pos(coord_text)
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Invalid board coordinate in token: {token!r}.") from exc

        if not is_valid(pos):
            raise ValueError(f"Invalid board coordinate in token: {token!r}.")
        if pos in seen:
            raise ValueError(f"Duplicate board coordinate in input: {token[:2]!r}.")

        seen.add(pos)
        board.cells[pos] = BLACK if color_text == "b" else WHITE

    black_count = board.marble_count(BLACK)
    white_count = board.marble_count(WHITE)
    if black_count > INITIAL_MARBLES_PER_PLAYER or white_count > INITIAL_MARBLES_PER_PLAYER:
        raise ValueError("Input contains more than 14 marbles for one side.")

    board.score = {
        BLACK: INITIAL_MARBLES_PER_PLAYER - white_count,
        WHITE: INITIAL_MARBLES_PER_PLAYER - black_count,
    }
    player = BLACK if turn == "b" else WHITE
    return board, player


def dump_position_list_state(board: Board) -> str:
    """Serialize the board into the compact comma-separated `C5b`/`H9w` format."""
    tokens = [f"{pos_to_str(pos).upper()}b" for pos in sorted(board.get_marbles(BLACK))]
    tokens.extend(f"{pos_to_str(pos).upper()}w" for pos in sorted(board.get_marbles(WHITE)))
    return ",".join(tokens)


def generate_next_state_strings(board: Board, player: int) -> List[str]:
    """Generate serialized depth-1 child states in compact position-list format."""
    return [dump_position_list_state(child) for child in generate_next_states(board, player)]


def expand_position_list_text(text: str) -> List[str]:
    """Expand a compact position-list input text block into serialized child states."""
    board, player = load_position_list_state(text)
    return generate_next_state_strings(board, player)


def export_position_list_states(input_path: str, output_path: Optional[str] = None) -> int:
    """Expand one compact state file and optionally write the serialized child states.

    :return: Number of child states generated.
    """
    child_states = expand_position_list_text(Path(input_path).read_text(encoding="utf-8"))
    if output_path:
        write_position_list_lines(child_states, output_path)
    else:
        print("\n".join(child_states))

    return len(child_states)


def expected_output_path_for_input(input_path: str) -> Path:
    """Resolve the conventional expected output path for a state-space input file."""
    source = Path(input_path)
    if source.parent.name != "state_space_inputs":
        raise ValueError(
            "Cannot infer expected output path. "
            "Input must be inside a `state_space_inputs/` directory."
        )

    output_dir = source.parent.parent / "state_space_outputs"
    return output_dir / f"{source.stem}.board"


def generated_output_path_for_expected(expected_path: str) -> Path:
    """Return a sibling path for storing generated output beside an expected `.board` file."""
    expected = Path(expected_path)
    return expected.with_name(f"{expected.stem}.generated{expected.suffix}")


def write_position_list_lines(lines: List[str], output_path: str) -> Path:
    """Write serialized child-state lines to disk with a trailing newline when non-empty."""
    resolved = Path(output_path)
    content = "\n".join(lines)
    resolved.write_text(f"{content}\n" if content else "", encoding="utf-8")
    return resolved


def compare_position_list_lines(actual: List[str], expected: List[str]) -> PositionListComparison:
    """Compare generated child-state strings against expected lines, preserving order."""
    mismatches = []
    for index, (actual_line, expected_line) in enumerate(zip(actual, expected), start=1):
        if actual_line != expected_line:
            mismatches.append((index, actual_line, expected_line))

    first_mismatch_line: Optional[int] = None
    first_actual: Optional[str] = None
    first_expected: Optional[str] = None
    if mismatches:
        first_mismatch_line, first_actual, first_expected = mismatches[0]

    return PositionListComparison(
        actual_lines=len(actual),
        expected_lines=len(expected),
        exact_match=actual == expected,
        same_set=set(actual) == set(expected),
        mismatch_count=len(mismatches),
        first_mismatch_line=first_mismatch_line,
        first_actual=first_actual,
        first_expected=first_expected,
        extra_actual_lines=max(0, len(actual) - len(expected)),
        extra_expected_lines=max(0, len(expected) - len(actual)),
    )


def compare_position_list_files(
    input_path: str,
    expected_path: Optional[str] = None,
) -> Tuple[PositionListComparison, Path]:
    """Generate child states from an input file and compare them to an expected file."""
    resolved_expected = Path(expected_path) if expected_path else expected_output_path_for_input(input_path)
    actual = expand_position_list_text(Path(input_path).read_text(encoding="utf-8"))
    expected = resolved_expected.read_text(encoding="utf-8").splitlines()
    return compare_position_list_lines(actual, expected), resolved_expected


def compare_and_save_position_list_files(
    input_path: str,
    expected_path: Optional[str] = None,
    generated_output_path: Optional[str] = None,
) -> Tuple[PositionListComparison, Path, Path]:
    """Compare generated child states to expected output and save the generated lines."""
    resolved_expected = Path(expected_path) if expected_path else expected_output_path_for_input(input_path)
    actual = expand_position_list_text(Path(input_path).read_text(encoding="utf-8"))
    expected = resolved_expected.read_text(encoding="utf-8").splitlines()
    resolved_generated = write_position_list_lines(
        actual,
        generated_output_path or str(generated_output_path_for_expected(str(resolved_expected))),
    )
    return compare_position_list_lines(actual, expected), resolved_expected, resolved_generated


def format_position_list_comparison(
    comparison: PositionListComparison,
    input_path: str,
    expected_path: str,
    generated_path: Optional[str] = None,
) -> str:
    """Render a human-readable comparison summary for CLI output."""
    lines = [
        f"input={input_path}",
        f"expected={expected_path}",
    ]
    if generated_path is not None:
        lines.append(f"generated={generated_path}")
    lines.extend(
        [
        f"actual_lines={comparison.actual_lines}",
        f"expected_lines={comparison.expected_lines}",
        f"exact_match={comparison.exact_match}",
        f"same_set={comparison.same_set}",
        f"mismatch_count={comparison.mismatch_count}",
        ]
    )

    if comparison.first_mismatch_line is not None:
        lines.append(f"first_mismatch_line={comparison.first_mismatch_line}")
        lines.append("actual:")
        lines.append(comparison.first_actual or "")
        lines.append("expected:")
        lines.append(comparison.first_expected or "")

    if comparison.extra_actual_lines:
        lines.append(f"extra_actual_lines={comparison.extra_actual_lines}")
    if comparison.extra_expected_lines:
        lines.append(f"extra_expected_lines={comparison.extra_expected_lines}")

    return "\n".join(lines)


def categorize_moves(moves: List[Move]) -> dict:
    """Categorize moves by type for display/analysis.

    :param moves: List of Move objects to categorize.
    :return: Dict mapping category names to lists of moves.
    """
    categories = {
        "single_inline": [],
        "double_inline": [],
        "double_broadside": [],
        "triple_inline": [],
        "triple_broadside": [],
    }

    for move in moves:
        count = move.count
        inline = move.is_inline
        if count == 1:
            categories["single_inline"].append(move)
        elif count == 2:
            categories["double_inline" if inline else "double_broadside"].append(move)
        else:
            categories["triple_inline" if inline else "triple_broadside"].append(move)

    return categories


def print_state_space_summary(board: Board, player: int):
    """Print a summary of legal move space for the given player.

    :param board: Current board state to analyze.
    :param player: Color constant (BLACK or WHITE) whose moves to summarize.
    """
    pname = "Black" if player == BLACK else "White"
    marbles = board.get_marbles(player)

    legal_moves = generate_legal_moves(board, player)
    categories = categorize_moves(legal_moves)

    print(f"\n=== State Space for {pname} ({len(marbles)} marbles) ===")
    print(f"  Legal moves (unique): {len(legal_moves)}")
    print()
    print("  Breakdown:")
    for category_name, cat_moves in categories.items():
        label = category_name.replace("_", " ").title()
        print(f"    {label:25s}: {len(cat_moves)}")

    # Show push moves
    push_moves = []
    for move in legal_moves:
        if move.is_inline and move.count > 1:
            direction = move.direction
            _, leading = move._leading_trailing()
            ahead = neighbor(leading, direction)
            opponent = WHITE if player == BLACK else BLACK
            if is_valid(ahead) and board.cells.get(ahead) == opponent:
                push_moves.append(move)

    if push_moves:
        print(f"\n  Push moves (sumito): {len(push_moves)}")
        for push_move in push_moves:
            print(f"    {push_move.to_notation(pushed=True)}")

    print()
