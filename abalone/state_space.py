"""State-space generator for legal Abalone moves.

This module provides only legal, deduplicated move expansion:
  - generate_legal_moves(): all unique legal moves for a position
"""

from typing import List, Set, Tuple

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
)

# 3 "positive" directions (one per axis, to avoid counting pairs/triples twice)
POSITIVE_DIRS: List[Direction] = [(0, 1), (1, 0), (1, 1)]


def generate_legal_moves(board: Board, player: int) -> List[Move]:
    """Generate all unique legal moves for the given player.

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

    for marble in marbles:
        # 1-stone moves
        for direction in DIRECTIONS:
            _add([marble], direction)

        # 2-stone moves
        for line_dir in POSITIVE_DIRS:
            second = neighbor(marble, line_dir)
            if second in marble_set:
                pair = [marble, second]
                for direction in DIRECTIONS:
                    _add(pair, direction)

        # 3-stone moves
        for line_dir in POSITIVE_DIRS:
            second = neighbor(marble, line_dir)
            third = neighbor(second, line_dir)
            if second in marble_set and third in marble_set:
                triple = [marble, second, third]
                for direction in DIRECTIONS:
                    _add(triple, direction)

    return moves


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
