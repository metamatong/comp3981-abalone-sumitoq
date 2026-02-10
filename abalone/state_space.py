"""
State space generator for Abalone.

Theoretical max per turn (without legality):
  14 marbles × (6 + 6×6 + 3×6) = 14 × 60 = 840 raw moves

  Per marble:
    - 1-stone: 6 directions                        = 6
    - 2-stone: 6 neighbor slots × 6 move directions = 36
    - 3-stone: 3 axes × 6 move directions            = 18
                                                 Total: 60

This module provides:
  - generate_raw_moves():  all 840 raw moves (no legality checks, may have duplicates)
  - generate_legal_moves(): all unique legal moves for a position
"""

from .board import (
    Board, Move, Position, Direction,
    DIRECTIONS, AXES, VALID_POSITIONS,
    BLACK, WHITE, EMPTY,
    neighbor, is_valid, opposite_dir, pos_to_str,
)
from typing import List, Set, Tuple


# 3 "positive" directions (one per axis, to avoid counting pairs/triples twice)
POSITIVE_DIRS: List[Direction] = [(0, 1), (1, 0), (1, 1)]


def generate_raw_moves(board: Board, player: int) -> List[Move]:
    """
    Generate all 840 raw moves per the formula:
      14 × (6 + 6×6 + 3×6) = 840

    For each marble:
      - 1-stone: try all 6 directions
      - 2-stone: for each of 6 neighbor directions, form a pair,
                 then try all 6 move directions
      - 3-stone: for each of 3 axes (positive dir), form a triple
                 (marble at one end), then try all 6 move directions

    No legality or validity checks. Includes duplicates
    (pairs counted from both ends, triples counted from all 3 members).
    """
    raw = []
    marbles = board.get_marbles(player)

    for m in marbles:
        # --- 1-stone: 6 moves ---
        for d in DIRECTIONS:
            raw.append(Move(marbles=(m,), direction=d))

        # --- 2-stone: 6 neighbors × 6 directions = 36 moves ---
        for nd in DIRECTIONS:
            m2 = neighbor(m, nd)
            pair = tuple(sorted([m, m2]))
            for d in DIRECTIONS:
                raw.append(Move(marbles=pair, direction=d))

        # --- 3-stone: 3 axes × 6 directions = 18 moves ---
        for pd in POSITIVE_DIRS:
            # Extend in positive direction: m, m+d, m+2d
            m2 = neighbor(m, pd)
            m3 = neighbor(m2, pd)
            triple = tuple(sorted([m, m2, m3]))
            for d in DIRECTIONS:
                raw.append(Move(marbles=triple, direction=d))

    return raw


def generate_legal_moves(board: Board, player: int) -> List[Move]:
    """
    Generate all unique legal moves for the given player.
    Properly deduplicates and validates each move.
    """
    moves: List[Move] = []
    seen: Set[Tuple[Tuple[Position, ...], Direction]] = set()
    marbles = board.get_marbles(player)
    marble_set = set(marbles)

    def _add(marbles_list, direction):
        key = (tuple(sorted(marbles_list)), direction)
        if key in seen:
            return
        seen.add(key)
        move = Move(marbles=tuple(sorted(marbles_list)), direction=direction)
        if board.is_legal_move(move, player):
            moves.append(move)

    for m in marbles:
        # 1-stone moves
        for d in DIRECTIONS:
            _add([m], d)

        # 2-stone moves
        for pd in POSITIVE_DIRS:
            m2 = neighbor(m, pd)
            if m2 in marble_set:
                pair = [m, m2]
                for d in DIRECTIONS:
                    _add(pair, d)

        # 3-stone moves
        for pd in POSITIVE_DIRS:
            m2 = neighbor(m, pd)
            m3 = neighbor(m2, pd)
            if m2 in marble_set and m3 in marble_set:
                triple = [m, m2, m3]
                for d in DIRECTIONS:
                    _add(triple, d)

    return moves


def categorize_moves(moves: List[Move]) -> dict:
    """Categorize moves by type for display/analysis."""
    cats = {
        'single_inline': [],
        'double_inline': [],
        'double_broadside': [],
        'triple_inline': [],
        'triple_broadside': [],
    }
    for move in moves:
        n = move.count
        inline = move.is_inline
        if n == 1:
            cats['single_inline'].append(move)
        elif n == 2:
            cats['double_inline' if inline else 'double_broadside'].append(move)
        else:
            cats['triple_inline' if inline else 'triple_broadside'].append(move)
    return cats


def print_state_space_summary(board: Board, player: int):
    """Print a summary of the state space for the given player."""
    pname = "Black" if player == BLACK else "White"
    marbles = board.get_marbles(player)

    # Raw moves (840 theoretical)
    raw = generate_raw_moves(board, player)

    # Legal moves (deduplicated)
    legal = generate_legal_moves(board, player)
    cats = categorize_moves(legal)

    print(f"\n=== State Space for {pname} ({len(marbles)} marbles) ===")
    print(f"  Raw moves (with duplicates, no legality): {len(raw)}")
    print(f"  Legal moves (unique):                     {len(legal)}")
    print()
    print(f"  Breakdown:")
    for cat_name, cat_moves in cats.items():
        label = cat_name.replace('_', ' ').title()
        print(f"    {label:25s}: {len(cat_moves)}")

    # Show push moves
    push_moves = []
    for move in legal:
        if move.is_inline and move.count > 1:
            d = move.direction
            _, leading = move._leading_trailing()
            ahead = neighbor(leading, d)
            opponent = WHITE if player == BLACK else BLACK
            if is_valid(ahead) and board.cells.get(ahead) == opponent:
                push_moves.append(move)

    if push_moves:
        print(f"\n  Push moves (sumito): {len(push_moves)}")
        for pm in push_moves:
            print(f"    {pm.to_notation(pushed=True)}")

    print()
