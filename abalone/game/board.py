"""
Abalone board representation and core logic.

Coordinate system:
  - Rows: a (0) at bottom to i (8) at top
  - Cols: 1 (left) to 9 (right)
  - 6 hex directions: E, W, NW, SE, NE, SW

Board layout:
            O O O O O           i
           O O O O O O          h
          . . O O O . .         g
         . . . . . . . .        f
        . . . . . . . . .       e
         . . . . . . . .        d
          . . @ @ @ . .         c
           @ @ @ @ @ @          b
            @ @ @ @ @           a
"""

import random as _random
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass

Position = Tuple[int, int]
Direction = Tuple[int, int]

# --- Constants ---

BLACK = 1
WHITE = 2
EMPTY = 0

ROW_LETTERS = 'abcdefghi'

# 6 hex directions as (row_delta, col_delta)
DIRECTIONS: List[Direction] = [
    (0, 1),    # E  (right)
    (0, -1),   # W  (left)
    (1, 0),    # NW (up-left: row increases, same col)
    (-1, 0),   # SE (down-right: row decreases, same col)
    (1, 1),    # NE (up-right)
    (-1, -1),  # SW (down-left)
]

DIRECTION_NAMES: Dict[Direction, str] = {
    (0, 1): 'E', (0, -1): 'W',
    (1, 0): 'NW', (-1, 0): 'SE',
    (1, 1): 'NE', (-1, -1): 'SW',
}

NAME_TO_DIR: Dict[str, Direction] = {v: k for k, v in DIRECTION_NAMES.items()}

# 3 axes (pairs of opposite directions)
AXES = [
    ((0, 1), (0, -1)),    # E-W
    ((1, 0), (-1, 0)),    # NW-SE
    ((1, 1), (-1, -1)),   # NE-SW
]


def _build_valid_positions() -> Set[Position]:
    """Build the canonical set of playable coordinates on the Abalone hex board."""
    positions = set()
    for r in range(9):
        if r <= 4:
            for c in range(1, 6 + r):
                positions.add((r, c))
        else:
            for c in range(r - 3, 10):
                positions.add((r, c))
    return positions


VALID_POSITIONS: Set[Position] = _build_valid_positions()


# --- Zobrist Hashing ---
# Pre-computed random numbers for incrementally-maintained board hashing.
# keyed by (position, color) -> random 64-bit int.

def _build_zobrist_table() -> Dict[Tuple[Position, int], int]:
    """Build a deterministic Zobrist random table for all (position, color) pairs."""
    rng = _random.Random(0xABA10E)  # fixed seed for determinism
    table: Dict[Tuple[Position, int], int] = {}
    for pos in sorted(VALID_POSITIONS):
        for color in (BLACK, WHITE):
            table[(pos, color)] = rng.getrandbits(64)
    # Side-to-move keys
    table[('side', BLACK)] = rng.getrandbits(64)
    table[('side', WHITE)] = rng.getrandbits(64)
    return table

ZOBRIST: Dict = _build_zobrist_table()


def pos_to_str(pos: Position) -> str:
    """Convert an internal `(row, col)` position into notation like `e5`."""
    return f"{ROW_LETTERS[pos[0]]}{pos[1]}"


def str_to_pos(s: str) -> Position:
    """Convert notation like `e5` into an internal `(row, col)` tuple."""
    return (ROW_LETTERS.index(s[0]), int(s[1]))


def neighbor(pos: Position, d: Direction) -> Position:
    """Return the adjacent coordinate reached by applying direction `d` to `pos`."""
    return (pos[0] + d[0], pos[1] + d[1])


def is_valid(pos: Position) -> bool:
    """Return whether a coordinate is on the playable board."""
    return pos in VALID_POSITIONS


def opposite_dir(d: Direction) -> Direction:
    """Return the opposite direction vector for `d`."""
    return (-d[0], -d[1])


# --- Move ---

@dataclass(frozen=True)
class Move:
    """Immutable move payload used by both game logic and search."""

    marbles: Tuple[Position, ...]   # positions of marbles being moved
    direction: Direction             # movement direction

    @property
    def count(self) -> int:
        """Number of marbles in this move."""
        return len(self.marbles)

    @property
    def is_inline(self) -> bool:
        """Return whether movement is along the marble line (inline) vs broadside."""
        if self.count == 1:
            return True
        m0, m1 = sorted(self.marbles)[:2]
        line_dir = (m1[0] - m0[0], m1[1] - m0[1])
        return self.direction == line_dir or self.direction == opposite_dir(line_dir)

    def leading_trailing(self):
        """Return (trailing, leading) marble based on movement direction."""
        d = self.direction
        s = sorted(self.marbles,
                   key=lambda m: m[0] * d[0] + m[1] * d[1])
        return s[0], s[-1]  # trailing, leading

    def to_notation(self, pushed=False) -> str:
        """Return CLI/web notation for the move, optionally marking push candidates."""
        if self.is_inline:
            trailing, leading = self.leading_trailing()
            goal = neighbor(leading, self.direction)
            push_str = "*" if pushed else ""
            return f"{self.count}:{pos_to_str(trailing)}{pos_to_str(goal)}{push_str}"
        else:
            sorted_m = sorted(self.marbles)
            start = pos_to_str(sorted_m[0])
            end = pos_to_str(sorted_m[-1])
            dir_name = DIRECTION_NAMES[self.direction]
            return f"{self.count}:{start}-{end}>{dir_name}"

    def __repr__(self):
        """Represent moves using compact Abalone notation for debugging output."""
        return self.to_notation()


# --- Board ---

STANDARD_BLACK = [
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
    (2, 3), (2, 4), (2, 5),
]

STANDARD_WHITE = [
    (6, 5), (6, 6), (6, 7),
    (7, 4), (7, 5), (7, 6), (7, 7), (7, 8), (7, 9),
    (8, 5), (8, 6), (8, 7), (8, 8), (8, 9),
]

BELGIAN_DAISY_BLACK = [
    (0, 1), (0, 2), (1, 1), (1, 2), (1, 3), (2, 2), (2, 3),       # bottom-left flower
    (6, 7), (6, 8), (7, 7), (7, 8), (7, 9), (8, 8), (8, 9),       # top-right flower
]

BELGIAN_DAISY_WHITE = [
    (0, 4), (0, 5), (1, 4), (1, 5), (1, 6), (2, 5), (2, 6),       # bottom-right flower
    (6, 4), (6, 5), (7, 4), (7, 5), (7, 6), (8, 5), (8, 6),       # top-left flower
]

GERMAN_DAISY_BLACK = [
    (1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2), (3, 3),       # bottom-left flower
    (5, 7), (5, 8), (6, 7), (6, 8), (6, 9), (7, 8), (7, 9),       # top-right flower
]

GERMAN_DAISY_WHITE = [
    (1, 5), (1, 6), (2, 5), (2, 6), (2, 7), (3, 6), (3, 7),       # bottom-right flower
    (5, 3), (5, 4), (6, 3), (6, 4), (6, 5), (7, 4), (7, 5),       # top-left flower
]

LAYOUTS = {
    "standard": (STANDARD_BLACK, STANDARD_WHITE),
    "belgian_daisy": (BELGIAN_DAISY_BLACK, BELGIAN_DAISY_WHITE),
    "german_daisy": (GERMAN_DAISY_BLACK, GERMAN_DAISY_WHITE),
}


class Board:
    """Mutable board state and authoritative move legality/application rules."""

    def __init__(self):
        """Initialize an empty board and zeroed capture score."""
        self.cells: Dict[Position, int] = {pos: EMPTY for pos in VALID_POSITIONS}
        self.score: Dict[int, int] = {BLACK: 0, WHITE: 0}
        self.zhash: int = 0  # Zobrist hash maintained incrementally

    def recompute_zhash(self):
        """Recompute Zobrist hash from scratch (used after bulk setup)."""
        h = 0
        for pos, color in self.cells.items():
            if color != EMPTY:
                h ^= ZOBRIST[(pos, color)]
        self.zhash = h

    def setup_standard(self):
        """Set marbles to the standard Abalone starting position."""
        self.clear()
        for pos in STANDARD_BLACK:
            self.cells[pos] = BLACK
        for pos in STANDARD_WHITE:
            self.cells[pos] = WHITE
        self.recompute_zhash()

    def setup_layout(self, name: str):
        """Set up board using a named layout."""
        if name not in LAYOUTS:
            raise ValueError(f"Unknown layout: {name}")
        black_positions, white_positions = LAYOUTS[name]
        self.clear()
        for pos in black_positions:
            self.cells[pos] = BLACK
        for pos in white_positions:
            self.cells[pos] = WHITE
        self.recompute_zhash()

    def clear(self):
        """Remove all marbles and reset scores."""
        for pos in VALID_POSITIONS:
            self.cells[pos] = EMPTY
        self.score = {BLACK: 0, WHITE: 0}
        self.zhash = 0

    def copy(self) -> 'Board':
        """Create a deep copy of board cells, score state, and Zobrist hash."""
        b = Board()
        b.cells = dict(self.cells)
        b.score = dict(self.score)
        b.zhash = self.zhash
        return b

    def get(self, pos: Position) -> Optional[int]:
        """Return marble color at a coordinate, or `None` for unknown coordinates."""
        return self.cells.get(pos)

    def get_marbles(self, player: int) -> List[Position]:
        """Return sorted coordinates for all marbles owned by `player`."""
        return sorted(pos for pos, v in self.cells.items() if v == player)

    def marble_count(self, player: int) -> int:
        """Return number of marbles currently on board for `player`."""
        return sum(1 for v in self.cells.values() if v == player)

    # --- Move validation ---

    def is_legal_move(self, move: Move, player: int) -> bool:
        """Validate move ownership, marble formation, and direction-specific constraints."""
        opponent = WHITE if player == BLACK else BLACK

        # All marbles must belong to the player
        for m in move.marbles:
            if self.cells.get(m) != player:
                return False

        # Marbles must form a contiguous line
        if not self._marbles_in_line(move.marbles):
            return False

        if move.is_inline:
            return self._check_inline(move, player, opponent)
        else:
            return self._check_broadside(move)

    def _marbles_in_line(self, marbles: Tuple[Position, ...]) -> bool:
        """Return whether selected marbles form a contiguous straight line."""
        if len(marbles) <= 1:
            return True
        s = sorted(marbles)
        d = (s[1][0] - s[0][0], s[1][1] - s[0][1])
        if d not in DIRECTIONS:
            return False
        for i in range(2, len(s)):
            expected = (s[0][0] + i * d[0], s[0][1] + i * d[1])
            if s[i] != expected:
                return False
        return True

    def _check_inline(self, move: Move, player: int, opponent: int) -> bool:
        """Validate an inline move, including sumito push rules."""
        d = move.direction
        _, leading = move.leading_trailing()
        ahead = neighbor(leading, d)

        if not is_valid(ahead):
            return False  # can't move own marble off board

        if self.cells[ahead] == EMPTY:
            return True

        if self.cells[ahead] == player:
            return False  # can't push own marble

        # Count contiguous opponent marbles ahead
        pushed_count = 0
        pos = ahead
        while is_valid(pos) and self.cells[pos] == opponent:
            pushed_count += 1
            pos = neighbor(pos, d)

        # Must outnumber
        if pushed_count >= move.count:
            return False

        # Space after pushed marbles must be empty or off-board
        if is_valid(pos) and self.cells[pos] != EMPTY:
            return False

        return True

    def _check_broadside(self, move: Move) -> bool:
        """Validate broadside movement where every destination must be empty and valid."""
        d = move.direction
        for m in move.marbles:
            dest = neighbor(m, d)
            if not is_valid(dest):
                return False
            if self.cells[dest] != EMPTY:
                return False
        return True

    # --- Apply move ---

    def _zhash_set(self, pos: Position, color: int):
        """XOR a (pos, color) pair into the Zobrist hash."""
        if color != EMPTY:
            self.zhash ^= ZOBRIST[(pos, color)]

    def apply_move(self, move: Move, player: int) -> dict:
        """Apply move, mutating the board. Returns result info."""
        result = {'pushed': [], 'pushoff': False}
        opponent = WHITE if player == BLACK else BLACK

        if move.is_inline:
            self._apply_inline(move, player, opponent, result)
        else:
            self._apply_broadside(move, player)

        return result

    def apply_move_undo(self, move: Move, player: int) -> dict:
        """Apply move and return an undo token for fast rollback in search.

        The undo token captures all cell changes and score deltas so that
        ``undo_move`` can restore the board without allocating a copy.
        """
        opponent = WHITE if player == BLACK else BLACK
        # snapshot changes for undo
        old_cells: List[Tuple[Position, int]] = []
        old_score_player = self.score[player]
        old_score_opponent = self.score[opponent]
        old_zhash = self.zhash

        d = move.direction
        if move.is_inline:
            _, leading = move.leading_trailing()
            # Find pushed opponent marbles
            pushed = []
            pos = neighbor(leading, d)
            while is_valid(pos) and self.cells[pos] == opponent:
                pushed.append(pos)
                pos = neighbor(pos, d)

            # Handle push-off
            pushoff = pushed and not is_valid(pos)
            if pushoff:
                self.score[player] += 1

            # Move pushed (farthest first)
            for i in range(len(pushed) - 1, -1, -1):
                p = pushed[i]
                old_cells.append((p, self.cells[p]))
                self._zhash_set(p, self.cells[p])  # remove old
                dest = neighbor(p, d)
                if is_valid(dest):
                    old_cells.append((dest, self.cells[dest]))
                    self.cells[dest] = opponent
                    self._zhash_set(dest, opponent)  # add new
                self.cells[p] = EMPTY

            # Move own marbles (leading first)
            sorted_by_dir = sorted(
                move.marbles,
                key=lambda m: m[0] * d[0] + m[1] * d[1],
                reverse=True
            )
            for marble in sorted_by_dir:
                old_cells.append((marble, self.cells[marble]))
                self._zhash_set(marble, self.cells[marble])  # remove old
                dest = neighbor(marble, d)
                old_cells.append((dest, self.cells[dest]))
                self.cells[dest] = player
                self._zhash_set(dest, player)  # add new
                self.cells[marble] = EMPTY
        else:
            # Broadside
            for m in move.marbles:
                old_cells.append((m, self.cells[m]))
                self._zhash_set(m, self.cells[m])  # remove old
                self.cells[m] = EMPTY
            for m in move.marbles:
                dest = neighbor(m, d)
                old_cells.append((dest, self.cells[dest]))
                self.cells[dest] = player
                self._zhash_set(dest, player)  # add new

        return {
            'old_cells': old_cells,
            'player': player,
            'opponent': opponent,
            'old_score_p': old_score_player,
            'old_score_o': old_score_opponent,
            'old_zhash': old_zhash,
        }

    def undo_move(self, undo_info: dict):
        """Restore board from undo token produced by apply_move_undo."""
        player = undo_info['player']
        opponent = undo_info['opponent']
        # Restore cells in reverse order
        for pos, val in reversed(undo_info['old_cells']):
            self.cells[pos] = val
        self.score[player] = undo_info['old_score_p']
        self.score[opponent] = undo_info['old_score_o']
        self.zhash = undo_info['old_zhash']

    def _apply_inline(self, move: Move, player: int, opponent: int, result: dict):
        """Apply an inline move, including pushes and potential push-off scoring."""
        d = move.direction
        _, leading = move.leading_trailing()

        # Find pushed opponent marbles
        pushed = []
        pos = neighbor(leading, d)
        while is_valid(pos) and self.cells[pos] == opponent:
            pushed.append(pos)
            pos = neighbor(pos, d)
        # pos = first cell after pushed chain (empty or off-board)

        result['pushed'] = [pos_to_str(p) for p in pushed]

        # Handle push-off
        if pushed and not is_valid(pos):
            result['pushoff'] = True
            self.score[player] += 1

        # Move pushed opponent marbles (farthest first)
        for i in range(len(pushed) - 1, -1, -1):
            p = pushed[i]
            self._zhash_set(p, self.cells[p])
            dest = neighbor(p, d)
            if is_valid(dest):
                self._zhash_set(dest, self.cells[dest])
                self.cells[dest] = opponent
                self._zhash_set(dest, opponent)
            self.cells[p] = EMPTY

        # Move own marbles (leading first to avoid overwriting)
        sorted_by_dir = sorted(
            move.marbles,
            key=lambda m: m[0] * d[0] + m[1] * d[1],
            reverse=True
        )
        for marble in sorted_by_dir:
            self._zhash_set(marble, self.cells[marble])
            dest = neighbor(marble, d)
            self._zhash_set(dest, self.cells[dest])
            self.cells[dest] = player
            self._zhash_set(dest, player)
            self.cells[marble] = EMPTY

    def _apply_broadside(self, move: Move, player: int):
        """Apply a broadside move after legality has already been established."""
        d = move.direction
        # Clear old, set new (safe because broadside dests are always empty)
        for m in move.marbles:
            self._zhash_set(m, self.cells[m])
            self.cells[m] = EMPTY
        for m in move.marbles:
            dest = neighbor(m, d)
            self.cells[dest] = player
            self._zhash_set(dest, player)

    # --- Display ---

    def display(self, last_move: Optional[Move] = None) -> str:
        """
        Display board with coordinates visible on every empty cell.
        Each cell is 2 chars: empty = coordinate (e.g. e5), black = @@, white = OO.

             OO OO OO OO OO               i
           OO OO OO OO OO OO              h
         g3 g4 OO OO OO g8 g9             g
        f2 f3 f4 f5 f6 f7 f8 f9           f
       e1 e2 e3 e4 e5 e6 e7 e8 e9         e
        d1 d2 d3 d4 d5 d6 d7 d8           d
         c1 c2 @@ @@ @@ c6 c7             c
           @@ @@ @@ @@ @@ @@              b
             @@ @@ @@ @@ @@               a
        """
        lines = []
        lines.append("")
        lines.append(f"  Score: Black(@@) {self.score[BLACK]} - {self.score[WHITE]} White(OO)")
        lines.append("")

        for r in range(8, -1, -1):
            letter = ROW_LETTERS[r]
            indent = abs(r - 4)
            if r <= 4:
                cols = range(1, 6 + r)
            else:
                cols = range(r - 3, 10)

            cells = []
            for c in cols:
                val = self.cells[(r, c)]
                if val == BLACK:
                    cells.append('@@')
                elif val == WHITE:
                    cells.append('OO')
                else:
                    cells.append(f"{letter}{c}")

            row_str = f"  {'  ' * indent}{' '.join(cells)}"
            row_str = f"{row_str:<40s}{letter}"
            lines.append(row_str)

        lines.append("")
        return '\n'.join(lines)
