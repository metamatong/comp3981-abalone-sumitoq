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

from typing import Tuple, Set, List, Optional, Dict
from dataclasses import dataclass, field

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


def pos_to_str(pos: Position) -> str:
    return f"{ROW_LETTERS[pos[0]]}{pos[1]}"


def str_to_pos(s: str) -> Position:
    return (ROW_LETTERS.index(s[0]), int(s[1]))


def neighbor(pos: Position, d: Direction) -> Position:
    return (pos[0] + d[0], pos[1] + d[1])


def is_valid(pos: Position) -> bool:
    return pos in VALID_POSITIONS


def opposite_dir(d: Direction) -> Direction:
    return (-d[0], -d[1])


# --- Move ---

@dataclass(frozen=True)
class Move:
    marbles: Tuple[Position, ...]   # positions of marbles being moved
    direction: Direction             # movement direction

    @property
    def count(self) -> int:
        return len(self.marbles)

    @property
    def is_inline(self) -> bool:
        if self.count == 1:
            return True
        m0, m1 = sorted(self.marbles)[:2]
        line_dir = (m1[0] - m0[0], m1[1] - m0[1])
        return self.direction == line_dir or self.direction == opposite_dir(line_dir)

    def _leading_trailing(self):
        """Return (trailing, leading) marble based on movement direction."""
        d = self.direction
        s = sorted(self.marbles,
                   key=lambda m: m[0] * d[0] + m[1] * d[1])
        return s[0], s[-1]  # trailing, leading

    def to_notation(self, pushed=False) -> str:
        if self.is_inline:
            trailing, leading = self._leading_trailing()
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
    (0, 1), (0, 2), (1, 1), (1, 2), (1, 3), (2, 1), (2, 2),       # bottom-left flower
    (6, 8), (6, 9), (7, 7), (7, 8), (7, 9), (8, 8), (8, 9),       # top-right flower
]

BELGIAN_DAISY_WHITE = [
    (0, 4), (0, 5), (1, 4), (1, 5), (1, 6), (2, 6), (2, 7),       # bottom-right flower
    (6, 3), (6, 4), (7, 4), (7, 5), (7, 6), (8, 5), (8, 6),       # top-left flower
]

GERMAN_DAISY_BLACK = [
    (1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2),       # bottom-left flower
    (5, 8), (5, 9), (6, 7), (6, 8), (6, 9), (7, 8), (7, 9),       # top-right flower
]

GERMAN_DAISY_WHITE = [
    (1, 5), (1, 6), (2, 5), (2, 6), (2, 7), (3, 7), (3, 8),       # bottom-right flower
    (5, 2), (5, 3), (6, 3), (6, 4), (6, 5), (7, 4), (7, 5),       # top-left flower
]

LAYOUTS = {
    "standard": (STANDARD_BLACK, STANDARD_WHITE),
    "belgian_daisy": (BELGIAN_DAISY_BLACK, BELGIAN_DAISY_WHITE),
    "german_daisy": (GERMAN_DAISY_BLACK, GERMAN_DAISY_WHITE),
}


class Board:
    def __init__(self):
        self.cells: Dict[Position, int] = {pos: EMPTY for pos in VALID_POSITIONS}
        self.score: Dict[int, int] = {BLACK: 0, WHITE: 0}

    def setup_standard(self):
        self.clear()
        for pos in STANDARD_BLACK:
            self.cells[pos] = BLACK
        for pos in STANDARD_WHITE:
            self.cells[pos] = WHITE

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

    def clear(self):
        for pos in VALID_POSITIONS:
            self.cells[pos] = EMPTY
        self.score = {BLACK: 0, WHITE: 0}

    def copy(self) -> 'Board':
        b = Board()
        b.cells = dict(self.cells)
        b.score = dict(self.score)
        return b

    def get(self, pos: Position) -> Optional[int]:
        return self.cells.get(pos)

    def get_marbles(self, player: int) -> List[Position]:
        return sorted(pos for pos, v in self.cells.items() if v == player)

    def marble_count(self, player: int) -> int:
        return sum(1 for v in self.cells.values() if v == player)

    # --- Move validation ---

    def is_legal_move(self, move: Move, player: int) -> bool:
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
        d = move.direction
        _, leading = move._leading_trailing()
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
        d = move.direction
        for m in move.marbles:
            dest = neighbor(m, d)
            if not is_valid(dest):
                return False
            if self.cells[dest] != EMPTY:
                return False
        return True

    # --- Apply move ---

    def apply_move(self, move: Move, player: int) -> dict:
        """Apply move, mutating the board. Returns result info."""
        result = {'pushed': [], 'pushoff': False}
        opponent = WHITE if player == BLACK else BLACK

        if move.is_inline:
            self._apply_inline(move, player, opponent, result)
        else:
            self._apply_broadside(move, player)

        return result

    def _apply_inline(self, move: Move, player: int, opponent: int, result: dict):
        d = move.direction
        _, leading = move._leading_trailing()

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
            dest = neighbor(pushed[i], d)
            if is_valid(dest):
                self.cells[dest] = opponent
            # else: pushed off, marble lost
            self.cells[pushed[i]] = EMPTY

        # Move own marbles (leading first to avoid overwriting)
        sorted_by_dir = sorted(
            move.marbles,
            key=lambda m: m[0] * d[0] + m[1] * d[1],
            reverse=True
        )
        for marble in sorted_by_dir:
            dest = neighbor(marble, d)
            self.cells[dest] = player
            self.cells[marble] = EMPTY

    def _apply_broadside(self, move: Move, player: int):
        d = move.direction
        # Clear old, set new (safe because broadside dests are always empty)
        for m in move.marbles:
            self.cells[m] = EMPTY
        for m in move.marbles:
            self.cells[neighbor(m, d)] = player

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
