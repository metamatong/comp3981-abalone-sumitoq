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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

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
DIRECTION_SET = frozenset(DIRECTIONS)

DIRECTION_NAMES: Dict[Direction, str] = {
    (0, 1): 'E', (0, -1): 'W',
    (1, 0): 'NW', (-1, 0): 'SE',
    (1, 1): 'NE', (-1, -1): 'SW',
}

NAME_TO_DIR: Dict[str, Direction] = {v: k for k, v in DIRECTION_NAMES.items()}
DIRECTION_INDEX: Dict[Direction, int] = {direction: index for index, direction in enumerate(DIRECTIONS)}
OPPOSITE_DIRECTION: Dict[Direction, Direction] = {
    direction: (-direction[0], -direction[1])
    for direction in DIRECTIONS
}

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
ORDERED_VALID_POSITIONS: Tuple[Position, ...] = tuple(sorted(VALID_POSITIONS))
POSITION_INDEX: Dict[Position, int] = {
    position: index
    for index, position in enumerate(ORDERED_VALID_POSITIONS)
}


def _build_neighbor_table() -> Dict[Position, Tuple[Optional[Position], ...]]:
    """Precompute valid neighbors for every board cell and direction."""
    table: Dict[Position, Tuple[Optional[Position], ...]] = {}
    for pos in ORDERED_VALID_POSITIONS:
        neighbors: List[Optional[Position]] = []
        row, col = pos
        for dr, dc in DIRECTIONS:
            candidate = (row + dr, col + dc)
            neighbors.append(candidate if candidate in VALID_POSITIONS else None)
        table[pos] = tuple(neighbors)
    return table


NEIGHBOR_TABLE: Dict[Position, Tuple[Optional[Position], ...]] = _build_neighbor_table()


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
    return OPPOSITE_DIRECTION.get(d, (-d[0], -d[1]))


def _canonicalize_marbles(marbles: Tuple[Position, ...]) -> Tuple[Position, ...]:
    """Return a canonical marble tuple without paying general sort costs for <=3 items."""
    count = len(marbles)
    if count <= 1:
        return tuple(marbles)
    if count == 2:
        first, second = marbles
        return (first, second) if first <= second else (second, first)
    if count == 3:
        first, second, third = marbles
        if first > second:
            first, second = second, first
        if second > third:
            second, third = third, second
        if first > second:
            first, second = second, first
        return (first, second, third)
    return tuple(sorted(marbles))


def _ordered_marbles_for_direction(
    marbles: Tuple[Position, ...],
    direction: Direction,
) -> Tuple[Position, ...]:
    """Order marbles from trailing to leading for a move direction."""
    count = len(marbles)
    if count <= 1:
        return marbles

    dr, dc = direction
    decorated = [
        (marble[0] * dr + marble[1] * dc, marble)
        for marble in marbles
    ]

    if count == 2:
        first, second = decorated
        if first[0] <= second[0]:
            return (first[1], second[1])
        return (second[1], first[1])

    if count == 3:
        first, second, third = decorated
        if first[0] > second[0]:
            first, second = second, first
        if second[0] > third[0]:
            second, third = third, second
        if first[0] > second[0]:
            first, second = second, first
        return (first[1], second[1], third[1])

    return tuple(marble for _, marble in sorted(decorated))


# --- Move ---

@dataclass(frozen=True, slots=True)
class Move:
    """Immutable move payload used by both game logic and search."""

    marbles: Tuple[Position, ...]   # positions of marbles being moved
    direction: Direction             # movement direction
    _count: int = field(init=False, repr=False, compare=False)
    _sorted_marbles: Tuple[Position, ...] = field(init=False, repr=False, compare=False)
    _ordered_marbles: Tuple[Position, ...] = field(init=False, repr=False, compare=False)
    _line_dir: Optional[Direction] = field(init=False, repr=False, compare=False)
    _is_inline: bool = field(init=False, repr=False, compare=False)
    _leading: Position = field(init=False, repr=False, compare=False)
    _trailing: Position = field(init=False, repr=False, compare=False)
    _ordering_key: Tuple[int, Position, int, Position, str] = field(init=False, repr=False, compare=False)
    _notation_plain: Optional[str] = field(init=False, repr=False, compare=False, default=None)
    _notation_pushed: Optional[str] = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self):
        """Normalize marble order and cache derived geometry once."""
        self._set_cached_geometry(tuple(self.marbles), self.direction, canonical=False)

    @classmethod
    def from_canonical(cls, marbles: Tuple[Position, ...], direction: Direction) -> "Move":
        """Build a move when the marble tuple is already in canonical order."""
        move = object.__new__(cls)
        move._set_cached_geometry(marbles, direction, canonical=True)
        return move

    def _set_cached_geometry(
        self,
        marbles: Tuple[Position, ...],
        direction: Direction,
        *,
        canonical: bool,
    ) -> None:
        """Populate the canonical move state and derived caches."""
        sorted_marbles = marbles if canonical else _canonicalize_marbles(marbles)
        direction = (int(direction[0]), int(direction[1]))
        count = len(sorted_marbles)

        if count <= 1:
            ordered_marbles = sorted_marbles
            line_dir = None
            is_inline = True
        else:
            line_dir = (
                sorted_marbles[1][0] - sorted_marbles[0][0],
                sorted_marbles[1][1] - sorted_marbles[0][1],
            )
            is_inline = direction == line_dir or direction == OPPOSITE_DIRECTION.get(line_dir, opposite_dir(line_dir))
            ordered_marbles = _ordered_marbles_for_direction(sorted_marbles, direction)

        leading = ordered_marbles[-1]
        trailing = ordered_marbles[0]
        if is_inline:
            ordering_key = (
                count,
                trailing,
                1,
                neighbor(leading, direction),
                "",
            )
        else:
            ordering_key = (
                count,
                sorted_marbles[0],
                0,
                sorted_marbles[-1],
                DIRECTION_NAMES[direction],
            )

        object.__setattr__(self, "marbles", sorted_marbles)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "_count", count)
        object.__setattr__(self, "_sorted_marbles", sorted_marbles)
        object.__setattr__(self, "_ordered_marbles", ordered_marbles)
        object.__setattr__(self, "_line_dir", line_dir)
        object.__setattr__(self, "_is_inline", is_inline)
        object.__setattr__(self, "_leading", leading)
        object.__setattr__(self, "_trailing", trailing)
        object.__setattr__(self, "_ordering_key", ordering_key)
        object.__setattr__(self, "_notation_plain", None)
        object.__setattr__(self, "_notation_pushed", None)

    @property
    def count(self) -> int:
        """Number of marbles in this move."""
        return self._count

    @property
    def is_inline(self) -> bool:
        """Return whether movement is along the marble line (inline) vs broadside."""
        return self._is_inline

    def leading_trailing(self):
        """Return (trailing, leading) marble based on movement direction."""
        return self._trailing, self._leading

    @property
    def ordering_key(self) -> Tuple[int, Position, int, Position, str]:
        """Return a deterministic sort key equivalent to plain notation order."""
        return self._ordering_key

    def to_notation(self, pushed=False) -> str:
        """Return CLI/web notation for the move, optionally marking push candidates."""
        cached = self._notation_pushed if pushed else self._notation_plain
        if cached is not None:
            return cached

        if self.is_inline:
            goal = neighbor(self._leading, self.direction)
            push_str = "*" if pushed else ""
            notation = f"{self.count}:{pos_to_str(self._trailing)}{pos_to_str(goal)}{push_str}"
        else:
            start = pos_to_str(self._sorted_marbles[0])
            end = pos_to_str(self._sorted_marbles[-1])
            dir_name = DIRECTION_NAMES[self.direction]
            notation = f"{self.count}:{start}-{end}>{dir_name}"

        if pushed:
            object.__setattr__(self, "_notation_pushed", notation)
        else:
            object.__setattr__(self, "_notation_plain", notation)
        return notation

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
        for pos in ORDERED_VALID_POSITIONS:
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

    def to_compact_token(self) -> str:
        """Serialize occupied cells plus capture score into a compact deterministic token."""
        occupied = []
        for pos in sorted(self.cells):
            color = self.cells[pos]
            if color == EMPTY:
                continue
            occupied.append(f"{pos_to_str(pos)}{'b' if color == BLACK else 'w'}")
        return f"{','.join(occupied)}|{self.score[BLACK]}-{self.score[WHITE]}"

    @classmethod
    def from_compact_token(cls, token: str) -> 'Board':
        """Rebuild a board from ``to_compact_token`` output."""
        board = cls()
        board.clear()
        token = str(token or "").strip()
        if not token:
            return board

        occupied_part, _, score_part = token.partition("|")
        if occupied_part:
            for item in occupied_part.split(","):
                item = item.strip()
                if not item:
                    continue
                color_char = item[-1].lower()
                pos = str_to_pos(item[:-1])
                board.cells[pos] = BLACK if color_char == "b" else WHITE
        if score_part:
            black_score, white_score = score_part.split("-", 1)
            board.score = {BLACK: int(black_score), WHITE: int(white_score)}
        board.recompute_zhash()
        return board

    def get(self, pos: Position) -> Optional[int]:
        """Return marble color at a coordinate, or `None` for unknown coordinates."""
        return self.cells.get(pos)

    def get_marbles(self, player: int) -> List[Position]:
        """Return sorted coordinates for all marbles owned by `player`."""
        cells = self.cells
        return [pos for pos in ORDERED_VALID_POSITIONS if cells[pos] == player]

    def marble_count(self, player: int) -> int:
        """Return number of marbles currently on board for `player`."""
        return sum(1 for v in self.cells.values() if v == player)

    # --- Move validation ---

    def is_legal_move(self, move: Move, player: int) -> bool:
        """Validate move ownership, marble formation, and direction-specific constraints."""
        if not self._owns_marbles(move.marbles, player):
            return False
        if not self._marbles_in_line(move.marbles):
            return False
        return self._is_directionally_legal(move, player)

    def is_generated_move_legal(self, move: Move, player: int) -> bool:
        """Validate a generated move when ownership and formation are already guaranteed."""
        return self._is_directionally_legal(move, player)

    def is_generated_move_legal_raw(
        self,
        marbles: Tuple[Position, ...],
        direction: Direction,
        player: int,
    ) -> bool:
        """Validate a generated move candidate without constructing a `Move` first."""
        direction_index = DIRECTION_INDEX[direction]
        count = len(marbles)
        if count <= 1:
            return self._check_single_raw(marbles[0], direction_index)

        line_dir = (
            marbles[1][0] - marbles[0][0],
            marbles[1][1] - marbles[0][1],
        )
        if direction == line_dir or direction == OPPOSITE_DIRECTION.get(line_dir, opposite_dir(line_dir)):
            leading = marbles[-1] if direction == line_dir else marbles[0]
            opponent = WHITE if player == BLACK else BLACK
            return self._check_inline_raw(leading, count, direction_index, player, opponent)

        return self._check_broadside_raw(marbles, direction_index)

    def _owns_marbles(self, marbles: Tuple[Position, ...], player: int) -> bool:
        """Return whether all selected coordinates belong to the given player."""
        cells = self.cells
        for marble in marbles:
            if cells.get(marble) != player:
                return False
        return True

    def _marbles_in_line(self, marbles: Tuple[Position, ...]) -> bool:
        """Return whether selected marbles form a contiguous straight line."""
        if len(marbles) <= 1:
            return True
        start = marbles[0]
        d = (marbles[1][0] - start[0], marbles[1][1] - start[1])
        if d not in DIRECTION_SET:
            return False
        for index in range(2, len(marbles)):
            expected = (start[0] + index * d[0], start[1] + index * d[1])
            if marbles[index] != expected:
                return False
        return True

    def _is_directionally_legal(self, move: Move, player: int) -> bool:
        """Validate the occupancy and push rules once shape has been established."""
        opponent = WHITE if player == BLACK else BLACK
        if move.is_inline:
            return self._check_inline_raw(move._leading, move.count, DIRECTION_INDEX[move.direction], player, opponent)
        return self._check_broadside_raw(move.marbles, DIRECTION_INDEX[move.direction])

    def _check_single_raw(self, marble: Position, direction_index: int) -> bool:
        """Validate a generated single-marble move."""
        cells = self.cells
        ahead = NEIGHBOR_TABLE[marble][direction_index]
        return ahead is not None and cells[ahead] == EMPTY

    def _check_inline_raw(
        self,
        leading: Position,
        count: int,
        direction_index: int,
        player: int,
        opponent: int,
    ) -> bool:
        """Validate an inline move, including sumito push rules."""
        cells = self.cells
        ahead = NEIGHBOR_TABLE[leading][direction_index]

        if ahead is None:
            return False  # can't move own marble off board

        ahead_value = cells[ahead]
        if ahead_value == EMPTY:
            return True

        if ahead_value == player:
            return False  # can't push own marble

        # Count contiguous opponent marbles ahead
        pushed_count = 0
        pos = ahead
        while pos is not None and cells[pos] == opponent:
            pushed_count += 1
            pos = NEIGHBOR_TABLE[pos][direction_index]

        # Must outnumber
        if pushed_count >= count:
            return False

        # Space after pushed marbles must be empty or off-board
        if pos is not None and cells[pos] != EMPTY:
            return False

        return True

    def _check_broadside_raw(self, marbles: Tuple[Position, ...], direction_index: int) -> bool:
        """Validate broadside movement where every destination must be empty and valid."""
        cells = self.cells
        neighbors = NEIGHBOR_TABLE
        for m in marbles:
            dest = neighbors[m][direction_index]
            if dest is None:
                return False
            if cells[dest] != EMPTY:
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

        cells = self.cells
        d = move.direction
        direction_index = DIRECTION_INDEX[d]
        if move.is_inline:
            # Find pushed opponent marbles
            pushed = []
            pos = NEIGHBOR_TABLE[move._leading][direction_index]
            while pos is not None and cells[pos] == opponent:
                pushed.append(pos)
                pos = NEIGHBOR_TABLE[pos][direction_index]

            # Handle push-off
            pushoff = pushed and pos is None
            if pushoff:
                self.score[player] += 1

            # Move pushed (farthest first)
            for i in range(len(pushed) - 1, -1, -1):
                p = pushed[i]
                old_cells.append((p, cells[p]))
                self._zhash_set(p, cells[p])  # remove old
                dest = NEIGHBOR_TABLE[p][direction_index]
                if dest is not None:
                    old_cells.append((dest, cells[dest]))
                    cells[dest] = opponent
                    self._zhash_set(dest, opponent)  # add new
                cells[p] = EMPTY

            # Move own marbles (leading first)
            for marble in reversed(move._ordered_marbles):
                old_cells.append((marble, cells[marble]))
                self._zhash_set(marble, cells[marble])  # remove old
                dest = NEIGHBOR_TABLE[marble][direction_index]
                old_cells.append((dest, cells[dest]))
                cells[dest] = player
                self._zhash_set(dest, player)  # add new
                cells[marble] = EMPTY
        else:
            # Broadside
            for m in move.marbles:
                old_cells.append((m, cells[m]))
                self._zhash_set(m, cells[m])  # remove old
                cells[m] = EMPTY
            for m in move.marbles:
                dest = NEIGHBOR_TABLE[m][direction_index]
                old_cells.append((dest, cells[dest]))
                cells[dest] = player
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
        cells = self.cells
        direction_index = DIRECTION_INDEX[move.direction]

        # Find pushed opponent marbles
        pushed = []
        pos = NEIGHBOR_TABLE[move._leading][direction_index]
        while pos is not None and cells[pos] == opponent:
            pushed.append(pos)
            pos = NEIGHBOR_TABLE[pos][direction_index]
        # pos = first cell after pushed chain (empty or off-board)

        result['pushed'] = [pos_to_str(p) for p in pushed]

        # Handle push-off
        if pushed and pos is None:
            result['pushoff'] = True
            self.score[player] += 1

        # Move pushed opponent marbles (farthest first)
        for i in range(len(pushed) - 1, -1, -1):
            p = pushed[i]
            self._zhash_set(p, cells[p])
            dest = NEIGHBOR_TABLE[p][direction_index]
            if dest is not None:
                self._zhash_set(dest, cells[dest])
                cells[dest] = opponent
                self._zhash_set(dest, opponent)
            cells[p] = EMPTY

        # Move own marbles (leading first to avoid overwriting)
        for marble in reversed(move._ordered_marbles):
            self._zhash_set(marble, cells[marble])
            dest = NEIGHBOR_TABLE[marble][direction_index]
            self._zhash_set(dest, cells[dest])
            cells[dest] = player
            self._zhash_set(dest, player)
            cells[marble] = EMPTY

    def _apply_broadside(self, move: Move, player: int):
        """Apply a broadside move after legality has already been established."""
        cells = self.cells
        direction_index = DIRECTION_INDEX[move.direction]
        # Clear old, set new (safe because broadside dests are always empty)
        for m in move.marbles:
            self._zhash_set(m, cells[m])
            cells[m] = EMPTY
        for m in move.marbles:
            dest = NEIGHBOR_TABLE[m][direction_index]
            cells[dest] = player
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
