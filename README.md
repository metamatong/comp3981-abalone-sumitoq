# Abalone

A fully playable [Abalone](https://en.wikipedia.org/wiki/Abalone_(board_game)) board game with a web UI and a state-space generator for building AI players.

## Quick Start

```bash
# Web UI (opens browser automatically)
python run.py

# Terminal UI
python run.py cli

# State-space analysis
python run.py state
```

No external dependencies — runs on Python 3.8+ standard library only.

## How to Play

Abalone is a two-player strategy game on a hexagonal board. Each player has 14 marbles. Push 6 of your opponent's marbles off the board to win.

### Rules

- **Move** 1, 2, or 3 of your marbles per turn (must be in a line).
- **Inline move**: marbles move along their line direction. Can push opponent marbles.
- **Broadside move**: marbles move perpendicular to their line. No pushing allowed.
- **Sumito (push)**: you must outnumber the opponent's inline marbles to push them (2v1, 3v1, 3v2).
- **Win condition**: first player to push 6 opponent marbles off the board.

### Web UI Controls

1. Click 1–3 of your marbles to select them (blue glow).
2. Valid destinations light up as blue dots.
3. Click a destination to execute the move.
4. **Undo** / **Clear** / **New Game** buttons below the board.

## Coordinate System

```
         i5 i6 i7 i8 i9           I
        h4 h5 h6 h7 h8 h9         H
       g3 g4 g5 g6 g7 g8 g9       G
      f2 f3 f4 f5 f6 f7 f8 f9     F
     e1 e2 e3 e4 e5 e6 e7 e8 e9   E
      d1 d2 d3 d4 d5 d6 d7 d8     D
       c1 c2 c3 c4 c5 c6 c7       C
        b1 b2 b3 b4 b5 b6         B
         a1 a2 a3 a4 a5           A
```

- **Rows**: `a` (bottom) to `i` (top)
- **Columns**: `1` (left) to `9` (right)
- **6 directions**: E, W, NE, NW, SE, SW

## Move Notation

| Type | Format | Example | Meaning |
|------|--------|---------|---------|
| Inline | `{n}:{trailing}{goal}` | `3:h7e7` | 3 stones at h7,g7,f7 move toward e7 |
| Single | `1:{from}{to}` | `1:e5e6` | Stone at e5 moves to e6 |
| Broadside | `{n}:{end1}-{end2}>{DIR}` | `3:c3-c5>NW` | 3 stones c3–c5 all move NW |
| Push | `*` suffix | `3:e3e6*` | Move pushed an opponent marble |

- **trailing** = back marble (before move)
- **goal** = where the front marble ends up (after move)

## State-Space Generator

For building AI / search algorithms. Every turn has a theoretical maximum of **840 raw moves**:

```
14 marbles × (6 single + 6×6 double + 3×6 triple) = 14 × 60 = 840
```

### Usage

```python
from abalone.board import Board, Move, BLACK, WHITE
from abalone.state_space import generate_legal_moves, generate_raw_moves

board = Board()
board.setup_standard()

# All 840 raw moves (no legality, includes duplicates)
raw = generate_raw_moves(board, BLACK)
len(raw)  # 840

# All legal moves for current position (deduplicated, validated)
legal = generate_legal_moves(board, BLACK)
len(legal)  # 44 from starting position

# Each Move object
for move in legal:
    move.marbles     # tuple of (row, col) positions
    move.direction   # (dr, dc)
    move.count       # 1, 2, or 3
    move.is_inline   # True/False
    move.to_notation()  # "3:h7e7"

# Apply a move
result = board.apply_move(move, BLACK)
# result = {'pushed': ['e6'], 'pushoff': False}

# Undo via copy
snapshot = board.copy()
board.apply_move(move, BLACK)
board = snapshot  # restore
```

### Starting Position (44 legal moves per side)

| Category | Count |
|----------|-------|
| Single inline | 14 |
| Double inline | 12 |
| Double broadside | 8 |
| Triple inline | 8 |
| Triple broadside | 2 |
| **Total** | **44** |

## Architecture

```
Browser (HTML/JS)              Python (abalone/)
  render board    ◄── JSON ──  board.py      Board, Move, validation, apply
  click events    ── POST ──►  server.py     HTTP API, game state
                               state_space.py  move generation (raw + legal)
                               game.py       CLI game loop
                               main.py       entry point
```

All game logic runs in Python. The browser is a pure display layer — it renders the JSON state and sends clicks back as API calls.

```
abalone/
  __init__.py
  board.py         # Board, Move, positions, validation, apply, display
  state_space.py   # generate_raw_moves(), generate_legal_moves()
  game.py          # CLI game loop with text UI
  server.py        # HTTP server + JSON API
  main.py          # CLI entry point
  static/
    index.html     # Web UI (SVG hex board, JS interaction)
run.py             # Runner: web / cli / state
```

## API Endpoints (Web UI)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/state` | Current board, score, legal moves, history |
| POST | `/api/move` | Apply a move `{marbles: ["e5"], direction: [0,1]}` |
| POST | `/api/undo` | Undo last move |
| POST | `/api/reset` | New game |
