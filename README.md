# Abalone

A fully playable [Abalone](https://en.wikipedia.org/wiki/Abalone_(board_game)) board game with a web UI and a state-space generator for building AI players.

## Quick Start

```bash
# Web UI (opens browser automatically)
python run.py

# Terminal UI
python run.py cli

# Terminal with AI vs AI
python run.py cli --mode ava --depth 2

# Terminal with Human vs AI (human as white)
python run.py cli --mode hva --human-side white --depth 2

# State-space analysis
python run.py state

# State-space: root + exactly one depth-1 child (selected by legal move index)
python run.py state --state-depth-one --state-player black --state-child-index 0
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

1. On launch, choose mode from the modal (`Human vs Human`, `Human (Black) vs AI`, `AI vs AI`).
2. Click 1–3 of your marbles to select them (blue glow).
3. Valid destinations light up as blue dots.
4. Click a destination to execute the move.
5. Use **AI Move** to manually trigger one AI turn, or let AI turns auto-play in AI-controlled turns.
6. Use **Change Mode** to reopen the mode modal.

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

For building AI / search algorithms. This module returns only legal, deduplicated moves.

### CLI State-Space Modes

- `python run.py state`
  - Prints legal state-space summaries for both players on the initial board.
- `python run.py state --state-depth-one --state-player black --state-child-index N`
  - Prints exactly two nodes:
    - root state summary for the selected root player,
    - one depth-1 child after applying legal move index `N`.
  - Re-run with different `N` to inspect different children.

### Usage

```python
from abalone.board import Board, Move, BLACK, WHITE
from abalone.state_space import generate_legal_moves

board = Board()
board.setup_standard()

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
  render board    ◄── JSON ──  game/board.py    Board, Move, validation, apply
  click events    ── POST ──►  game/server.py   HTTP API, session + mode config
                               state_space.py   legal move generation
                               game/cli.py      CLI game loop
                               game/main.py     CLI entry point
                               players/*        minimax bot + heuristics + validator
```

All game logic runs in Python. The browser is a pure display layer — it renders the JSON state and sends clicks back as API calls.

```
abalone/
  __init__.py
  board.py         # compatibility shim -> game/board.py
  game/
    board.py       # Board, Move, positions, validation, apply, display
    cli.py         # CLI game loop with text UI and AI turns
    main.py        # CLI entry point
    server.py      # HTTP server + JSON API
    session.py     # shared runtime state for CLI/web
    config.py      # mode/controller configuration
  players/
    agent.py       # choose_move(...) public interface
    minimax.py     # minimax + alpha-beta search
    heuristics.py  # pluggable board evaluation
    validator.py   # shared move payload + legality validation
    types.py       # AgentConfig
  state_space.py   # generate_legal_moves()
  server.py        # compatibility shim -> game/server.py
  main.py          # compatibility shim -> game/main.py
  static/
    index.html     # Web UI — HTML structure and layout
    style.css      # Web UI — all CSS styles
    script.js      # Web UI — all JS logic (state, rendering, API calls)
run.py             # Runner: web / cli / state
```

## API Endpoints (Web UI)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/state` | Current board, score, legal moves, history |
| POST | `/api/move` | Apply a move `{marbles: ["e5"], direction: [0,1]}` |
| POST | `/api/agent-move` | Apply one AI move for the current AI-controlled turn |
| POST | `/api/config` | Set mode/config `{mode, human_side, ai_depth}` |
| POST | `/api/undo` | Undo last move |
| POST | `/api/reset` | New game |
| POST | `/api/pause` | Toggle pause/resume |
