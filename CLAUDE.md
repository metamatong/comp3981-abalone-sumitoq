# Abalone — Project Guide

## What Is This?

A fully playable Abalone board game (web UI + terminal) with a state-space generator for AI research. Pure Python 3.8+ — zero external dependencies.

## Quick Start

```bash
python run.py          # Web UI at http://localhost:8080 (opens browser)
python run.py cli      # Terminal UI
python run.py state    # State-space analysis (prints legal move summary)
```

## Project Structure

```
run.py                  # Entry point — routes to web/cli/state mode
abalone/
  board.py              # Core: Board, Move, positions, validation, apply
  state_space.py        # Move generation: raw (840) and legal (deduplicated)
  game.py               # CLI game loop with undo, move parsing, text display
  server.py             # HTTP server + JSON API (global game state)
  main.py               # CLI/state-space entry point
  static/
    index.html          # Web UI — HTML structure and layout
    style.css           # Web UI — all CSS styles
    script.js           # Web UI — all JS logic (state, rendering, API calls)
```

## Architecture

All game logic lives in Python. The browser is a stateless display layer — it fetches JSON state from the server and sends user actions as API calls.

```
Browser (static/)              Python (abalone/)
  SVG board render  <── GET /api/state ──  board.py + state_space.py
  click to move     ── POST /api/move ──>  server.py  (global state)
  undo / reset      ── POST /api/undo|reset ──>
```

## Key Conventions

- **Coordinate system**: rows `a`–`i` (0–8), columns `1`–`9`. Position = `(row, col)` tuple.
- **Player constants**: `BLACK = 1`, `WHITE = 2`, `EMPTY = 0`.
- **Directions**: 6 hex directions as `(dr, dc)` tuples — E, W, NW, SE, NE, SW.
- **Move dataclass**: `Move(marbles, direction)` is frozen/immutable for hashability.
- **Type hints**: used throughout with `typing` module aliases (`Position`, `Direction`).
- **No external deps**: only Python stdlib (`http.server`, `json`, `dataclasses`, `typing`).

## API Endpoints

| Method | Path         | Body                                          | Returns                    |
|--------|--------------|-----------------------------------------------|----------------------------|
| GET    | `/api/state` | —                                             | Full game state as JSON    |
| POST   | `/api/move`  | `{"marbles": ["e5"], "direction": [0, 1]}`    | `{"ok": true, "result": …}` |
| POST   | `/api/undo`  | —                                             | `{"ok": true}`             |
| POST   | `/api/reset` | —                                             | `{"ok": true}`             |

## Common Tasks

- **Add a new API endpoint**: edit `server.py` — add route in `do_GET`/`do_POST`, implement handler function.
- **Change game rules**: edit `board.py` — validation in `is_legal_move()`, execution in `apply_move()`.
- **Modify UI**: edit files in `abalone/static/` — styles in `style.css`, logic in `script.js`, layout in `index.html`.
- **Change move generation**: edit `state_space.py` — `generate_legal_moves()` for deduplication/validation.

## Testing

No formal test suite. Verify manually:

```bash
python run.py state    # Should show 44 legal moves from starting position
python run.py cli      # Interactive play — type "help" for commands
python run.py          # Web UI — click marbles and test moves
```
