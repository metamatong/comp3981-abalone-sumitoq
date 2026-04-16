# Abalone — Project Guide

## What Is This?

A fully playable Abalone board game (web UI + terminal) with AI opponents, a state-space generator, local AI benchmarking, and adaptive heuristic tuning. Python 3 with a compiled C native extension (`abalone._native`) for move generation, evaluation, and search.

## Quick Start

Build the native extension first, then run:

```bash
python3 setup.py build_ext --inplace   # Build native extension
python3 run.py                          # Web UI (opens browser)
python3 run.py cli                      # Terminal UI
python3 run.py state                    # State-space analysis
python3 run.py match                    # Benchmark two presets
python3 run.py duel                     # Single duel, gauntlet, or tuning
```

## Project Structure

```
run.py                  # Entry point — routes to web/cli/state/match/duel mode
abalone/
  _native_src/          # C source files for the abalone._native extension
  ai/                   # Native-backed search, heuristics, agent types
  eval/                 # Reusable gauntlet runners + adaptive tuning
  game/                 # Board, CLI, server, session, config, duel orchestration
  players/              # Preset registry + team-owned AI configs
    teams/              # Kyle / Abdullah / Cole / Jonah + tournament agents
  state_space.py        # Python wrapper over native legal move generation
  static/               # Web UI (HTML, CSS, JS)
tests/                  # Unit and integration tests
dist/                   # Packaged executable (SumitoQ.exe)
```

## Key Conventions

- **Coordinate system**: rows `a`–`i` (0–8), columns `1`–`9`. Position = `(row, col)` tuple.
- **Player constants**: `BLACK = 1`, `WHITE = 2`, `EMPTY = 0`.
- **Directions**: 6 hex directions as `(dr, dc)` tuples — E, W, NW, SE, NE, SW.
- **Move dataclass**: `Move(marbles, direction)` is frozen/immutable for hashability.
- **Native extension**: heavy computation (move generation, evaluation, search) runs in compiled C via `abalone._native`.
- **Type hints**: used throughout with `typing` module aliases (`Position`, `Direction`).

## API Endpoints (Web UI)

| Method | Path             | Description                                      |
|--------|------------------|--------------------------------------------------|
| GET    | `/api/state`     | Full game state as JSON                          |
| POST   | `/api/move`      | Apply a move `{marbles: ["e5"], direction: [0,1]}` |
| POST   | `/api/agent-move`| Apply one AI move for current AI-controlled turn |
| POST   | `/api/config`    | Set game config (mode, layout, timers, AI presets) |
| POST   | `/api/undo`      | Undo last move                                   |
| POST   | `/api/reset`     | New game                                         |
| POST   | `/api/pause`     | Toggle pause/resume                              |
| POST   | `/api/resign`    | Current player resigns                           |

## Common Tasks

- **Add a new API endpoint**: edit `abalone/game/server.py`.
- **Change game rules**: edit `abalone/game/board.py` — `is_legal_move()`, `apply_move()`.
- **Modify UI**: edit files in `abalone/static/`.
- **Change AI heuristics**: edit team files in `abalone/players/teams/<name>/heuristic.py`.
- **Change move generation**: edit native C source in `abalone/_native_src/movegen.c`.
- **Modify search**: edit `abalone/_native_src/search.c` or `abalone/ai/minimax.py`.

## Testing

```bash
python3 -m unittest discover -s tests    # Run all tests
python3 run.py state                     # Should show 44 legal moves from starting position
python3 run.py cli                       # Interactive play — type "help" for commands
python3 run.py                           # Web UI — click marbles and test moves
```

## Packaging

Build the executable with PyInstaller using `SumitoQ.spec`:

```bash
python3 setup.py build_ext --inplace
python3 -m PyInstaller --noconfirm --clean SumitoQ.spec
```

Produces `dist/SumitoQ.exe`. End users don't need Python or build tools.
