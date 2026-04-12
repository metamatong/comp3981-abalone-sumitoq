# Abalone

A fully playable [Abalone](https://en.wikipedia.org/wiki/Abalone_(board_game)) board game with a web UI, a state-space generator, local AI benchmarking, and adaptive heuristic tuning.

## Native Setup

This is the native-only branch. Move generation, heuristic evaluation, and minimax search now run through the compiled `abalone._native` extension, so you must build the extension before running `run.py`.

Use the same Python interpreter for both install/build steps and runtime. On some Windows machines that may be `python`, `py -3`, or a full interpreter path instead of `python3`.

### Windows

Install Visual Studio Build Tools with the C++ workload:

```powershell
winget install --id Microsoft.VisualStudio.2022.BuildTools --exact --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools"
```

Then make sure your Python build packages are installed:

```powershell
python3 -m ensurepip --upgrade
python3 -m pip install --upgrade pip setuptools wheel
```

Then open a new terminal in this repo and build the extension:

```powershell
python3 setup.py build_ext --inplace
```

If the compiler is still not found, open `Developer PowerShell for VS 2022` and retry the build there.

### macOS

Install Apple's command-line developer tools:

```bash
xcode-select --install
```

Make sure Python build packages are installed:

```bash
python3 -m ensurepip --upgrade
python3 -m pip install --upgrade pip setuptools wheel
```

Then build the extension:

```bash
python3 setup.py build_ext --inplace
```

## Quick Start

After the native extension is built:

```bash
# Canonical web entrypoint
python3 run.py

# Web UI on a custom port
python3 run.py web 9090

# Terminal UI
python3 run.py cli

# Terminal with AI vs AI using selected presets
python3 run.py cli --mode ava --depth 2 --black-ai default --white-ai kyle

# Terminal with Human vs AI (human as white)
python3 run.py cli --mode hva --human-side white --depth 2 --black-ai default --white-ai jonah

# Local benchmark rounds (colors swap each round)
python3 run.py match --black-ai default --white-ai cole --rounds 3 --depth 2 --seed 7

# Single AI-vs-AI simulation report
python3 run.py duel --black-ai default --white-ai cole --depth 2

# One-agent gauntlet (two games vs every other agent, one per color)
# Uses one worker per CPU by default; add --jobs 1 to force serial execution.
# A live stderr progress line animates while games are running.
python3 run.py duel --agent jonah --all-opponents --depth 2

# One-agent gauntlet with an explicit worker count
python3 run.py duel --agent jonah --all-opponents --depth 2 --jobs 4

# Adaptive heuristic tuning: repeated one-vs-all gauntlets with checkpointed weight updates
python3 run.py duel --agent jonah --all-opponents --tune --iterations 20 --depth 2 --jobs 4

# Resume a prior tuning run from its checkpoint
python3 run.py duel --all-opponents --tune --resume-from abalone/eval_runs/jonah/<config>/<timestamp>/checkpoint.json --iterations 40

# State-space analysis
python3 run.py state

# State-space: root + exactly one depth-1 child (selected by legal move index)
python3 run.py state --state-depth-one --state-player black --state-child-index 0

# State-space: expand one custom input file and save all next states
python3 run.py state --state-input-file Test1.input --state-output-file Test1.board

# State-space: verify generated child states against expected output
python3 run.py state --state-input-file abalone/state_space_inputs/Test1.input --state-verify

# Test suite
python3 -m unittest discover -s tests
```

The web server automatically tries nearby ports if the requested one is busy.

This branch requires Python 3 plus a local C toolchain on the machine that builds the extension:
- Windows: MSVC Build Tools
- macOS: Xcode Command Line Tools

If the native extension is missing, the CLI now fails fast with a short preflight message instead of crashing inside worker processes.

## Create Executable

Build the native extension first, then build the executable with the matching Python interpreter. The PyInstaller build must use the same Python version that produced `abalone._native` or the packaged app will fail its native preflight at startup.

Recommended Windows build flow:

```powershell
# 1. Build the required native extension in-place.
py -3.13 setup.py build_ext --inplace

# 2. Package from the checked-in spec so PyInstaller includes:
#    - abalone/static
#    - the compiled abalone._native extension (.pyd)
py -3.13 -m PyInstaller --noconfirm --clean SumitoQ.spec
```

This produces `dist\SumitoQ.exe`.

End users running the finished `.exe` do not need Python, `setuptools`, or MSVC Build Tools installed.

### Packaging Notes

- `pip install PyInstaller` and `py -3.13 -m PyInstaller ...` must target the same Python installation. If PyInstaller is installed under Python 3.12 but you run `py -3.13 -m PyInstaller`, the build will fail with `No module named PyInstaller`.
- `SumitoQ.spec` is the source of truth for packaging. It explicitly includes the native extension module `abalone._native` and the static web assets.
- If you switch Python versions, rebuild the native extension before rebuilding the executable.
- For easier debugging, you can temporarily build a folder-based package instead:

```powershell
py -3.13 -m PyInstaller --noconfirm --clean --onedir SumitoQ.spec
```

### Troubleshooting Executable Startup

If the packaged app exits with:

```text
Native engine not built for this branch.
```

then the bundled executable is missing `abalone._native` or was built against a different Python version than the native extension.

Check these first:

- Re-run `py -3.13 setup.py build_ext --inplace` before packaging.
- Rebuild from `SumitoQ.spec` instead of packaging `run.py` directly.
- Inspect `build\SumitoQ\warn-SumitoQ.txt`. `abalone._native` should not appear there as a missing module.

## CLI Usage

The terminal entrypoint is:

```bash
python3 run.py cli
```

### AI Preset IDs

Use these preset IDs with `--black-ai` and `--white-ai`:

| Preset ID | Owner | Style |
|-----------|-------|-------|
| `default` | Shared | Shared baseline heuristic from `abalone/ai/heuristics.py` |
| `kyle` | Kyle | Member-owned preset from `abalone/players/teams/kyle/` |
| `abdullah` | Abdullah | Member-owned preset from `abalone/players/teams/abdullah/` |
| `cole` | Cole | Member-owned preset from `abalone/players/teams/cole/` |
| `jonah` | Jonah | Member-owned preset from `abalone/players/teams/jonah/` |
| `tournament-standard` | Tournament | Layout-tuned preset from `abalone/players/teams/standard/` |
| `tournament-belgian` | Tournament | Layout-tuned preset from `abalone/players/teams/belgian_daisy/` |
| `tournament-german` | Tournament | Layout-tuned preset from `abalone/players/teams/german_daisy/` |

### Where To Customize AI

- Shared default AI: edit `abalone/ai/heuristics.py`
- Kyle AI: edit `abalone/players/teams/kyle/heuristic.py`
- Abdullah AI: edit `abalone/players/teams/abdullah/heuristic.py`
- Cole AI: edit `abalone/players/teams/cole/heuristic.py`
- Jonah AI: edit `abalone/players/teams/jonah/heuristic.py`
- Each member preset currently starts with the same copied baseline weights, but those files are independent and can diverge without affecting `default`
- Adaptive tuning runs do not rewrite these source files; mutable weights are checkpointed under `abalone/eval_runs/`

### Enabling Quiescence Search

Quiescence search is configured per preset through `AgentDefinition.max_quiescence_depth` in each team's `agents.py`.

- `0` disables quiescence search. This is the default.
- `2` is the recommended starting point for normal play.
- `3` is reasonable for experiments if move times are still acceptable.
- Higher values are mainly useful for analysis runs and can become expensive.

Example:

```python
AGENTS = [
    AgentDefinition(
        id="kyle",
        label="Kyle",
        owner="Kyle",
        evaluator=evaluate_kyle,
        default_depth=4,
        tie_break="lexicographic",
        max_quiescence_depth=2,
    )
]
```

If you call the search API directly, you can also override this at runtime with `AgentConfig(max_quiescence_depth=...)`. Setting `AgentConfig(max_quiescence_depth=0)` forces quiescence off even if the preset enables it.

### Common CLI Modes

```bash
# Human vs Human
python3 run.py cli --mode hvh

# Human vs AI, human is black, white uses Jonah's preset
python3 run.py cli --mode hva --human-side black --depth 2 --white-ai jonah

# Human vs AI, human is white, black uses baseline
python3 run.py cli --mode hva --human-side white --depth 2 --black-ai default

# AI vs AI, black and white use different presets
python3 run.py cli --mode ava --depth 2 --black-ai kyle --white-ai cole
```

### CLI Flags That Matter For AI

| Flag | Meaning |
|------|---------|
| `--mode` | `hvh`, `hva`, or `ava` |
| `--human-side` | Human color in `hva` mode |
| `--depth` | Shared search-depth override for AI players |
| `--black-ai` | Preset ID for black when black is AI-controlled |
| `--white-ai` | Preset ID for white when white is AI-controlled |

### Search Behavior

- Black's first AI move is a random legal move, as required for the project.
- After the opening move, all AI turns use the native-backed search path exposed through `abalone/ai/minimax.py`.
- Native root search defaults to up to `4` worker threads when `ABALONE_NATIVE_ROOT_THREADS` is unset, and it scales down on lower-core hosts.
- Set `ABALONE_NATIVE_ROOT_THREADS` to override that per-move native thread count for local benchmarking or tuning.
- If the resolved native worker count is below `2`, the engine automatically falls back to the serial native reference path.
- The search uses iterative deepening and respects the remaining per-turn budget.
- When `max_quiescence_depth > 0`, leaf nodes can extend through a bounded quiescence search over tactical push/contact moves.
- Requests that include root candidates stay on the serial native path, so they are not suitable for raw threading benchmarks.
- `--jobs` only controls match/gauntlet process parallelism and does not change the per-move native root thread count.
- If a search runs out of time before completing a depth, it returns the best fully completed result so far.
- If no depth finishes in time, it falls back to an immediate legal move.

## How to Play

Abalone is a two-player strategy game on a hexagonal board. Each player has 14 marbles. Push 6 of your opponent's marbles off the board to win.

### Rules

- **Move** 1, 2, or 3 of your marbles per turn (must be in a line).
- **Inline move**: marbles move along their line direction. Can push opponent marbles.
- **Broadside move**: marbles move perpendicular to their line. No pushing allowed.
- **Sumito (push)**: you must outnumber the opponent's inline marbles to push them (2v1, 3v1, 3v2).
- **Win condition**: first player to push 6 opponent marbles off the board.

### Game-Over Conditions

A game can end in four ways:

| Condition | Trigger | Winner |
|-----------|---------|--------|
| Score | A player pushes 6 opponent marbles off | That player |
| Timeout | Combined game clock runs out | Player with more captures (or draw) |
| Max moves | Move limit reached (default 500) | Player with more captures (or draw) |
| Resign | A player resigns | Opponent |

### Web UI Controls

1. On launch, choose mode from the modal (`Human vs Human`, `Human vs AI`, `AI vs AI`).
2. In AI-controlled modes, select the `Black AI` and/or `White AI` preset by color.
3. In `Human vs AI`, choose the human color; only the AI-controlled side selector is shown.
4. Click 1â€“3 of your marbles to select them (blue glow).
5. Valid destinations light up as blue dots.
6. Click a destination to execute the move.
7. AI turns auto-play in AI-controlled turns.
8. Use **New Game** to reopen the game config modal.

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
| Broadside | `{n}:{end1}-{end2}>{DIR}` | `3:c3-c5>NW` | 3 stones c3â€“c5 all move NW |
| Push | `*` suffix | `3:e3e6*` | Move pushed an opponent marble |

- **trailing** = back marble (before move)
- **goal** = where the front marble ends up (after move)

## Board Layouts

Three starting layouts are available, selectable via CLI or `/api/config`:

| Layout | Description |
|--------|-------------|
| `standard` | Classic Abalone starting position (default) |
| `belgian_daisy` | Belgian Daisy formation |
| `german_daisy` | German Daisy formation |

## Timers & Clocks

The game tracks two timing systems, both configurable via `/api/config`:

- **Game clock** (`game_time_ms`, default 30 min) â€” total shared time split equally between players. When the combined time runs out, the game ends.
- **Per-turn time limit** (`player1_time_per_turn_s`, `player2_time_per_turn_s`, default 30s each) â€” if a player exceeds their turn limit, their turn is automatically skipped.
- **Move duration** â€” each move in the history records `duration_ms`, the wall-clock time the player took.

In CLI and benchmark usage:

- `player1_time_per_turn_s` maps to Black
- `player2_time_per_turn_s` maps to White
- The AI reserves a small safety buffer before the hard turn limit when searching

## Benchmark Mode

Use `python3 run.py match` to compare two presets locally. One round plays two games with colors swapped.

```bash
python3 run.py match \
  --black-ai default \
  --white-ai cole \
  --rounds 3 \
  --depth 2 \
  --move-time-s 5 \
  --max-moves 50 \
  --seed 7
```

### Benchmark Flags

| Flag | Meaning |
|------|---------|
| `--black-ai` | Preset assigned to black in game 1 of each round |
| `--white-ai` | Preset assigned to white in game 1 of each round |
| `--rounds` | Number of side-swapped rounds |
| `--depth` | Shared search-depth override |
| `--layout` | `standard`, `belgian_daisy`, or `german_daisy` |
| `--move-time-s` | Per-turn time limit for both colors |
| `--max-moves` | Move cap before draw/tiebreak |
| `--seed` | Base seed for reproducible random black openings |

The summary prints:

- wins, losses, and draws
- captures
- partial search count (`timeout_partial` / fallback outcomes)
- average move time
- average completed depth

## Duel And Adaptive Tuning

Use `python3 run.py duel` for single-game reports, one-vs-all gauntlets, and long-running heuristic tuning.

### Single Duel

```bash
python3 run.py duel --black-ai default --white-ai cole --depth 2
```

This runs one AI-vs-AI game and prints the winner, score, move count, time usage, and the active heuristic weights for both sides.

### One-Vs-All Gauntlet

```bash
python3 run.py duel --agent jonah --all-opponents --depth 2 --jobs 4
```

- The target agent plays every other registered agent twice: once as black and once as white.
- `--jobs` defaults to the CPU count; use `--jobs 1` to force serial execution.
- A live stderr progress line is shown while games are running.
- Results are sorted back into schedule order before reporting, even when workers finish out of order.

### Adaptive Tuning Mode

```bash
python3 run.py duel \
  --agent jonah \
  --all-opponents \
  --tune \
  --iterations 20 \
  --depth 2 \
  --layout standard \
  --move-time-s 5 \
  --max-moves 80 \
  --seed 7 \
  --jobs 4
```

One tuning iteration does the following:

1. Runs the full side-swapped gauntlet for the target agent against every other registered agent.
2. Analyzes each completed match using the tuned agent's heuristic telemetry and search diagnostics.
3. Aggregates likely reasons for losses and weak draws, including edge exposure, fragmentation, mobility collapse, weak push threats, poor center control, material loss, and search instability.
4. Adjusts the target agent's heuristic weights and starts the next iteration.

During a tuning run, the console output is intentionally minimal:

- a live progress line for the current gauntlet, labeled as `Iteration X/Y`
- one completed-iteration result line such as `Iteration 5/20: W=3 D=0 L=7 capture_diff=-5 partial_searches=0 rejected.`
- a final summary block when the run finishes

Checkpoint files are still written throughout the run, but those save-path messages are not printed during normal execution.

### Tuning Flags

| Flag | Meaning |
|------|---------|
| `--agent` | Tuned preset ID for fresh tuning runs |
| `--all-opponents` | Required for tuning and gauntlet mode |
| `--tune` | Enable adaptive heuristic tuning |
| `--iterations` | Number of tuning iterations to run |
| `--resume-from` | Resume from an existing `checkpoint.json` |
| `--depth` | Shared search-depth override; omit to use each preset's default depth |
| `--layout` | `standard`, `belgian_daisy`, or `german_daisy` |
| `--move-time-s` | Per-turn time limit for both colors |
| `--max-moves` | Move cap before draw/tiebreak |
| `--seed` | Base seed for reproducible random black openings |
| `--jobs` | Worker process count for gauntlet/tuning mode |

### Tuning Artifacts

Each tuning run writes artifacts under:

```text
abalone/eval_runs/<agent>/<config-slug>/<timestamp>/
```

| File | Purpose |
|------|---------|
| `checkpoint.json` | Source of truth for baseline, current, and best weights; config; progress; latest analyses; and artifact paths |
| `matches.jsonl` | One JSON line per completed game, including telemetry summaries and inferred reasons |
| `iterations.jsonl` | One JSON line per completed tuning iteration, including the score tuple, accepted/rejected status, and weight deltas |

Notes:

- Runtime tuning uses generated checkpoint files, not the original preset source files.
- If you interrupt the process, the current weights are flushed to `checkpoint.json` before the command exits.
- Resume with `--resume-from <checkpoint.json>` to continue from the saved state.
- `current_weights` are the candidate weights being evaluated or prepared for the next iteration.
- `best_weights` are the strongest weights seen so far under the optimizer's full score ordering; they are not always the most recent weights.

### Final Tuning Summary

When tuning completes, the CLI prints a summary in this shape:

```text
Final summary:
Time elapsed: 10m42s
Iterations completed: 10
Layout: belgian_daisy
Agent: tournament-belgian
Max moves: 80

Average win rate 60.00%
Best win rate 80.00% (Iterations, 2, 4, 7)

Best heuristic:
marble 50000.000->50284.038 (+284.038, x1.006)
center 85.000->117.256 (+32.256, x1.379)
cohesion 70.000->94.304 (+24.304, x1.347)
cluster 25.000->31.095 (+6.095, x1.244)
edge_pressure 121.000->129.261 (+8.261, x1.068)
formation 88.000->110.470 (+22.470, x1.255)
push 330.000->436.821 (+106.821, x1.324)
mobility 38.000->49.448 (+11.448, x1.301)
stability 52.000->64.274 (+12.274, x1.236)
```

Interpretation:

- `Average win rate` is the mean win rate across all completed tuning iterations.
- `Best win rate` lists every iteration that matched the peak win rate.
- `Best heuristic` is the single best-so-far weight profile stored in `best_weights`, shown as a diff from the baseline heuristic.
- If several iterations share the same peak win rate, the printed heuristic still comes from the one that won the optimizer's full tie-break ordering, not from an average of those iterations.

## State-Space Generator

For building AI / search algorithms. This module returns only legal, deduplicated moves.

### Generating Output Files from an Input File

Given a `.input` file describing a board position, you can generate **two** output files:

| File | Extension | Contents |
|------|-----------|----------|
| Board states | `.board` | One line per legal move â€” the resulting board state in compact token format |
| Move notations | `.move` | One line per legal move â€” the move notation (e.g. `1:c5c6`, `3:d5g8`) |

Both files have the same number of lines in the same order â€” line *N* of the `.move` file is the move that produces line *N* of the `.board` file.

**To generate both files, run:**

```bash
python3 run.py state --state-input-file abalone/state_space_inputs/Test1.input --state-output-file abalone/state_space_outputs/Test1.board
```

This produces:
- `abalone/state_space_outputs/Test1.board` â€” resulting board states
- `abalone/state_space_outputs/Test1.move` â€” move notations

**To generate from Test2:**

```bash
python3 run.py state --state-input-file abalone/state_space_inputs/Test2.input --state-output-file abalone/state_space_outputs/Test2.board
```

You can also point at any custom input/output paths:

```bash
python3 run.py state --state-input-file my_board.input --state-output-file my_board.board
```

The `.move` file is always created alongside the `.board` file automatically (same directory, same stem, `.move` extension).

### Input File Format

Each `.input` file has two parts:

1. **Line 1**: `b` or `w` â€” which player moves next
2. **Line 2+**: comma-separated marble tokens like `C5b,D5b,E7w,...`
   - Each token is `<Col><Row><color>` where color is `b` (black) or `w` (white)

Example (`Test1.input`):

```
b
C5b,D5b,E4b,E5b,E6b,F5b,F6b,F7b,F8b,G6b,H6b,C3w,C4w,D3w,D4w,D6w,E7w,F4w,G5w,G7w,G8w,G9w,H7w,H8w,H9w
```

### Output File Formats

**`.board` file** â€” each line is a resulting board state:
```
C6b,D5b,E4b,E5b,E6b,F5b,F6b,F7b,F8b,G6b,H6b,C3w,C4w,D3w,D4w,D6w,E7w,F4w,G5w,G7w,G8w,G9w,H7w,H8w,H9w
B5b,D5b,E4b,E5b,E6b,F5b,F6b,F7b,F8b,G6b,H6b,C3w,C4w,D3w,D4w,D6w,E7w,F4w,G5w,G7w,G8w,G9w,H7w,H8w,H9w
...
```

**`.move` file** â€” each line is a move notation:
```
1:c5c6
1:c5b5
1:c5b4
3:d5g8
...
```

### Other State-Space CLI Modes

```bash
# Print legal move summary for both players on the standard starting board
python3 run.py state

# Show root summary + one depth-1 child (by legal move index)
python3 run.py state --state-depth-one --state-player black --state-child-index 0

# Verify generated board states against an expected .board file
python3 run.py state --state-input-file abalone/state_space_inputs/Test1.input --state-verify
```

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

```text
Browser (HTML/JS)              Python (abalone/)
  render board    <- JSON ->   game/board.py    Board, Move, validation, apply
  click events    -> POST ->   game/server.py   HTTP API, session + mode config
                               game/duel.py     duel, gauntlet, and tuning CLI orchestration
                               eval/*           reusable gauntlet runners + adaptive tuning
                               state_space.py   Python wrapper over native legal move generation
                               game/cli.py      CLI game loop
                               game/main.py     CLI entry point
                               ai/*             native-backed search + heuristic wrappers + types
                               players/*        preset registry + team-owned AI configs
```

Gameplay/session orchestration stays in Python. The browser is still a pure display layer, but the heavy backend work now lives in the compiled native extension.

```text
abalone/
  __init__.py
  board.py         # compatibility shim -> game/board.py
  _native_src/     # native extension sources for the required abalone._native module
    common.h       # shared native constants, structs, and function declarations
    tables.c       # lookup tables, hashes, and core move/board helpers
    movegen.c      # legal move generation and move ordering
    eval.c         # weighted heuristic evaluation
    search.c       # apply-move, TT, iterative deepening, and alpha-beta search
    module.c       # Python extension binding layer
  ai/
    agent.py       # shared opening rule + choose_move(...) interface
    minimax.py     # Python adapter over the native search engine
    heuristics.py  # shared heuristic metadata + native-backed weighted evaluators
    defaults.py    # shared default AI preset
    types.py       # AgentDefinition + AgentConfig
  eval/
    __init__.py
    gauntlet.py    # reusable gauntlet execution, telemetry analysis, checkpoints, tuning loop
  game/
    board.py       # Board, Move, positions, validation, apply, display
    cli.py         # CLI game loop with text UI and AI turns
    duel.py        # single duel, one-vs-all gauntlet, and adaptive tuning CLI
    main.py        # CLI entry point
    server.py      # HTTP server + JSON API
    session.py     # shared runtime state, timers, clocks, runtime weight overrides, telemetry
    config.py      # mode/layout/timer configuration
  players/
    registry.py    # selectable AI preset registry + runtime override helpers
    teams/         # Kyle / Abdullah / Cole / Jonah agent packages
      kyle/        # Kyle-owned agents.py + heuristic.py
      abdullah/    # Abdullah-owned agents.py + heuristic.py
      cole/        # Cole-owned agents.py + heuristic.py
      jonah/       # Jonah-owned agents.py + heuristic.py
    validator.py   # shared move payload + legality validation
    agent.py       # compatibility shim -> ai/agent.py
    minimax.py     # compatibility shim -> ai/minimax.py
    heuristics.py  # compatibility shim -> ai/heuristics.py
    types.py       # compatibility shim -> ai/types.py
  state_space.py   # Python wrapper over native generate_legal_moves()
  server.py        # compatibility shim -> game/server.py
  main.py          # compatibility shim -> game/main.py
  static/
    index.html     # Web UI - HTML structure and layout
    style.css      # Web UI - all CSS styles
    script.js      # Web UI - all JS logic (state, rendering, API calls)
tests/
  test_duel.py
  test_duel_tuning.py
  test_last_move_indicator.py
  test_players.py
  test_session_modes.py
run.py             # Runner: web / cli / state / match / duel
```

The extension is still built as one compiled module, `abalone._native`, but its C implementation is split across multiple source files for maintainability without changing the Python API.

## API Endpoints (Web UI)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/state` | Current board, score, timers, legal moves, history, available AI presets |
| POST | `/api/move` | Apply a move `{marbles: ["e5"], direction: [0,1]}` |
| POST | `/api/agent-move` | Apply one AI move for the current AI-controlled turn |
| POST | `/api/config` | Set config (see fields below) |
| POST | `/api/undo` | Undo last move |
| POST | `/api/reset` | New game (resets board, timers, history) |
| POST | `/api/pause` | Toggle pause/resume |
| POST | `/api/resign` | Current player resigns; opponent wins |

### `/api/config` Fields

All fields are optional â€” send only the ones you want to change.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | string | `"hvh"` | `hvh` (human-human), `hva` (human-AI), `ava` (AI-AI) |
| `human_side` | string/int | `"black"` | Human color in `hva` mode |
| `ai_depth` | int | `2` | Minimax search depth (1â€“5) |
| `black_ai_id` | string | `"default"` | Selected AI preset for black |
| `white_ai_id` | string | `"default"` | Selected AI preset for white |
| `board_layout` | string | `"standard"` | `standard`, `belgian_daisy`, or `german_daisy` |
| `game_time_ms` | int | `1800000` | Total shared game clock in ms (0 = unlimited) |
| `max_moves` | int | `500` | Move limit before game ends (0 = unlimited) |
| `player1_time_per_turn_s` | int | `30` | Black's per-turn time limit in seconds |
| `player2_time_per_turn_s` | int | `30` | White's per-turn time limit in seconds |

Fixed `--seed` values make the random black opening reproducible across benchmark runs.

