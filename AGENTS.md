# Abalone Project AGENTS Guide

## Mission
Build and maintain a complete Abalone project with four outcomes:

1. A working game implementation.
2. An HTML GUI to visualize and play it.
3. A local in-repo agent that can play using minimax.
4. A testing and evaluation framework that compares heuristic strategies.

## Non-Goals
- No external AI services or model APIs.
- No OpenAI/LLM-based move generation.
- No duplicated rule engines outside core board logic.

## Current Status
- Complete: `1) working game`, `2) HTML GUI`, `3) basic minimax agent integrated`.
- Complete integrations:
  - Modular runtime split into `abalone/game/` and `abalone/players/`.
  - Play modes available in CLI and Web: `Human vs Human`, `Human vs AI`, `AI vs AI`.
  - Web mode-selection modal and AI move API wiring.
  - State-space CLI supports both initial summaries and root+one-child depth-1 expansion.
  - Initial regression tests for minimax legality/determinism and mode turn-gating.
- Remaining: `4) full heuristic strategy benchmarking framework` and broader test coverage.

## Source of Truth
- Game rules and legality are centralized in `abalone/game/board.py`.
- `abalone/board.py` is a compatibility shim that re-exports from `abalone/game/board.py`.
- Move generation is centralized in `abalone/state_space.py`.
- Session/mode runtime state is centralized in `abalone/game/session.py`.
- AI/search code must consume these interfaces and must not re-implement board legality rules.

## Current Architecture
- `abalone/game/`
  - `board.py`: board representation, legality, move application, display.
  - `config.py`: mode/controller config (`hvh`, `hva`, `ava`) and normalization.
  - `session.py`: shared game runtime state and turn execution (human + AI).
  - `cli.py`: terminal game loop.
  - `main.py`: CLI entry point and args.
  - `server.py`: HTTP server and JSON API routes.
- `abalone/players/`
  - `agent.py`: public API (`choose_move`, `choose_move_with_info`).
  - `minimax.py`: deterministic minimax + alpha-beta + move ordering.
  - `heuristics.py`: current baseline heuristic presets.
  - `validator.py`: shared move payload + legality validation wrapper.
  - `types.py`: `AgentConfig`.
- `abalone/state_space.py`
  - Central legal move generation only (raw move generation removed).
- `abalone/static/`
  - `index.html`, `script.js`, `style.css` for Web UI.
  - Includes mode selection modal and AI turn controls.
- Compatibility shims
  - `abalone/board.py`, `abalone/main.py`, `abalone/server.py`.

## Implemented API Surface
- `GET /api/state`
- `POST /api/move`
- `POST /api/agent-move`
- `POST /api/config`
- `POST /api/undo`
- `POST /api/reset`
- `POST /api/pause`

## Implemented CLI State-Space Options
- `python run.py state`
  - Print legal state-space summaries for the initial board (both players).
- `python run.py state --state-depth-one --state-player black|white --state-child-index N`
  - Print root summary plus one selected depth-1 child node.
  - To inspect additional children, rerun with a different `N`.

## High-Level Plan (Remaining Work)

### Phase 3: Heuristic Strategies (Expand)
- Extend heuristics beyond current baseline:
  - group cohesion/connectivity,
  - edge risk / push-off danger,
  - richer center control features.
- Add weighted strategy registry so heuristic sets can be swapped by name/config.
- Add deterministic board fixtures for comparing heuristic behavior.

### Phase 4: Evaluation Framework
- Add `abalone/eval/` package:
  - `simulate.py`: AI-vs-AI match runner with fixed seeds/config.
  - `metrics.py`: win/loss/draw, score differential, nodes/time per move.
  - `strategies.py`: strategy definitions and parameter registry.
- Run repeated side-swapped matches and output JSON + human summary.

### Phase 5: Quality Gate
- Expand tests to cover:
  - board/rules invariants,
  - minimax edge cases and deterministic choices,
  - API behavior for all controller/mode combinations,
  - reproducible evaluation pipeline basics.
- Update docs for running experiments and interpreting metrics.

## Definition of Done
- Human can play against AI via CLI and web UI.
- AI only generates legal moves.
- At least two heuristic strategies can be benchmarked head-to-head.
- Evaluation runs are reproducible with fixed seeds.
- Test suite covers rules, search correctness, mode control, and evaluation pipeline basics.
