# Abalone Project AGENTS Guide

## Mission
Build a complete Abalone project with four outcomes:

1. A working game implementation.
2. An HTML GUI to visualize and play it.
3. An agent that can play using minimax.
4. A testing framework that compares heuristic evaluation strategies.

## Non-Goals
- No external AI services or model APIs.
- No OpenAI/LLM-based move generation.
- The agent must be implemented locally in this codebase using search + heuristics.

## Current Status
- Complete: `1) working game`, `2) HTML GUI`.
- Remaining: `3) minimax agent`, `4) heuristic evaluation framework`.

## Source of Truth
- Game rules and legality must remain centralized in `abalone/board.py`.
- Move generation should remain centralized in `abalone/state_space.py`.
- Any AI/search code must consume these interfaces rather than re-implement rules.

## Target Architecture
- `abalone/ai/`
  - `agent.py`: public agent interface (`choose_move(board, player, config)`).
  - `minimax.py`: minimax + alpha-beta pruning.
  - `heuristics.py`: pluggable heuristic functions and weighted evaluators.
  - `ordering.py` (optional): move ordering helpers for search performance.
- `abalone/eval/`
  - `simulate.py`: run AI-vs-AI matches with fixed seeds/config.
  - `metrics.py`: win rate, average score margin, average decision time.
  - `strategies.py`: strategy registry (heuristic sets + search parameters).
- `tests/`
  - Unit tests for rules invariants and heuristic features.
  - Integration tests for deterministic agent behavior on fixed positions.

## High-Level Plan

### Phase 1: AI Foundation
- Define a stable agent interface and config object (depth, time budget, heuristic name).
- Add board utility support needed for search (fast copy and/or apply+undo workflow).
- Create deterministic fixtures for a small set of known board states.

### Phase 2: Minimax Agent
- Implement minimax with alpha-beta pruning.
- Add terminal-state detection (win/loss conditions) and depth cutoffs.
- Add basic move ordering (push moves, captures, center-improving moves first).
- Add a simple CLI hook to request an AI move from the current game state.

### Phase 3: Heuristic Strategies
- Implement baseline heuristics:
  - material advantage (score / marbles pushed off),
  - center control,
  - group cohesion / connectivity,
  - mobility (legal move count),
  - edge risk (marbles near push-off danger).
- Support weighted combinations so strategies can be swapped without code changes.

### Phase 4: Evaluation Framework
- Build match runner for round-robin strategy comparisons.
- Run repeated games per pairing with side swapping and fixed seeds.
- Collect and report metrics:
  - win/loss/draw rate,
  - average score differential,
  - average nodes/time per move.
- Output machine-readable results (JSON) and a concise human summary.

### Phase 5: Integration + Quality Gate
- Add a local HTTP route for agent turns (example: `POST /api/agent-move`) that calls in-repo minimax code.
- Add UI control to trigger AI move and show chosen move notation.
- Add regression tests for:
  - rules correctness,
  - minimax legality,
  - deterministic evaluation runs.
- Update docs with how to run games, AI, and evaluation experiments.

## Definition of Done
- Human can play against AI via CLI and web UI.
- AI only generates legal moves.
- At least two heuristic strategies can be benchmarked head-to-head.
- Evaluation runs are reproducible with fixed seeds.
- Test suite covers rules, search correctness, and evaluation pipeline basics.
