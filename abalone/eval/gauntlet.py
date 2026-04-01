"""Reusable AI-vs-AI game, gauntlet, and adaptive tuning runners."""

import json
import os
import random
import re
import time
import inspect
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..ai.heuristics import (
    FEATURE_ORDER,
    WEIGHT_TUNING_RULES,
    evaluate_breakdown,
    normalize_weights,
    weights_from_multipliers,
    weights_to_multipliers,
)
from ..ai.agent import choose_move_with_info
from ..ai.types import AgentConfig
from ..game.board import BLACK, WHITE, Board
from ..game.config import GameConfig
from ..game.session import GameSession
from ..players.registry import build_runtime_agent, get_agent_weights, list_agents
from ..state_space import generate_legal_moves

CHECKPOINT_VERSION = 2
RUNS_ROOT = Path("abalone") / "eval_runs"
PARTIAL_SEARCH_SOURCES = frozenset({"timeout_partial", "timeout_fallback", "timeout_fallback_partial"})
SEARCH_INSTABILITY_SOURCES = PARTIAL_SEARCH_SOURCES | frozenset({"repeat_avoidance"})
OUTCOME_WEIGHTS = {"loss": 3.0, "weak_draw": 1.0, "draw": 0.0, "win": 0.0}
PHASE_STEP_SCALES = {"exploration": 1.0, "refinement": 0.45, "recovery": 1.35}
PHASE_JITTER_SCALES = {"exploration": 0.35, "refinement": 0.12, "recovery": 0.50}
TRAINING_SEED_PAIRS = 2
VALIDATION_SEED_PAIRS = 1
MAX_LOSS_CRITICAL_TURNS = 3
MAX_WEAK_DRAW_CRITICAL_TURNS = 2
MAX_ANALYSIS_ROOT_CANDIDATES = 3
ANALYSIS_DEPTH_BONUS = 1
ANALYSIS_TIME_BUDGET_MS = 1500
ANALYSIS_MISSED_WIN_THRESHOLD = 250.0
WEAK_DRAW_LATE_MATERIAL_THRESHOLD = -0.75
WEAK_DRAW_INSTABILITY_THRESHOLD = 2
REASON_TO_WEIGHT_DIRECTIONS = {
    "edge_exposure": {"edge_pressure": 1.0, "stability": 0.8, "center": 0.4},
    "fragmentation": {"cohesion": 1.0, "cluster": 0.8, "stability": 0.7, "mobility": 0.2},
    "mobility_collapse": {"mobility": 1.0, "center": 0.4, "cluster": -0.2},
    "weak_push_threat": {"push": 1.0, "formation": 0.9, "mobility": 0.25},
    "poor_center_control": {"center": 1.0, "mobility": 0.3, "edge_pressure": 0.2},
    "material_loss": {"marble": 0.6, "edge_pressure": 0.5, "stability": 0.5, "center": 0.15},
    "search_instability": {"stability": 0.4, "marble": 0.2, "push": -0.25, "formation": -0.15},
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _round_float(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _sanitize_component(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "default"


def _config_slug(config: dict) -> str:
    return "-".join(
        [
            f"layout-{_sanitize_component(config['layout'])}",
            f"depth-{config['depth'] if config['depth'] is not None else 'preset'}",
            f"move-{config['move_time_s']}",
            f"max-{config['max_moves']}",
            f"seed-{config['seed']}",
            f"iter-{config['iterations']}",
        ]
    )


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temp_path.replace(path)


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")


def winner_agent_id(winner: Optional[int], black_ai_id: str, white_ai_id: str) -> Optional[str]:
    """Resolve the winning AI preset identifier from the winning color."""
    if winner == BLACK:
        return black_ai_id
    if winner == WHITE:
        return white_ai_id
    return None


def build_all_opponents_jobs(
    agent_id: str,
    depth: Optional[int],
    layout: str,
    move_time_s: int,
    max_moves: int,
    seed: Optional[int],
) -> List[dict]:
    """Build two color-swapped game jobs against every registered opponent."""
    opponents = [agent.id for agent in list_agents() if agent.id != agent_id]
    jobs = []
    game_offset = 0
    for opponent_id in opponents:
        for black_ai_id, white_ai_id in ((agent_id, opponent_id), (opponent_id, agent_id)):
            jobs.append(
                {
                    "index": len(jobs),
                    "black_ai_id": black_ai_id,
                    "white_ai_id": white_ai_id,
                    "depth": depth,
                    "layout": layout,
                    "move_time_s": move_time_s,
                    "max_moves": max_moves,
                    "opening_seed": seed + game_offset if seed is not None else None,
                    "agent_color": "black" if black_ai_id == agent_id else "white",
                }
            )
            game_offset += 1
    return jobs


def build_tuning_jobs(
    agent_id: str,
    depth: Optional[int],
    layout: str,
    move_time_s: int,
    max_moves: int,
    seed: int,
    opponents: Sequence[str],
    training_seed_pairs: int = TRAINING_SEED_PAIRS,
    validation_seed_pairs: int = VALIDATION_SEED_PAIRS,
) -> Dict[str, List[dict]]:
    """Build per-iteration train/validation schedules with repeated seed pairs."""
    schedules = {"train": [], "validation": []}
    index = 0
    seed_cursor = int(seed)
    for split, pair_count in (("train", training_seed_pairs), ("validation", validation_seed_pairs)):
        for pair_index in range(pair_count):
            for opponent_id in opponents:
                for black_ai_id, white_ai_id in ((agent_id, opponent_id), (opponent_id, agent_id)):
                    schedules[split].append(
                        {
                            "index": index,
                            "black_ai_id": black_ai_id,
                            "white_ai_id": white_ai_id,
                            "depth": depth,
                            "layout": layout,
                            "move_time_s": move_time_s,
                            "max_moves": max_moves,
                            "opening_seed": seed_cursor,
                            "agent_color": "black" if black_ai_id == agent_id else "white",
                            "schedule_split": split,
                            "pair_index": pair_index,
                        }
                    )
                    index += 1
                    seed_cursor += 1
    return schedules


def resolve_worker_count(requested_jobs: Optional[int], game_count: int) -> int:
    """Resolve the effective worker count for a gauntlet run."""
    if game_count <= 0:
        return 0
    requested = requested_jobs if requested_jobs is not None else (os.cpu_count() or 1)
    return max(1, min(requested, game_count))


def run_game_session(
    black_ai_id: str,
    white_ai_id: str,
    depth: Optional[int],
    layout: str,
    move_time_s: int,
    max_moves: int,
    opening_seed: Optional[int],
    agent_weight_overrides: Optional[Dict[str, Dict[str, float]]] = None,
    telemetry_agent_ids: Optional[Sequence[str]] = None,
) -> GameSession:
    """Run a full AI-vs-AI game and return the finished session."""
    config = GameConfig(
        mode="ava",
        ai_depth=depth,
        black_ai_id=black_ai_id,
        white_ai_id=white_ai_id,
        board_layout=layout,
        max_moves=max_moves,
        player1_time_per_turn_s=move_time_s,
        player2_time_per_turn_s=move_time_s,
    )
    session = GameSession(
        config=config,
        opening_seed=opening_seed,
        agent_weight_overrides=agent_weight_overrides,
        telemetry_agent_ids=set(telemetry_agent_ids or ()),
    )
    session.reset()

    while not session.status()["game_over"]:
        result = session.apply_agent_move()
        if "error" in result:
            raise RuntimeError(result["error"])

    return session


def serialize_history(session: GameSession) -> List[dict]:
    """Return a JSON-serializable move history for reporting and analysis."""
    history = []
    for entry in session.move_history:
        result = entry.get("result") or {}
        history.append(
            {
                "notation": entry["move"].to_notation(pushed=bool(result.get("pushed"))),
                "player": entry.get("player"),
                "source": entry.get("source"),
                "agent_id": entry.get("agent_id"),
                "agent_label": entry.get("agent_label"),
                "duration_ms": int(entry.get("duration_ms", 0) or 0),
                "pushoff": bool(result.get("pushoff")),
                "pushed": list(result.get("pushed", [])),
                "search": entry.get("search"),
            }
        )
    return history


def run_scheduled_game(
    job: dict,
    agent_weight_overrides: Optional[Dict[str, Dict[str, float]]] = None,
    telemetry_agent_ids: Optional[Sequence[str]] = None,
) -> dict:
    """Run one scheduled gauntlet job and return a serializable summary."""
    start_time = time.perf_counter()
    session = run_game_session(
        black_ai_id=job["black_ai_id"],
        white_ai_id=job["white_ai_id"],
        depth=job["depth"],
        layout=job["layout"],
        move_time_s=job["move_time_s"],
        max_moves=job["max_moves"],
        opening_seed=job["opening_seed"],
        agent_weight_overrides=agent_weight_overrides,
        telemetry_agent_ids=telemetry_agent_ids,
    )
    elapsed_s = time.perf_counter() - start_time
    status = session.status()
    return {
        "index": job["index"],
        "black_ai_id": job["black_ai_id"],
        "white_ai_id": job["white_ai_id"],
        "schedule_split": job.get("schedule_split", "train"),
        "winner_ai_id": winner_agent_id(status.get("winner"), job["black_ai_id"], job["white_ai_id"]),
        "winner_tiebreak": status.get("winner_tiebreak"),
        "game_over_reason": status.get("game_over_reason"),
        "time_used_ms": status.get("time_used_ms", {BLACK: 0, WHITE: 0}),
        "score": dict(session.board.score),
        "moves": len(session.move_history),
        "duration_s": elapsed_s,
        "agent_color": job.get("agent_color"),
        "history": serialize_history(session),
    }


def _shutdown_executor(executor: ProcessPoolExecutor) -> None:
    """Best-effort executor shutdown that cancels queued work when supported."""
    try:
        executor.shutdown(cancel_futures=True)
    except TypeError:
        executor.shutdown()


def _run_serial(
    jobs: Iterable[dict],
    total: int,
    agent_weight_overrides: Optional[Dict[str, Dict[str, float]]],
    telemetry_agent_ids: Optional[Sequence[str]],
    on_game_complete: Optional[Callable[[int, int, dict], None]],
    start_completed: int = 0,
) -> List[dict]:
    games = []
    completed = start_completed
    for job in jobs:
        game = run_scheduled_game(
            job,
            agent_weight_overrides=agent_weight_overrides,
            telemetry_agent_ids=telemetry_agent_ids,
        )
        games.append(game)
        completed += 1
        if on_game_complete is not None:
            on_game_complete(completed, total, game)
    return games


def run_scheduled_games(
    scheduled_games: List[dict],
    worker_count: int,
    agent_weight_overrides: Optional[Dict[str, Dict[str, float]]] = None,
    telemetry_agent_ids: Optional[Sequence[str]] = None,
    on_game_complete: Optional[Callable[[int, int, dict], None]] = None,
    on_warning: Optional[Callable[[str], None]] = None,
) -> List[dict]:
    """Run scheduled gauntlet jobs, using parallel workers when available."""
    if worker_count <= 1:
        return _run_serial(
            scheduled_games,
            len(scheduled_games),
            agent_weight_overrides,
            telemetry_agent_ids,
            on_game_complete,
        )

    games = []
    total = len(scheduled_games)
    job_iter = iter(scheduled_games)
    completed = 0

    try:
        executor = ProcessPoolExecutor(max_workers=worker_count)
    except (BrokenProcessPool, OSError, PermissionError) as exc:
        if on_warning is not None:
            on_warning(
                "Warning: parallel duel execution is unavailable "
                f"({exc.__class__.__name__}: {exc}). Continuing serially."
            )
        return _run_serial(
            scheduled_games,
            total,
            agent_weight_overrides,
            telemetry_agent_ids,
            on_game_complete,
        )

    futures = {}
    try:
        while len(futures) < min(worker_count, total):
            job = next(job_iter, None)
            if job is None:
                break
            future = executor.submit(run_scheduled_game, job, agent_weight_overrides, telemetry_agent_ids)
            futures[future] = job

        while futures:
            try:
                future = next(as_completed(list(futures.keys())))
            except KeyboardInterrupt:
                for pending in futures:
                    pending.cancel()
                _shutdown_executor(executor)
                raise

            job = futures.pop(future)
            try:
                game = future.result()
            except (BrokenProcessPool, OSError, PermissionError) as exc:
                if on_warning is not None:
                    on_warning(
                        "Warning: parallel duel execution is unavailable "
                        f"({exc.__class__.__name__}: {exc}). Continuing serially."
                    )
                remaining_jobs = [job]
                remaining_jobs.extend(futures[pending] for pending in futures)
                remaining_jobs.extend(list(job_iter))
                for pending in futures:
                    pending.cancel()
                _shutdown_executor(executor)
                games.extend(
                    _run_serial(
                        remaining_jobs,
                        total,
                        agent_weight_overrides,
                        telemetry_agent_ids,
                        on_game_complete,
                        start_completed=completed,
                    )
                )
                return sorted(games, key=lambda item: item["index"])

            games.append(game)
            completed += 1
            if on_game_complete is not None:
                on_game_complete(completed, total, game)

            next_job = next(job_iter, None)
            if next_job is not None:
                next_future = executor.submit(
                    run_scheduled_game,
                    next_job,
                    agent_weight_overrides,
                    telemetry_agent_ids,
                )
                futures[next_future] = next_job
    finally:
        _shutdown_executor(executor)

    return sorted(games, key=lambda item: item["index"])


def _opponent_for_game(game: dict, target_agent_id: str) -> str:
    if game["black_ai_id"] == target_agent_id:
        return game["white_ai_id"]
    return game["black_ai_id"]


def _outcome_for_game(game: dict, target_agent_id: str) -> str:
    winner = game.get("winner_ai_id")
    if winner == target_agent_id:
        return "win"
    if winner is None:
        return "draw"
    return "loss"


def _target_color(game: dict, target_agent_id: str) -> int:
    return BLACK if game["black_ai_id"] == target_agent_id else WHITE


def _phase_for_turn(turn_index: int, total_turns: int) -> str:
    if total_turns <= 0:
        return "midgame"
    ratio = float(turn_index + 1) / float(total_turns)
    if ratio <= 0.33:
        return "opening"
    if ratio <= 0.75:
        return "midgame"
    return "late"


def _feature_values(blob: Optional[dict]) -> Dict[str, float]:
    features = {}
    source = (blob or {}).get("features") or {}
    for key in FEATURE_ORDER:
        features[key] = float(source.get(key, 0.0) or 0.0)
    return features


def _feature_total(blob: Optional[dict]) -> float:
    return float((blob or {}).get("total", 0.0) or 0.0)


def _exact_legal_move_count(board_token: Optional[str], player: int) -> int:
    if not board_token:
        return 0
    board = Board.from_compact_token(board_token)
    return len(generate_legal_moves(board, player))


def _late_turns(turns: Sequence[dict]) -> Sequence[dict]:
    if not turns:
        return ()
    start_index = max(0, len(turns) - max(1, len(turns) // 3))
    return turns[start_index:]


def _teacher_root_analysis(
    *,
    board_token: Optional[str],
    player: int,
    agent_id: str,
    requested_depth: int,
    analysis_weights: Optional[Dict[str, float]],
    fallback_weights: Optional[Dict[str, float]],
) -> Optional[dict]:
    if not board_token:
        return None

    teacher_weights = analysis_weights or fallback_weights
    if not teacher_weights:
        return None

    board = Board.from_compact_token(board_token)
    teacher_agent = build_runtime_agent(agent_id, teacher_weights)
    analysis_evaluator_id = "analysis-fixed" if analysis_weights else "analysis-fallback-best"
    result = choose_move_with_info(
        board,
        player,
        agent=teacher_agent,
        config=AgentConfig(
            depth=max(int(requested_depth or 0) + ANALYSIS_DEPTH_BONUS, 2),
            time_budget_ms=ANALYSIS_TIME_BUDGET_MS,
            root_candidate_limit=MAX_ANALYSIS_ROOT_CANDIDATES,
            analysis_evaluator_id=analysis_evaluator_id,
            board_token_before=board_token,
        ),
    )
    teacher_move = result.move
    if teacher_move is None:
        return None

    teacher_board = Board.from_compact_token(board_token)
    teacher_board.apply_move(teacher_move, player)
    teacher_breakdown = evaluate_breakdown(teacher_board, player, teacher_weights)

    return {
        "analysis_evaluator_id": analysis_evaluator_id,
        "teacher_move": teacher_move.to_notation(),
        "teacher_score": _round_float(result.score, 6),
        "teacher_depth": int(result.completed_depth),
        "teacher_candidates": result.root_candidates or [],
        "teacher_post": {
            "features": {key: _round_float(teacher_breakdown["features"][key], 6) for key in FEATURE_ORDER},
            "total": _round_float(teacher_breakdown["total"], 6),
        },
    }


def _feature_delta_summary(counterfactuals: Sequence[dict]) -> Dict[str, float]:
    totals = {key: 0.0 for key in FEATURE_ORDER}
    for item in counterfactuals:
        for key, value in (item.get("feature_delta") or {}).items():
            totals[key] = totals.get(key, 0.0) + float(value or 0.0)
    return {
        key: _round_float(value, 6)
        for key, value in totals.items()
        if abs(value) > 1e-9
    }


def analyze_match_result(
    game: dict,
    target_agent_id: str,
    *,
    analysis_weights: Optional[Dict[str, float]] = None,
    fallback_analysis_weights: Optional[Dict[str, float]] = None,
) -> dict:
    """Analyze one game using phase-aware critical turns and offline teacher reruns."""
    target_color = _target_color(game, target_agent_id)
    opponent_color = WHITE if target_color == BLACK else BLACK
    opponent_id = _opponent_for_game(game, target_agent_id)
    outcome = _outcome_for_game(game, target_agent_id)
    target_score = int(game["score"][target_color])
    opponent_score = int(game["score"][opponent_color])
    capture_diff = target_score - opponent_score

    target_entries = [entry for entry in game.get("history", []) if entry.get("agent_id") == target_agent_id]
    decision_sources = []
    completed_depths = []
    requested_depths = []
    move_durations = []
    feature_samples = []
    turns = []
    phase_breakdown = {
        "opening": {"turns": 0, "criticality": 0.0},
        "midgame": {"turns": 0, "criticality": 0.0},
        "late": {"turns": 0, "criticality": 0.0},
    }

    total_target_turns = len(target_entries)
    for turn_index, entry in enumerate(target_entries):
        search = entry.get("search") or {}
        pre_move = search.get("pre_move") or {}
        post_move = search.get("post_move") or {}
        pre_features = _feature_values(pre_move)
        post_features = _feature_values(post_move)
        if pre_move.get("features"):
            feature_samples.append(pre_features)

        decision_source = search.get("decision_source") or "unknown"
        requested_depth = int(search.get("depth", 0) or 0)
        completed_depth = float(search.get("completed_depth", 0) or 0)
        move_duration = float(entry.get("duration_ms", 0) or 0)
        board_token_before = search.get("board_token_before")
        board_token_after = search.get("board_token_after")
        exact_before = _exact_legal_move_count(board_token_before, target_color)
        exact_after = _exact_legal_move_count(board_token_after, target_color)
        eval_drop = max(0.0, _feature_total(pre_move) - _feature_total(post_move))
        center_drop = max(0.0, pre_features["center"] - post_features["center"])
        edge_drop = max(0.0, pre_features["edge_pressure"] - post_features["edge_pressure"])
        cohesion_drop = max(0.0, pre_features["cohesion"] - post_features["cohesion"])
        cluster_drop = max(0.0, pre_features["cluster"] - post_features["cluster"])
        stability_drop = max(0.0, pre_features["stability"] - post_features["stability"])
        mobility_drop = max(0.0, float(exact_before - exact_after))
        phase = _phase_for_turn(turn_index, total_target_turns)
        phase_weight = {"opening": 1.0, "midgame": 1.2, "late": 1.45}[phase]
        search_penalty = 0.0
        if decision_source in PARTIAL_SEARCH_SOURCES:
            search_penalty += 2.0
        elif decision_source == "repeat_avoidance" and eval_drop > 0.0:
            search_penalty += 0.75
        criticality = phase_weight * (
            (eval_drop / 200.0)
            + max(0.0, -post_features["marble"]) * 1.5
            + (center_drop * 0.03)
            + (edge_drop * 0.04)
            + ((cohesion_drop + cluster_drop + stability_drop) * 0.02)
            + (mobility_drop * 0.18)
            + search_penalty
        )

        phase_breakdown[phase]["turns"] += 1
        phase_breakdown[phase]["criticality"] += criticality
        turns.append(
            {
                "turn_index": turn_index,
                "phase": phase,
                "decision_source": decision_source,
                "requested_depth": requested_depth,
                "completed_depth": completed_depth,
                "duration_ms": move_duration,
                "board_token_before": board_token_before,
                "board_token_after": board_token_after,
                "pre_features": pre_features,
                "post_features": post_features,
                "pre_total": _feature_total(pre_move),
                "post_total": _feature_total(post_move),
                "exact_legal_moves_before": exact_before,
                "exact_legal_moves_after": exact_after,
                "criticality": criticality,
                "notation": entry.get("notation") or search.get("notation"),
            }
        )
        decision_sources.append(decision_source)
        completed_depths.append(completed_depth)
        requested_depths.append(float(requested_depth))
        move_durations.append(move_duration)

    feature_averages = {
        key: _round_float(_mean([float(sample.get(key, 0.0)) for sample in feature_samples]), 4)
        for key in FEATURE_ORDER
    }
    partial_searches = sum(1 for source in decision_sources if source in PARTIAL_SEARCH_SOURCES)
    unstable_decisions = sum(
        1
        for turn in turns
        if turn["decision_source"] in PARTIAL_SEARCH_SOURCES
        or (turn["decision_source"] == "repeat_avoidance" and turn["criticality"] > 0.0)
    )
    avg_completed_depth = _round_float(_mean(completed_depths), 4)
    avg_requested_depth = _round_float(_mean(requested_depths), 4)
    avg_move_ms = _round_float(_mean(move_durations), 3)

    late_turns = list(_late_turns(turns))
    late_material_avg = _mean([float(turn["post_features"]["marble"]) for turn in late_turns]) if late_turns else 0.0
    provisional_weak_draw = (
        outcome == "draw"
        and (
            late_material_avg <= WEAK_DRAW_LATE_MATERIAL_THRESHOLD
            or unstable_decisions >= WEAK_DRAW_INSTABILITY_THRESHOLD
        )
    )

    critical_turn_limit = 0
    if outcome == "loss":
        critical_turn_limit = MAX_LOSS_CRITICAL_TURNS
    elif outcome == "draw":
        critical_turn_limit = MAX_WEAK_DRAW_CRITICAL_TURNS
    critical_turns = sorted(turns, key=lambda item: (-item["criticality"], -item["turn_index"]))[:critical_turn_limit]

    counterfactuals = []
    for turn in critical_turns:
        teacher = _teacher_root_analysis(
            board_token=turn.get("board_token_before"),
            player=target_color,
            agent_id=target_agent_id,
            requested_depth=turn["requested_depth"],
            analysis_weights=analysis_weights,
            fallback_weights=fallback_analysis_weights,
        )
        if teacher is None:
            continue
        teacher_features = teacher["teacher_post"]["features"]
        feature_delta = {
            key: _round_float(float(teacher_features.get(key, 0.0)) - float(turn["post_features"].get(key, 0.0)), 6)
            for key in FEATURE_ORDER
        }
        total_delta = _round_float(float(teacher["teacher_post"]["total"]) - float(turn["post_total"]), 6)
        counterfactuals.append(
            {
                "turn_index": turn["turn_index"],
                "phase": turn["phase"],
                "board_token_before": turn.get("board_token_before"),
                "chosen_move": turn.get("notation"),
                "chosen_score": _round_float(turn["post_total"], 6),
                "teacher_move": teacher["teacher_move"],
                "teacher_score": teacher["teacher_score"],
                "teacher_depth": teacher["teacher_depth"],
                "analysis_evaluator_id": teacher["analysis_evaluator_id"],
                "total_delta": total_delta,
                "feature_delta": {key: value for key, value in feature_delta.items() if abs(value) > 1e-9},
                "teacher_candidates": teacher["teacher_candidates"],
            }
        )

    missed_win = any(float(item.get("total_delta", 0.0)) >= ANALYSIS_MISSED_WIN_THRESHOLD for item in counterfactuals)
    weak_draw = bool(provisional_weak_draw or (outcome == "draw" and missed_win))
    tuning_outcome = "weak_draw" if weak_draw else outcome
    drives_tuning = outcome == "loss" or weak_draw

    reasons = []
    if drives_tuning:
        reason_map = {
            "edge_exposure": {
                "summary": "Critical turns increased edge exposure or reduced local support near the rim.",
                "score": 0.0,
                "evidence": {"avg_edge_pressure": feature_averages["edge_pressure"], "turns": []},
            },
            "fragmentation": {
                "summary": "Critical turns loosened group structure and reduced support.",
                "score": 0.0,
                "evidence": {
                    "avg_cohesion": feature_averages["cohesion"],
                    "avg_cluster": feature_averages["cluster"],
                    "avg_stability": feature_averages["stability"],
                    "turns": [],
                },
            },
            "mobility_collapse": {
                "summary": "Critical turns left the agent with fewer exact legal options or lower mobility.",
                "score": 0.0,
                "evidence": {"avg_mobility": feature_averages["mobility"], "turns": []},
            },
            "weak_push_threat": {
                "summary": "Critical turns failed to preserve pressure, formations, or pushing threats.",
                "score": 0.0,
                "evidence": {"avg_push": feature_averages["push"], "avg_formation": feature_averages["formation"], "turns": []},
            },
            "poor_center_control": {
                "summary": "Critical turns ceded too much center influence or positional balance.",
                "score": 0.0,
                "evidence": {"avg_center": feature_averages["center"], "turns": []},
            },
            "material_loss": {
                "summary": "The game drifted into a persistent material deficit.",
                "score": max(0.0, float(opponent_score - target_score)) + abs(min(feature_averages["marble"], 0.0)),
                "evidence": {
                    "target_score": target_score,
                    "opponent_score": opponent_score,
                    "capture_diff": capture_diff,
                    "avg_marble_feature": feature_averages["marble"],
                    "turns": [],
                },
            },
            "search_instability": {
                "summary": "Critical turns relied on partial or unstable search decisions.",
                "score": 0.0,
                "evidence": {
                    "partial_searches": partial_searches,
                    "unstable_decisions": unstable_decisions,
                    "avg_completed_depth": avg_completed_depth,
                    "avg_requested_depth": avg_requested_depth,
                    "avg_move_ms": avg_move_ms,
                    "turns": [],
                },
            },
        }

        counterfactual_by_turn = {item["turn_index"]: item for item in counterfactuals}
        for turn in critical_turns:
            turn_index = int(turn["turn_index"])
            post_features = turn["post_features"]
            if post_features["edge_pressure"] < -1.0:
                reason_map["edge_exposure"]["score"] += abs(post_features["edge_pressure"])
                reason_map["edge_exposure"]["evidence"]["turns"].append(turn_index)
            fragmentation_pressure = (
                abs(min(post_features["cohesion"], 0.0))
                + abs(min(post_features["cluster"], 0.0))
                + abs(min(post_features["stability"], 0.0))
            )
            if fragmentation_pressure > 0.0:
                reason_map["fragmentation"]["score"] += fragmentation_pressure
                reason_map["fragmentation"]["evidence"]["turns"].append(turn_index)
            mobility_pressure = max(0.0, float(turn["exact_legal_moves_before"] - turn["exact_legal_moves_after"]))
            mobility_pressure += abs(min(post_features["mobility"], 0.0))
            if mobility_pressure > 0.0:
                reason_map["mobility_collapse"]["score"] += mobility_pressure
                reason_map["mobility_collapse"]["evidence"]["turns"].append(turn_index)
            push_pressure = abs(min(post_features["push"], 0.0)) + abs(min(post_features["formation"], 0.0))
            if push_pressure > 0.0:
                reason_map["weak_push_threat"]["score"] += push_pressure
                reason_map["weak_push_threat"]["evidence"]["turns"].append(turn_index)
            if post_features["center"] < -2.0:
                reason_map["poor_center_control"]["score"] += abs(post_features["center"])
                reason_map["poor_center_control"]["evidence"]["turns"].append(turn_index)
            if turn["decision_source"] in PARTIAL_SEARCH_SOURCES or (
                turn["decision_source"] == "repeat_avoidance" and turn["criticality"] > 0.0
            ):
                reason_map["search_instability"]["score"] += 1.0 + max(
                    0.0,
                    (float(turn["requested_depth"]) * 0.6) - float(turn["completed_depth"]),
                )
                reason_map["search_instability"]["evidence"]["turns"].append(turn_index)

            counterfactual = counterfactual_by_turn.get(turn_index)
            if counterfactual:
                delta = counterfactual.get("feature_delta") or {}
                reason_map["edge_exposure"]["score"] += max(0.0, float(delta.get("edge_pressure", 0.0)))
                reason_map["fragmentation"]["score"] += max(0.0, float(delta.get("cohesion", 0.0))) + max(0.0, float(delta.get("cluster", 0.0))) + max(0.0, float(delta.get("stability", 0.0)))
                reason_map["mobility_collapse"]["score"] += max(0.0, float(delta.get("mobility", 0.0)))
                reason_map["weak_push_threat"]["score"] += max(0.0, float(delta.get("push", 0.0))) + max(0.0, float(delta.get("formation", 0.0)))
                reason_map["poor_center_control"]["score"] += max(0.0, float(delta.get("center", 0.0)))
                reason_map["material_loss"]["score"] += max(0.0, float(delta.get("marble", 0.0)))

        for code, payload in reason_map.items():
            if payload["score"] <= 0.0:
                continue
            payload["evidence"]["turns"] = sorted(set(payload["evidence"].get("turns", [])))
            reasons.append(
                {
                    "code": code,
                    "score": _round_float(payload["score"], 4),
                    "summary": payload["summary"],
                    "evidence": payload["evidence"],
                }
            )
        reasons.sort(key=lambda item: (-item["score"], item["code"]))

    for phase in phase_breakdown.values():
        phase["criticality"] = _round_float(phase["criticality"], 4)

    return {
        "analysis_version": CHECKPOINT_VERSION,
        "target_agent_id": target_agent_id,
        "opponent_id": opponent_id,
        "target_color": "black" if target_color == BLACK else "white",
        "outcome": outcome,
        "tuning_outcome": tuning_outcome,
        "weak_draw": weak_draw,
        "drives_tuning": drives_tuning,
        "target_score": target_score,
        "opponent_score": opponent_score,
        "capture_diff": capture_diff,
        "feature_averages": feature_averages,
        "feature_delta_summary": _feature_delta_summary(counterfactuals),
        "phase_breakdown": phase_breakdown,
        "critical_turns": [
            {
                "turn_index": turn["turn_index"],
                "phase": turn["phase"],
                "criticality": _round_float(turn["criticality"], 6),
                "decision_source": turn["decision_source"],
                "notation": turn.get("notation"),
                "exact_legal_moves_before": turn["exact_legal_moves_before"],
                "exact_legal_moves_after": turn["exact_legal_moves_after"],
                "board_token_before": turn.get("board_token_before"),
            }
            for turn in critical_turns
        ],
        "counterfactual_summary": counterfactuals,
        "search_summary": {
            "move_count": len(target_entries),
            "partial_searches": partial_searches,
            "unstable_decisions": unstable_decisions,
            "avg_completed_depth": avg_completed_depth,
            "avg_requested_depth": avg_requested_depth,
            "avg_move_ms": avg_move_ms,
            "late_material_avg": _round_float(late_material_avg, 4),
        },
        "reasons": reasons,
    }


def aggregate_iteration_analysis(match_records: Sequence[dict]) -> dict:
    """Aggregate per-game reasons into an iteration-level diagnosis."""
    aggregates = {}
    feature_adjustments = {key: 0.0 for key in FEATURE_ORDER}
    for record in match_records:
        analysis = record["analysis"]
        outcome_weight = OUTCOME_WEIGHTS.get(analysis.get("tuning_outcome") or analysis["outcome"], 0.0)
        if outcome_weight <= 0.0:
            continue
        for reason in analysis["reasons"]:
            bucket = aggregates.setdefault(
                reason["code"],
                {
                    "code": reason["code"],
                    "weighted_score": 0.0,
                    "occurrences": 0,
                    "losses": 0,
                    "draws": 0,
                    "opponents": set(),
                    "summary": reason["summary"],
                },
            )
            bucket["weighted_score"] += float(reason["score"]) * outcome_weight
            bucket["occurrences"] += 1
            bucket["losses"] += 1 if analysis["outcome"] == "loss" else 0
            bucket["draws"] += 1 if analysis.get("weak_draw") else 0
            bucket["opponents"].add(analysis["opponent_id"])
        for key, value in (analysis.get("feature_delta_summary") or {}).items():
            feature_adjustments[key] = feature_adjustments.get(key, 0.0) + (float(value or 0.0) * outcome_weight)

    top_reasons = []
    for bucket in aggregates.values():
        top_reasons.append(
            {
                "code": bucket["code"],
                "weighted_score": _round_float(bucket["weighted_score"], 4),
                "occurrences": bucket["occurrences"],
                "losses": bucket["losses"],
                "draws": bucket["draws"],
                "opponent_count": len(bucket["opponents"]),
                "summary": bucket["summary"],
            }
        )

    top_reasons.sort(
        key=lambda item: (
            -item["weighted_score"],
            -item["occurrences"],
            -item["opponent_count"],
            item["code"],
        )
    )
    return {
        "top_reasons": top_reasons,
        "feature_delta_summary": {
            key: _round_float(value, 6)
            for key, value in feature_adjustments.items()
            if abs(value) > 1e-9
        },
    }


def _score_subset(match_records: Sequence[dict], split: Optional[str] = None) -> dict:
    relevant = [
        record
        for record in match_records
        if split is None or record.get("schedule_split", "train") == split
    ]
    wins = sum(1 for record in relevant if record["analysis"]["outcome"] == "win")
    draws = sum(1 for record in relevant if record["analysis"]["outcome"] == "draw")
    losses = sum(1 for record in relevant if record["analysis"]["outcome"] == "loss")
    match_points = (wins * 2) + draws
    capture_diff = sum(int(record["analysis"]["capture_diff"]) for record in relevant)
    partial_searches = sum(int(record["analysis"]["search_summary"]["partial_searches"]) for record in relevant)

    total_target_moves = sum(int(record["analysis"]["search_summary"]["move_count"]) for record in relevant)
    total_depth = sum(
        float(record["analysis"]["search_summary"]["avg_completed_depth"]) * int(record["analysis"]["search_summary"]["move_count"])
        for record in relevant
    )
    total_move_ms = sum(
        float(record["analysis"]["search_summary"]["avg_move_ms"]) * int(record["analysis"]["search_summary"]["move_count"])
        for record in relevant
    )
    avg_completed_depth = _round_float(total_depth / total_target_moves, 4) if total_target_moves else 0.0
    avg_move_ms = _round_float(total_move_ms / total_target_moves, 3) if total_target_moves else 0.0
    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "match_points": match_points,
        "capture_diff": capture_diff,
        "partial_searches": partial_searches,
        "avg_completed_depth": avg_completed_depth,
        "avg_move_ms": avg_move_ms,
        "games": len(relevant),
    }


def _validation_tuple(score: Optional[dict]) -> Optional[Tuple[float, ...]]:
    if score is None:
        return None
    return (
        int(score["match_points"]),
        int(score["capture_diff"]),
        -int(score["partial_searches"]),
        float(score["avg_completed_depth"]),
        -float(score["avg_move_ms"]),
    )


def score_iteration(match_records: Sequence[dict]) -> dict:
    """Score one full tuning iteration with explicit train/validation splits."""
    training = _score_subset(match_records, "train")
    validation = _score_subset(match_records, "validation")
    overall = _score_subset(match_records, None)
    return {
        "training": training,
        "validation": validation,
        "overall": overall,
        "comparison_key": {
            "training_match_points": int(training["match_points"]),
            "validation": _validation_tuple(validation),
        },
    }


def is_better_score(candidate: dict, incumbent: Optional[dict]) -> bool:
    """Apply the explicit train/validation acceptance rule."""
    if incumbent is None:
        return True
    candidate_training = candidate["training"]
    incumbent_training = incumbent["training"]
    if int(candidate_training["match_points"]) <= int(incumbent_training["match_points"]):
        return False
    candidate_validation = _validation_tuple(candidate["validation"])
    incumbent_validation = _validation_tuple(incumbent["validation"])
    if candidate_validation is None:
        return True
    if incumbent_validation is None:
        return True
    return candidate_validation >= incumbent_validation


def determine_phase(iteration_index: int, total_iterations: int, stagnation_count: int) -> str:
    """Pick the current optimizer phase for the next proposal."""
    if stagnation_count >= 3:
        return "recovery"
    exploration_window = max(2, total_iterations // 3)
    if iteration_index <= exploration_window:
        return "exploration"
    return "refinement"


def propose_next_weights(
    baseline_weights: Dict[str, float],
    current_weights: Dict[str, float],
    best_weights: Dict[str, float],
    top_reasons: Sequence[dict],
    feature_delta_summary: Optional[Dict[str, float]],
    iteration_index: int,
    total_iterations: int,
    stagnation_count: int,
    seed: int,
) -> dict:
    """Generate the next heuristic candidate using a hybrid exploration/refinement loop."""
    phase = determine_phase(iteration_index, total_iterations, stagnation_count)
    current_multipliers = weights_to_multipliers(current_weights, baseline_weights)
    best_multipliers = weights_to_multipliers(best_weights, baseline_weights)
    base_multipliers = dict(best_multipliers if phase != "exploration" or stagnation_count else current_multipliers)
    proposed = dict(base_multipliers)
    phase_scale = PHASE_STEP_SCALES[phase]
    jitter_scale = PHASE_JITTER_SCALES[phase]
    evidence_applied = False

    for key, delta_value in (feature_delta_summary or {}).items():
        if key not in WEIGHT_TUNING_RULES:
            continue
        rules = WEIGHT_TUNING_RULES[key]
        strength = max(-1.5, min(1.5, float(delta_value) / 12.0))
        if abs(strength) <= 1e-9:
            continue
        proposed[key] *= 1.0 + (rules["step"] * phase_scale * 0.35 * strength)
        evidence_applied = True

    if not evidence_applied:
        for reason in top_reasons[:3]:
            directions = REASON_TO_WEIGHT_DIRECTIONS.get(reason["code"], {})
            strength = min(2.0, max(0.25, float(reason["weighted_score"]) / max(1.0, 4.0 * float(reason["occurrences"]))))
            for key, direction in directions.items():
                rules = WEIGHT_TUNING_RULES[key]
                delta = direction * rules["step"] * phase_scale * strength
                proposed[key] *= 1.0 + delta

    for index, key in enumerate(FEATURE_ORDER):
        rules = WEIGHT_TUNING_RULES[key]
        jitter_rng = random.Random((seed * 10007) + (iteration_index * 7919) + ((stagnation_count + 1) * 97) + (index * 53))
        jitter = (jitter_rng.random() - 0.5) * rules["step"] * jitter_scale
        proposed[key] *= 1.0 + jitter
        proposed[key] = max(rules["min_multiplier"], min(rules["max_multiplier"], proposed[key]))

    weights = weights_from_multipliers(baseline_weights, proposed)
    return {
        "phase": phase,
        "weights": {key: _round_float(weights[key], 6) for key in FEATURE_ORDER},
        "multipliers": {key: _round_float(proposed[key], 6) for key in FEATURE_ORDER},
    }


def _weight_delta(current: Dict[str, float], reference: Dict[str, float]) -> Dict[str, float]:
    return {
        key: _round_float(current[key] - reference[key], 6)
        for key in FEATURE_ORDER
        if abs(current[key] - reference[key]) > 1e-9
    }


def _default_tuning_runner(
    scheduled_games: List[dict],
    worker_count: int,
    agent_weight_overrides: Dict[str, Dict[str, float]],
    telemetry_agent_ids: Sequence[str],
    on_game_complete: Callable[[int, int, dict], None],
    on_warning: Callable[[str], None],
    iteration_index: Optional[int] = None,
    total_iterations: Optional[int] = None,
) -> List[dict]:
    del iteration_index, total_iterations
    return run_scheduled_games(
        scheduled_games,
        worker_count,
        agent_weight_overrides=agent_weight_overrides,
        telemetry_agent_ids=telemetry_agent_ids,
        on_game_complete=on_game_complete,
        on_warning=on_warning,
    )


def _initial_state(config: dict, baseline_weights: Dict[str, float], run_dir: Path) -> dict:
    checkpoint_path = run_dir / "checkpoint.json"
    matches_path = run_dir / "matches.jsonl"
    iterations_path = run_dir / "iterations.jsonl"
    critical_positions_path = run_dir / "critical_positions.jsonl"
    baseline_weights = normalize_weights(baseline_weights)
    baseline_multipliers = {key: 1.0 for key in FEATURE_ORDER}
    return {
        "version": CHECKPOINT_VERSION,
        "analysis_version": CHECKPOINT_VERSION,
        "status": "running",
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "config": config,
        "paths": {
            "run_dir": str(run_dir.resolve()),
            "checkpoint": str(checkpoint_path.resolve()),
            "matches": str(matches_path.resolve()),
            "iterations": str(iterations_path.resolve()),
            "critical_positions": str(critical_positions_path.resolve()),
        },
        "baseline_weights": {key: _round_float(baseline_weights[key], 6) for key in FEATURE_ORDER},
        "baseline_multipliers": baseline_multipliers,
        "analysis_weights": {key: _round_float(baseline_weights[key], 6) for key in FEATURE_ORDER},
        "current_weights": {key: _round_float(baseline_weights[key], 6) for key in FEATURE_ORDER},
        "current_multipliers": dict(baseline_multipliers),
        "best_weights": {key: _round_float(baseline_weights[key], 6) for key in FEATURE_ORDER},
        "best_multipliers": dict(baseline_multipliers),
        "best_score": None,
        "stagnation_count": 0,
        "latest_analyses": [],
        "progress": {
            "completed_iterations": 0,
            "current_iteration": None,
            "completed_matches_total": 0,
            "current_iteration_matches": 0,
            "matches_per_iteration": int(config["games_per_iteration"]),
        },
        "active_iteration": None,
        "iteration_summaries": [],
        "train_schedule": [],
        "validation_schedule": [],
        "fixed_position_suite_result": None,
        "timing": {
            "started_at": _utc_now(),
            "completed_at": None,
            "elapsed_s": None,
        },
        "final_summary": None,
    }


def save_checkpoint(state: dict) -> None:
    """Persist the checkpoint JSON atomically."""
    state["updated_at"] = _utc_now()
    checkpoint_path = Path(state["paths"]["checkpoint"])
    _atomic_write_json(checkpoint_path, state)


def load_checkpoint(path: str) -> dict:
    """Load a checkpoint from disk."""
    checkpoint_path = Path(path)
    with checkpoint_path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    if int(state.get("version", 1)) != CHECKPOINT_VERSION:
        raise ValueError(
            f"Checkpoint version {state.get('version', 1)} is not supported by this evaluation pipeline."
        )
    return state


def _build_run_dir(config: dict, runs_root: Optional[Path]) -> Path:
    root = runs_root or RUNS_ROOT
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return root / config["agent_id"] / _config_slug(config) / timestamp


def _build_match_log_record(iteration_index: int, game: dict, analysis: dict) -> dict:
    return {
        "analysis_version": analysis.get("analysis_version", CHECKPOINT_VERSION),
        "iteration": iteration_index,
        "game_index": game["index"],
        "schedule_split": game.get("schedule_split", "train"),
        "black_ai_id": game["black_ai_id"],
        "white_ai_id": game["white_ai_id"],
        "opponent_id": analysis["opponent_id"],
        "target_color": analysis["target_color"],
        "winner_ai_id": game.get("winner_ai_id"),
        "winner_tiebreak": game.get("winner_tiebreak"),
        "game_over_reason": game.get("game_over_reason"),
        "score": game["score"],
        "target_score": analysis["target_score"],
        "opponent_score": analysis["opponent_score"],
        "outcome": analysis["outcome"],
        "capture_diff": analysis["capture_diff"],
        "moves": game["moves"],
        "duration_s": _round_float(game["duration_s"], 4),
        "time_used_ms": game.get("time_used_ms", {BLACK: 0, WHITE: 0}),
        "feature_averages": analysis["feature_averages"],
        "feature_delta_summary": analysis.get("feature_delta_summary", {}),
        "phase_breakdown": analysis.get("phase_breakdown", {}),
        "critical_turns": analysis.get("critical_turns", []),
        "counterfactual_summary": analysis.get("counterfactual_summary", []),
        "telemetry_summary": analysis["search_summary"],
        "weak_draw": analysis.get("weak_draw", False),
        "tuning_outcome": analysis.get("tuning_outcome", analysis["outcome"]),
        "reasons": analysis["reasons"],
        "analysis": analysis,
    }


def _load_board_from_input(path: Path) -> Tuple[Board, int]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    player = BLACK if lines[0].lower().startswith("b") else WHITE
    board = Board()
    board.clear()
    if len(lines) > 1:
        for token in lines[1].split(","):
            token = token.strip()
            if not token:
                continue
            pos = token[:-1]
            color = token[-1].lower()
            board.cells[(ord(pos[0].lower()) - ord("a"), int(pos[1:]))] = BLACK if color == "b" else WHITE
    board.recompute_zhash()
    return board, player


def _run_fixed_position_suite(weights: Dict[str, float]) -> dict:
    cases = []
    suite_dir = Path("abalone") / "state_space_inputs"
    for case_name in ("Test1.input", "Test2.input", "Test3.input"):
        path = suite_dir / case_name
        if not path.exists():
            continue
        board, player = _load_board_from_input(path)
        breakdown = evaluate_breakdown(board, player, weights)
        cases.append(
            {
                "case": case_name,
                "player": "black" if player == BLACK else "white",
                "total": _round_float(breakdown["total"], 6),
                "legal_moves": len(generate_legal_moves(board, player)),
            }
        )
    return {
        "cases": cases,
        "average_total": _round_float(_mean([float(case["total"]) for case in cases]), 6) if cases else 0.0,
    }


def _parse_utc_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _elapsed_time_seconds(started_at: Optional[str], completed_at: Optional[str]) -> float:
    started = _parse_utc_timestamp(started_at)
    completed = _parse_utc_timestamp(completed_at)
    if started is None or completed is None:
        return 0.0
    return _round_float(max(0.0, (completed - started).total_seconds()), 3)


def _format_elapsed_time(seconds: float) -> str:
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    total_seconds = int(round(seconds))
    minutes, secs = divmod(total_seconds, 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{secs:02d}s"


def _prepare_resume_state(state: dict, iterations: int, jobs: Optional[int]) -> dict:
    if int(state.get("version", 1)) != CHECKPOINT_VERSION:
        raise ValueError(
            f"Resume is only supported for checkpoint version {CHECKPOINT_VERSION}; found {state.get('version', 1)}."
        )
    state["config"]["iterations"] = max(int(iterations), int(state["config"]["iterations"]))
    if jobs is not None:
        state["config"]["jobs"] = int(jobs)
    state.setdefault("analysis_version", CHECKPOINT_VERSION)
    state.setdefault("analysis_weights", dict(state.get("baseline_weights", {})))
    state.setdefault("train_schedule", [])
    state.setdefault("validation_schedule", [])
    state.setdefault("fixed_position_suite_result", None)
    state.setdefault("iteration_summaries", [])
    state.setdefault("progress", {})
    state["progress"].setdefault("matches_per_iteration", int(state["config"].get("games_per_iteration", 0)))
    state.setdefault(
        "timing",
        {
            "started_at": state.get("created_at"),
            "completed_at": None,
            "elapsed_s": None,
        },
    )
    state.setdefault("final_summary", None)
    active_iteration = state.get("active_iteration")
    if active_iteration and active_iteration.get("status") != "completed":
        state["current_weights"] = dict(active_iteration["candidate_weights"])
        state["current_multipliers"] = dict(active_iteration["candidate_multipliers"])
        state["progress"]["current_iteration"] = active_iteration["index"]
        state["progress"]["current_iteration_matches"] = 0
        active_iteration["status"] = "restarting"
    save_checkpoint(state)
    return state


def _iteration_win_rate(score: dict, games_per_iteration: int) -> float:
    if games_per_iteration <= 0:
        return 0.0
    return _round_float((float(score["wins"]) * 100.0) / float(games_per_iteration), 4)


def _weight_change_summary(baseline_weights: Dict[str, float], result_weights: Dict[str, float]) -> List[dict]:
    changes = []
    for key in FEATURE_ORDER:
        baseline = float(baseline_weights[key])
        result = float(result_weights[key])
        delta = result - baseline
        if abs(delta) <= 1e-9:
            continue
        multiplier = 1.0 if baseline == 0.0 else (result / baseline)
        changes.append(
            {
                "feature": key,
                "baseline": _round_float(baseline, 6),
                "result": _round_float(result, 6),
                "delta": _round_float(delta, 6),
                "multiplier": _round_float(multiplier, 6),
            }
        )
    return changes


def _format_weight_change_summary(changes: Sequence[dict]) -> str:
    if not changes:
        return "no weight changes"
    parts = []
    for change in changes:
        delta_label = f"{float(change['delta']):+.3f}"
        parts.append(
            f"{change['feature']} {float(change['baseline']):.3f}->{float(change['result']):.3f} "
            f"({delta_label}, x{float(change['multiplier']):.3f})"
        )
    return "; ".join(parts)


def _format_weight_change_lines(changes: Sequence[dict]) -> List[str]:
    if not changes:
        return ["no weight changes"]
    lines = []
    for change in changes:
        lines.append(
            f"{change['feature']} {float(change['baseline']):.3f}->{float(change['result']):.3f} "
            f"({float(change['delta']):+.3f}, x{float(change['multiplier']):.3f})"
        )
    return lines


def _build_final_summary(state: dict) -> dict:
    summaries = list(state.get("iteration_summaries", []))
    average_win_rate = _round_float(_mean([float(item.get("win_rate", 0.0)) for item in summaries]), 4) if summaries else 0.0
    if summaries:
        best_win_rate = _round_float(max(float(item.get("win_rate", 0.0)) for item in summaries), 4)
        best_iterations = [
            int(item["iteration"])
            for item in summaries
            if abs(float(item.get("win_rate", 0.0)) - best_win_rate) <= 1e-9
        ]
    else:
        best_win_rate = 0.0
        best_iterations = []
    timing = state.get("timing", {})
    completed_at = timing.get("completed_at") or _utc_now()
    elapsed_s = _elapsed_time_seconds(timing.get("started_at") or state.get("created_at"), completed_at)
    return {
        "average_win_rate": average_win_rate,
        "best_win_rate": best_win_rate,
        "best_win_rate_iteration": best_iterations[0] if best_iterations else None,
        "best_win_rate_iterations": best_iterations,
        "total_time_s": elapsed_s,
        "iterations_completed": int(state.get("progress", {}).get("completed_iterations", 0)),
        "agent_id": state["config"]["agent_id"],
        "layout": state["config"]["layout"],
        "max_moves": int(state["config"]["max_moves"]),
        "weight_changes": _weight_change_summary(state["baseline_weights"], state["best_weights"]),
        "fixed_position_suite_result": state.get("fixed_position_suite_result"),
    }
def _run_tuning_iteration_runner(
    runner: Callable,
    scheduled_games: List[dict],
    worker_count: int,
    agent_weight_overrides: Dict[str, Dict[str, float]],
    telemetry_agent_ids: Sequence[str],
    on_game_complete: Callable[[int, int, dict], None],
    on_warning: Callable[[str], None],
    iteration_index: int,
    total_iterations: int,
) -> List[dict]:
    try:
        signature = inspect.signature(runner)
    except (TypeError, ValueError):
        signature = None

    if signature is not None:
        parameters = signature.parameters
        if "iteration_index" in parameters or "total_iterations" in parameters:
            return runner(
                scheduled_games,
                worker_count,
                agent_weight_overrides,
                telemetry_agent_ids,
                on_game_complete,
                on_warning,
                iteration_index=iteration_index,
                total_iterations=total_iterations,
            )

    return runner(
        scheduled_games,
        worker_count,
        agent_weight_overrides,
        telemetry_agent_ids,
        on_game_complete,
        on_warning,
    )


def run_tuning_loop(
    *,
    agent_id: Optional[str],
    iterations: int,
    depth: Optional[int],
    layout: str,
    move_time_s: int,
    max_moves: int,
    seed: int,
    jobs: Optional[int],
    resume_from: Optional[str] = None,
    runs_root: Optional[Path] = None,
    run_gauntlet_iteration: Optional[Callable[[List[dict], int, Dict[str, Dict[str, float]], Sequence[str], Callable[[int, int, dict], None], Callable[[str], None]], List[dict]]] = None,
    announce: Optional[Callable[[str], None]] = None,
) -> Path:
    """Run the adaptive gauntlet tuning loop and return the checkpoint path."""
    announce = announce or print
    gauntlet_runner = run_gauntlet_iteration or _default_tuning_runner

    if resume_from:
        state = _prepare_resume_state(load_checkpoint(resume_from), iterations, jobs)
        checkpoint_path = Path(state["paths"]["checkpoint"])
    else:
        if not agent_id:
            raise ValueError("agent_id is required to start a fresh tuning run")
        baseline_weights = get_agent_weights(agent_id)
        opponents = [agent.id for agent in list_agents() if agent.id != agent_id]
        config = {
            "agent_id": agent_id,
            "depth": depth,
            "layout": layout,
            "move_time_s": move_time_s,
            "max_moves": max_moves,
            "seed": int(seed),
            "jobs": jobs,
            "iterations": int(iterations),
            "opponents": opponents,
            "training_seed_pairs": TRAINING_SEED_PAIRS,
            "validation_seed_pairs": VALIDATION_SEED_PAIRS,
            "games_per_iteration": len(opponents) * 2 * (TRAINING_SEED_PAIRS + VALIDATION_SEED_PAIRS),
        }
        run_dir = _build_run_dir(config, runs_root)
        state = _initial_state(config, baseline_weights, run_dir)
        save_checkpoint(state)
        checkpoint_path = Path(state["paths"]["checkpoint"])

    agent_id = state["config"]["agent_id"]
    total_iterations = int(state["config"]["iterations"])
    matches_path = Path(state["paths"]["matches"])
    iterations_path = Path(state["paths"]["iterations"])
    critical_positions_path = Path(state["paths"]["critical_positions"])

    next_iteration = int(state["progress"]["completed_iterations"]) + 1
    active_iteration = state.get("active_iteration")
    if active_iteration and active_iteration.get("status") != "completed":
        next_iteration = int(active_iteration["index"])

    while next_iteration <= total_iterations:
        schedules = build_tuning_jobs(
            agent_id=agent_id,
            depth=state["config"]["depth"],
            layout=state["config"]["layout"],
            move_time_s=state["config"]["move_time_s"],
            max_moves=state["config"]["max_moves"],
            seed=state["config"]["seed"] + ((next_iteration - 1) * int(state["config"]["games_per_iteration"])),
            opponents=state["config"]["opponents"],
            training_seed_pairs=int(state["config"].get("training_seed_pairs", TRAINING_SEED_PAIRS)),
            validation_seed_pairs=int(state["config"].get("validation_seed_pairs", VALIDATION_SEED_PAIRS)),
        )
        scheduled_games = list(schedules["train"]) + list(schedules["validation"])
        worker_count = resolve_worker_count(state["config"].get("jobs"), len(scheduled_games))
        candidate_weights = normalize_weights(state["current_weights"])
        candidate_multipliers = weights_to_multipliers(candidate_weights, state["baseline_weights"])
        best_weights_before = normalize_weights(state["best_weights"])
        best_score_before = state.get("best_score")
        fixed_position_suite_result = _run_fixed_position_suite(candidate_weights)

        state["status"] = "running"
        state["progress"]["current_iteration"] = next_iteration
        state["progress"]["current_iteration_matches"] = 0
        state["train_schedule"] = [
            {
                "index": job["index"],
                "opponent_id": _opponent_for_game(job, agent_id),
                "opening_seed": job["opening_seed"],
                "agent_color": job["agent_color"],
            }
            for job in schedules["train"]
        ]
        state["validation_schedule"] = [
            {
                "index": job["index"],
                "opponent_id": _opponent_for_game(job, agent_id),
                "opening_seed": job["opening_seed"],
                "agent_color": job["agent_color"],
            }
            for job in schedules["validation"]
        ]
        state["active_iteration"] = {
            "index": next_iteration,
            "status": "running",
            "phase": determine_phase(next_iteration, total_iterations, int(state.get("stagnation_count", 0))),
            "candidate_weights": {key: _round_float(candidate_weights[key], 6) for key in FEATURE_ORDER},
            "candidate_multipliers": {key: _round_float(candidate_multipliers[key], 6) for key in FEATURE_ORDER},
            "completed_matches": 0,
            "games_total": len(scheduled_games),
        }
        save_checkpoint(state)
        match_records = []

        def on_warning(message: str) -> None:
            announce(message)

        def on_game_complete(completed: int, total: int, game: dict) -> None:
            analysis = analyze_match_result(
                game,
                agent_id,
                analysis_weights=normalize_weights(state.get("analysis_weights", state["baseline_weights"])),
                fallback_analysis_weights=normalize_weights(state["best_weights"]),
            )
            record = _build_match_log_record(next_iteration, game, analysis)
            match_records.append(record)
            _append_jsonl(matches_path, record)
            for critical in analysis.get("counterfactual_summary", []):
                _append_jsonl(
                    critical_positions_path,
                    {
                        "analysis_version": CHECKPOINT_VERSION,
                        "iteration": next_iteration,
                        "game_index": game["index"],
                        "opponent_id": analysis["opponent_id"],
                        "target_color": analysis["target_color"],
                        **critical,
                    },
                )
            state["progress"]["completed_matches_total"] += 1
            state["progress"]["current_iteration_matches"] = completed
            state["latest_analyses"] = analysis["reasons"][:3]
            state["active_iteration"]["completed_matches"] = completed
            state["active_iteration"]["latest_game"] = {
                "game_index": game["index"],
                "opponent_id": analysis["opponent_id"],
                "outcome": analysis["outcome"],
                "capture_diff": analysis["capture_diff"],
            }
            save_checkpoint(state)

        try:
            _run_tuning_iteration_runner(
                gauntlet_runner,
                scheduled_games,
                worker_count,
                {agent_id: candidate_weights},
                [agent_id],
                on_game_complete,
                on_warning,
                next_iteration,
                total_iterations,
            )
        except KeyboardInterrupt:
            state["status"] = "interrupted"
            state["active_iteration"]["status"] = "interrupted"
            state["active_iteration"]["completed_matches"] = len(match_records)
            state["latest_analyses"] = aggregate_iteration_analysis(match_records)["top_reasons"][:3]
            save_checkpoint(state)
            announce("Interrupted. Current heuristics preserved.")
            return checkpoint_path

        aggregate = aggregate_iteration_analysis(match_records)
        score = score_iteration(match_records)
        overall_score = score["overall"]
        training_score = score["training"]
        validation_score = score["validation"]
        win_rate = _iteration_win_rate(overall_score, len(scheduled_games))
        improved = is_better_score(score, best_score_before)
        if improved:
            state["best_weights"] = {key: _round_float(candidate_weights[key], 6) for key in FEATURE_ORDER}
            state["best_multipliers"] = {key: _round_float(candidate_multipliers[key], 6) for key in FEATURE_ORDER}
            state["best_score"] = score
            state["fixed_position_suite_result"] = fixed_position_suite_result
            state["stagnation_count"] = 0
        else:
            state["stagnation_count"] = int(state.get("stagnation_count", 0)) + 1

        iteration_record = {
            "iteration": next_iteration,
            "phase": state["active_iteration"]["phase"],
            "accepted_as_best": improved,
            "score": score,
            "training_score": training_score,
            "validation_score": validation_score,
            "overall_score": overall_score,
            "win_rate": win_rate,
            "top_reasons": aggregate["top_reasons"][:5],
            "feature_delta_summary": aggregate.get("feature_delta_summary", {}),
            "fixed_position_suite_result": fixed_position_suite_result,
            "candidate_weights": {key: _round_float(candidate_weights[key], 6) for key in FEATURE_ORDER},
            "candidate_multipliers": {key: _round_float(candidate_multipliers[key], 6) for key in FEATURE_ORDER},
            "weight_diff_from_previous_best": _weight_delta(candidate_weights, best_weights_before),
        }
        _append_jsonl(iterations_path, iteration_record)

        state.setdefault("iteration_summaries", [])
        state["iteration_summaries"] = [
            item for item in state["iteration_summaries"] if int(item["iteration"]) != next_iteration
        ]
        state["iteration_summaries"].append(
            {
                "iteration": next_iteration,
                "wins": overall_score["wins"],
                "draws": overall_score["draws"],
                "losses": overall_score["losses"],
                "match_points": overall_score["match_points"],
                "capture_diff": overall_score["capture_diff"],
                "training_match_points": training_score["match_points"],
                "validation_match_points": validation_score["match_points"],
                "win_rate": win_rate,
                "accepted_as_best": improved,
            }
        )
        state["iteration_summaries"].sort(key=lambda item: int(item["iteration"]))

        state["latest_analyses"] = aggregate["top_reasons"][:5]
        state["progress"]["completed_iterations"] = next_iteration
        state["progress"]["current_iteration"] = None
        state["progress"]["current_iteration_matches"] = 0
        state["active_iteration"] = None

        announce(
            f"[Iteration {next_iteration}/{total_iterations}] "
            f"W={overall_score['wins']} D={overall_score['draws']} L={overall_score['losses']} "
            f"capture_diff={overall_score['capture_diff']} partial_searches={overall_score['partial_searches']} "
            f"{'accepted' if improved else 'rejected'}."
        )

        if next_iteration >= total_iterations:
            state["status"] = "completed"
            state.setdefault("timing", {})
            state["timing"]["started_at"] = state["timing"].get("started_at") or state.get("created_at")
            state["timing"]["completed_at"] = _utc_now()
            state["timing"]["elapsed_s"] = _elapsed_time_seconds(
                state["timing"].get("started_at"),
                state["timing"].get("completed_at"),
            )
            state["final_summary"] = _build_final_summary(state)
            save_checkpoint(state)
            announce("Final summary:")
            announce(f"Time elapsed: {_format_elapsed_time(float(state['final_summary']['total_time_s']))}")
            announce(f"Iterations completed: {state['final_summary']['iterations_completed']}")
            announce(f"Layout: {state['final_summary']['layout']}")
            announce(f"Agent: {state['final_summary']['agent_id']}")
            announce(f"Max moves: {state['final_summary']['max_moves']}")
            announce("")
            announce(f"Average win rate {state['final_summary']['average_win_rate']:.2f}%")
            announce(
                f"Best win rate {state['final_summary']['best_win_rate']:.2f}% "
                f"(Iteration(s): {', '.join(str(item) for item in state['final_summary']['best_win_rate_iterations'])})"
            )
            announce("")
            announce("Best heuristic:")
            for line in _format_weight_change_lines(state['final_summary']['weight_changes']):
                announce(line)
            return checkpoint_path

        proposal = propose_next_weights(
            baseline_weights=state["baseline_weights"],
            current_weights=candidate_weights,
            best_weights=normalize_weights(state["best_weights"]),
            top_reasons=aggregate["top_reasons"],
            feature_delta_summary=aggregate.get("feature_delta_summary", {}),
            iteration_index=next_iteration + 1,
            total_iterations=total_iterations,
            stagnation_count=int(state.get("stagnation_count", 0)),
            seed=int(state["config"]["seed"]),
        )
        state["current_weights"] = proposal["weights"]
        state["current_multipliers"] = proposal["multipliers"]
        state["optimizer"] = {
            "next_phase": proposal["phase"],
            "stagnation_count": int(state.get("stagnation_count", 0)),
        }
        save_checkpoint(state)
        next_iteration += 1

    state["status"] = "completed"
    save_checkpoint(state)
    return checkpoint_path







