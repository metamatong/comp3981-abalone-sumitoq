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
    normalize_weights,
    weights_from_multipliers,
    weights_to_multipliers,
)
from ..game.board import BLACK, WHITE
from ..game.config import GameConfig
from ..game.session import GameSession
from ..players.registry import get_agent_weights, list_agents

CHECKPOINT_VERSION = 1
RUNS_ROOT = Path("abalone") / "eval_runs"
PARTIAL_SEARCH_SOURCES = frozenset({"timeout_partial", "timeout_fallback", "timeout_fallback_partial"})
SEARCH_INSTABILITY_SOURCES = PARTIAL_SEARCH_SOURCES | frozenset({"repeat_avoidance"})
OUTCOME_WEIGHTS = {"loss": 3.0, "draw": 1.0, "win": 0.0}
PHASE_STEP_SCALES = {"exploration": 1.0, "refinement": 0.45, "recovery": 1.35}
PHASE_JITTER_SCALES = {"exploration": 0.35, "refinement": 0.12, "recovery": 0.50}
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


def analyze_match_result(game: dict, target_agent_id: str) -> dict:
    """Infer likely reasons for a game result from the tuned agent's telemetry."""
    target_color = _target_color(game, target_agent_id)
    opponent_color = WHITE if target_color == BLACK else BLACK
    opponent_id = _opponent_for_game(game, target_agent_id)
    outcome = _outcome_for_game(game, target_agent_id)
    target_score = int(game["score"][target_color])
    opponent_score = int(game["score"][opponent_color])
    capture_diff = target_score - opponent_score

    target_entries = [entry for entry in game.get("history", []) if entry.get("agent_id") == target_agent_id]
    feature_samples = []
    decision_sources = []
    completed_depths = []
    requested_depths = []
    move_durations = []
    for entry in target_entries:
        search = entry.get("search") or {}
        pre_move = search.get("pre_move") or {}
        features = pre_move.get("features")
        if features:
            feature_samples.append(features)
        decision_sources.append(search.get("decision_source") or "unknown")
        completed_depths.append(float(search.get("completed_depth", 0) or 0))
        requested_depths.append(float(search.get("depth", 0) or 0))
        move_durations.append(float(entry.get("duration_ms", 0) or 0))

    feature_averages = {
        key: _round_float(_mean([float(sample.get(key, 0.0)) for sample in feature_samples]), 4)
        for key in FEATURE_ORDER
    }
    partial_searches = sum(1 for source in decision_sources if source in PARTIAL_SEARCH_SOURCES)
    unstable_decisions = sum(1 for source in decision_sources if source in SEARCH_INSTABILITY_SOURCES)
    avg_completed_depth = _round_float(_mean(completed_depths), 4)
    avg_requested_depth = _round_float(_mean(requested_depths), 4)
    avg_move_ms = _round_float(_mean(move_durations), 3)

    reasons = []

    def add_reason(code: str, severity: float, summary: str, evidence: dict) -> None:
        if severity <= 0.0:
            return
        reasons.append(
            {
                "code": code,
                "score": _round_float(severity, 4),
                "summary": summary,
                "evidence": evidence,
            }
        )

    if feature_averages["edge_pressure"] < -1.0:
        add_reason(
            "edge_exposure",
            abs(feature_averages["edge_pressure"]),
            "The agent repeatedly operated from more exposed edge positions than its opponent.",
            {
                "avg_edge_pressure": feature_averages["edge_pressure"],
                "avg_stability": feature_averages["stability"],
            },
        )

    fragmentation_pressure = (
        abs(min(feature_averages["cohesion"], 0.0))
        + abs(min(feature_averages["cluster"], 0.0))
        + abs(min(feature_averages["stability"], 0.0))
    )
    if fragmentation_pressure >= 1.5:
        add_reason(
            "fragmentation",
            fragmentation_pressure,
            "The agent's group structure became looser and less stable than the opponent's.",
            {
                "avg_cohesion": feature_averages["cohesion"],
                "avg_cluster": feature_averages["cluster"],
                "avg_stability": feature_averages["stability"],
            },
        )

    if feature_averages["mobility"] < -1.0:
        add_reason(
            "mobility_collapse",
            abs(feature_averages["mobility"]),
            "The agent spent too many turns with fewer practical movement options.",
            {"avg_mobility": feature_averages["mobility"]},
        )

    push_pressure = abs(min(feature_averages["push"], 0.0)) + abs(min(feature_averages["formation"], 0.0))
    if push_pressure >= 1.5:
        add_reason(
            "weak_push_threat",
            push_pressure,
            "The agent generated weaker pushing threats and formations than its opponent.",
            {
                "avg_push": feature_averages["push"],
                "avg_formation": feature_averages["formation"],
            },
        )

    if feature_averages["center"] < -2.0:
        add_reason(
            "poor_center_control",
            abs(feature_averages["center"]),
            "The agent ceded too much central control over the course of the game.",
            {"avg_center": feature_averages["center"]},
        )

    material_pressure = max(0.0, float(opponent_score - target_score)) + abs(min(feature_averages["marble"], 0.0))
    if material_pressure > 0.0:
        add_reason(
            "material_loss",
            material_pressure,
            "The opponent converted the game into a material advantage.",
            {
                "target_score": target_score,
                "opponent_score": opponent_score,
                "capture_diff": capture_diff,
                "avg_marble_feature": feature_averages["marble"],
            },
        )

    depth_gap = max(0.0, (avg_requested_depth * 0.6) - avg_completed_depth)
    instability_pressure = float(partial_searches) + (0.5 * float(unstable_decisions)) + depth_gap
    if instability_pressure > 0.0:
        add_reason(
            "search_instability",
            instability_pressure,
            "The agent relied on partial or unstable search decisions too often.",
            {
                "partial_searches": partial_searches,
                "unstable_decisions": unstable_decisions,
                "avg_completed_depth": avg_completed_depth,
                "avg_requested_depth": avg_requested_depth,
                "avg_move_ms": avg_move_ms,
            },
        )

    reasons.sort(key=lambda item: (-item["score"], item["code"]))
    return {
        "target_agent_id": target_agent_id,
        "opponent_id": opponent_id,
        "target_color": "black" if target_color == BLACK else "white",
        "outcome": outcome,
        "target_score": target_score,
        "opponent_score": opponent_score,
        "capture_diff": capture_diff,
        "feature_averages": feature_averages,
        "search_summary": {
            "move_count": len(target_entries),
            "partial_searches": partial_searches,
            "unstable_decisions": unstable_decisions,
            "avg_completed_depth": avg_completed_depth,
            "avg_requested_depth": avg_requested_depth,
            "avg_move_ms": avg_move_ms,
        },
        "reasons": reasons,
    }


def aggregate_iteration_analysis(match_records: Sequence[dict]) -> dict:
    """Aggregate per-game reasons into an iteration-level diagnosis."""
    aggregates = {}
    for record in match_records:
        analysis = record["analysis"]
        outcome_weight = OUTCOME_WEIGHTS[analysis["outcome"]]
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
            bucket["draws"] += 1 if analysis["outcome"] == "draw" else 0
            bucket["opponents"].add(analysis["opponent_id"])

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
    return {"top_reasons": top_reasons}


def score_iteration(match_records: Sequence[dict]) -> dict:
    """Score one full gauntlet using the lexicographic objective."""
    wins = sum(1 for record in match_records if record["analysis"]["outcome"] == "win")
    draws = sum(1 for record in match_records if record["analysis"]["outcome"] == "draw")
    losses = sum(1 for record in match_records if record["analysis"]["outcome"] == "loss")
    match_points = (wins * 2) + draws
    capture_diff = sum(int(record["analysis"]["capture_diff"]) for record in match_records)
    partial_searches = sum(int(record["analysis"]["search_summary"]["partial_searches"]) for record in match_records)

    total_target_moves = sum(int(record["analysis"]["search_summary"]["move_count"]) for record in match_records)
    total_depth = sum(
        float(record["analysis"]["search_summary"]["avg_completed_depth"]) * int(record["analysis"]["search_summary"]["move_count"])
        for record in match_records
    )
    total_move_ms = sum(
        float(record["analysis"]["search_summary"]["avg_move_ms"]) * int(record["analysis"]["search_summary"]["move_count"])
        for record in match_records
    )
    avg_completed_depth = _round_float(total_depth / total_target_moves, 4) if total_target_moves else 0.0
    avg_move_ms = _round_float(total_move_ms / total_target_moves, 3) if total_target_moves else 0.0

    comparison_key = [
        int(match_points),
        int(capture_diff),
        -int(partial_searches),
        float(avg_completed_depth),
        -float(avg_move_ms),
    ]
    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "match_points": match_points,
        "capture_diff": capture_diff,
        "partial_searches": partial_searches,
        "avg_completed_depth": avg_completed_depth,
        "avg_move_ms": avg_move_ms,
        "comparison_key": comparison_key,
    }


def _score_tuple(score: Optional[dict]) -> Optional[Tuple[float, ...]]:
    if score is None:
        return None
    return tuple(score["comparison_key"])


def is_better_score(candidate: dict, incumbent: Optional[dict]) -> bool:
    """Return whether a candidate gauntlet score beats the incumbent score."""
    incumbent_key = _score_tuple(incumbent)
    if incumbent_key is None:
        return True
    return _score_tuple(candidate) > incumbent_key


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
    baseline_weights = normalize_weights(baseline_weights)
    baseline_multipliers = {key: 1.0 for key in FEATURE_ORDER}
    return {
        "version": CHECKPOINT_VERSION,
        "status": "running",
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "config": config,
        "paths": {
            "run_dir": str(run_dir.resolve()),
            "checkpoint": str(checkpoint_path.resolve()),
            "matches": str(matches_path.resolve()),
            "iterations": str(iterations_path.resolve()),
        },
        "baseline_weights": {key: _round_float(baseline_weights[key], 6) for key in FEATURE_ORDER},
        "baseline_multipliers": baseline_multipliers,
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
            "matches_per_iteration": len(config["opponents"]) * 2,
        },
        "active_iteration": None,
        "iteration_summaries": [],
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
        return json.load(handle)


def _build_run_dir(config: dict, runs_root: Optional[Path]) -> Path:
    root = runs_root or RUNS_ROOT
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return root / config["agent_id"] / _config_slug(config) / timestamp


def _build_match_log_record(iteration_index: int, game: dict, analysis: dict) -> dict:
    return {
        "iteration": iteration_index,
        "game_index": game["index"],
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
        "telemetry_summary": analysis["search_summary"],
        "reasons": analysis["reasons"],
        "analysis": analysis,
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
    state["config"]["iterations"] = max(int(iterations), int(state["config"]["iterations"]))
    if jobs is not None:
        state["config"]["jobs"] = int(jobs)
    state.setdefault("iteration_summaries", [])
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
        }
        run_dir = _build_run_dir(config, runs_root)
        state = _initial_state(config, baseline_weights, run_dir)
        save_checkpoint(state)
        checkpoint_path = Path(state["paths"]["checkpoint"])

    agent_id = state["config"]["agent_id"]
    total_iterations = int(state["config"]["iterations"])
    matches_path = Path(state["paths"]["matches"])
    iterations_path = Path(state["paths"]["iterations"])

    next_iteration = int(state["progress"]["completed_iterations"]) + 1
    active_iteration = state.get("active_iteration")
    if active_iteration and active_iteration.get("status") != "completed":
        next_iteration = int(active_iteration["index"])

    while next_iteration <= total_iterations:
        scheduled_games = build_all_opponents_jobs(
            agent_id=agent_id,
            depth=state["config"]["depth"],
            layout=state["config"]["layout"],
            move_time_s=state["config"]["move_time_s"],
            max_moves=state["config"]["max_moves"],
            seed=state["config"]["seed"] + ((next_iteration - 1) * (len(state["config"]["opponents"]) * 2)),
        )
        worker_count = resolve_worker_count(state["config"].get("jobs"), len(scheduled_games))
        candidate_weights = normalize_weights(state["current_weights"])
        candidate_multipliers = weights_to_multipliers(candidate_weights, state["baseline_weights"])
        best_weights_before = normalize_weights(state["best_weights"])
        best_score_before = state.get("best_score")

        state["status"] = "running"
        state["progress"]["current_iteration"] = next_iteration
        state["progress"]["current_iteration_matches"] = 0
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
            analysis = analyze_match_result(game, agent_id)
            record = _build_match_log_record(next_iteration, game, analysis)
            match_records.append(record)
            _append_jsonl(matches_path, record)
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
        win_rate = _iteration_win_rate(score, len(scheduled_games))
        improved = is_better_score(score, best_score_before)
        if improved:
            state["best_weights"] = {key: _round_float(candidate_weights[key], 6) for key in FEATURE_ORDER}
            state["best_multipliers"] = {key: _round_float(candidate_multipliers[key], 6) for key in FEATURE_ORDER}
            state["best_score"] = score
            state["stagnation_count"] = 0
        else:
            state["stagnation_count"] = int(state.get("stagnation_count", 0)) + 1

        iteration_record = {
            "iteration": next_iteration,
            "phase": state["active_iteration"]["phase"],
            "accepted_as_best": improved,
            "score": score,
            "win_rate": win_rate,
            "top_reasons": aggregate["top_reasons"][:5],
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
                "wins": score["wins"],
                "draws": score["draws"],
                "losses": score["losses"],
                "match_points": score["match_points"],
                "capture_diff": score["capture_diff"],
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
            f"W={score['wins']} D={score['draws']} L={score['losses']} "
            f"capture_diff={score['capture_diff']} partial_searches={score['partial_searches']} "
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







