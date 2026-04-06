"""Native-backed deterministic minimax search."""

from dataclasses import dataclass
import time
from typing import Dict, List, Optional

from ..game.board import Board, Move
from ..native import search_weighted as native_search_weighted
from .defaults import DEFAULT_AGENT
from .heuristics import FEATURE_ORDER
from .types import AgentConfig, AgentDefinition, resolve_agent_config


@dataclass(frozen=True)
class SearchResult:
    """Search output including chosen move and diagnostic metadata."""

    move: Optional[Move]
    score: float
    nodes: int
    elapsed_ms: float
    depth: int
    completed_depth: int
    decision_source: str
    timed_out: bool
    time_budget_ms: Optional[int]
    agent_id: str
    agent_label: str
    root_candidates: Optional[List[Dict[str, object]]] = None
    analysis_evaluator_id: Optional[str] = None
    board_token_before: Optional[str] = None

    def as_dict(self) -> Dict[str, object]:
        """Serialize search diagnostics for CLI/API responses."""
        notation = None if self.move is None else self.move.to_notation()
        payload = {
            "notation": notation,
            "score": self.score,
            "nodes": self.nodes,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "depth": self.depth,
            "completed_depth": self.completed_depth,
            "decision_source": self.decision_source,
            "timed_out": self.timed_out,
            "time_budget_ms": self.time_budget_ms,
            "agent_id": self.agent_id,
            "agent_label": self.agent_label,
        }
        if self.root_candidates:
            payload["root_candidates"] = list(self.root_candidates)
        if self.analysis_evaluator_id:
            payload["analysis_evaluator_id"] = self.analysis_evaluator_id
        if self.board_token_before:
            payload["board_token_before"] = self.board_token_before
        return payload


def _resolve_decision_source(
    *,
    move: Optional[Move],
    completed_depth: int,
    timed_out: bool,
    avoidance_applied: bool,
) -> str:
    """Map native search metadata onto the public decision-source strings."""
    decision_source = "search"
    if move is None:
        decision_source = "search"
    elif completed_depth == 0 and not timed_out:
        decision_source = "fallback"
    elif timed_out and completed_depth == 0:
        decision_source = "timeout_fallback_partial"
    elif timed_out:
        decision_source = "timeout_partial"
    if avoidance_applied and decision_source == "search":
        decision_source = "repeat_avoidance"
    return decision_source


def _normalize_root_candidates(raw_candidates, limit: int):
    """Normalize native root-candidate payloads to the public response shape."""
    if limit <= 0 or not raw_candidates:
        return None
    root_candidates = [
        {
            "notation": candidate["move"].to_notation(),
            "score": round(float(candidate["score"]), 6),
            "depth": int(candidate["depth"]),
        }
        for candidate in raw_candidates
        if candidate.get("move") is not None
    ]
    if not root_candidates:
        return None
    return sorted(
        root_candidates,
        key=lambda item: (-float(item["score"]), str(item["notation"])),
    )[:limit]


def search_best_move(
    board: Board,
    player: int,
    agent: Optional[AgentDefinition] = None,
    config: Optional[AgentConfig] = None,
) -> SearchResult:
    """Return best move and diagnostics for `player` from the native engine."""
    resolved_agent = agent or DEFAULT_AGENT
    resolved_config = resolve_agent_config(resolved_agent, config)
    agent_weights = getattr(resolved_agent.evaluator, "weights", None)
    if not agent_weights:
        raise ValueError(
            "Native search on this branch requires agents whose evaluators expose "
            "deterministic heuristic weights."
        )

    ordered_weights = tuple(float(agent_weights[key]) for key in FEATURE_ORDER)
    native_start = time.perf_counter()
    native_result = native_search_weighted(
        board,
        player,
        ordered_weights,
        depth=resolved_config.depth,
        time_budget_ms=resolved_config.time_budget_ms,
        tie_break=resolved_config.tie_break,
        avoid_move=resolved_config.avoid_move,
        root_candidate_limit=resolved_config.root_candidate_limit,
    )
    elapsed_ms = (time.perf_counter() - native_start) * 1000.0

    move = native_result["move"]
    completed_depth = int(native_result["completed_depth"])
    timed_out = bool(native_result["timed_out"])
    avoidance_applied = bool(native_result.get("avoidance_applied"))
    return SearchResult(
        move=move,
        score=float(native_result["score"]),
        nodes=int(native_result["nodes"]),
        elapsed_ms=elapsed_ms,
        depth=resolved_config.depth,
        completed_depth=completed_depth,
        decision_source=_resolve_decision_source(
            move=move,
            completed_depth=completed_depth,
            timed_out=timed_out,
            avoidance_applied=avoidance_applied,
        ),
        timed_out=timed_out,
        time_budget_ms=resolved_config.time_budget_ms,
        agent_id=resolved_agent.id,
        agent_label=resolved_agent.label,
        root_candidates=_normalize_root_candidates(
            native_result.get("root_candidates"),
            resolved_config.root_candidate_limit,
        ),
        analysis_evaluator_id=resolved_config.analysis_evaluator_id,
        board_token_before=resolved_config.board_token_before,
    )
