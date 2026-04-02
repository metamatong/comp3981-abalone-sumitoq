"""Heuristic feature helpers and weighted evaluation functions for minimax."""

from collections import deque
from typing import Callable, Dict, List, Set, Tuple

from ..game.board import BLACK, WHITE, Board, DIRECTIONS, VALID_POSITIONS, is_valid, Position

CENTER = (4, 5)

# Weights for the default balanced heuristic.
DEFAULT_WEIGHTS = {
    "marble": 50000.0,   # marble count advantage (dominant term; losing a marble is catastrophic)
    "center": 40.0,      # proximity of own marbles to board center (lower squared distance = better)
    "cohesion": 35.0,    # total friendly adjacency count (tight groups are harder to push off)
    "cluster": 30.0,     # size of largest connected friendly group (encourages staying together)
    "edge_pressure": 70.0,  # combined own edge exposure and opponent rim pressure from one shared scan
    "formation": 50.0,   # inline 2- and 3-marble formations (prerequisite for push moves)
    "push": 120.0,       # estimated push opportunities against opponent (most actionable threat)
    "mobility": 20.0,    # difference in legal move count (more options = better flexibility)
    "stability": 25.0,   # marbles with >=2 friendly neighbors (well-supported, harder to isolate)
}

FEATURE_ORDER = tuple(DEFAULT_WEIGHTS.keys())
SUPPORTED_WEIGHT_KEYS = frozenset(FEATURE_ORDER)

# Runtime tuning metadata used by the adaptive gauntlet optimizer.
WEIGHT_TUNING_RULES = {
    "marble": {"step": 0.04, "min_multiplier": 0.75, "max_multiplier": 1.75},
    "center": {"step": 0.18, "min_multiplier": 0.20, "max_multiplier": 3.00},
    "cohesion": {"step": 0.16, "min_multiplier": 0.20, "max_multiplier": 3.00},
    "cluster": {"step": 0.14, "min_multiplier": 0.20, "max_multiplier": 3.00},
    "edge_pressure": {"step": 0.18, "min_multiplier": 0.20, "max_multiplier": 3.50},
    "formation": {"step": 0.16, "min_multiplier": 0.20, "max_multiplier": 3.00},
    "push": {"step": 0.18, "min_multiplier": 0.20, "max_multiplier": 3.50},
    "mobility": {"step": 0.18, "min_multiplier": 0.20, "max_multiplier": 3.50},
    "stability": {"step": 0.16, "min_multiplier": 0.20, "max_multiplier": 3.00},
}


def _opponent(player: int) -> int:
    return WHITE if player == BLACK else BLACK


def _center_distance_sum(marbles: List[Position]) -> int:
    """Return total squared distance from center using the board's hex geometry."""
    total = 0
    for r, c in marbles:
        dr, dc = r - CENTER[0], c - CENTER[1]
        if (dr >= 0 and dc >= 0) or (dr <= 0 and dc <= 0):
            dist = max(abs(dr), abs(dc))
        else:
            dist = abs(dr) + abs(dc)
        total += dist ** 2
    return total


def _structure_profile(marbles: Set[Position]) -> Tuple[int, int, int]:
    """Return adjacency score, largest cluster size, and stability count from one scan."""
    visited: Set[Position] = set()
    adjacency = 0
    largest_cluster_size = 0
    stable_count = 0

    for marble in marbles:
        if marble in visited:
            continue

        queue = deque([marble])
        visited.add(marble)
        component_size = 0

        while queue:
            row, col = queue.popleft()
            component_size += 1
            support = 0

            for dr, dc in DIRECTIONS:
                next_pos = (row + dr, col + dc)
                if next_pos not in marbles:
                    continue
                adjacency += 1
                support += 1
                if next_pos not in visited:
                    visited.add(next_pos)
                    queue.append(next_pos)

            if support >= 2:
                stable_count += 1

        if component_size > largest_cluster_size:
            largest_cluster_size = component_size

    return adjacency, largest_cluster_size, stable_count


def _extract_feature_context(board: Board, player: int) -> Dict[str, object]:
    """Collect reusable marble lists, sets, and derived scans for one evaluation."""
    opp = _opponent(player)
    player_marbles_list = board.get_marbles(player)
    opp_marbles_list = board.get_marbles(opp)
    player_marbles_set = set(player_marbles_list)
    opp_marbles_set = set(opp_marbles_list)
    player_adjacency, player_cluster, player_stability = _structure_profile(player_marbles_set)
    opp_adjacency, opp_cluster, opp_stability = _structure_profile(opp_marbles_set)
    player_risk, player_pressure = _edge_profile(player_marbles_list)
    opp_risk, opp_pressure = _edge_profile(opp_marbles_list)
    return {
        "player_marbles_list": player_marbles_list,
        "opp_marbles_list": opp_marbles_list,
        "player_marbles_set": player_marbles_set,
        "opp_marbles_set": opp_marbles_set,
        "player_adjacency": player_adjacency,
        "opp_adjacency": opp_adjacency,
        "player_cluster": player_cluster,
        "opp_cluster": opp_cluster,
        "player_stability": player_stability,
        "opp_stability": opp_stability,
        "player_risk": player_risk,
        "opp_risk": opp_risk,
        "player_pressure": player_pressure,
        "opp_pressure": opp_pressure,
    }


def _features_from_context(context: Dict[str, object]) -> Dict[str, float]:
    """Build the full feature map from a reusable extraction context."""
    player_marbles_list = context["player_marbles_list"]
    opp_marbles_list = context["opp_marbles_list"]
    player_marbles_set = context["player_marbles_set"]
    opp_marbles_set = context["opp_marbles_set"]
    return {
        "marble": float(len(player_marbles_list) - len(opp_marbles_list)),
        "center": float(_center_distance_sum(opp_marbles_list) - _center_distance_sum(player_marbles_list)),
        "cohesion": float(context["player_adjacency"] - context["opp_adjacency"]),
        "cluster": float(context["player_cluster"] - context["opp_cluster"]),
        "edge_pressure": float(
            (context["opp_risk"] + context["opp_pressure"])
            - (context["player_risk"] + context["player_pressure"])
        ),
        "formation": formation_strength(player_marbles_set, opp_marbles_set),
        "push": push_potential(player_marbles_set, opp_marbles_set),
        "mobility": mobility(player_marbles_set, opp_marbles_set),
        "stability": float(context["player_stability"] - context["opp_stability"]),
    }


def normalize_weights(
    weights: Dict[str, float],
    baseline: Dict[str, float] = DEFAULT_WEIGHTS,
    *,
    fill_missing: bool = False,
) -> Dict[str, float]:
    """Validate a weight mapping and optionally fill missing keys from a baseline."""
    unknown_keys = sorted(key for key in weights if key not in SUPPORTED_WEIGHT_KEYS)
    if unknown_keys:
        joined = ", ".join(unknown_keys)
        raise ValueError(f"Unsupported heuristic weight key(s): {joined}")
    if fill_missing:
        resolved = dict(baseline)
        resolved.update(weights)
        return {key: float(resolved[key]) for key in FEATURE_ORDER}
    return {key: float(weights[key]) for key in FEATURE_ORDER if key in weights}


def extract_features(board: Board, player: int) -> Dict[str, float]:
    """Return the raw heuristic feature values for `player` on the current board."""
    return _features_from_context(_extract_feature_context(board, player))


def evaluate_breakdown(board: Board, player: int, weights: Dict[str, float]) -> Dict[str, Dict[str, float] | float]:
    """Return raw features plus weighted contributions for reporting and tuning."""
    resolved = normalize_weights(weights, fill_missing=True)
    features = extract_features(board, player)
    contributions = {key: resolved[key] * features[key] for key in FEATURE_ORDER}
    return {
        "features": features,
        "contributions": contributions,
        "total": sum(contributions.values()),
    }


def weights_to_multipliers(
    weights: Dict[str, float],
    baseline: Dict[str, float] = DEFAULT_WEIGHTS,
) -> Dict[str, float]:
    """Express absolute weights as baseline-relative multipliers."""
    resolved_weights = normalize_weights(weights, fill_missing=True)
    resolved_baseline = normalize_weights(baseline, fill_missing=True)
    multipliers = {}
    for key in FEATURE_ORDER:
        baseline_value = resolved_baseline[key]
        if baseline_value == 0.0:
            multipliers[key] = 1.0 if resolved_weights[key] == 0.0 else resolved_weights[key]
        else:
            multipliers[key] = resolved_weights[key] / baseline_value
    return multipliers


def clamp_multiplier(key: str, multiplier: float) -> float:
    """Clamp a multiplier using the per-weight tuning rules."""
    rules = WEIGHT_TUNING_RULES[key]
    return max(rules["min_multiplier"], min(rules["max_multiplier"], multiplier))


def weights_from_multipliers(
    baseline: Dict[str, float],
    multipliers: Dict[str, float],
) -> Dict[str, float]:
    """Convert baseline-relative multipliers back into absolute weights."""
    resolved_baseline = normalize_weights(baseline, fill_missing=True)
    resolved_multipliers = {key: float(multipliers.get(key, 1.0)) for key in FEATURE_ORDER}
    weights = {}
    for key in FEATURE_ORDER:
        clamped = clamp_multiplier(key, resolved_multipliers[key])
        weights[key] = resolved_baseline[key] * clamped
    return weights


def marble_advantage(player_marbles: List[Position], opp_marbles: List[Position]) -> float:
    """Difference in marble count."""
    return float(len(player_marbles) - len(opp_marbles))


def center_control(player_marbles: List[Position], opp_marbles: List[Position]) -> float:
    """Prefer marbles closer to center, squared to heavily penalize edges."""
    return float(_center_distance_sum(opp_marbles) - _center_distance_sum(player_marbles))


def cohesion(player_marbles: Set[Position], opp_marbles: Set[Position]) -> float:
    """Friendly adjacency count."""
    def adjacency(marbles):
        score = 0
        for r, c in marbles:
            for dr, dc in DIRECTIONS:
                if (r + dr, c + dc) in marbles:
                    score += 1
        return score

    return float(adjacency(player_marbles) - adjacency(opp_marbles))


def largest_cluster(player_marbles: Set[Position], opp_marbles: Set[Position]) -> float:
    """Largest connected group size."""
    def cluster_size(marbles):
        visited = set()
        largest = 0

        for marble in marbles:
            if marble in visited:
                continue

            queue = deque([marble])
            visited.add(marble)
            size = 0

            while queue:
                r, c = queue.popleft()
                size += 1

                for dr, dc in DIRECTIONS:
                    next_pos = (r + dr, c + dc)
                    if next_pos in marbles and next_pos not in visited:
                        visited.add(next_pos)
                        queue.append(next_pos)

            largest = max(largest, size)

        return largest

    return float(cluster_size(player_marbles) - cluster_size(opp_marbles))


def _edge_profile(marbles: List[Position]) -> Tuple[int, int]:
    """Return combined edge-risk and rim-pressure points from one pass."""
    risk_points = 0
    pressure_points = 0

    for r, c in marbles:
        edge_dist = 2
        for dr, dc in DIRECTIONS:
            pos1 = (r + dr, c + dc)
            if not is_valid(pos1):
                edge_dist = 0
                break
            pos2 = (pos1[0] + dr, pos1[1] + dc)
            if not is_valid(pos2):
                edge_dist = min(edge_dist, 1)

        if edge_dist == 0:
            risk_points += 2
            pressure_points += 1
        elif edge_dist == 1:
            risk_points += 1
            pressure_points += 1

    return risk_points, pressure_points


def edge_pressure(player_marbles: List[Position], opp_marbles: List[Position]) -> float:
    """Combine own edge exposure and opponent rim pressure from one shared scan."""
    player_risk, player_pressure = _edge_profile(player_marbles)
    opp_risk, opp_pressure = _edge_profile(opp_marbles)
    return float((opp_risk + opp_pressure) - (player_risk + player_pressure))


def formation_strength(player_marbles: Set[Position], opp_marbles: Set[Position]) -> float:
    """Inline formations useful for pushes."""
    def formation(marbles):
        score = 0
        for r, c in marbles:
            for dr, dc in DIRECTIONS:
                m1 = (r + dr, c + dc)
                if m1 in marbles:
                    score += 1
                    m2 = (r + 2 * dr, c + 2 * dc)
                    if m2 in marbles:
                        score += 3
        return score

    return float(formation(player_marbles) - formation(opp_marbles))


def push_potential(player_marbles: Set[Position], opp_marbles: Set[Position]) -> float:
    """Estimate potential pushes."""
    def pushes(marbles, opponent):
        score = 0
        for r, c in marbles:
            for dr, dc in DIRECTIONS:
                m1 = (r + dr, c + dc)
                if m1 in marbles:
                    m2 = (r + 2 * dr, c + 2 * dc)
                    if m2 in opponent:
                        score += 2
                    elif m2 in marbles:
                        m3 = (r + 3 * dr, c + 3 * dc)
                        if m3 in opponent:
                            score += 3
        return score

    return float(pushes(player_marbles, opp_marbles) - pushes(opp_marbles, player_marbles))


def mobility(player_marbles: Set[Position], opp_marbles: Set[Position]) -> float:
    """Fast mobility approximation: count empty valid neighbors per marble.

    This approximates true legal-move count at O(marbles × 6) instead of
    the O(marbles² × directions) cost of full move generation.
    """
    all_occupied = player_marbles | opp_marbles

    def freedom(marbles):
        count = 0
        for r, c in marbles:
            for dr, dc in DIRECTIONS:
                nb = (r + dr, c + dc)
                if nb in VALID_POSITIONS and nb not in all_occupied:
                    count += 1
        return count

    return float(freedom(player_marbles) - freedom(opp_marbles))


def stability(player_marbles: Set[Position], opp_marbles: Set[Position]) -> float:
    """Pieces supported by friendly neighbors."""
    def stable(marbles):
        count = 0
        for r, c in marbles:
            support = 0
            for dr, dc in DIRECTIONS:
                if (r + dr, c + dc) in marbles:
                    support += 1
            if support >= 2:
                count += 1
        return count

    return float(stable(player_marbles) - stable(opp_marbles))


def evaluate_with_weights(board: Board, player: int, weights: Dict[str, float]) -> float:
    """Score a board using a weighted combination of shared heuristic features."""
    resolved_weights = normalize_weights(weights)
    features = _features_from_context(_extract_feature_context(board, player))
    return sum(
        resolved_weights[key] * features[key]
        for key in FEATURE_ORDER
        if resolved_weights.get(key, 0.0) != 0.0
    )


def build_weighted_evaluator(weights: Dict[str, float]) -> Callable[[Board, int], float]:
    """Create an evaluator callable from a shared weight dictionary."""
    resolved = normalize_weights(weights, fill_missing=True)

    def evaluator(board: Board, player: int) -> float:
        return evaluate_with_weights(board, player, resolved)

    # Attach weights for reporting/inspection (e.g., benchmarking summaries).
    evaluator.weights = dict(resolved)
    return evaluator


def evaluate_board(board: Board, player: int) -> float:
    """Default balanced heuristic evaluation."""
    return evaluate_with_weights(board, player, DEFAULT_WEIGHTS)


# Attach baseline weights for reporting/inspection.
evaluate_board.weights = dict(DEFAULT_WEIGHTS)
