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

SUPPORTED_WEIGHT_KEYS = frozenset(DEFAULT_WEIGHTS.keys())


def _opponent(player: int) -> int:
    return WHITE if player == BLACK else BLACK


def marble_advantage(player_marbles: List[Position], opp_marbles: List[Position]) -> float:
    """Difference in marble count."""
    return float(len(player_marbles) - len(opp_marbles))


def center_control(player_marbles: List[Position], opp_marbles: List[Position]) -> float:
    """Prefer marbles closer to center, squared to heavily penalize edges."""
    def dist_sum(marbles):
        total = 0
        for r, c in marbles:
            dr, dc = r - CENTER[0], c - CENTER[1]
            if (dr >= 0 and dc >= 0) or (dr <= 0 and dc <= 0):
                dist = max(abs(dr), abs(dc))
            else:
                dist = abs(dr) + abs(dc)
            total += dist ** 2
        return total

    return float(dist_sum(opp_marbles) - dist_sum(player_marbles))


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
    opp = _opponent(player)

    # Extract board marbles ONLY ONCE!
    player_marbles_list = board.get_marbles(player)
    opp_marbles_list = board.get_marbles(opp)

    player_marbles_set = set(player_marbles_list)
    opp_marbles_set = set(opp_marbles_list)

    score = 0.0

    if weights.get("marble", 0.0) != 0.0:
        score += weights["marble"] * marble_advantage(player_marbles_list, opp_marbles_list)

    if weights.get("center", 0.0) != 0.0:
        score += weights["center"] * center_control(player_marbles_list, opp_marbles_list)

    if weights.get("cohesion", 0.0) != 0.0:
        score += weights["cohesion"] * cohesion(player_marbles_set, opp_marbles_set)

    if weights.get("cluster", 0.0) != 0.0:
        score += weights["cluster"] * largest_cluster(player_marbles_set, opp_marbles_set)

    if weights.get("edge_pressure", 0.0) != 0.0:
        score += weights["edge_pressure"] * edge_pressure(player_marbles_list, opp_marbles_list)

    if weights.get("formation", 0.0) != 0.0:
        score += weights["formation"] * formation_strength(player_marbles_set, opp_marbles_set)

    if weights.get("push", 0.0) != 0.0:
        score += weights["push"] * push_potential(player_marbles_set, opp_marbles_set)

    if weights.get("mobility", 0.0) != 0.0:
        score += weights["mobility"] * mobility(player_marbles_set, opp_marbles_set)

    if weights.get("stability", 0.0) != 0.0:
        score += weights["stability"] * stability(player_marbles_set, opp_marbles_set)

    return score


def build_weighted_evaluator(weights: Dict[str, float]) -> Callable[[Board, int], float]:
    """Create an evaluator callable from a shared weight dictionary."""
    resolved = dict(weights)
    unknown_keys = sorted(key for key in resolved if key not in SUPPORTED_WEIGHT_KEYS)
    if unknown_keys:
        joined = ", ".join(unknown_keys)
        raise ValueError(f"Unsupported heuristic weight key(s): {joined}")

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
