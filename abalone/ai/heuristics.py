"""Heuristic feature helpers and weighted evaluation functions for minimax."""

from collections import deque
from typing import Callable, Dict, List

from ..game.board import BLACK, WHITE, Board, DIRECTIONS, Move, is_valid
from ..state_space import generate_legal_moves

CENTER = (4, 5)

# Weights for the default balanced heuristic.
DEFAULT_WEIGHTS = {
    "marble": 50000.0,   # marble count advantage (dominant term; losing a marble is catastrophic)
    "center": 40.0,      # proximity of own marbles to board center (lower squared distance = better)
    "cohesion": 35.0,    # total friendly adjacency count (tight groups are harder to push off)
    "cluster": 30.0,     # size of largest connected friendly group (encourages staying together)
    "edge": 60.0,        # penalizes own marbles near the board edge (high push-off risk)
    "formation": 50.0,   # inline 2- and 3-marble formations (prerequisite for push moves)
    "push": 120.0,       # estimated push opportunities against opponent (most actionable threat)
    "threat": 80.0,      # opponent marbles near edges under friendly pressure (imminent capture)
    "mobility": 20.0,    # difference in legal move count (more options = better flexibility)
    "stability": 25.0,   # marbles with >=2 friendly neighbors (well-supported, harder to isolate)
}


def _opponent(player: int) -> int:
    return WHITE if player == BLACK else BLACK


def marble_advantage(board: Board, player: int) -> float:
    """Difference in marble count."""
    opp = _opponent(player)
    return float(board.marble_count(player) - board.marble_count(opp))


def center_control(board: Board, player: int) -> float:
    """Prefer marbles closer to center, squared to heavily penalize edges."""
    opp = _opponent(player)

    def dist_sum(color):
        total = 0
        for r, c in board.get_marbles(color):
            dr, dc = r - CENTER[0], c - CENTER[1]
            if (dr >= 0 and dc >= 0) or (dr <= 0 and dc <= 0):
                dist = max(abs(dr), abs(dc))
            else:
                dist = abs(dr) + abs(dc)
            total += dist ** 2
        return total

    return float(dist_sum(opp) - dist_sum(player))


def cohesion(board: Board, player: int) -> float:
    """Friendly adjacency count."""
    opp = _opponent(player)

    def adjacency(color):
        marbles = set(board.get_marbles(color))
        score = 0
        for r, c in marbles:
            for dr, dc in DIRECTIONS:
                if (r + dr, c + dc) in marbles:
                    score += 1
        return score

    return float(adjacency(player) - adjacency(opp))


def largest_cluster(board: Board, player: int) -> float:
    """Largest connected group size."""
    opp = _opponent(player)

    def cluster_size(color):
        marbles = set(board.get_marbles(color))
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

    return float(cluster_size(player) - cluster_size(opp))


def edge_risk(board: Board, player: int) -> float:
    """Penalize marbles near edges."""
    opp = _opponent(player)

    def risk(color):
        score = 0
        for r, c in board.get_marbles(color):
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
                score += 2
            elif edge_dist == 1:
                score += 1

        return score

    return float(risk(opp) - risk(player))


def formation_strength(board: Board, player: int) -> float:
    """Inline formations useful for pushes."""
    opp = _opponent(player)

    def formation(color):
        marbles = set(board.get_marbles(color))
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

    return float(formation(player) - formation(opp))


def push_potential(board: Board, player: int) -> float:
    """Estimate potential pushes."""
    opp = _opponent(player)

    def pushes(color):
        marbles = set(board.get_marbles(color))
        opponent_marbles = set(board.get_marbles(_opponent(color)))
        score = 0

        for r, c in marbles:
            for dr, dc in DIRECTIONS:
                m1 = (r + dr, c + dc)
                if m1 in marbles:
                    m2 = (r + 2 * dr, c + 2 * dc)
                    if m2 in opponent_marbles:
                        score += 2
                    elif m2 in marbles:
                        m3 = (r + 3 * dr, c + 3 * dc)
                        if m3 in opponent_marbles:
                            score += 3

        return score

    return float(pushes(player) - pushes(opp))


def threat_pressure(board: Board, player: int) -> float:
    """Opponent marbles near edges under pressure."""
    opp = _opponent(player)

    def threats(color):
        opponent = set(board.get_marbles(_opponent(color)))
        score = 0

        for r, c in opponent:
            edge_dist = 2
            for dr, dc in DIRECTIONS:
                pos1 = (r + dr, c + dc)
                if not is_valid(pos1):
                    edge_dist = 0
                    break
                pos2 = (pos1[0] + dr, pos1[1] + dc)
                if not is_valid(pos2):
                    edge_dist = min(edge_dist, 1)

            if edge_dist <= 1:
                score += 1

        return score

    return float(threats(player) - threats(opp))


def mobility(player_moves: List[Move], opp_moves: List[Move]) -> float:
    """Legal move difference."""
    return float(len(player_moves) - len(opp_moves))


def stability(board: Board, player: int) -> float:
    """Pieces supported by friendly neighbors."""
    opp = _opponent(player)

    def stable(color):
        marbles = set(board.get_marbles(color))
        count = 0

        for r, c in marbles:
            support = 0
            for dr, dc in DIRECTIONS:
                if (r + dr, c + dc) in marbles:
                    support += 1
            if support >= 2:
                count += 1

        return count

    return float(stable(player) - stable(opp))


def evaluate_with_weights(board: Board, player: int, weights: Dict[str, float]) -> float:
    """Score a board using a weighted combination of shared heuristic features."""
    opp = _opponent(player)
    player_moves = generate_legal_moves(board, player)
    opp_moves = generate_legal_moves(board, opp)

    return (
        weights["marble"] * marble_advantage(board, player)
        + weights["center"] * center_control(board, player)
        + weights["cohesion"] * cohesion(board, player)
        + weights["cluster"] * largest_cluster(board, player)
        + weights["edge"] * edge_risk(board, player)
        + weights["formation"] * formation_strength(board, player)
        + weights["push"] * push_potential(board, player)
        + weights["threat"] * threat_pressure(board, player)
        + weights["mobility"] * mobility(player_moves, opp_moves)
        + weights["stability"] * stability(board, player)
    )


def build_weighted_evaluator(weights: Dict[str, float]) -> Callable[[Board, int], float]:
    """Create an evaluator callable from a shared weight dictionary."""
    resolved = dict(weights)

    def evaluator(board: Board, player: int) -> float:
        return evaluate_with_weights(board, player, resolved)

    return evaluator


def evaluate_board(board: Board, player: int) -> float:
    """Default balanced heuristic evaluation."""
    return evaluate_with_weights(board, player, DEFAULT_WEIGHTS)
