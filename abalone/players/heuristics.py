"""Heuristic evaluation for minimax search."""

from ..game.board import BLACK, WHITE, Board
from ..state_space import generate_legal_moves

CENTER = (4, 5)

HEURISTIC_PRESETS = {
    "balanced": {
        "score": 1400.0,
        "material": 40.0,
        "center": 4.0,
        "mobility": 1.0,
    },
    "material": {
        "score": 1800.0,
        "material": 50.0,
        "center": 0.0,
        "mobility": 0.0,
    },
}


def _opponent(player: int) -> int:
    """Return the opposing color constant."""
    return WHITE if player == BLACK else BLACK


def score_advantage(board: Board, player: int) -> float:
    """Captured-marble score difference from `player` perspective."""
    opp = _opponent(player)
    return float(board.score[player] - board.score[opp])


def material_advantage(board: Board, player: int) -> float:
    """On-board marble-count difference from `player` perspective."""
    opp = _opponent(player)
    return float(board.marble_count(player) - board.marble_count(opp))


def center_control(board: Board, player: int) -> float:
    """Prefer positions where `player` marbles are closer to board center."""
    opp = _opponent(player)

    def center_sum(color: int) -> int:
        """Total Manhattan-like distance from each marble to the board center."""
        total = 0
        for r, c in board.get_marbles(color):
            total += abs(r - CENTER[0]) + abs(c - CENTER[1])
        return total

    return float(center_sum(opp) - center_sum(player))


def mobility(board: Board, player: int) -> float:
    """Legal-move count difference between player and opponent."""
    opp = _opponent(player)
    return float(len(generate_legal_moves(board, player)) - len(generate_legal_moves(board, opp)))


def evaluate_board(board: Board, player: int, preset: str = "balanced") -> float:
    """Weighted sum heuristic used by minimax to evaluate non-terminal states."""
    weights = HEURISTIC_PRESETS.get(preset, HEURISTIC_PRESETS["balanced"])
    return (
        weights["score"] * score_advantage(board, player)
        + weights["material"] * material_advantage(board, player)
        + weights["center"] * center_control(board, player)
        + weights["mobility"] * mobility(board, player)
    )
