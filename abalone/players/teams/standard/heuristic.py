"""Standard board tournament heuristic."""

from ....ai.heuristics import build_weighted_evaluator

# Standard board: 10 min game clock, 5s per move, no move limit.
STANDARD_WEIGHTS = {
    "marble": 20000.0,  # Material advantage (number of marbles)
    "center": 450.0,    # Distance to the center of the board
    "cohesion": 70.0,   # Grouping of marbles together
    "cluster": 50.0,    # Number of contiguous marble groups
    "edge_pressure": 130.0,  # Combined own edge exposure and opponent rim pressure
    "formation": 25.0,  # Maintaining strong defensive shapes
    "push": 300.0,      # Ability to push opponent marbles
    "mobility": 50.0,   # Number of available legal moves
    "stability": 20.0,  # How safe marbles are from being pushed
}

evaluate_standard = build_weighted_evaluator(STANDARD_WEIGHTS)
