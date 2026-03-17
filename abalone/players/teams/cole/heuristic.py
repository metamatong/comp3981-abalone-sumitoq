"""Cole's heuristic configuration."""

from ....ai.heuristics import build_weighted_evaluator

# Starts from the shared baseline. Cole can tune only this file later.
COLE_WEIGHTS = {
    "marble": 50000.0,    # Material advantage (number of marbles)
    "center": 40.0,       # Distance to the center of the board
    "cohesion": 35.0,     # Grouping of marbles together
    "cluster": 30.0,      # Number of contiguous marble groups
    "edge": 60.0,         # Distance from the edge of the board
    "formation": 50.0,    # Maintaining strong defensive shapes
    "push": 120.0,        # Ability to push opponent marbles
    "threat": 80.0,       # Direct threats to push opponent off board
    "mobility": 20.0,     # Number of available legal moves
    "stability": 25.0,    # How safe marbles are from being pushed
}

evaluate_cole = build_weighted_evaluator(COLE_WEIGHTS)
