"""Cole's heuristic configuration."""

from ....ai.heuristics import build_weighted_evaluator

# Starts from the shared baseline. Cole can tune only this file later.
COLE_WEIGHTS = {
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
# TODO results are consistently different browser vs terminal

evaluate_cole = build_weighted_evaluator(COLE_WEIGHTS)
