"""Jonah's heuristic configuration."""

from ....ai.heuristics import build_weighted_evaluator

# Starts from the shared baseline. Jonah can tune only this file later.
JONAH_WEIGHTS = {
    "marble": 50000.0,    # Material advantage (number of marbles)
    "center": 450.0,      # Distance to the center of the board
    "cohesion": 100.0,    # Grouping of marbles together
    "cluster": 30.0,      # Number of contiguous marble groups
    "edge": 40.0,         # Distance from the edge of the board
    "formation": 120.0,   # Maintaining strong defensive shapes
    "push": 120.0,        # Ability to push opponent marbles
    "threat": 100.0,      # Direct threats to push opponent off board
    "mobility": 20.0,     # Number of available legal moves
    "stability": 30.0,    # How safe marbles are from being pushed
}

evaluate_jonah = build_weighted_evaluator(JONAH_WEIGHTS)
