"""Kyle's heuristic configuration."""

from ....ai.heuristics import build_weighted_evaluator

# Starts from the shared baseline. Kyle can tune only this file later.
KYLE_WEIGHTS = {
    "marble": 50000.0,   # Material advantage must stay dominant.
    "center": 85.0,      # Useful positional signal, but no longer dominates tactical terms.
    "cohesion": 70.0,    # Keep groups connected without over-constraining movement.
    "cluster": 25.0,     # Cohesion does most of the group-shape work.
    "edge": 82.0,        # Penalize exposure, but less aggressively than the prior depth-2 tuning.
    "formation": 88.0,   # Reward strong lines and wedges that deeper search can exploit.
    "push": 330.0,       # Restore meaningful offensive pressure for depth-3 play.
    "threat": 160.0,     # Keep edge-pressure conversion important.
    "mobility": 38.0,    # More freedom helps avoid passive depth-3 lines.
    "stability": 52.0,   # Enough safety to avoid blunders without freezing the agent.
}

evaluate_kyle = build_weighted_evaluator(KYLE_WEIGHTS)
