"""Tournament Belgian Daisy 1 heuristic."""

from ....ai.heuristics import build_weighted_evaluator

# Belgian Daisy: 80 total game moves, 10s per move.
# original belgian weights
TOURNAMENT_BELGIAN_1_WEIGHTS = {
    "marble": 49635.333,   
    "center": 107.150,     
    "cohesion": 85.425,    
    "cluster": 30.164,     
    "edge_pressure": 125.891,
    "formation": 103.780,  
    "push": 418.161,       
    "mobility": 30.245,    
    "stability": 63.321,   
}

evaluate_tournament_belgian_1 = build_weighted_evaluator(TOURNAMENT_BELGIAN_1_WEIGHTS)
