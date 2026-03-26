"""Jonah's heuristic configuration."""

from ....ai.heuristics import build_weighted_evaluator

# Starts from the shared baseline. Jonah can tune only this file later.
JONAH_WEIGHTS = {
    # Material
    
    # Marble count difference: your number of marbles minus the opponent's.
    # This is the material term, so losing a marble changes the score sharply.
    "marble": 40000.0,  

    # Positional control

    # Center control sums squared distance from the board center for both sides.
    # It returns opponent distance minus your distance, so central marbles score better.
    "center": 500.0,    
    # Mobility approximates freedom by counting empty valid neighboring spaces per marble.
    # It is a fast proxy for legal move options rather than full move generation.
    "mobility": 20.0,   

    # Structure and coordination

    # Cohesion counts friendly adjacencies in the six board directions.
    # The score is your total touching-neighbor count minus the opponent's.
    "cohesion": 100.0,  
    # Cluster measures the size of each side's largest connected marble group.
    # It rewards keeping one big connected cluster instead of split fragments.
    "cluster": 30.0,    
    # Formation strength looks for inline friendly 2- and 3-marble chains.
    # Longer straight formations score more because they are useful for strong moves and pushes.
    "formation": 120.0, 

    # Defensive resilience

    # Edge risk marks marbles on the edge or one step from it as more vulnerable.
    # The feature is opponent edge-risk minus your edge-risk, so safer shapes score higher.
    "edge": 40.0,       
    # Stability counts marbles that have at least two friendly neighbors.
    # That favors well-supported pieces that are harder to isolate.
    "stability": 30.0,  

    # Aggressive pressure

    # Push potential scans for inline friendly groups with an enemy marble directly ahead.
    # Bigger aligned groups score more because they represent stronger push chances.
    "push": 120.0,      
    # Threat counts opponent marbles that are on the edge or one step from it.
    # In the current code this is exposure-based, not a full check for actual friendly pressure.
    "threat": 100.0,    
}

evaluate_jonah = build_weighted_evaluator(JONAH_WEIGHTS)
