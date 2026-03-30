"""Abdullah's heuristic center-focused and stable."""

from ....ai.heuristics import (
    marble_advantage,
    center_control,
    cohesion,
    edge_risk,
    formation_strength,
    push_potential,
    threat_pressure,
    mobility,
    stability,
)
from ....game.board import BLACK, WHITE, DIRECTIONS, is_valid


ABDULLAH_WEIGHTS = {
    "opening": {
        "marble": 50000.0,
        "center": 165.0,
        "cohesion": 70.0,
        "edge": 90.0,
        "formation": 65.0,
        "push": 70.0,
        "threat": 60.0,
        "mobility": 55.0,
        "stability": 65.0,
        "own_edge_penalty": 120.0,
        "imminent_push": 2200.0,
        "imminent_threat": 1800.0,
    },
    "midgame": {
        "marble": 50000.0,
        "center": 115.0,
        "cohesion": 50.0,
        "edge": 95.0,
        "formation": 85.0,
        "push": 130.0,
        "threat": 120.0,
        "mobility": 40.0,
        "stability": 45.0,
        "own_edge_penalty": 150.0,
        "imminent_push": 4200.0,
        "imminent_threat": 3400.0,
    },
    "endgame": {
        "marble": 50000.0,
        "center": 70.0,
        "cohesion": 30.0,
        "edge": 110.0,
        "formation": 95.0,
        "push": 180.0,
        "threat": 170.0,
        "mobility": 25.0,
        "stability": 25.0,
        "own_edge_penalty": 190.0,
        "imminent_push": 7600.0,
        "imminent_threat": 6000.0,
    },
}


def _opponent(player: int) -> int:
    """Return the opposing player."""
    return WHITE if player == BLACK else BLACK


def _edge_distance(pos) -> int:
    """Return approximate edge distance: 0=edge, 1=near edge, 2=safer."""
    r = pos[0]
    c = pos[1]
    dist = 2

    for dr, dc in DIRECTIONS:
        nb1 = (r + dr, c + dc)
        if not is_valid(nb1):
            return 0

        nb2 = (nb1[0] + dr, nb1[1] + dc)
        if not is_valid(nb2):
            dist = min(dist, 1)

    return dist


def _phase(player_list, opp_list) -> str:
    """Estimate game phase from material loss."""
    lost_total = (14 - len(player_list)) + (14 - len(opp_list))

    if lost_total >= 4:
        return "endgame"

    if lost_total >= 2:
        return "midgame"

    return "opening"


def own_edge_penalty(player_marbles: set) -> float:
    """Positive penalty for our own marbles being too close to the edge."""
    score = 0.0

    for marble in player_marbles:
        d = _edge_distance(marble)

        if d == 0:
            score += 6.0
        elif d == 1:
            score += 2.0

    return score


def _find_pushoff_threats(attacker: set, defender: set) -> float:
    """Count one-move push-off threats."""
    score = 0.0

    for marble in defender:
        if _edge_distance(marble) != 0:
            continue

        r = marble[0]
        c = marble[1]

        for dr, dc in DIRECTIONS:
            if is_valid((r + dr, c + dc)):
                continue

            pdr = -dr
            pdc = -dc

            def_chain = []
            cur = (r, c)

            while cur in defender and len(def_chain) < 3:
                def_chain.append(cur)
                cur = (cur[0] + pdr, cur[1] + pdc)

            if not def_chain:
                continue

            att_chain = []
            cur = (def_chain[-1][0] + pdr, def_chain[-1][1] + pdc)

            while cur in attacker and len(att_chain) < 3:
                att_chain.append(cur)
                cur = (cur[0] + pdr, cur[1] + pdc)

            if len(att_chain) > len(def_chain):
                score += float(len(def_chain))

    return score


def _synergy_bonus(
    phase_name: str,
    own_center: float,
    own_push: float,
    opp_push: float,
    own_threat: float,
    opp_threat: float,
    own_imminent_threat: float,
    opp_imminent_threat: float,
) -> float:
    """Add a few controlled bonuses without making the evaluator noisy."""
    bonus = 0.0

    if own_center > 0 and own_push > opp_push:
        bonus += 80.0

    if own_center > 0 and own_threat > opp_threat:
        bonus += 60.0

    if phase_name == "opening" and own_center > 0:
        bonus += 50.0

    if phase_name == "endgame" and own_imminent_threat > 0:
        bonus += 140.0

    if opp_imminent_threat > own_imminent_threat:
        if phase_name == "endgame":
            bonus -= 240.0 * (opp_imminent_threat - own_imminent_threat)
        else:
            bonus -= 150.0 * (opp_imminent_threat - own_imminent_threat)

    return bonus


def evaluate_abdullah(board, player: int) -> float:
    """Evaluate the board using a center-focused and stable heuristic."""
    opp = _opponent(player)

    player_list = board.get_marbles(player)
    opp_list = board.get_marbles(opp)

    player_set = set(player_list)
    opp_set = set(opp_list)

    phase_name = _phase(player_list, opp_list)
    w = ABDULLAH_WEIGHTS[phase_name]

    own_marble = marble_advantage(player_list, opp_list)
    own_center = center_control(player_list, opp_list)
    own_cohesion = cohesion(player_set, opp_set)
    own_edge = edge_risk(player_list, opp_list)
    own_formation = formation_strength(player_set, opp_set)
    own_push = push_potential(player_set, opp_set)
    opp_push = push_potential(opp_set, player_set)
    own_threat = threat_pressure(player_set, opp_set)
    opp_threat = threat_pressure(opp_set, player_set)
    own_mobility = mobility(player_set, opp_set)
    own_stability = stability(player_set, opp_set)

    own_edge_exposure = own_edge_penalty(player_set)
    opp_edge_exposure = own_edge_penalty(opp_set)

    own_imminent_threat = _find_pushoff_threats(player_set, opp_set)
    opp_imminent_threat = _find_pushoff_threats(opp_set, player_set)

    score = 0.0

    score += w["marble"] * own_marble
    score += w["center"] * own_center
    score += w["cohesion"] * own_cohesion
    score += w["edge"] * own_edge
    score += w["formation"] * own_formation
    score += w["push"] * (own_push - opp_push)
    score += w["threat"] * (own_threat - opp_threat)
    score += w["mobility"] * own_mobility
    score += w["stability"] * own_stability

    score -= w["own_edge_penalty"] * own_edge_exposure
    score += w["own_edge_penalty"] * opp_edge_exposure

    score += w["imminent_threat"] * own_imminent_threat
    score -= w["imminent_push"] * opp_imminent_threat

    score += _synergy_bonus(
        phase_name,
        own_center,
        own_push,
        opp_push,
        own_threat,
        opp_threat,
        own_imminent_threat,
        opp_imminent_threat,
    )

    return score


evaluate_abdullah.weights = ABDULLAH_WEIGHTS