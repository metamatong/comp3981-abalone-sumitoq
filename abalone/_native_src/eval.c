/* Computes heuristic feature scores for the native minimax evaluator. */
#include "common.h"

/* Sums squared distances from the board center for a list of marbles. */
static int
sum_squared_center_distance(const uint8_t *marbles, int count)
{
    int total = 0;
    int marble_idx;
    for (marble_idx = 0; marble_idx < count; ++marble_idx) {
        int row_delta = (int) g_rows[marbles[marble_idx]] - 4;
        int col_delta = (int) g_cols[marbles[marble_idx]] - 5;
        int dist;
        if ((row_delta >= 0 && col_delta >= 0) || (row_delta <= 0 && col_delta <= 0)) {
            dist = abs(row_delta) > abs(col_delta) ? abs(row_delta) : abs(col_delta);
        } else {
            dist = abs(row_delta) + abs(col_delta);
        }
        total += dist * dist;
    }
    return total;
}

/* Measures connectivity, component size, and local support for one side. */
static void
compute_structure_profile(uint64_t player_bits, int *adjacency, int *largest_cluster_size, int *stable_count)
{
    uint64_t visited = 0;
    int cell_idx;
    *adjacency = 0;
    *largest_cluster_size = 0;
    *stable_count = 0;

    for (cell_idx = 0; cell_idx < CELL_COUNT; ++cell_idx) {
        uint64_t bit = bit_for((uint8_t) cell_idx);
        int component_size = 0;
        uint8_t queue[14];
        int head = 0;
        int tail = 0;
        if (!(player_bits & bit) || (visited & bit)) {
            continue;
        }

        queue[tail++] = (uint8_t) cell_idx;
        visited |= bit;

        while (head < tail) {
            uint8_t current_cell = queue[head++];
            int friendly_neighbors = 0;
            int direction_idx;
            component_size += 1;

            for (direction_idx = 0; direction_idx < DIR_COUNT; ++direction_idx) {
                uint8_t neighbor = g_neighbors[current_cell][direction_idx];
                if (neighbor == INVALID_INDEX) {
                    continue;
                }
                if (!(player_bits & bit_for(neighbor))) {
                    continue;
                }
                *adjacency += 1;
                friendly_neighbors += 1;
                if (!(visited & bit_for(neighbor))) {
                    visited |= bit_for(neighbor);
                    queue[tail++] = neighbor;
                }
            }

            if (friendly_neighbors >= 2) {
                *stable_count += 1;
            }
        }

        if (component_size > *largest_cluster_size) {
            *largest_cluster_size = component_size;
        }
    }
}

/* Accumulates precomputed edge danger and pressure values for a marble list. */
static void
accumulate_edge_profile(const uint8_t *marbles, int count, int *risk_points, int *pressure_points)
{
    int marble_idx;
    *risk_points = 0;
    *pressure_points = 0;
    for (marble_idx = 0; marble_idx < count; ++marble_idx) {
        *risk_points += g_edge_risk[marbles[marble_idx]];
        *pressure_points += g_edge_pressure[marbles[marble_idx]];
    }
}

/* Rewards aligned pairs and triples that create stronger formations. */
static int
score_formations(uint64_t player_bits)
{
    int cell_idx;
    int score = 0;
    for (cell_idx = 0; cell_idx < CELL_COUNT; ++cell_idx) {
        int direction_idx;
        if (!(player_bits & bit_for((uint8_t) cell_idx))) {
            continue;
        }
        for (direction_idx = 0; direction_idx < DIR_COUNT; ++direction_idx) {
            uint8_t neighbor1 = g_neighbors[cell_idx][direction_idx];
            if (neighbor1 == INVALID_INDEX || !(player_bits & bit_for(neighbor1))) {
                continue;
            }
            score += 1;
            {
                uint8_t neighbor2 = g_neighbors[neighbor1][direction_idx];
                if (neighbor2 != INVALID_INDEX && (player_bits & bit_for(neighbor2))) {
                    score += 3;
                }
            }
        }
    }
    return score;
}

/* Counts immediate and stronger multi-marble push threats. */
static int
score_push_pressure(uint64_t player_bits, uint64_t opponent_bits)
{
    int cell_idx;
    int score = 0;
    for (cell_idx = 0; cell_idx < CELL_COUNT; ++cell_idx) {
        int direction_idx;
        if (!(player_bits & bit_for((uint8_t) cell_idx))) {
            continue;
        }
        for (direction_idx = 0; direction_idx < DIR_COUNT; ++direction_idx) {
            uint8_t neighbor1 = g_neighbors[cell_idx][direction_idx];
            uint8_t neighbor2;
            if (neighbor1 == INVALID_INDEX || !(player_bits & bit_for(neighbor1))) {
                continue;
            }
            neighbor2 = g_neighbors[neighbor1][direction_idx];
            if (neighbor2 != INVALID_INDEX && (opponent_bits & bit_for(neighbor2))) {
                score += 2;
            } else if (neighbor2 != INVALID_INDEX && (player_bits & bit_for(neighbor2))) {
                uint8_t neighbor3 = g_neighbors[neighbor2][direction_idx];
                if (neighbor3 != INVALID_INDEX && (opponent_bits & bit_for(neighbor3))) {
                    score += 3;
                }
            }
        }
    }
    return score;
}

/* Counts open neighboring cells as a simple mobility estimate. */
static int
score_mobility(uint64_t player_bits, uint64_t opponent_bits)
{
    uint64_t occupied = player_bits | opponent_bits;
    int cell_idx;
    int score = 0;
    for (cell_idx = 0; cell_idx < CELL_COUNT; ++cell_idx) {
        int direction_idx;
        if (!(player_bits & bit_for((uint8_t) cell_idx))) {
            continue;
        }
        for (direction_idx = 0; direction_idx < DIR_COUNT; ++direction_idx) {
            uint8_t neighbor = g_neighbors[cell_idx][direction_idx];
            if (neighbor != INVALID_INDEX && !(occupied & bit_for(neighbor))) {
                score += 1;
            }
        }
    }
    return score;
}

/* Evaluates a board by combining the weighted feature differences for both sides. */
double
evaluate_weighted_native(const BoardState *board, int player, const double *weights)
{
    int opponent = player == BLACK ? WHITE : BLACK;
    uint8_t player_marbles[14];
    uint8_t opponent_marbles[14];
    int player_count = list_marbles(board, player, player_marbles);
    int opponent_count = list_marbles(board, opponent, opponent_marbles);
    int player_adjacency;
    int opponent_adjacency;
    int player_cluster;
    int opponent_cluster;
    int player_stability;
    int opponent_stability;
    int player_risk;
    int opponent_risk;
    int player_pressure;
    int opponent_pressure;
    double total = 0.0;

    compute_structure_profile(board->bits[player], &player_adjacency, &player_cluster, &player_stability);
    compute_structure_profile(board->bits[opponent], &opponent_adjacency, &opponent_cluster, &opponent_stability);
    accumulate_edge_profile(player_marbles, player_count, &player_risk, &player_pressure);
    accumulate_edge_profile(opponent_marbles, opponent_count, &opponent_risk, &opponent_pressure);

    total += weights[0] * (double) (player_count - opponent_count);
    total += weights[1] * (double) (
        sum_squared_center_distance(opponent_marbles, opponent_count) -
        sum_squared_center_distance(player_marbles, player_count)
    );
    total += weights[2] * (double) (player_adjacency - opponent_adjacency);
    total += weights[3] * (double) (player_cluster - opponent_cluster);
    total += weights[4] * (double) ((opponent_risk + opponent_pressure) - (player_risk + player_pressure));
    total += weights[5] * (double) (score_formations(board->bits[player]) - score_formations(board->bits[opponent]));
    total += weights[6] * (double) (
        score_push_pressure(board->bits[player], board->bits[opponent]) -
        score_push_pressure(board->bits[opponent], board->bits[player])
    );
    total += weights[7] * (double) (
        score_mobility(board->bits[player], board->bits[opponent]) -
        score_mobility(board->bits[opponent], board->bits[player])
    );
    total += weights[8] * (double) (player_stability - opponent_stability);
    return total;
}
