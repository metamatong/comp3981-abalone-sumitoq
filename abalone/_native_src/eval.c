#include "common.h"

static int
center_distance_sum(const uint8_t *marbles, int count)
{
    int total = 0;
    int idx;
    for (idx = 0; idx < count; ++idx) {
        int dr = (int) g_rows[marbles[idx]] - 4;
        int dc = (int) g_cols[marbles[idx]] - 5;
        int dist;
        if ((dr >= 0 && dc >= 0) || (dr <= 0 && dc <= 0)) {
            dist = abs(dr) > abs(dc) ? abs(dr) : abs(dc);
        } else {
            dist = abs(dr) + abs(dc);
        }
        total += dist * dist;
    }
    return total;
}

static void
structure_profile(uint64_t bits, int *adjacency, int *largest_cluster_size, int *stable_count)
{
    uint64_t visited = 0;
    int idx;
    *adjacency = 0;
    *largest_cluster_size = 0;
    *stable_count = 0;

    for (idx = 0; idx < CELL_COUNT; ++idx) {
        uint64_t bit = bit_for((uint8_t) idx);
        int component_size = 0;
        uint8_t queue[14];
        int head = 0;
        int tail = 0;
        if (!(bits & bit) || (visited & bit)) {
            continue;
        }

        queue[tail++] = (uint8_t) idx;
        visited |= bit;

        while (head < tail) {
            uint8_t current = queue[head++];
            int support = 0;
            int dir;
            component_size += 1;

            for (dir = 0; dir < DIR_COUNT; ++dir) {
                uint8_t neighbor = g_neighbors[current][dir];
                if (neighbor == INVALID_INDEX) {
                    continue;
                }
                if (!(bits & bit_for(neighbor))) {
                    continue;
                }
                *adjacency += 1;
                support += 1;
                if (!(visited & bit_for(neighbor))) {
                    visited |= bit_for(neighbor);
                    queue[tail++] = neighbor;
                }
            }

            if (support >= 2) {
                *stable_count += 1;
            }
        }

        if (component_size > *largest_cluster_size) {
            *largest_cluster_size = component_size;
        }
    }
}

static void
edge_profile(const uint8_t *marbles, int count, int *risk_points, int *pressure_points)
{
    int idx;
    *risk_points = 0;
    *pressure_points = 0;
    for (idx = 0; idx < count; ++idx) {
        *risk_points += g_edge_risk[marbles[idx]];
        *pressure_points += g_edge_pressure[marbles[idx]];
    }
}

static int
formation_score(uint64_t bits)
{
    int idx;
    int score = 0;
    for (idx = 0; idx < CELL_COUNT; ++idx) {
        int dir;
        if (!(bits & bit_for((uint8_t) idx))) {
            continue;
        }
        for (dir = 0; dir < DIR_COUNT; ++dir) {
            uint8_t m1 = g_neighbors[idx][dir];
            if (m1 == INVALID_INDEX || !(bits & bit_for(m1))) {
                continue;
            }
            score += 1;
            {
                uint8_t m2 = g_neighbors[m1][dir];
                if (m2 != INVALID_INDEX && (bits & bit_for(m2))) {
                    score += 3;
                }
            }
        }
    }
    return score;
}

static int
push_score(uint64_t bits, uint64_t opponent_bits)
{
    int idx;
    int score = 0;
    for (idx = 0; idx < CELL_COUNT; ++idx) {
        int dir;
        if (!(bits & bit_for((uint8_t) idx))) {
            continue;
        }
        for (dir = 0; dir < DIR_COUNT; ++dir) {
            uint8_t m1 = g_neighbors[idx][dir];
            uint8_t m2;
            if (m1 == INVALID_INDEX || !(bits & bit_for(m1))) {
                continue;
            }
            m2 = g_neighbors[m1][dir];
            if (m2 != INVALID_INDEX && (opponent_bits & bit_for(m2))) {
                score += 2;
            } else if (m2 != INVALID_INDEX && (bits & bit_for(m2))) {
                uint8_t m3 = g_neighbors[m2][dir];
                if (m3 != INVALID_INDEX && (opponent_bits & bit_for(m3))) {
                    score += 3;
                }
            }
        }
    }
    return score;
}

static int
mobility_score(uint64_t player_bits, uint64_t opponent_bits)
{
    uint64_t occupied = player_bits | opponent_bits;
    int idx;
    int score = 0;
    for (idx = 0; idx < CELL_COUNT; ++idx) {
        int dir;
        if (!(player_bits & bit_for((uint8_t) idx))) {
            continue;
        }
        for (dir = 0; dir < DIR_COUNT; ++dir) {
            uint8_t neighbor = g_neighbors[idx][dir];
            if (neighbor != INVALID_INDEX && !(occupied & bit_for(neighbor))) {
                score += 1;
            }
        }
    }
    return score;
}

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

    structure_profile(board->bits[player], &player_adjacency, &player_cluster, &player_stability);
    structure_profile(board->bits[opponent], &opponent_adjacency, &opponent_cluster, &opponent_stability);
    edge_profile(player_marbles, player_count, &player_risk, &player_pressure);
    edge_profile(opponent_marbles, opponent_count, &opponent_risk, &opponent_pressure);

    total += weights[0] * (double) (player_count - opponent_count);
    total += weights[1] * (double) (center_distance_sum(opponent_marbles, opponent_count) - center_distance_sum(player_marbles, player_count));
    total += weights[2] * (double) (player_adjacency - opponent_adjacency);
    total += weights[3] * (double) (player_cluster - opponent_cluster);
    total += weights[4] * (double) ((opponent_risk + opponent_pressure) - (player_risk + player_pressure));
    total += weights[5] * (double) (formation_score(board->bits[player]) - formation_score(board->bits[opponent]));
    total += weights[6] * (double) (push_score(board->bits[player], board->bits[opponent]) - push_score(board->bits[opponent], board->bits[player]));
    total += weights[7] * (double) (mobility_score(board->bits[player], board->bits[opponent]) - mobility_score(board->bits[opponent], board->bits[player]));
    total += weights[8] * (double) (player_stability - opponent_stability);
    return total;
}
