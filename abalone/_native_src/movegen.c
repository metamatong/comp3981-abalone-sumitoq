#include "common.h"

static int
check_single_raw(const BoardState *board, uint8_t marble, uint8_t dir_idx)
{
    uint8_t ahead = g_neighbors[marble][dir_idx];
    return ahead != INVALID_INDEX && board->cells[ahead] == EMPTY;
}

static int
check_inline_raw(const BoardState *board, uint8_t leading, int count, uint8_t dir_idx, int player, int opponent)
{
    int pushed_count = 0;
    uint8_t ahead = g_neighbors[leading][dir_idx];
    uint8_t pos;

    if (ahead == INVALID_INDEX) {
        return 0;
    }
    if (board->cells[ahead] == EMPTY) {
        return 1;
    }
    if (board->cells[ahead] == (uint8_t) player) {
        return 0;
    }

    pos = ahead;
    while (pos != INVALID_INDEX && board->cells[pos] == (uint8_t) opponent) {
        pushed_count += 1;
        pos = g_neighbors[pos][dir_idx];
    }
    if (pushed_count >= count) {
        return 0;
    }
    if (pos != INVALID_INDEX && board->cells[pos] != EMPTY) {
        return 0;
    }
    return 1;
}

static int
check_broadside_raw(const BoardState *board, const uint8_t *marbles, int count, uint8_t dir_idx)
{
    int idx;
    for (idx = 0; idx < count; ++idx) {
        uint8_t dest = g_neighbors[marbles[idx]][dir_idx];
        if (dest == INVALID_INDEX) {
            return 0;
        }
        if (board->cells[dest] != EMPTY) {
            return 0;
        }
    }
    return 1;
}

static int
is_generated_move_legal_raw_native(const BoardState *board, const uint8_t *marbles, int count, uint8_t dir_idx, int player)
{
    if (count <= 1) {
        return check_single_raw(board, marbles[0], dir_idx);
    }

    {
        int line_dir = dir_index_from_delta(
            (int) g_rows[marbles[1]] - (int) g_rows[marbles[0]],
            (int) g_cols[marbles[1]] - (int) g_cols[marbles[0]]
        );
        if (line_dir >= 0 && (dir_idx == (uint8_t) line_dir || dir_idx == OPPOSITE_DIR[line_dir])) {
            int opponent = player == BLACK ? WHITE : BLACK;
            uint8_t leading = dir_idx == (uint8_t) line_dir ? marbles[count - 1] : marbles[0];
            return check_inline_raw(board, leading, count, dir_idx, player, opponent);
        }
    }

    return check_broadside_raw(board, marbles, count, dir_idx);
}

static uint32_t
move_key(uint8_t dir_idx, const uint8_t *marbles, int count)
{
    uint32_t m0 = marbles[0];
    uint32_t m1 = count > 1 ? marbles[1] : 61;
    uint32_t m2 = count > 2 ? marbles[2] : 61;
    return (uint32_t) dir_idx + 6U * (m0 + 62U * (m1 + 62U * (m2 + 62U * (uint32_t) (count - 1))));
}

static int
seen_insert(uint32_t *keys, uint8_t *used, size_t capacity, uint32_t key)
{
    size_t idx = ((size_t) key * 2654435761U) & (capacity - 1);
    while (used[idx]) {
        if (keys[idx] == key) {
            return 0;
        }
        idx = (idx + 1) & (capacity - 1);
    }
    used[idx] = 1;
    keys[idx] = key;
    return 1;
}

static int
add_generated_move(
    const BoardState *board,
    int player,
    const uint8_t *marbles,
    int count,
    uint8_t dir_idx,
    uint32_t *seen_keys,
    uint8_t *seen_used,
    NativeMove *moves,
    int *move_count
)
{
    uint8_t canonical[3] = {0, 0, 0};
    uint32_t key;
    NativeMove move;
    int idx;

    for (idx = 0; idx < count; ++idx) {
        canonical[idx] = marbles[idx];
    }
    canonicalize_indices(canonical, count);
    key = move_key(dir_idx, canonical, count);
    if (!seen_insert(seen_keys, seen_used, 512U, key)) {
        return 0;
    }
    if (!is_generated_move_legal_raw_native(board, canonical, count, dir_idx, player)) {
        return 0;
    }
    if (*move_count >= MAX_MOVES) {
        return -1;
    }
    build_move(canonical, count, dir_idx, &move);
    moves[*move_count] = move;
    *move_count += 1;
    return 0;
}

int
list_marbles(const BoardState *board, int player, uint8_t *marbles)
{
    int idx;
    int count = 0;
    for (idx = 0; idx < CELL_COUNT; ++idx) {
        if (board->cells[idx] == (uint8_t) player) {
            marbles[count++] = (uint8_t) idx;
        }
    }
    return count;
}

int
generate_legal_moves_native(const BoardState *board, int player, NativeMove *moves)
{
    uint8_t marbles[14];
    uint32_t seen_keys[512];
    uint8_t seen_used[512];
    uint64_t player_bits = board->bits[player];
    int marble_count = list_marbles(board, player, marbles);
    int move_count = 0;
    int marble_idx;

    memset(seen_keys, 0, sizeof(seen_keys));
    memset(seen_used, 0, sizeof(seen_used));

    for (marble_idx = 0; marble_idx < marble_count; ++marble_idx) {
        uint8_t marble = marbles[marble_idx];
        int ref_idx;
        for (ref_idx = 0; ref_idx < DIR_COUNT; ++ref_idx) {
            uint8_t dir_idx = REFERENCE_DIRS[ref_idx];
            uint8_t candidate[3];
            uint8_t fwd1;
            uint8_t fwd2;

            candidate[0] = marble;
            if (add_generated_move(board, player, candidate, 1, dir_idx, seen_keys, seen_used, moves, &move_count) < 0) {
                return -1;
            }

            fwd1 = g_neighbors[marble][dir_idx];
            if (fwd1 == INVALID_INDEX || !(player_bits & bit_for(fwd1))) {
                continue;
            }

            candidate[1] = fwd1;
            if (add_generated_move(board, player, candidate, 2, dir_idx, seen_keys, seen_used, moves, &move_count) < 0) {
                return -1;
            }

            fwd2 = g_neighbors[fwd1][dir_idx];
            if (fwd2 == INVALID_INDEX || !(player_bits & bit_for(fwd2))) {
                continue;
            }

            candidate[2] = fwd2;
            if (add_generated_move(board, player, candidate, 3, dir_idx, seen_keys, seen_used, moves, &move_count) < 0) {
                return -1;
            }
        }
    }

    for (marble_idx = 0; marble_idx < marble_count; ++marble_idx) {
        uint8_t marble = marbles[marble_idx];
        int line_dir_idx;
        for (line_dir_idx = 0; line_dir_idx < 3; ++line_dir_idx) {
            uint8_t line_dir = POSITIVE_DIRS[line_dir_idx];
            uint8_t second = g_neighbors[marble][line_dir];
            int ref_idx;

            if (second == INVALID_INDEX || !(player_bits & bit_for(second))) {
                continue;
            }

            for (ref_idx = 0; ref_idx < DIR_COUNT; ++ref_idx) {
                uint8_t dir_idx = REFERENCE_DIRS[ref_idx];
                uint8_t pair[2] = {marble, second};
                if (add_generated_move(board, player, pair, 2, dir_idx, seen_keys, seen_used, moves, &move_count) < 0) {
                    return -1;
                }
            }

            {
                uint8_t third = g_neighbors[second][line_dir];
                if (third == INVALID_INDEX || !(player_bits & bit_for(third))) {
                    continue;
                }
                for (ref_idx = 0; ref_idx < DIR_COUNT; ++ref_idx) {
                    uint8_t dir_idx = REFERENCE_DIRS[ref_idx];
                    uint8_t triple[3] = {marble, second, third};
                    if (add_generated_move(board, player, triple, 3, dir_idx, seen_keys, seen_used, moves, &move_count) < 0) {
                        return -1;
                    }
                }
            }
        }
    }

    return move_count;
}

static int
is_push_move(const BoardState *board, int player, const NativeMove *move)
{
    int opponent = player == BLACK ? WHITE : BLACK;
    uint8_t ahead;
    if (!move->is_inline || move->count < 2) {
        return 0;
    }
    ahead = g_neighbors[move->leading][move->dir_idx];
    return ahead != INVALID_INDEX && board->cells[ahead] == (uint8_t) opponent;
}

static int
compare_ordered_move(
    const BoardState *board,
    int player,
    const NativeMove *left,
    const NativeMove *right,
    const NativeMove *tt_move,
    const NativeMove *killer_move
)
{
    int left_tt = move_has_value(tt_move) && move_equal(left, tt_move);
    int right_tt = move_has_value(tt_move) && move_equal(right, tt_move);
    if (left_tt != right_tt) {
        return left_tt ? -1 : 1;
    }

    {
        int left_killer = move_has_value(killer_move) && move_equal(left, killer_move);
        int right_killer = move_has_value(killer_move) && move_equal(right, killer_move);
        if (left_killer != right_killer) {
            return left_killer ? -1 : 1;
        }
    }

    {
        int left_push = is_push_move(board, player, left);
        int right_push = is_push_move(board, player, right);
        if (left_push != right_push) {
            return left_push ? -1 : 1;
        }
    }

    if (left->count != right->count) {
        return left->count > right->count ? -1 : 1;
    }

    return compare_ordering_key_tail(left, right);
}

void
order_moves(
    const BoardState *board,
    int player,
    NativeMove *moves,
    int count,
    const NativeMove *tt_move,
    const NativeMove *killer_move
)
{
    int idx;
    for (idx = 1; idx < count; ++idx) {
        NativeMove current = moves[idx];
        int pos = idx - 1;
        while (pos >= 0 && compare_ordered_move(board, player, &current, &moves[pos], tt_move, killer_move) < 0) {
            moves[pos + 1] = moves[pos];
            pos -= 1;
        }
        moves[pos + 1] = current;
    }
}
