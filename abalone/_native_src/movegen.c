/* Generates legal moves and applies deterministic native move ordering. */
#include "common.h"

/* Checks whether a single marble can slide into the requested neighbor cell. */
static int
can_move_single(const BoardState *board, uint8_t marble, uint8_t dir_idx)
{
    uint8_t ahead = g_neighbors[marble][dir_idx];
    return ahead != INVALID_INDEX && board->cells[ahead] == EMPTY;
}

/* Validates an inline move, including legal sumito pushes. */
static int
can_move_inline(const BoardState *board, uint8_t leading, int count, uint8_t dir_idx, int player, int opponent)
{
    int pushed_count = 0;
    uint8_t ahead = g_neighbors[leading][dir_idx];
    uint8_t scan_pos;

    if (ahead == INVALID_INDEX) {
        return 0;
    }
    if (board->cells[ahead] == EMPTY) {
        return 1;
    }
    if (board->cells[ahead] == (uint8_t) player) {
        return 0;
    }

    scan_pos = ahead;
    while (scan_pos != INVALID_INDEX && board->cells[scan_pos] == (uint8_t) opponent) {
        pushed_count += 1;
        scan_pos = g_neighbors[scan_pos][dir_idx];
    }
    if (pushed_count >= count) {
        return 0;
    }
    if (scan_pos != INVALID_INDEX && board->cells[scan_pos] != EMPTY) {
        return 0;
    }
    return 1;
}

/* Validates a broadside move by checking that every destination cell is empty. */
static int
can_move_broadside(const BoardState *board, const uint8_t *marbles, int count, uint8_t dir_idx)
{
    int marble_idx;
    for (marble_idx = 0; marble_idx < count; ++marble_idx) {
        uint8_t dest = g_neighbors[marbles[marble_idx]][dir_idx];
        if (dest == INVALID_INDEX) {
            return 0;
        }
        if (board->cells[dest] != EMPTY) {
            return 0;
        }
    }
    return 1;
}

/* Dispatches a generated move candidate to the correct legality check. */
static int
is_generated_move_legal(const BoardState *board, const uint8_t *marbles, int count, uint8_t dir_idx, int player)
{
    if (count <= 1) {
        return can_move_single(board, marbles[0], dir_idx);
    }

    {
        int line_dir = dir_index_from_delta(
            (int) g_rows[marbles[1]] - (int) g_rows[marbles[0]],
            (int) g_cols[marbles[1]] - (int) g_cols[marbles[0]]
        );
        if (line_dir >= 0 && (dir_idx == (uint8_t) line_dir || dir_idx == OPPOSITE_DIR[line_dir])) {
            int opponent = player == BLACK ? WHITE : BLACK;
            uint8_t leading = dir_idx == (uint8_t) line_dir ? marbles[count - 1] : marbles[0];
            return can_move_inline(board, leading, count, dir_idx, player, opponent);
        }
    }

    return can_move_broadside(board, marbles, count, dir_idx);
}

/* Encodes a canonical move into a compact deduplication key. */
static uint32_t
encode_move_key(uint8_t dir_idx, const uint8_t *marbles, int count)
{
    uint32_t m0 = marbles[0];
    uint32_t m1 = count > 1 ? marbles[1] : 61;
    uint32_t m2 = count > 2 ? marbles[2] : 61;
    return (uint32_t) dir_idx + 6U * (m0 + 62U * (m1 + 62U * (m2 + 62U * (uint32_t) (count - 1))));
}

/* Inserts a move key into the local open-addressed dedupe table. */
static int
insert_seen_key(uint32_t *keys, uint8_t *used, size_t capacity, uint32_t key)
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

/* Canonicalizes, validates, and appends a generated move if it is new. */
static int
append_generated_move(
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
    int marble_idx;

    for (marble_idx = 0; marble_idx < count; ++marble_idx) {
        canonical[marble_idx] = marbles[marble_idx];
    }
    canonicalize_indices(canonical, count);
    key = encode_move_key(dir_idx, canonical, count);
    if (!insert_seen_key(seen_keys, seen_used, 512U, key)) {
        return 0;
    }
    if (!is_generated_move_legal(board, canonical, count, dir_idx, player)) {
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

/* Collects the current player's marble indices into a compact array. */
int
list_marbles(const BoardState *board, int player, uint8_t *marbles)
{
    int cell_idx;
    int count = 0;
    for (cell_idx = 0; cell_idx < CELL_COUNT; ++cell_idx) {
        if (board->cells[cell_idx] == (uint8_t) player) {
            marbles[count++] = (uint8_t) cell_idx;
        }
    }
    return count;
}

/* Generates every legal move for the requested player without duplicates. */
int
generate_legal_moves_native(const BoardState *board, int player, NativeMove *moves)
{
    uint8_t marbles[14];
    uint32_t seen_keys[512];
    uint8_t seen_used[512];
    uint64_t player_bits = board->bits[player];
    int marble_count = list_marbles(board, player, marbles);
    int move_count = 0;
    int marble_list_index;

    memset(seen_keys, 0, sizeof(seen_keys));
    memset(seen_used, 0, sizeof(seen_used));

    for (marble_list_index = 0; marble_list_index < marble_count; ++marble_list_index) {
        uint8_t marble = marbles[marble_list_index];
        int ref_dir_index;
        for (ref_dir_index = 0; ref_dir_index < DIR_COUNT; ++ref_dir_index) {
            uint8_t dir_idx = REFERENCE_DIRS[ref_dir_index];
            uint8_t candidate[3];
            uint8_t second_marble;
            uint8_t third_marble;

            candidate[0] = marble;
            if (append_generated_move(board, player, candidate, 1, dir_idx, seen_keys, seen_used, moves, &move_count) < 0) {
                return -1;
            }

            second_marble = g_neighbors[marble][dir_idx];
            if (second_marble == INVALID_INDEX || !(player_bits & bit_for(second_marble))) {
                continue;
            }

            candidate[1] = second_marble;
            if (append_generated_move(board, player, candidate, 2, dir_idx, seen_keys, seen_used, moves, &move_count) < 0) {
                return -1;
            }

            third_marble = g_neighbors[second_marble][dir_idx];
            if (third_marble == INVALID_INDEX || !(player_bits & bit_for(third_marble))) {
                continue;
            }

            candidate[2] = third_marble;
            if (append_generated_move(board, player, candidate, 3, dir_idx, seen_keys, seen_used, moves, &move_count) < 0) {
                return -1;
            }
        }
    }

    for (marble_list_index = 0; marble_list_index < marble_count; ++marble_list_index) {
        uint8_t marble = marbles[marble_list_index];
        int line_dir_idx;
        for (line_dir_idx = 0; line_dir_idx < 3; ++line_dir_idx) {
            uint8_t line_dir = POSITIVE_DIRS[line_dir_idx];
            uint8_t second = g_neighbors[marble][line_dir];
            int ref_dir_index;

            if (second == INVALID_INDEX || !(player_bits & bit_for(second))) {
                continue;
            }

            for (ref_dir_index = 0; ref_dir_index < DIR_COUNT; ++ref_dir_index) {
                uint8_t dir_idx = REFERENCE_DIRS[ref_dir_index];
                uint8_t pair[2] = {marble, second};
                if (append_generated_move(board, player, pair, 2, dir_idx, seen_keys, seen_used, moves, &move_count) < 0) {
                    return -1;
                }
            }

            {
                uint8_t third = g_neighbors[second][line_dir];
                if (third == INVALID_INDEX || !(player_bits & bit_for(third))) {
                    continue;
                }
                for (ref_dir_index = 0; ref_dir_index < DIR_COUNT; ++ref_dir_index) {
                    uint8_t dir_idx = REFERENCE_DIRS[ref_dir_index];
                    uint8_t triple[3] = {marble, second, third};
                    if (append_generated_move(board, player, triple, 3, dir_idx, seen_keys, seen_used, moves, &move_count) < 0) {
                        return -1;
                    }
                }
            }
        }
    }

    return move_count;
}

/* Reports whether a move begins by pushing an opposing marble inline. */
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

/* Compares two moves under TT, killer, push, and deterministic fallback priorities. */
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

/* Sorts moves in place so alpha-beta sees the most promising ones first. */
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
    int move_index;
    for (move_index = 1; move_index < count; ++move_index) {
        NativeMove current_move = moves[move_index];
        int insert_pos = move_index - 1;
        while (insert_pos >= 0 &&
                compare_ordered_move(board, player, &current_move, &moves[insert_pos], tt_move, killer_move) < 0) {
            moves[insert_pos + 1] = moves[insert_pos];
            insert_pos -= 1;
        }
        moves[insert_pos + 1] = current_move;
    }
}
