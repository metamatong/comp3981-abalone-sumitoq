#include "common.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

const int8_t DIR_DR[DIR_COUNT] = {0, 0, 1, -1, 1, -1};
const int8_t DIR_DC[DIR_COUNT] = {1, -1, 0, 0, 1, -1};
const uint8_t OPPOSITE_DIR[DIR_COUNT] = {1, 0, 3, 2, 5, 4};
const uint8_t REFERENCE_DIRS[DIR_COUNT] = {4, 0, 3, 5, 1, 2};
const uint8_t POSITIVE_DIRS[3] = {0, 2, 4};
const uint8_t DIR_NAME_RANK[DIR_COUNT] = {0, 5, 2, 3, 1, 4};

int g_tables_ready = 0;
int8_t g_pos_index[9][10];
uint8_t g_rows[CELL_COUNT];
uint8_t g_cols[CELL_COUNT];
uint8_t g_neighbors[CELL_COUNT][DIR_COUNT];
uint64_t g_adj_masks[CELL_COUNT];
uint8_t g_edge_risk[CELL_COUNT];
uint8_t g_edge_pressure[CELL_COUNT];
uint64_t g_zobrist[CELL_COUNT][3];
uint64_t g_side_zobrist[3];

static uint64_t
splitmix64_next(uint64_t *state)
{
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

void
init_tables(void)
{
    int row;
    int col;
    int idx;
    uint64_t seed;

    if (g_tables_ready) {
        return;
    }

    memset(g_pos_index, -1, sizeof(g_pos_index));
    idx = 0;
    for (row = 0; row < 9; ++row) {
        int start = row <= 4 ? 1 : row - 3;
        int end = row <= 4 ? 5 + row : 9;
        for (col = start; col <= end; ++col) {
            g_rows[idx] = (uint8_t) row;
            g_cols[idx] = (uint8_t) col;
            g_pos_index[row][col] = (int8_t) idx;
            idx += 1;
        }
    }

    for (idx = 0; idx < CELL_COUNT; ++idx) {
        int dir;
        uint64_t mask = 0;
        uint8_t risk = 2;
        uint8_t pressure = 0;
        for (dir = 0; dir < DIR_COUNT; ++dir) {
            int next_row = (int) g_rows[idx] + DIR_DR[dir];
            int next_col = (int) g_cols[idx] + DIR_DC[dir];
            uint8_t neighbor = INVALID_INDEX;
            if (next_row >= 0 && next_row < 9 && next_col >= 0 && next_col < 10) {
                int8_t mapped = g_pos_index[next_row][next_col];
                if (mapped >= 0) {
                    neighbor = (uint8_t) mapped;
                    mask |= (1ULL << neighbor);
                }
            }
            g_neighbors[idx][dir] = neighbor;
        }
        g_adj_masks[idx] = mask;

        for (col = 0; col < DIR_COUNT; ++col) {
            uint8_t pos1 = g_neighbors[idx][col];
            if (pos1 == INVALID_INDEX) {
                risk = 0;
                break;
            }
            if (g_neighbors[pos1][col] == INVALID_INDEX && risk > 1) {
                risk = 1;
            }
        }
        if (risk == 0 || risk == 1) {
            pressure = 1;
        }
        g_edge_risk[idx] = risk == 0 ? 2 : (risk == 1 ? 1 : 0);
        g_edge_pressure[idx] = pressure;
    }

    seed = 0xABA10EULL;
    for (idx = 0; idx < CELL_COUNT; ++idx) {
        g_zobrist[idx][BLACK] = splitmix64_next(&seed);
        g_zobrist[idx][WHITE] = splitmix64_next(&seed);
    }
    g_side_zobrist[BLACK] = splitmix64_next(&seed);
    g_side_zobrist[WHITE] = splitmix64_next(&seed);

    g_tables_ready = 1;
}

double
monotonic_seconds(void)
{
#ifdef _WIN32
    static LARGE_INTEGER frequency;
    LARGE_INTEGER now;
    if (frequency.QuadPart == 0) {
        QueryPerformanceFrequency(&frequency);
    }
    QueryPerformanceCounter(&now);
    return (double) now.QuadPart / (double) frequency.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + ((double) ts.tv_nsec / 1000000000.0);
#endif
}

int
dir_index_from_delta(int dr, int dc)
{
    int dir;
    for (dir = 0; dir < DIR_COUNT; ++dir) {
        if (DIR_DR[dir] == dr && DIR_DC[dir] == dc) {
            return dir;
        }
    }
    return -1;
}

void
move_clear(NativeMove *move)
{
    memset(move, 0, sizeof(*move));
}

int
move_has_value(const NativeMove *move)
{
    return move->count > 0;
}

int
move_equal(const NativeMove *left, const NativeMove *right)
{
    int idx;
    if (left->count != right->count || left->dir_idx != right->dir_idx) {
        return 0;
    }
    for (idx = 0; idx < left->count; ++idx) {
        if (left->marbles[idx] != right->marbles[idx]) {
            return 0;
        }
    }
    return 1;
}

void
canonicalize_indices(uint8_t *marbles, int count)
{
    if (count == 2) {
        if (marbles[0] > marbles[1]) {
            uint8_t tmp = marbles[0];
            marbles[0] = marbles[1];
            marbles[1] = tmp;
        }
        return;
    }
    if (count == 3) {
        if (marbles[0] > marbles[1]) {
            uint8_t tmp = marbles[0];
            marbles[0] = marbles[1];
            marbles[1] = tmp;
        }
        if (marbles[1] > marbles[2]) {
            uint8_t tmp = marbles[1];
            marbles[1] = marbles[2];
            marbles[2] = tmp;
        }
        if (marbles[0] > marbles[1]) {
            uint8_t tmp = marbles[0];
            marbles[0] = marbles[1];
            marbles[1] = tmp;
        }
    }
}

static void
ordered_for_direction(const uint8_t *marbles, int count, uint8_t dir_idx, uint8_t *ordered)
{
    if (count <= 1) {
        ordered[0] = marbles[0];
        return;
    }

    if (count == 2) {
        int score0 = (int) g_rows[marbles[0]] * DIR_DR[dir_idx] + (int) g_cols[marbles[0]] * DIR_DC[dir_idx];
        int score1 = (int) g_rows[marbles[1]] * DIR_DR[dir_idx] + (int) g_cols[marbles[1]] * DIR_DC[dir_idx];
        if (score0 <= score1) {
            ordered[0] = marbles[0];
            ordered[1] = marbles[1];
        } else {
            ordered[0] = marbles[1];
            ordered[1] = marbles[0];
        }
        return;
    }

    {
        uint8_t a = marbles[0];
        uint8_t b = marbles[1];
        uint8_t c = marbles[2];
        int sa = (int) g_rows[a] * DIR_DR[dir_idx] + (int) g_cols[a] * DIR_DC[dir_idx];
        int sb = (int) g_rows[b] * DIR_DR[dir_idx] + (int) g_cols[b] * DIR_DC[dir_idx];
        int sc = (int) g_rows[c] * DIR_DR[dir_idx] + (int) g_cols[c] * DIR_DC[dir_idx];

        if (sa > sb) {
            int tmp_s = sa;
            uint8_t tmp_p = a;
            sa = sb;
            a = b;
            sb = tmp_s;
            b = tmp_p;
        }
        if (sb > sc) {
            int tmp_s = sb;
            uint8_t tmp_p = b;
            sb = sc;
            b = c;
            sc = tmp_s;
            c = tmp_p;
        }
        if (sa > sb) {
            uint8_t tmp_p = a;
            a = b;
            b = tmp_p;
        }

        ordered[0] = a;
        ordered[1] = b;
        ordered[2] = c;
    }
}

void
build_move(const uint8_t *canonical_marbles, int count, uint8_t dir_idx, NativeMove *move)
{
    int idx;
    int line_dir;

    move_clear(move);
    move->count = (uint8_t) count;
    move->dir_idx = dir_idx;
    for (idx = 0; idx < count; ++idx) {
        move->marbles[idx] = canonical_marbles[idx];
    }

    if (count <= 1) {
        move->is_inline = 1;
        move->ordered[0] = canonical_marbles[0];
    } else {
        line_dir = dir_index_from_delta(
            (int) g_rows[canonical_marbles[1]] - (int) g_rows[canonical_marbles[0]],
            (int) g_cols[canonical_marbles[1]] - (int) g_cols[canonical_marbles[0]]
        );
        move->is_inline = (line_dir >= 0) && (dir_idx == (uint8_t) line_dir || dir_idx == OPPOSITE_DIR[line_dir]);
        ordered_for_direction(canonical_marbles, count, dir_idx, move->ordered);
    }

    move->leading = move->ordered[count - 1];
    move->trailing = move->ordered[0];
    if (move->is_inline) {
        move->order_pos1 = move->trailing;
        move->order_inline_flag = 1;
        move->order_pos2 = g_neighbors[move->leading][dir_idx];
        move->order_dir_rank = 0;
    } else {
        move->order_pos1 = canonical_marbles[0];
        move->order_inline_flag = 0;
        move->order_pos2 = canonical_marbles[count - 1];
        move->order_dir_rank = DIR_NAME_RANK[dir_idx];
    }
}

int
compare_ordering_key_tail(const NativeMove *left, const NativeMove *right)
{
    if (left->order_pos1 != right->order_pos1) {
        return left->order_pos1 < right->order_pos1 ? -1 : 1;
    }
    if (left->order_inline_flag != right->order_inline_flag) {
        return left->order_inline_flag < right->order_inline_flag ? -1 : 1;
    }
    if (left->order_pos2 != right->order_pos2) {
        return left->order_pos2 < right->order_pos2 ? -1 : 1;
    }
    if (left->order_dir_rank != right->order_dir_rank) {
        return left->order_dir_rank < right->order_dir_rank ? -1 : 1;
    }
    return 0;
}

static int
compare_ordering_key_full(const NativeMove *left, const NativeMove *right)
{
    if (left->count != right->count) {
        return left->count < right->count ? -1 : 1;
    }
    return compare_ordering_key_tail(left, right);
}

int
prefer_by_tie_break(int tie_break_lexicographic, const NativeMove *candidate, const NativeMove *incumbent)
{
    if (!move_has_value(incumbent)) {
        return 1;
    }
    if (!tie_break_lexicographic) {
        return 0;
    }
    return compare_ordering_key_full(candidate, incumbent) < 0;
}

int
board_init(BoardState *board, const uint8_t *cells, int black_score, int white_score)
{
    int idx;
    memset(board, 0, sizeof(*board));
    board->scores[BLACK] = black_score;
    board->scores[WHITE] = white_score;

    for (idx = 0; idx < CELL_COUNT; ++idx) {
        uint8_t cell = cells[idx];
        if (cell > WHITE) {
            return 0;
        }
        board->cells[idx] = cell;
        if (cell == BLACK || cell == WHITE) {
            board->bits[cell] |= bit_for((uint8_t) idx);
            board->zhash ^= g_zobrist[idx][cell];
        }
    }
    return 1;
}

void
board_set_cell(BoardState *board, uint8_t idx, uint8_t color)
{
    uint8_t old_color = board->cells[idx];
    if (old_color == color) {
        return;
    }

    if (old_color == BLACK || old_color == WHITE) {
        board->bits[old_color] &= ~bit_for(idx);
        board->zhash ^= g_zobrist[idx][old_color];
    }

    board->cells[idx] = color;

    if (color == BLACK || color == WHITE) {
        board->bits[color] |= bit_for(idx);
        board->zhash ^= g_zobrist[idx][color];
    }
}
