/* Initializes shared lookup tables, hashing data, and move-ordering helpers. */
#include "common.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
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

#ifdef _WIN32
static INIT_ONCE g_tables_once = INIT_ONCE_STATIC_INIT;
#else
static pthread_once_t g_tables_once = PTHREAD_ONCE_INIT;
#endif

/* Advances the SplitMix64 generator used for deterministic Zobrist seeds. */
static uint64_t
next_splitmix64(uint64_t *generator_state)
{
    uint64_t mixed = (*generator_state += 0x9E3779B97F4A7C15ULL);
    mixed = (mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9ULL;
    mixed = (mixed ^ (mixed >> 27)) * 0x94D049BB133111EBULL;
    return mixed ^ (mixed >> 31);
}

/* Builds board-coordinate tables, adjacency caches, edge metrics, and Zobrist keys. */
static void
init_tables_impl(void)
{
    int board_row;
    int board_col;
    int cell_idx;
    uint64_t seed;

    memset(g_pos_index, -1, sizeof(g_pos_index));
    cell_idx = 0;
    for (board_row = 0; board_row < 9; ++board_row) {
        int start_col = board_row <= 4 ? 1 : board_row - 3;
        int end_col = board_row <= 4 ? 5 + board_row : 9;
        for (board_col = start_col; board_col <= end_col; ++board_col) {
            g_rows[cell_idx] = (uint8_t) board_row;
            g_cols[cell_idx] = (uint8_t) board_col;
            g_pos_index[board_row][board_col] = (int8_t) cell_idx;
            cell_idx += 1;
        }
    }

    for (cell_idx = 0; cell_idx < CELL_COUNT; ++cell_idx) {
        int direction_idx;
        uint64_t mask = 0;
        uint8_t edge_risk = 2;
        uint8_t edge_pressure = 0;
        for (direction_idx = 0; direction_idx < DIR_COUNT; ++direction_idx) {
            int next_row = (int) g_rows[cell_idx] + DIR_DR[direction_idx];
            int next_col = (int) g_cols[cell_idx] + DIR_DC[direction_idx];
            uint8_t neighbor = INVALID_INDEX;
            if (next_row >= 0 && next_row < 9 && next_col >= 0 && next_col < 10) {
                int8_t mapped_index = g_pos_index[next_row][next_col];
                if (mapped_index >= 0) {
                    neighbor = (uint8_t) mapped_index;
                    mask |= bit_for(neighbor);
                }
            }
            g_neighbors[cell_idx][direction_idx] = neighbor;
        }
        g_adj_masks[cell_idx] = mask;

        for (direction_idx = 0; direction_idx < DIR_COUNT; ++direction_idx) {
            uint8_t neighbor = g_neighbors[cell_idx][direction_idx];
            if (neighbor == INVALID_INDEX) {
                edge_risk = 0;
                break;
            }
            if (g_neighbors[neighbor][direction_idx] == INVALID_INDEX && edge_risk > 1) {
                edge_risk = 1;
            }
        }
        if (edge_risk == 0 || edge_risk == 1) {
            edge_pressure = 1;
        }
        g_edge_risk[cell_idx] = edge_risk == 0 ? 2 : (edge_risk == 1 ? 1 : 0);
        g_edge_pressure[cell_idx] = edge_pressure;
    }

    seed = 0xABA10EULL;
    for (cell_idx = 0; cell_idx < CELL_COUNT; ++cell_idx) {
        g_zobrist[cell_idx][BLACK] = next_splitmix64(&seed);
        g_zobrist[cell_idx][WHITE] = next_splitmix64(&seed);
    }
    g_side_zobrist[BLACK] = next_splitmix64(&seed);
    g_side_zobrist[WHITE] = next_splitmix64(&seed);

    g_tables_ready = 1;
}

#ifdef _WIN32
static BOOL CALLBACK
init_tables_once(PINIT_ONCE init_once, PVOID parameter, PVOID *context)
{
    (void) init_once;
    (void) parameter;
    (void) context;
    init_tables_impl();
    return TRUE;
}
#endif

void
init_tables(void)
{
    if (g_tables_ready) {
        return;
    }
#ifdef _WIN32
    InitOnceExecuteOnce(&g_tables_once, init_tables_once, NULL, NULL);
#else
    pthread_once(&g_tables_once, init_tables_impl);
#endif
}

/* Returns a monotonic wall-clock timestamp in seconds for time-budget checks. */
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

/* Maps a row/column delta to the native direction index, or -1 when invalid. */
int
dir_index_from_delta(int dr, int dc)
{
    int direction_idx;
    for (direction_idx = 0; direction_idx < DIR_COUNT; ++direction_idx) {
        if (DIR_DR[direction_idx] == dr && DIR_DC[direction_idx] == dc) {
            return direction_idx;
        }
    }
    return -1;
}

/* Resets a move struct to its empty sentinel state. */
void
move_clear(NativeMove *move)
{
    memset(move, 0, sizeof(*move));
}

/* Reports whether a move struct currently holds a real move. */
int
move_has_value(const NativeMove *move)
{
    return move->count > 0;
}

/* Checks whether two move payloads describe the same move. */
int
move_equal(const NativeMove *left, const NativeMove *right)
{
    int marble_idx;
    if (left->count != right->count || left->dir_idx != right->dir_idx) {
        return 0;
    }
    for (marble_idx = 0; marble_idx < left->count; ++marble_idx) {
        if (left->marbles[marble_idx] != right->marbles[marble_idx]) {
            return 0;
        }
    }
    return 1;
}

/* Sorts marble indices into canonical ascending order for deduplication. */
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

/* Orders a marble line from trailing to leading along a given movement direction. */
static void
sort_marbles_for_direction(const uint8_t *marbles, int count, uint8_t dir_idx, uint8_t *ordered)
{
    if (count <= 1) {
        ordered[0] = marbles[0];
        return;
    }

    if (count == 2) {
        int projection0 = (int) g_rows[marbles[0]] * DIR_DR[dir_idx] + (int) g_cols[marbles[0]] * DIR_DC[dir_idx];
        int projection1 = (int) g_rows[marbles[1]] * DIR_DR[dir_idx] + (int) g_cols[marbles[1]] * DIR_DC[dir_idx];
        if (projection0 <= projection1) {
            ordered[0] = marbles[0];
            ordered[1] = marbles[1];
        } else {
            ordered[0] = marbles[1];
            ordered[1] = marbles[0];
        }
        return;
    }

    {
        uint8_t first = marbles[0];
        uint8_t second = marbles[1];
        uint8_t third = marbles[2];
        int first_projection = (int) g_rows[first] * DIR_DR[dir_idx] + (int) g_cols[first] * DIR_DC[dir_idx];
        int second_projection = (int) g_rows[second] * DIR_DR[dir_idx] + (int) g_cols[second] * DIR_DC[dir_idx];
        int third_projection = (int) g_rows[third] * DIR_DR[dir_idx] + (int) g_cols[third] * DIR_DC[dir_idx];

        if (first_projection > second_projection) {
            int tmp_projection = first_projection;
            uint8_t tmp_index = first;
            first_projection = second_projection;
            first = second;
            second_projection = tmp_projection;
            second = tmp_index;
        }
        if (second_projection > third_projection) {
            int tmp_projection = second_projection;
            uint8_t tmp_index = second;
            second_projection = third_projection;
            second = third;
            third_projection = tmp_projection;
            third = tmp_index;
        }
        if (first_projection > second_projection) {
            uint8_t tmp_index = first;
            first = second;
            second = tmp_index;
        }

        ordered[0] = first;
        ordered[1] = second;
        ordered[2] = third;
    }
}

/* Builds cached move metadata used by move ordering and move application. */
void
build_move(const uint8_t *canonical_marbles, int count, uint8_t dir_idx, NativeMove *move)
{
    int marble_idx;
    int line_dir;

    move_clear(move);
    move->count = (uint8_t) count;
    move->dir_idx = dir_idx;
    for (marble_idx = 0; marble_idx < count; ++marble_idx) {
        move->marbles[marble_idx] = canonical_marbles[marble_idx];
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
        sort_marbles_for_direction(canonical_marbles, count, dir_idx, move->ordered);
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

/* Compares the stable ordering-key suffix used for deterministic tie-breaking. */
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

/* Compares full move ordering keys, including group size, for root tie breaks. */
static int
compare_ordering_key_full(const NativeMove *left, const NativeMove *right)
{
    if (left->count != right->count) {
        return left->count < right->count ? -1 : 1;
    }
    return compare_ordering_key_tail(left, right);
}

/* Chooses the better move when scores tie and lexicographic ordering is enabled. */
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

/* Initializes a native board from compact cell data and capture counts. */
int
board_init(BoardState *board, const uint8_t *cells, int black_score, int white_score)
{
    int cell_idx;
    memset(board, 0, sizeof(*board));
    board->scores[BLACK] = black_score;
    board->scores[WHITE] = white_score;

    for (cell_idx = 0; cell_idx < CELL_COUNT; ++cell_idx) {
        uint8_t cell = cells[cell_idx];
        if (cell > WHITE) {
            return 0;
        }
        board->cells[cell_idx] = cell;
        if (cell == BLACK || cell == WHITE) {
            board->bits[cell] |= bit_for((uint8_t) cell_idx);
            board->zhash ^= g_zobrist[cell_idx][cell];
        }
    }
    return 1;
}

/* Updates a single board cell while keeping bitboards and hashes in sync. */
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
