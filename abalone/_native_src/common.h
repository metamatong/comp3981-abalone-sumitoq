/* Shared constants, structs, and native function declarations for the Abalone extension. */
#ifndef ABALONE_NATIVE_COMMON_H
#define ABALONE_NATIVE_COMMON_H

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define CELL_COUNT 61
#define DIR_COUNT 6
#define FEATURE_COUNT 9
#define MAX_MOVES 256
#define INVALID_INDEX 255

#define EMPTY 0
#define BLACK 1
#define WHITE 2

#define EXACT 0
#define LOWERBOUND 1
#define UPPERBOUND 2

#define TT_MODE_FULL 0
#define TT_MODE_QUIESCENCE 1

extern const int8_t DIR_DR[DIR_COUNT];
extern const int8_t DIR_DC[DIR_COUNT];
extern const uint8_t OPPOSITE_DIR[DIR_COUNT];
extern const uint8_t REFERENCE_DIRS[DIR_COUNT];
extern const uint8_t POSITIVE_DIRS[3];
extern const uint8_t DIR_NAME_RANK[DIR_COUNT];

extern int g_tables_ready;
extern int8_t g_pos_index[9][10];
extern uint8_t g_rows[CELL_COUNT];
extern uint8_t g_cols[CELL_COUNT];
extern uint8_t g_neighbors[CELL_COUNT][DIR_COUNT];
extern uint64_t g_adj_masks[CELL_COUNT];
extern uint8_t g_edge_risk[CELL_COUNT];
extern uint8_t g_edge_pressure[CELL_COUNT];
extern uint64_t g_zobrist[CELL_COUNT][3];
extern uint64_t g_side_zobrist[3];

typedef struct {
    uint8_t cells[CELL_COUNT];
    uint64_t bits[3];
    int scores[3];
    uint64_t zhash;
} BoardState;

typedef struct {
    uint8_t count;
    uint8_t dir_idx;
    uint8_t is_inline;
    uint8_t marbles[3];
    uint8_t ordered[3];
    uint8_t leading;
    uint8_t trailing;
    uint8_t order_pos1;
    uint8_t order_inline_flag;
    uint8_t order_pos2;
    uint8_t order_dir_rank;
} NativeMove;

typedef struct {
    NativeMove move;
    double score;
    int depth;
} RootCandidate;

typedef struct {
    uint64_t key;
    double value;
    NativeMove move;
    int depth;
    uint8_t flag;
    uint8_t used;
} TTEntry;

typedef struct {
    TTEntry *entries;
    size_t capacity;
    size_t size;
} TTTable;

typedef struct {
    double weights[FEATURE_COUNT];
    int tie_break_lexicographic;
    double deadline_at;
    TTTable tt;
    const TTTable *tt_seed;
    const uint64_t *shared_root_alpha_bits;
    double root_alpha_floor;
    NativeMove killer_moves[64];
} SearchContext;

typedef struct {
    NativeMove move;
    double score;
    uint64_t nodes;
    int completed_depth;
    int timed_out;
    int avoidance_applied;
    RootCandidate root_candidates[MAX_MOVES];
    int root_candidate_count;
} SearchResultNative;

void init_tables(void);
double monotonic_seconds(void);
int dir_index_from_delta(int dr, int dc);
void move_clear(NativeMove *move);
int move_has_value(const NativeMove *move);
int move_equal(const NativeMove *left, const NativeMove *right);
void canonicalize_indices(uint8_t *marbles, int count);
void build_move(const uint8_t *canonical_marbles, int count, uint8_t dir_idx, NativeMove *move);
int compare_ordering_key_tail(const NativeMove *left, const NativeMove *right);
int prefer_by_tie_break(
    int tie_break_lexicographic,
    const NativeMove *candidate,
    const NativeMove *incumbent
);
int board_init(BoardState *board, const uint8_t *cells, int black_score, int white_score);
void board_set_cell(BoardState *board, uint8_t idx, uint8_t color);
int list_marbles(const BoardState *board, int player, uint8_t *marbles);
int generate_legal_moves_native(const BoardState *board, int player, NativeMove *moves);
double evaluate_weighted_native(const BoardState *board, int player, const double *weights);
void order_moves(
    const BoardState *board,
    int player,
    NativeMove *moves,
    int count,
    const NativeMove *tt_move,
    const NativeMove *killer_move
);
void apply_move_native(BoardState *board, const NativeMove *move, int player);
int search_weighted_native(
    const BoardState *board,
    int player,
    const double *weights,
    int requested_depth,
    int max_quiescence_depth,
    int has_deadline,
    int time_budget_ms,
    int has_remaining_game_moves,
    int remaining_game_moves,
    int tie_break_lexicographic,
    const NativeMove *avoid_move,
    int root_candidate_limit,
    SearchResultNative *out_result
);
int search_weighted_native_serial_for_testing(
    const BoardState *board,
    int player,
    const double *weights,
    int requested_depth,
    int max_quiescence_depth,
    int has_deadline,
    int time_budget_ms,
    int has_remaining_game_moves,
    int remaining_game_moves,
    int tie_break_lexicographic,
    const NativeMove *avoid_move,
    int root_candidate_limit,
    SearchResultNative *out_result
);
int debug_resolve_root_worker_count(int legal_count, unsigned int cpu_count);

/* Returns the bit mask for a single board index. */
static inline uint64_t
bit_for(uint8_t idx)
{
    return 1ULL << idx;
}

/* Reports whether either player has already reached the capture win condition. */
static inline int
board_terminal(const BoardState *board)
{
    return board->scores[BLACK] >= 6 || board->scores[WHITE] >= 6;
}

#endif
