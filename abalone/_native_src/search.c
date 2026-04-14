/* Applies moves and runs the native alpha-beta search with iterative deepening. */
#include "common.h"

#ifdef _WIN32
#include <process.h>
#include <windows.h>
#else
#include <pthread.h>
#include <unistd.h>
#endif

typedef enum {
    ROOT_JOB_PENDING = 0,
    ROOT_JOB_DONE = 1,
    ROOT_JOB_TIMED_OUT = 2,
    ROOT_JOB_ERROR = 3,
} RootJobStatus;

typedef struct {
    RootJobStatus status;
    double alpha;
    double value;
    uint64_t nodes;
    int needs_exact;
} RootJobResult;

#ifdef _WIN32
typedef HANDLE RootWorkerHandle;
typedef CRITICAL_SECTION RootMutex;
typedef CONDITION_VARIABLE RootCondition;
#else
typedef pthread_t RootWorkerHandle;
typedef pthread_mutex_t RootMutex;
typedef pthread_cond_t RootCondition;
#endif

typedef struct RootSearchThreadPool RootSearchThreadPool;

typedef struct {
    RootWorkerHandle handle;
    RootSearchThreadPool *pool;
    SearchContext ctx;
    int started;
} RootSearchWorker;

struct RootSearchThreadPool {
    RootMutex mutex;
    RootCondition work_available;
    RootCondition work_finished;
    int sync_ready;
    int stop;
    int generation;
    int finished_workers;
    int next_job_index;
    int end_job_index;
    int cancel_jobs;
    int worker_init_error;
    const BoardState *board;
    const NativeMove *ordered_root;
    int player;
    int opponent;
    int depth_after_move;
    int max_quiescence_depth;
    int root_remaining_game_moves;
    uint64_t alpha_bits;
    double deadline_at;
    const double *weights;
    int tie_break_lexicographic;
    const TTTable *tt_seed;
    TTTable tt_seed_snapshot;
    const NativeMove *killer_moves_seed;
    RootJobResult *results;
    int worker_count;
    RootSearchWorker *workers;
};

/* Initializes the cross-platform mutex used by the root worker pool. */
static int
root_mutex_init(RootMutex *mutex)
{
#ifdef _WIN32
    InitializeCriticalSection(mutex);
    return 1;
#else
    return pthread_mutex_init(mutex, NULL) == 0;
#endif
}

/* Releases the cross-platform mutex used by the root worker pool. */
static void
root_mutex_destroy(RootMutex *mutex)
{
#ifdef _WIN32
    DeleteCriticalSection(mutex);
#else
    pthread_mutex_destroy(mutex);
#endif
}

/* Acquires the cross-platform mutex used by the root worker pool. */
static void
root_mutex_lock(RootMutex *mutex)
{
#ifdef _WIN32
    EnterCriticalSection(mutex);
#else
    pthread_mutex_lock(mutex);
#endif
}

/* Releases the cross-platform mutex used by the root worker pool. */
static void
root_mutex_unlock(RootMutex *mutex)
{
#ifdef _WIN32
    LeaveCriticalSection(mutex);
#else
    pthread_mutex_unlock(mutex);
#endif
}

/* Initializes the condition variables used by the root worker pool. */
static void
root_condition_init(RootCondition *condition)
{
#ifdef _WIN32
    InitializeConditionVariable(condition);
#else
    pthread_cond_init(condition, NULL);
#endif
}

/* Releases the condition variables used by the root worker pool. */
static void
root_condition_destroy(RootCondition *condition)
{
#ifdef _WIN32
    (void) condition;
#else
    pthread_cond_destroy(condition);
#endif
}

/* Waits for either new work or worker completion within the root worker pool. */
static void
root_condition_wait(RootCondition *condition, RootMutex *mutex)
{
#ifdef _WIN32
    SleepConditionVariableCS(condition, mutex, INFINITE);
#else
    pthread_cond_wait(condition, mutex);
#endif
}

/* Wakes every waiter on a root worker pool condition variable. */
static void
root_condition_broadcast(RootCondition *condition)
{
#ifdef _WIN32
    WakeAllConditionVariable(condition);
#else
    pthread_cond_broadcast(condition);
#endif
}

/* Mixes a 64-bit board hash for transposition-table indexing. */
static uint64_t
mix_hash64(uint64_t value)
{
    value ^= value >> 33;
    value *= 0xff51afd7ed558ccdULL;
    value ^= value >> 33;
    value *= 0xc4ceb9fe1a85ec53ULL;
    value ^= value >> 33;
    return value;
}

/* Returns the machine's logical hardware concurrency. */
static unsigned int
hardware_thread_count(void)
{
#ifdef _WIN32
    SYSTEM_INFO system_info;
    GetSystemInfo(&system_info);
    return system_info.dwNumberOfProcessors > 0 ? system_info.dwNumberOfProcessors : 1U;
#else
    long cpu_count = sysconf(_SC_NPROCESSORS_ONLN);
    return cpu_count > 0 ? (unsigned int) cpu_count : 1U;
#endif
}

/* Reads an optional environment override for root worker count. */
static int
configured_root_worker_count(void)
{
    const char *value = getenv("ABALONE_NATIVE_ROOT_THREADS");
    int configured;

    if (value == NULL || *value == '\0') {
        return 0;
    }

    configured = atoi(value);
    return configured > 0 ? configured : 0;
}

/* Caps root-search worker count to remaining root moves using either an env override or default. */
static int
resolve_root_worker_count_with_cpu_count(int legal_count, unsigned int cpu_count)
{
    int remaining_root_moves = legal_count - 1;
    int max_workers;
    int configured_workers = configured_root_worker_count();

    if (remaining_root_moves <= 0 || cpu_count <= 1U) {
        return 0;
    }

    if (configured_workers > 0) {
        max_workers = configured_workers;
    } else {
        max_workers = (int) cpu_count - 1;
        if (max_workers > 4) {
            max_workers = 4;
        } else if (max_workers < 0) {
            max_workers = 0;
        }
    }
    return remaining_root_moves < max_workers ? remaining_root_moves : max_workers;
}

/* Reinterprets a double as its raw uint64 bit pattern. */
static uint64_t
double_to_bits(double value)
{
    union {
        double as_double;
        uint64_t as_bits;
    } converter;

    converter.as_double = value;
    return converter.as_bits;
}

/* Reinterprets a uint64 bit pattern as a double. */
static double
bits_to_double(uint64_t value)
{
    union {
        double as_double;
        uint64_t as_bits;
    } converter;

    converter.as_bits = value;
    return converter.as_double;
}

/* Atomically reads the shared root alpha published by the main thread. */
static double
root_alpha_load(const uint64_t *bits)
{
    if (bits == NULL) {
        return -INFINITY;
    }
#ifdef _WIN32
    return bits_to_double((uint64_t) InterlockedCompareExchange64((volatile LONG64 *) bits, 0, 0));
#else
    return bits_to_double(__atomic_load_n(bits, __ATOMIC_ACQUIRE));
#endif
}

/* Atomically stores the shared root alpha for a fresh iterative-deepening pass. */
static void
root_alpha_store(uint64_t *bits, double alpha)
{
#ifdef _WIN32
    InterlockedExchange64((volatile LONG64 *) bits, (LONG64) double_to_bits(alpha));
#else
    __atomic_store_n(bits, double_to_bits(alpha), __ATOMIC_RELEASE);
#endif
}

/* Atomically raises the shared root alpha when the main thread improves the root best score. */
static void
root_alpha_raise(uint64_t *bits, double alpha)
{
    uint64_t desired = double_to_bits(alpha);

    if (bits == NULL) {
        return;
    }

#ifdef _WIN32
    for (;;) {
        LONG64 observed = InterlockedCompareExchange64((volatile LONG64 *) bits, 0, 0);
        if (bits_to_double((uint64_t) observed) >= alpha) {
            return;
        }
        if (InterlockedCompareExchange64((volatile LONG64 *) bits, (LONG64) desired, observed) == observed) {
            return;
        }
    }
#else
    for (;;) {
        uint64_t observed = __atomic_load_n(bits, __ATOMIC_ACQUIRE);
        if (bits_to_double(observed) >= alpha) {
            return;
        }
        if (__atomic_compare_exchange_n(
                bits,
                &observed,
                desired,
                0,
                __ATOMIC_ACQ_REL,
                __ATOMIC_ACQUIRE)) {
            return;
        }
    }
#endif
}

/* Syncs the current search alpha with the shared root alpha used by worker threads. */
static void
sync_root_alpha(SearchContext *ctx, double *alpha)
{
    double shared_alpha = root_alpha_load(ctx->shared_root_alpha_bits);
    if (shared_alpha > ctx->root_alpha_floor) {
        ctx->root_alpha_floor = shared_alpha;
    }
    if (shared_alpha > *alpha) {
        *alpha = shared_alpha;
    }
}

/* Caps root-search worker count to available hardware and remaining root moves. */
static int
resolve_root_worker_count(int legal_count)
{
    return resolve_root_worker_count_with_cpu_count(legal_count, hardware_thread_count());
}

/* Internal test hook that resolves root worker count with a caller-supplied CPU count. */
int
debug_resolve_root_worker_count(int legal_count, unsigned int cpu_count)
{
    return resolve_root_worker_count_with_cpu_count(legal_count, cpu_count);
}

/* Advances an optional move budget while preserving -1 as "unlimited". */
static int
next_remaining_game_moves(int remaining_game_moves)
{
    if (remaining_game_moves < 0) {
        return -1;
    }
    if (remaining_game_moves == 0) {
        return 0;
    }
    return remaining_game_moves - 1;
}

/* Allocates an empty transposition table with power-of-two capacity. */
static int
tt_init(TTTable *table, size_t capacity)
{
    TTEntry *entries;
    size_t actual_capacity = 1;
    while (actual_capacity < capacity) {
        actual_capacity <<= 1;
    }
    entries = (TTEntry *) calloc(actual_capacity, sizeof(TTEntry));
    if (entries == NULL) {
        return 0;
    }
    table->entries = entries;
    table->capacity = actual_capacity;
    table->size = 0;
    return 1;
}

/* Clones an existing transposition table so worker threads inherit prior search state. */
static int
tt_clone(TTTable *table, const TTTable *source)
{
    size_t bytes;

    table->entries = NULL;
    table->capacity = 0;
    table->size = 0;
    if (source == NULL || source->capacity == 0 || source->entries == NULL) {
        return tt_init(table, 1024U);
    }

    bytes = sizeof(TTEntry) * source->capacity;
    table->entries = (TTEntry *) malloc(bytes);
    if (table->entries == NULL) {
        return 0;
    }
    memcpy(table->entries, source->entries, bytes);
    table->capacity = source->capacity;
    table->size = source->size;
    return 1;
}

/* Releases all transposition-table storage. */
static void
tt_free(TTTable *table)
{
    free(table->entries);
    table->entries = NULL;
    table->capacity = 0;
    table->size = 0;
}

/* Finds an existing transposition-table entry for the given key. */
static const TTEntry *
tt_lookup(const TTTable *table, uint64_t key)
{
    size_t idx;
    if (table->capacity == 0) {
        return NULL;
    }
    idx = (size_t) (mix_hash64(key) & (table->capacity - 1));
    while (table->entries[idx].used) {
        if (table->entries[idx].key == key) {
            return &table->entries[idx];
        }
        idx = (idx + 1) & (table->capacity - 1);
    }
    return NULL;
}

/* Probes a worker-local table and then its read-only seed snapshot. */
static const TTEntry *
tt_lookup_seeded(const SearchContext *ctx, uint64_t key)
{
    const TTEntry *local_entry = tt_lookup(&ctx->tt, key);
    const TTEntry *seed_entry = NULL;

    if (ctx->tt_seed != NULL) {
        seed_entry = tt_lookup(ctx->tt_seed, key);
    }
    if (local_entry == NULL) {
        return seed_entry;
    }
    if (seed_entry == NULL) {
        return local_entry;
    }
    return local_entry->depth >= seed_entry->depth ? local_entry : seed_entry;
}

/* Rebuilds the transposition table at a larger capacity. */
static int
tt_rehash(TTTable *table, size_t new_capacity)
{
    TTTable replacement;
    size_t idx;
    if (!tt_init(&replacement, new_capacity)) {
        return 0;
    }
    for (idx = 0; idx < table->capacity; ++idx) {
        if (table->entries[idx].used) {
            TTEntry entry = table->entries[idx];
            size_t insert_idx = (size_t) (mix_hash64(entry.key) & (replacement.capacity - 1));
            while (replacement.entries[insert_idx].used) {
                insert_idx = (insert_idx + 1) & (replacement.capacity - 1);
            }
            replacement.entries[insert_idx] = entry;
            replacement.entries[insert_idx].used = 1;
            replacement.size += 1;
        }
    }
    tt_free(table);
    *table = replacement;
    return 1;
}

/* Stores or replaces a transposition-table entry for the supplied key. */
static int
tt_store(TTTable *table, uint64_t key, int depth, double value, uint8_t flag, const NativeMove *move)
{
    size_t idx;
    if ((table->size + 1) * 10 >= table->capacity * 7) {
        if (!tt_rehash(table, table->capacity << 1)) {
            return 0;
        }
    }
    idx = (size_t) (mix_hash64(key) & (table->capacity - 1));
    while (table->entries[idx].used) {
        if (table->entries[idx].key == key) {
            table->entries[idx].depth = depth;
            table->entries[idx].value = value;
            table->entries[idx].flag = flag;
            table->entries[idx].move = *move;
            return 1;
        }
        idx = (idx + 1) & (table->capacity - 1);
    }

    table->entries[idx].used = 1;
    table->entries[idx].key = key;
    table->entries[idx].depth = depth;
    table->entries[idx].value = value;
    table->entries[idx].flag = flag;
    table->entries[idx].move = *move;
    table->size += 1;
    return 1;
}

/* Builds the transposition key for a board, side, search mode, and move budget. */
static uint64_t
tt_key_for_state(const BoardState *board, int to_move, int mode, int remaining_game_moves)
{
    uint64_t key = board->zhash ^ g_side_zobrist[to_move];
    uint64_t remaining_key = remaining_game_moves < 0 ? 1ULL : (uint64_t) remaining_game_moves + 2ULL;
    key ^= mix_hash64((((uint64_t) mode) << 32) ^ remaining_key);
    return key;
}

/* Checks whether the active search deadline has been reached. */
static int
deadline_reached(const SearchContext *ctx)
{
    return ctx->deadline_at >= 0.0 && monotonic_seconds() >= ctx->deadline_at;
}

/* Reports whether a player has already secured the sixth capture. */
static int
player_has_terminal_win(const BoardState *board, int player)
{
    int opponent = player == BLACK ? WHITE : BLACK;
    uint8_t marbles[14];

    if (board->scores[player] >= 6) {
        return 1;
    }
    return list_marbles(board, opponent, marbles) <= 8;
}

/* Reports whether either side has already reached a game-ending state. */
static int
is_terminal_position(const BoardState *board)
{
    return player_has_terminal_win(board, BLACK) || player_has_terminal_win(board, WHITE);
}

/* Returns a forced terminal score, preferring faster wins and slower losses. */
static int
terminal_value(const BoardState *board, int root_player, int depth_remaining, double *out_value)
{
    int opponent = root_player == BLACK ? WHITE : BLACK;
    double depth_bonus = depth_remaining > 0 ? (double) depth_remaining : 0.0;

    if (player_has_terminal_win(board, root_player)) {
        *out_value = TERMINAL_SCORE + depth_bonus;
        return 1;
    }
    if (player_has_terminal_win(board, opponent)) {
        *out_value = -TERMINAL_SCORE - depth_bonus;
        return 1;
    }
    return 0;
}

/* Reports whether a move is tactical enough to extend quiescence. */
static int
is_quiescence_move_native(const BoardState *board, int player, const NativeMove *move)
{
    int opponent = player == BLACK ? WHITE : BLACK;
    uint8_t ahead;
    if (!move->is_inline || move->count < 2) {
        return 0;
    }
    ahead = g_neighbors[move->leading][move->dir_idx];
    return ahead != INVALID_INDEX && board->cells[ahead] == (uint8_t) opponent;
}

/* Applies an inline move and resolves any pushed opposing marbles. */
static void
apply_move_inline(BoardState *board, const NativeMove *move, int player, int opponent)
{
    uint8_t pushed[3];
    int pushed_count = 0;
    int idx;
    uint8_t scan_pos = g_neighbors[move->leading][move->dir_idx];
    while (scan_pos != INVALID_INDEX && board->cells[scan_pos] == (uint8_t) opponent) {
        pushed[pushed_count++] = scan_pos;
        scan_pos = g_neighbors[scan_pos][move->dir_idx];
    }

    if (pushed_count > 0 && scan_pos == INVALID_INDEX) {
        board->scores[player] += 1;
    }

    for (idx = pushed_count - 1; idx >= 0; --idx) {
        uint8_t source = pushed[idx];
        uint8_t dest = g_neighbors[source][move->dir_idx];
        board_set_cell(board, source, EMPTY);
        if (dest != INVALID_INDEX) {
            board_set_cell(board, dest, (uint8_t) opponent);
        }
    }

    for (idx = move->count - 1; idx >= 0; --idx) {
        uint8_t source = move->ordered[idx];
        uint8_t dest = g_neighbors[source][move->dir_idx];
        board_set_cell(board, source, EMPTY);
        board_set_cell(board, dest, (uint8_t) player);
    }
}

/* Applies a broadside move by clearing then refilling the shifted cells. */
static void
apply_move_broadside(BoardState *board, const NativeMove *move, int player)
{
    int idx;
    for (idx = 0; idx < move->count; ++idx) {
        board_set_cell(board, move->marbles[idx], EMPTY);
    }
    for (idx = 0; idx < move->count; ++idx) {
        uint8_t dest = g_neighbors[move->marbles[idx]][move->dir_idx];
        board_set_cell(board, dest, (uint8_t) player);
    }
}

/* Applies either inline or broadside movement to the native board state. */
void
apply_move_native(BoardState *board, const NativeMove *move, int player)
{
    int opponent = player == BLACK ? WHITE : BLACK;
    if (move->is_inline) {
        apply_move_inline(board, move, player, opponent);
    } else {
        apply_move_broadside(board, move, player);
    }
}

/* Extends tactical leaf positions to reduce horizon effects. */
static int
quiescence_native(
    const BoardState *board,
    int to_move,
    int root_player,
    int remaining_depth,
    int remaining_game_moves,
    double alpha,
    double beta,
    SearchContext *ctx,
    uint64_t *nodes,
    double *out_value,
    NativeMove *out_move,
    uint8_t *out_flag
)
{
    double initial_alpha;
    double initial_beta;
    uint64_t key;
    const TTEntry *entry;
    NativeMove tt_move;
    NativeMove best_move;
    double stand_pat;
    int maximizing;
    double best_value;
    uint8_t flag;
    NativeMove legal_moves[MAX_MOVES];
    NativeMove tactical_moves[MAX_MOVES];
    int legal_count;
    int tactical_count = 0;
    int move_index;
    int opponent;

    if (deadline_reached(ctx)) {
        return 1;
    }
    sync_root_alpha(ctx, &alpha);
    initial_alpha = alpha;
    initial_beta = beta;
    *nodes += 1;

    key = tt_key_for_state(board, to_move, TT_MODE_QUIESCENCE, remaining_game_moves);
    entry = tt_lookup_seeded(ctx, key);
    if (entry != NULL && entry->depth >= remaining_depth) {
        if (entry->flag == EXACT) {
            *out_value = entry->value;
            *out_move = entry->move;
            if (out_flag != NULL) {
                *out_flag = EXACT;
            }
            return 0;
        }
        if (entry->flag == LOWERBOUND && entry->value > alpha) {
            alpha = entry->value;
        } else if (entry->flag == UPPERBOUND && entry->value < beta) {
            beta = entry->value;
        }
        if (alpha >= beta) {
            *out_value = entry->value;
            *out_move = entry->move;
            if (out_flag != NULL) {
                *out_flag = entry->flag;
            }
            return 0;
        }
        tt_move = entry->move;
    } else {
        move_clear(&tt_move);
    }

    if (terminal_value(board, root_player, remaining_depth, &stand_pat)) {
        move_clear(out_move);
        if (!tt_store(&ctx->tt, key, remaining_depth, stand_pat, EXACT, out_move)) {
            return -1;
        }
        *out_value = stand_pat;
        return 0;
    }

    stand_pat = evaluate_weighted_native(board, root_player, ctx->weights);
    if (remaining_depth <= 0 || board_terminal(board) || remaining_game_moves == 0) {
        move_clear(out_move);
        if (!tt_store(&ctx->tt, key, remaining_depth, stand_pat, EXACT, out_move)) {
            return -1;
        }
        *out_value = stand_pat;
        if (out_flag != NULL) {
            *out_flag = EXACT;
        }
        return 0;
    }

    maximizing = to_move == root_player;
    if (maximizing) {
        if (stand_pat >= beta) {
            move_clear(out_move);
            if (!tt_store(&ctx->tt, key, remaining_depth, stand_pat, LOWERBOUND, out_move)) {
                return -1;
            }
            *out_value = stand_pat;
            if (out_flag != NULL) {
                *out_flag = LOWERBOUND;
            }
            return 0;
        }
        if (stand_pat > alpha) {
            alpha = stand_pat;
        }
        best_value = stand_pat;
    } else {
        if (stand_pat <= alpha) {
            move_clear(out_move);
            if (!tt_store(&ctx->tt, key, remaining_depth, stand_pat, UPPERBOUND, out_move)) {
                return -1;
            }
            *out_value = stand_pat;
            if (out_flag != NULL) {
                *out_flag = UPPERBOUND;
            }
            return 0;
        }
        if (stand_pat < beta) {
            beta = stand_pat;
        }
        best_value = stand_pat;
    }

    legal_count = generate_legal_moves_native(board, to_move, legal_moves);
    if (legal_count < 0) {
        return -1;
    }
    for (move_index = 0; move_index < legal_count; ++move_index) {
        if (is_quiescence_move_native(board, to_move, &legal_moves[move_index])) {
            tactical_moves[tactical_count++] = legal_moves[move_index];
        }
    }
    if (tactical_count == 0) {
        move_clear(out_move);
        if (!tt_store(&ctx->tt, key, remaining_depth, stand_pat, EXACT, out_move)) {
            return -1;
        }
        *out_value = stand_pat;
        if (out_flag != NULL) {
            *out_flag = EXACT;
        }
        return 0;
    }

    move_clear(&best_move);
    order_moves(board, to_move, tactical_moves, tactical_count, &tt_move, &best_move);
    opponent = to_move == BLACK ? WHITE : BLACK;

    for (move_index = 0; move_index < tactical_count; ++move_index) {
        BoardState child = *board;
        double child_value = 0.0;
        NativeMove ignored;
        uint8_t child_flag = EXACT;
        int child_remaining_game_moves = next_remaining_game_moves(remaining_game_moves);
        int status;

        if (deadline_reached(ctx)) {
            return 1;
        }
        sync_root_alpha(ctx, &alpha);
        if (move_index > 0 && beta <= alpha) {
            break;
        }

        apply_move_native(&child, &tactical_moves[move_index], to_move);
        status = quiescence_native(
            &child,
            opponent,
            root_player,
            remaining_depth - 1,
            child_remaining_game_moves,
            alpha,
            beta,
            ctx,
            nodes,
            &child_value,
            &ignored,
            &child_flag
        );
        if (status != 0) {
            return status;
        }

        if (maximizing) {
            if (child_value > best_value ||
                    (child_value == best_value &&
                        prefer_by_tie_break(ctx->tie_break_lexicographic, &tactical_moves[move_index], &best_move))) {
                best_value = child_value;
                best_move = tactical_moves[move_index];
            }
            if (best_value > alpha) {
                alpha = best_value;
            }
            if (beta <= alpha) {
                break;
            }
        } else {
            if (child_value < best_value ||
                    (child_value == best_value &&
                        prefer_by_tie_break(ctx->tie_break_lexicographic, &tactical_moves[move_index], &best_move))) {
                best_value = child_value;
                best_move = tactical_moves[move_index];
            }
            if (best_value < beta) {
                beta = best_value;
            }
            if (beta <= alpha) {
                break;
            }
        }
    }

    flag = EXACT;
    if (ctx->root_alpha_floor > initial_alpha) {
        initial_alpha = ctx->root_alpha_floor;
    }
    if (best_value <= initial_alpha) {
        flag = UPPERBOUND;
    } else if (best_value >= initial_beta) {
        flag = LOWERBOUND;
    }
    if (!tt_store(&ctx->tt, key, remaining_depth, best_value, flag, &best_move)) {
        return -1;
    }

    *out_value = best_value;
    *out_move = best_move;
    if (out_flag != NULL) {
        *out_flag = flag;
    }
    return 0;
}

/* Recursively searches the game tree with alpha-beta pruning and TT support. */
static int
minimax_native(
    const BoardState *board,
    int to_move,
    int root_player,
    int depth,
    int remaining_game_moves,
    int max_quiescence_depth,
    double alpha,
    double beta,
    SearchContext *ctx,
    uint64_t *nodes,
    double *out_value,
    NativeMove *out_move,
    uint8_t *out_flag
)
{
    double initial_alpha;
    double initial_beta;
    uint64_t key;
    const TTEntry *entry;
    NativeMove tt_move;
    NativeMove best_move;
    NativeMove legal_moves[MAX_MOVES];
    int legal_count;
    int move_index;
    int maximizing;
    int opponent;
    double best_value;
    uint8_t flag;

    if (deadline_reached(ctx)) {
        return 1;
    }
    sync_root_alpha(ctx, &alpha);
    initial_alpha = alpha;
    initial_beta = beta;
    if (remaining_game_moves == 0) {
        *out_value = evaluate_weighted_native(board, root_player, ctx->weights);
        move_clear(out_move);
        if (out_flag != NULL) {
            *out_flag = EXACT;
        }
        return 0;
    }
    if (depth == 0 && !is_terminal_position(board) && max_quiescence_depth > 0) {
        return quiescence_native(
            board,
            to_move,
            root_player,
            max_quiescence_depth,
            remaining_game_moves,
            alpha,
            beta,
            ctx,
            nodes,
            out_value,
            out_move,
            out_flag
        );
    }
    *nodes += 1;

    key = tt_key_for_state(board, to_move, TT_MODE_FULL, remaining_game_moves);
    entry = tt_lookup_seeded(ctx, key);
    if (entry != NULL && entry->depth >= depth) {
        if (entry->flag == EXACT) {
            *out_value = entry->value;
            *out_move = entry->move;
            if (out_flag != NULL) {
                *out_flag = EXACT;
            }
            return 0;
        }
        if (entry->flag == LOWERBOUND && entry->value > alpha) {
            alpha = entry->value;
        } else if (entry->flag == UPPERBOUND && entry->value < beta) {
            beta = entry->value;
        }
        if (alpha >= beta) {
            *out_value = entry->value;
            *out_move = entry->move;
            if (out_flag != NULL) {
                *out_flag = entry->flag;
            }
            return 0;
        }
        tt_move = entry->move;
    } else {
        move_clear(&tt_move);
    }

    if (terminal_value(board, root_player, depth, out_value)) {
        move_clear(out_move);
        return 0;
    }

    if (depth == 0) {
        *out_value = evaluate_weighted_native(board, root_player, ctx->weights);
        move_clear(out_move);
        if (out_flag != NULL) {
            *out_flag = EXACT;
        }
        return 0;
    }

    legal_count = generate_legal_moves_native(board, to_move, legal_moves);
    if (legal_count < 0) {
        return -1;
    }
    if (legal_count == 0) {
        if (!terminal_value(board, root_player, depth, out_value)) {
            *out_value = evaluate_weighted_native(board, root_player, ctx->weights);
        }
        move_clear(out_move);
        if (out_flag != NULL) {
            *out_flag = EXACT;
        }
        return 0;
    }

    order_moves(
        board,
        to_move,
        legal_moves,
        legal_count,
        &tt_move,
        depth < (int) (sizeof(ctx->killer_moves) / sizeof(ctx->killer_moves[0])) ? &ctx->killer_moves[depth] : &tt_move
    );

    maximizing = to_move == root_player;
    opponent = to_move == BLACK ? WHITE : BLACK;
    move_clear(&best_move);
    best_value = maximizing ? -INFINITY : INFINITY;

    for (move_index = 0; move_index < legal_count; ++move_index) {
        BoardState child = *board;
        double child_value;
        NativeMove child_best_reply;
        uint8_t child_flag = EXACT;
        int child_remaining_game_moves = next_remaining_game_moves(remaining_game_moves);
        int status;

        if (deadline_reached(ctx)) {
            return 1;
        }
        sync_root_alpha(ctx, &alpha);
        if (move_index > 0 && beta <= alpha) {
            break;
        }

        apply_move_native(&child, &legal_moves[move_index], to_move);
        status = minimax_native(
            &child,
            opponent,
            root_player,
            depth - 1,
            child_remaining_game_moves,
            max_quiescence_depth,
            alpha,
            beta,
            ctx,
            nodes,
            &child_value,
            &child_best_reply,
            &child_flag
        );
        if (status != 0) {
            return status;
        }

        if (maximizing) {
            if (child_value > best_value ||
                    (child_value == best_value &&
                        prefer_by_tie_break(ctx->tie_break_lexicographic, &legal_moves[move_index], &best_move))) {
                best_value = child_value;
                best_move = legal_moves[move_index];
            }
            if (best_value > alpha) {
                alpha = best_value;
            }
            if (beta <= alpha) {
                if (depth < (int) (sizeof(ctx->killer_moves) / sizeof(ctx->killer_moves[0]))) {
                    ctx->killer_moves[depth] = legal_moves[move_index];
                }
                break;
            }
        } else {
            if (child_value < best_value ||
                    (child_value == best_value &&
                        prefer_by_tie_break(ctx->tie_break_lexicographic, &legal_moves[move_index], &best_move))) {
                best_value = child_value;
                best_move = legal_moves[move_index];
            }
            if (best_value < beta) {
                beta = best_value;
            }
            if (beta <= alpha) {
                if (depth < (int) (sizeof(ctx->killer_moves) / sizeof(ctx->killer_moves[0]))) {
                    ctx->killer_moves[depth] = legal_moves[move_index];
                }
                break;
            }
        }
    }

    flag = EXACT;
    if (ctx->root_alpha_floor > initial_alpha) {
        initial_alpha = ctx->root_alpha_floor;
    }
    if (best_value <= initial_alpha) {
        flag = UPPERBOUND;
    } else if (best_value >= initial_beta) {
        flag = LOWERBOUND;
    }
    if (!tt_store(&ctx->tt, key, depth, best_value, flag, &best_move)) {
        return -1;
    }

    *out_value = best_value;
    *out_move = best_move;
    if (out_flag != NULL) {
        *out_flag = flag;
    }
    return 0;
}

/* Initializes per-search context state from immutable search inputs. */
static void
init_search_context(
    SearchContext *ctx,
    const double *weights,
    int tie_break_lexicographic,
    double deadline_at
)
{
    memset(ctx, 0, sizeof(*ctx));
    memcpy(ctx->weights, weights, sizeof(double) * FEATURE_COUNT);
    ctx->tie_break_lexicographic = tie_break_lexicographic;
    ctx->deadline_at = deadline_at;
    ctx->tt_seed = NULL;
    ctx->shared_root_alpha_bits = NULL;
    ctx->root_alpha_floor = -INFINITY;
}

/* Evaluates one ordered root move using the existing recursive search logic. */
static int
evaluate_root_move(
    const BoardState *board,
    const NativeMove *move,
    int player,
    int opponent,
    int depth_after_move,
    int max_quiescence_depth,
    int root_remaining_game_moves,
    double alpha,
    double beta,
    SearchContext *ctx,
    uint64_t *nodes,
    double *out_value,
    int *out_exact
)
{
    BoardState child = *board;
    NativeMove ignored;
    uint8_t result_flag = EXACT;
    int child_remaining_game_moves = next_remaining_game_moves(root_remaining_game_moves);
    int status;

    if (deadline_reached(ctx)) {
        return 1;
    }

    apply_move_native(&child, move, player);
    *nodes += 1;
    status = minimax_native(
        &child,
        opponent,
        player,
        depth_after_move,
        child_remaining_game_moves,
        max_quiescence_depth,
        alpha,
        beta,
        ctx,
        nodes,
        out_value,
        &ignored,
        &result_flag
    );
    if (status == 0 && out_exact != NULL) {
        *out_exact = result_flag == EXACT;
    }
    return status;
}

/* Resets one worker's reusable local context for a new iterative-deepening depth. */
static int
prepare_root_search_worker(RootSearchWorker *worker)
{
    RootSearchThreadPool *pool = worker->pool;

    tt_free(&worker->ctx.tt);
    init_search_context(&worker->ctx, pool->weights, pool->tie_break_lexicographic, pool->deadline_at);
    worker->ctx.tt_seed = pool->tt_seed;
    worker->ctx.shared_root_alpha_bits = &pool->alpha_bits;
    worker->ctx.root_alpha_floor = root_alpha_load(worker->ctx.shared_root_alpha_bits);
    if (pool->killer_moves_seed != NULL) {
        memcpy(worker->ctx.killer_moves, pool->killer_moves_seed, sizeof(worker->ctx.killer_moves));
    }
    return tt_init(&worker->ctx.tt, 1024U);
}

/* Drains queued root jobs on one persistent worker thread. */
static void
root_search_worker_loop(RootSearchWorker *worker)
{
    RootSearchThreadPool *pool = worker->pool;
    int seen_generation = 0;

    memset(&worker->ctx, 0, sizeof(worker->ctx));

    for (;;) {
        root_mutex_lock(&pool->mutex);
        while (!pool->stop && seen_generation == pool->generation) {
            root_condition_wait(&pool->work_available, &pool->mutex);
        }
        if (pool->stop) {
            root_mutex_unlock(&pool->mutex);
            break;
        }
        seen_generation = pool->generation;
        root_mutex_unlock(&pool->mutex);

        if (!prepare_root_search_worker(worker)) {
            root_mutex_lock(&pool->mutex);
            pool->worker_init_error = 1;
            pool->cancel_jobs = 1;
            pool->finished_workers += 1;
            root_condition_broadcast(&pool->work_finished);
            root_mutex_unlock(&pool->mutex);
            continue;
        }

        for (;;) {
            int job_index;
            double value = 0.0;
            uint64_t nodes = 0;
            int exact = 1;
            int status;

            root_mutex_lock(&pool->mutex);
            if (pool->cancel_jobs || pool->next_job_index >= pool->end_job_index) {
                pool->finished_workers += 1;
                root_condition_broadcast(&pool->work_finished);
                root_mutex_unlock(&pool->mutex);
                break;
            }
            job_index = pool->next_job_index;
            pool->next_job_index += 1;
            root_mutex_unlock(&pool->mutex);

            worker->ctx.root_alpha_floor = root_alpha_load(worker->ctx.shared_root_alpha_bits);

            status = evaluate_root_move(
                pool->board,
                &pool->ordered_root[job_index],
                pool->player,
                pool->opponent,
                pool->depth_after_move,
                pool->max_quiescence_depth,
                pool->root_remaining_game_moves,
                worker->ctx.root_alpha_floor,
                INFINITY,
                &worker->ctx,
                &nodes,
                &value,
                &exact
            );

            root_mutex_lock(&pool->mutex);
            {
                RootJobResult *result = &pool->results[job_index];
                result->alpha = worker->ctx.root_alpha_floor;
                result->value = value;
                result->nodes = nodes;
                if (status == 0) {
                    result->status = ROOT_JOB_DONE;
                    result->needs_exact = !exact;
                } else if (status == 1) {
                    result->status = ROOT_JOB_TIMED_OUT;
                } else {
                    result->status = ROOT_JOB_ERROR;
                }
                if (result->status != ROOT_JOB_DONE) {
                    pool->cancel_jobs = 1;
                }
                root_condition_broadcast(&pool->work_finished);
            }
            root_mutex_unlock(&pool->mutex);
        }
    }

    tt_free(&worker->ctx.tt);
}

/* Starts one queued root-search iteration on the persistent worker pool. */
static int
root_search_thread_pool_begin(
    RootSearchThreadPool *pool,
    const BoardState *board,
    const NativeMove *ordered_root,
    int start_index,
    int end_index,
    int player,
    int opponent,
    int depth_after_move,
    int max_quiescence_depth,
    int root_remaining_game_moves,
    double alpha,
    double deadline_at,
    const double *weights,
    int tie_break_lexicographic,
    const TTTable *tt_seed,
    const NativeMove *killer_moves_seed,
    RootJobResult *results
)
{
    tt_free(&pool->tt_seed_snapshot);
    if (!tt_clone(&pool->tt_seed_snapshot, tt_seed)) {
        return 0;
    }

    root_mutex_lock(&pool->mutex);
    pool->board = board;
    pool->ordered_root = ordered_root;
    pool->player = player;
    pool->opponent = opponent;
    pool->depth_after_move = depth_after_move;
    pool->max_quiescence_depth = max_quiescence_depth;
    pool->root_remaining_game_moves = root_remaining_game_moves;
    root_alpha_store(&pool->alpha_bits, alpha);
    pool->deadline_at = deadline_at;
    pool->weights = weights;
    pool->tie_break_lexicographic = tie_break_lexicographic;
    pool->tt_seed = &pool->tt_seed_snapshot;
    pool->killer_moves_seed = killer_moves_seed;
    pool->results = results;
    pool->next_job_index = start_index;
    pool->end_job_index = end_index;
    pool->cancel_jobs = 0;
    pool->worker_init_error = 0;
    pool->finished_workers = 0;
    pool->generation += 1;
    root_condition_broadcast(&pool->work_available);
    root_mutex_unlock(&pool->mutex);
    return 1;
}

/* Waits until the requested ordered root result is available or the iteration aborts. */
static int
root_search_thread_pool_wait_for_result(RootSearchThreadPool *pool, int result_index, RootJobResult *out_result)
{
    int ready = 0;

    root_mutex_lock(&pool->mutex);
    while (pool->results[result_index].status == ROOT_JOB_PENDING &&
            pool->finished_workers < pool->worker_count &&
            !pool->worker_init_error) {
        root_condition_wait(&pool->work_finished, &pool->mutex);
    }
    if (pool->results[result_index].status != ROOT_JOB_PENDING) {
        *out_result = pool->results[result_index];
        ready = 1;
    }
    root_mutex_unlock(&pool->mutex);
    return ready;
}

/* Publishes a stronger root alpha so future queued jobs prune more aggressively. */
static void
root_search_thread_pool_update_alpha(RootSearchThreadPool *pool, double alpha)
{
    root_alpha_raise(&pool->alpha_bits, alpha);
}

/* Stops assigning new jobs in the current queued root-search iteration. */
static void
root_search_thread_pool_cancel(RootSearchThreadPool *pool)
{
    root_mutex_lock(&pool->mutex);
    pool->cancel_jobs = 1;
    root_mutex_unlock(&pool->mutex);
}

/* Waits for every worker to finish the current queued root-search iteration. */
static int
root_search_thread_pool_wait_idle(RootSearchThreadPool *pool)
{
    int worker_init_error;

    root_mutex_lock(&pool->mutex);
    while (pool->finished_workers < pool->worker_count) {
        root_condition_wait(&pool->work_finished, &pool->mutex);
    }
    worker_init_error = pool->worker_init_error;
    root_mutex_unlock(&pool->mutex);
    return worker_init_error ? 0 : 1;
}

#ifdef _WIN32
static unsigned __stdcall
root_search_worker_entry(void *raw_context)
{
    root_search_worker_loop((RootSearchWorker *) raw_context);
    return 0U;
}

static int
start_root_search_worker(RootSearchWorker *worker)
{
    uintptr_t raw_handle = _beginthreadex(NULL, 0, root_search_worker_entry, worker, 0, NULL);
    if (raw_handle == 0U) {
        return 0;
    }
    worker->handle = (HANDLE) raw_handle;
    worker->started = 1;
    return 1;
}

static void
join_root_search_worker(RootSearchWorker *worker)
{
    WaitForSingleObject(worker->handle, INFINITE);
    CloseHandle(worker->handle);
    worker->started = 0;
}
#else
static void *
root_search_worker_entry(void *raw_context)
{
    root_search_worker_loop((RootSearchWorker *) raw_context);
    return NULL;
}

static int
start_root_search_worker(RootSearchWorker *worker)
{
    if (pthread_create(&worker->handle, NULL, root_search_worker_entry, worker) != 0) {
        return 0;
    }
    worker->started = 1;
    return 1;
}

static void
join_root_search_worker(RootSearchWorker *worker)
{
    pthread_join(worker->handle, NULL);
    worker->started = 0;
}
#endif

/* Starts the persistent worker pool used for native root parallelism. */
static int
root_search_thread_pool_init(RootSearchThreadPool *pool, int worker_count)
{
    int worker_index;

    memset(pool, 0, sizeof(*pool));
    pool->worker_count = worker_count;
    if (worker_count <= 0) {
        return 1;
    }

    if (!root_mutex_init(&pool->mutex)) {
        return 0;
    }
    root_condition_init(&pool->work_available);
    root_condition_init(&pool->work_finished);
    pool->sync_ready = 1;

    pool->workers = (RootSearchWorker *) calloc((size_t) worker_count, sizeof(RootSearchWorker));
    if (pool->workers == NULL) {
        root_condition_destroy(&pool->work_available);
        root_condition_destroy(&pool->work_finished);
        root_mutex_destroy(&pool->mutex);
        pool->sync_ready = 0;
        return 0;
    }

    for (worker_index = 0; worker_index < worker_count; ++worker_index) {
        pool->workers[worker_index].pool = pool;
        if (!start_root_search_worker(&pool->workers[worker_index])) {
            root_mutex_lock(&pool->mutex);
            pool->stop = 1;
            root_condition_broadcast(&pool->work_available);
            root_mutex_unlock(&pool->mutex);
            while (--worker_index >= 0) {
                join_root_search_worker(&pool->workers[worker_index]);
            }
            free(pool->workers);
            pool->workers = NULL;
            root_condition_destroy(&pool->work_available);
            root_condition_destroy(&pool->work_finished);
            root_mutex_destroy(&pool->mutex);
            pool->sync_ready = 0;
            return 0;
        }
    }

    return 1;
}

/* Stops the persistent worker pool and releases all synchronization state. */
static void
root_search_thread_pool_destroy(RootSearchThreadPool *pool)
{
    int worker_index;

    if (pool->workers != NULL && pool->sync_ready) {
        root_mutex_lock(&pool->mutex);
        pool->stop = 1;
        root_condition_broadcast(&pool->work_available);
        root_mutex_unlock(&pool->mutex);
    }

    if (pool->workers != NULL) {
        for (worker_index = 0; worker_index < pool->worker_count; ++worker_index) {
            if (pool->workers[worker_index].started) {
                join_root_search_worker(&pool->workers[worker_index]);
            }
        }
        free(pool->workers);
        pool->workers = NULL;
    }

    tt_free(&pool->tt_seed_snapshot);
    pool->tt_seed = NULL;

    if (pool->sync_ready) {
        root_condition_destroy(&pool->work_available);
        root_condition_destroy(&pool->work_finished);
        root_mutex_destroy(&pool->mutex);
        pool->sync_ready = 0;
    }
}

/* Preserves the original serial native search implementation for single-core fallback. */
static int
search_weighted_native_serial(
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
)
{
    SearchContext ctx;
    NativeMove legal_moves[MAX_MOVES];
    int legal_count;
    int idx;
    int opponent = player == BLACK ? WHITE : BLACK;
    uint64_t total_nodes = 0;
    NativeMove best_move;
    double best_score = 0.0;
    int completed_depth = 0;
    int timed_out = 0;
    int avoidance_applied = 0;
    RootCandidate saved_candidates[MAX_MOVES];
    int saved_candidate_count = 0;
    int root_candidate_cap = root_candidate_limit;

    if (root_candidate_cap < 0) {
        root_candidate_cap = 0;
    } else if (root_candidate_cap > MAX_MOVES) {
        root_candidate_cap = MAX_MOVES;
    }

    init_search_context(
        &ctx,
        weights,
        tie_break_lexicographic,
        has_deadline ? monotonic_seconds() + ((double) time_budget_ms / 1000.0) : -1.0
    );
    if (!tt_init(&ctx.tt, 1024U)) {
        return -1;
    }

    if (terminal_value(board, player, requested_depth, &out_result->score)) {
        move_clear(&out_result->move);
        out_result->nodes = 0;
        out_result->completed_depth = 0;
        out_result->timed_out = 0;
        out_result->avoidance_applied = 0;
        out_result->root_candidate_count = 0;
        tt_free(&ctx.tt);
        return 0;
    }

    legal_count = generate_legal_moves_native(board, player, legal_moves);
    if (legal_count < 0) {
        tt_free(&ctx.tt);
        return -1;
    }
    if (legal_count == 0) {
        move_clear(&out_result->move);
        if (!terminal_value(board, player, requested_depth, &out_result->score)) {
            out_result->score = evaluate_weighted_native(board, player, weights);
        }
        out_result->nodes = 0;
        out_result->completed_depth = 0;
        out_result->timed_out = 0;
        out_result->avoidance_applied = 0;
        out_result->root_candidate_count = 0;
        tt_free(&ctx.tt);
        return 0;
    }

    if (move_has_value(avoid_move) && legal_count > 1) {
        NativeMove filtered[MAX_MOVES];
        int filtered_count = 0;
        for (idx = 0; idx < legal_count; ++idx) {
            if (!move_equal(&legal_moves[idx], avoid_move)) {
                filtered[filtered_count++] = legal_moves[idx];
            }
        }
        if (filtered_count > 0) {
            memcpy(legal_moves, filtered, sizeof(NativeMove) * filtered_count);
            legal_count = filtered_count;
            avoidance_applied = 1;
        }
    }

    best_move = legal_moves[0];

    for (idx = 1; idx <= requested_depth; ++idx) {
        uint64_t iteration_nodes = 0;
        NativeMove current_best_move = best_move;
        double current_best_score = -INFINITY;
        RootCandidate depth_candidates[MAX_MOVES];
        int depth_candidate_count = 0;
        const TTEntry *root_entry;
        NativeMove tt_move;
        NativeMove ordered_root[MAX_MOVES];
        double alpha = -INFINITY;
        double beta = INFINITY;
        int move_index;
        int root_remaining_game_moves = has_remaining_game_moves ? remaining_game_moves : -1;

        memset(ctx.killer_moves, 0, sizeof(ctx.killer_moves));

        memcpy(ordered_root, legal_moves, sizeof(NativeMove) * legal_count);
        root_entry = tt_lookup(&ctx.tt, tt_key_for_state(board, player, TT_MODE_FULL, root_remaining_game_moves));
        if (root_entry != NULL) {
            tt_move = root_entry->move;
        } else {
            move_clear(&tt_move);
        }
        order_moves(board, player, ordered_root, legal_count, &tt_move, idx < (int) (sizeof(ctx.killer_moves) / sizeof(ctx.killer_moves[0])) ? &ctx.killer_moves[idx] : &tt_move);

        for (move_index = 0; move_index < legal_count; ++move_index) {
            double child_value = 0.0;
            int status;

            status = evaluate_root_move(
                board,
                &ordered_root[move_index],
                player,
                opponent,
                idx - 1,
                max_quiescence_depth,
                root_remaining_game_moves,
                alpha,
                beta,
                &ctx,
                &iteration_nodes,
                &child_value,
                NULL
            );
            if (status == 1) {
                timed_out = 1;
                break;
            }
            if (status != 0) {
                tt_free(&ctx.tt);
                return -1;
            }

            if (child_value > current_best_score || (child_value == current_best_score && prefer_by_tie_break(ctx.tie_break_lexicographic, &ordered_root[move_index], &current_best_move))) {
                current_best_score = child_value;
                current_best_move = ordered_root[move_index];
            }

            if (root_candidate_cap > 0 && depth_candidate_count < MAX_MOVES) {
                depth_candidates[depth_candidate_count].move = ordered_root[move_index];
                depth_candidates[depth_candidate_count].score = child_value;
                depth_candidates[depth_candidate_count].depth = idx;
                depth_candidate_count += 1;
            }

            if (current_best_score > alpha) {
                alpha = current_best_score;
            }
        }

        total_nodes += iteration_nodes;

        if (timed_out) {
            if (current_best_score != -INFINITY) {
                best_score = current_best_score;
            }
            best_move = current_best_move;
            saved_candidate_count = depth_candidate_count;
            if (saved_candidate_count > 0) {
                memcpy(saved_candidates, depth_candidates, sizeof(RootCandidate) * depth_candidate_count);
            }
            break;
        }

        if (!tt_store(&ctx.tt, tt_key_for_state(board, player, TT_MODE_FULL, root_remaining_game_moves), idx, current_best_score, EXACT, &current_best_move)) {
            tt_free(&ctx.tt);
            return -1;
        }

        best_move = current_best_move;
        best_score = current_best_score;
        completed_depth = idx;
        saved_candidate_count = depth_candidate_count;
        if (saved_candidate_count > 0) {
            memcpy(saved_candidates, depth_candidates, sizeof(RootCandidate) * depth_candidate_count);
        }
    }

    out_result->move = best_move;
    out_result->score = best_score;
    out_result->nodes = total_nodes;
    out_result->completed_depth = completed_depth;
    out_result->timed_out = timed_out;
    out_result->avoidance_applied = avoidance_applied;
    out_result->root_candidate_count = saved_candidate_count;
    if (saved_candidate_count > 0) {
        memcpy(out_result->root_candidates, saved_candidates, sizeof(RootCandidate) * saved_candidate_count);
    }

    tt_free(&ctx.tt);
    return 0;
}

/* Parallelizes root move evaluation while preserving serial move reduction order. */
static int
search_weighted_native_threaded(
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
)
{
    SearchContext ctx;
    RootSearchThreadPool pool;
    NativeMove legal_moves[MAX_MOVES];
    int legal_count;
    int idx;
    int opponent = player == BLACK ? WHITE : BLACK;
    uint64_t total_nodes = 0;
    NativeMove best_move;
    double best_score = 0.0;
    int completed_depth = 0;
    int timed_out = 0;
    int avoidance_applied = 0;
    RootCandidate saved_candidates[MAX_MOVES];
    int saved_candidate_count = 0;
    int root_candidate_cap = root_candidate_limit;
    int threaded_worker_count;

    if (root_candidate_limit > 0) {
        return search_weighted_native_serial(
            board,
            player,
            weights,
            requested_depth,
            max_quiescence_depth,
            has_deadline,
            time_budget_ms,
            has_remaining_game_moves,
            remaining_game_moves,
            tie_break_lexicographic,
            avoid_move,
            root_candidate_limit,
            out_result
        );
    }
    if (requested_depth <= 1) {
        return search_weighted_native_serial(
            board,
            player,
            weights,
            requested_depth,
            max_quiescence_depth,
            has_deadline,
            time_budget_ms,
            has_remaining_game_moves,
            remaining_game_moves,
            tie_break_lexicographic,
            avoid_move,
            root_candidate_limit,
            out_result
        );
    }

    if (root_candidate_cap < 0) {
        root_candidate_cap = 0;
    } else if (root_candidate_cap > MAX_MOVES) {
        root_candidate_cap = MAX_MOVES;
    }

    memset(&pool, 0, sizeof(pool));

    legal_count = generate_legal_moves_native(board, player, legal_moves);
    if (legal_count < 0) {
        return -1;
    }
    if (legal_count == 0) {
        move_clear(&out_result->move);
        out_result->score = evaluate_weighted_native(board, player, weights);
        out_result->nodes = 0;
        out_result->completed_depth = 0;
        out_result->timed_out = 0;
        out_result->avoidance_applied = 0;
        out_result->root_candidate_count = 0;
        return 0;
    }

    if (move_has_value(avoid_move) && legal_count > 1) {
        NativeMove filtered[MAX_MOVES];
        int filtered_count = 0;
        for (idx = 0; idx < legal_count; ++idx) {
            if (!move_equal(&legal_moves[idx], avoid_move)) {
                filtered[filtered_count++] = legal_moves[idx];
            }
        }
        if (filtered_count > 0) {
            memcpy(legal_moves, filtered, sizeof(NativeMove) * filtered_count);
            legal_count = filtered_count;
            avoidance_applied = 1;
        }
    }

    threaded_worker_count = resolve_root_worker_count(legal_count);
    if (threaded_worker_count < 2) {
        return search_weighted_native_serial(
            board,
            player,
            weights,
            requested_depth,
            max_quiescence_depth,
            has_deadline,
            time_budget_ms,
            has_remaining_game_moves,
            remaining_game_moves,
            tie_break_lexicographic,
            avoid_move,
            root_candidate_limit,
            out_result
        );
    }
    if (!root_search_thread_pool_init(&pool, threaded_worker_count)) {
        return search_weighted_native_serial(
            board,
            player,
            weights,
            requested_depth,
            max_quiescence_depth,
            has_deadline,
            time_budget_ms,
            has_remaining_game_moves,
            remaining_game_moves,
            tie_break_lexicographic,
            avoid_move,
            root_candidate_limit,
            out_result
        );
    }

    init_search_context(
        &ctx,
        weights,
        tie_break_lexicographic,
        has_deadline ? monotonic_seconds() + ((double) time_budget_ms / 1000.0) : -1.0
    );
    if (!tt_init(&ctx.tt, 1024U)) {
        root_search_thread_pool_destroy(&pool);
        return -1;
    }

    best_move = legal_moves[0];

    for (idx = 1; idx <= requested_depth; ++idx) {
        uint64_t iteration_nodes = 0;
        NativeMove current_best_move = best_move;
        double current_best_score = -INFINITY;
        RootCandidate depth_candidates[MAX_MOVES];
        double root_scores[MAX_MOVES];
        uint8_t root_score_valid[MAX_MOVES];
        int depth_candidate_count = 0;
        const TTEntry *root_entry;
        NativeMove tt_move;
        NativeMove ordered_root[MAX_MOVES];
        double alpha = -INFINITY;
        double beta = INFINITY;
        int move_index;
        int root_remaining_game_moves = has_remaining_game_moves ? remaining_game_moves : -1;

        memset(ctx.killer_moves, 0, sizeof(ctx.killer_moves));
        memset(root_score_valid, 0, sizeof(root_score_valid));

        memcpy(ordered_root, legal_moves, sizeof(NativeMove) * legal_count);
        root_entry = tt_lookup(&ctx.tt, tt_key_for_state(board, player, TT_MODE_FULL, root_remaining_game_moves));
        if (root_entry != NULL) {
            tt_move = root_entry->move;
        } else {
            move_clear(&tt_move);
        }
        order_moves(
            board,
            player,
            ordered_root,
            legal_count,
            &tt_move,
            idx < (int) (sizeof(ctx.killer_moves) / sizeof(ctx.killer_moves[0])) ? &ctx.killer_moves[idx] : &tt_move
        );

        {
            double child_value = 0.0;
            int status = evaluate_root_move(
                board,
                &ordered_root[0],
                player,
                opponent,
                idx - 1,
                max_quiescence_depth,
                root_remaining_game_moves,
                alpha,
                beta,
                &ctx,
                &iteration_nodes,
                &child_value,
                NULL
            );
            if (status == 1) {
                timed_out = 1;
                total_nodes += iteration_nodes;
                break;
            }
            if (status != 0) {
                tt_free(&ctx.tt);
                root_search_thread_pool_destroy(&pool);
                return -1;
            }
            current_best_score = child_value;
            current_best_move = ordered_root[0];
            root_scores[0] = child_value;
            root_score_valid[0] = 1;
            if (root_candidate_cap > 0) {
                depth_candidates[depth_candidate_count].move = ordered_root[0];
                depth_candidates[depth_candidate_count].score = child_value;
                depth_candidates[depth_candidate_count].depth = idx;
                depth_candidate_count += 1;
            }
            if (current_best_score > alpha) {
                alpha = current_best_score;
            }
        }

        if (legal_count > 1) {
            int serial_seed_limit = 1;
            if (threaded_worker_count >= 2) {
                int desired_seed_limit = threaded_worker_count > 2 ? threaded_worker_count : 2;
                serial_seed_limit = legal_count < desired_seed_limit ? legal_count : desired_seed_limit;
            }

            for (move_index = 1; move_index < serial_seed_limit; ++move_index) {
                double child_value = 0.0;
                int status = evaluate_root_move(
                    board,
                    &ordered_root[move_index],
                    player,
                    opponent,
                    idx - 1,
                    max_quiescence_depth,
                    root_remaining_game_moves,
                    alpha,
                    beta,
                    &ctx,
                    &iteration_nodes,
                    &child_value,
                    NULL
                );
                if (status == 1) {
                    timed_out = 1;
                    break;
                }
                if (status != 0) {
                    tt_free(&ctx.tt);
                    root_search_thread_pool_destroy(&pool);
                    return -1;
                }
                root_scores[move_index] = child_value;
                root_score_valid[move_index] = 1;
                if (child_value > current_best_score ||
                        (child_value == current_best_score &&
                            prefer_by_tie_break(ctx.tie_break_lexicographic, &ordered_root[move_index], &current_best_move))) {
                    current_best_score = child_value;
                    current_best_move = ordered_root[move_index];
                }
                if (current_best_score > alpha) {
                    alpha = current_best_score;
                }
            }

            if (!timed_out && threaded_worker_count >= 2 && idx > 1 && legal_count > serial_seed_limit) {
                RootJobResult worker_results[MAX_MOVES];

                memset(worker_results, 0, sizeof(worker_results));
                if (!root_search_thread_pool_begin(
                        &pool,
                        board,
                        ordered_root,
                        serial_seed_limit,
                        legal_count,
                        player,
                        opponent,
                        idx - 1,
                        max_quiescence_depth,
                        root_remaining_game_moves,
                        alpha,
                        ctx.deadline_at,
                        weights,
                        tie_break_lexicographic,
                        &ctx.tt,
                        ctx.killer_moves,
                        worker_results)) {
                    tt_free(&ctx.tt);
                    root_search_thread_pool_destroy(&pool);
                    return -1;
                }

                for (move_index = serial_seed_limit; move_index < legal_count; ++move_index) {
                    RootJobResult result;
                    double child_value;

                    if (!root_search_thread_pool_wait_for_result(&pool, move_index, &result)) {
                        root_search_thread_pool_cancel(&pool);
                        if (!root_search_thread_pool_wait_idle(&pool)) {
                            tt_free(&ctx.tt);
                            root_search_thread_pool_destroy(&pool);
                            return -1;
                        }
                        tt_free(&ctx.tt);
                        root_search_thread_pool_destroy(&pool);
                        return -1;
                    }

                    child_value = result.value;
                    iteration_nodes += result.nodes;
                    if (result.status == ROOT_JOB_ERROR) {
                        root_search_thread_pool_cancel(&pool);
                        if (!root_search_thread_pool_wait_idle(&pool)) {
                            tt_free(&ctx.tt);
                            root_search_thread_pool_destroy(&pool);
                            return -1;
                        }
                        tt_free(&ctx.tt);
                        root_search_thread_pool_destroy(&pool);
                        return -1;
                    }
                    if (result.status != ROOT_JOB_DONE) {
                        timed_out = 1;
                        root_search_thread_pool_cancel(&pool);
                        break;
                    }
                    if (child_value >= alpha) {
                        int status = evaluate_root_move(
                            board,
                            &ordered_root[move_index],
                            player,
                            opponent,
                            idx - 1,
                            max_quiescence_depth,
                            root_remaining_game_moves,
                            alpha,
                            beta,
                            &ctx,
                            &iteration_nodes,
                            &child_value,
                            NULL
                        );
                        if (status == 1) {
                            timed_out = 1;
                            root_search_thread_pool_cancel(&pool);
                            break;
                        }
                        if (status != 0) {
                            root_search_thread_pool_cancel(&pool);
                            if (!root_search_thread_pool_wait_idle(&pool)) {
                                tt_free(&ctx.tt);
                                root_search_thread_pool_destroy(&pool);
                                return -1;
                            }
                            tt_free(&ctx.tt);
                            root_search_thread_pool_destroy(&pool);
                            return -1;
                        }
                    }
                    root_scores[move_index] = child_value;
                    root_score_valid[move_index] = 1;
                    if (child_value > current_best_score ||
                            (child_value == current_best_score &&
                                prefer_by_tie_break(ctx.tie_break_lexicographic, &ordered_root[move_index], &current_best_move))) {
                        current_best_score = child_value;
                        current_best_move = ordered_root[move_index];
                    }
                    if (current_best_score > alpha) {
                        alpha = current_best_score;
                        root_search_thread_pool_update_alpha(&pool, alpha);
                    }
                }

                if (!root_search_thread_pool_wait_idle(&pool)) {
                    tt_free(&ctx.tt);
                    root_search_thread_pool_destroy(&pool);
                    return -1;
                }
            } else if (!timed_out) {
                for (move_index = serial_seed_limit; move_index < legal_count; ++move_index) {
                    double child_value = 0.0;
                    int status = evaluate_root_move(
                        board,
                        &ordered_root[move_index],
                        player,
                        opponent,
                        idx - 1,
                        max_quiescence_depth,
                        root_remaining_game_moves,
                        alpha,
                        beta,
                        &ctx,
                        &iteration_nodes,
                        &child_value,
                        NULL
                    );
                    if (status == 1) {
                        timed_out = 1;
                        break;
                    }
                    if (status != 0) {
                        tt_free(&ctx.tt);
                        root_search_thread_pool_destroy(&pool);
                        return -1;
                    }
                    root_scores[move_index] = child_value;
                    root_score_valid[move_index] = 1;
                    if (child_value > current_best_score ||
                            (child_value == current_best_score &&
                                prefer_by_tie_break(ctx.tie_break_lexicographic, &ordered_root[move_index], &current_best_move))) {
                        current_best_score = child_value;
                        current_best_move = ordered_root[move_index];
                    }
                    if (root_candidate_cap > 0 && depth_candidate_count < MAX_MOVES) {
                        depth_candidates[depth_candidate_count].move = ordered_root[move_index];
                        depth_candidates[depth_candidate_count].score = child_value;
                        depth_candidates[depth_candidate_count].depth = idx;
                        depth_candidate_count += 1;
                    }
                    if (current_best_score > alpha) {
                        alpha = current_best_score;
                    }
                }
            }
        }

        if (!timed_out && threaded_worker_count >= 2) {
            int tied_count = 0;
            const double tie_epsilon = 1e-6;

            for (move_index = 0; move_index < legal_count; ++move_index) {
                if (root_score_valid[move_index] &&
                        fabs(root_scores[move_index] - current_best_score) <= tie_epsilon) {
                    tied_count += 1;
                }
            }

            if (tied_count > 1) {
                SearchContext tie_ctx;
                NativeMove tie_best_move;
                double tie_best_score = -INFINITY;

                init_search_context(&tie_ctx, weights, tie_break_lexicographic, ctx.deadline_at);
                if (!tt_init(&tie_ctx.tt, 1024U)) {
                    tt_free(&ctx.tt);
                    root_search_thread_pool_destroy(&pool);
                    return -1;
                }

                move_clear(&tie_best_move);
                for (move_index = 0; move_index < legal_count; ++move_index) {
                    double tie_value = 0.0;
                    uint64_t tie_nodes = 0;
                    int status;

                    if (!root_score_valid[move_index] ||
                            fabs(root_scores[move_index] - current_best_score) > tie_epsilon) {
                        continue;
                    }

                    status = evaluate_root_move(
                        board,
                        &ordered_root[move_index],
                        player,
                        opponent,
                        idx - 1,
                        max_quiescence_depth,
                        root_remaining_game_moves,
                        -INFINITY,
                        INFINITY,
                        &tie_ctx,
                        &tie_nodes,
                        &tie_value,
                        NULL
                    );
                    iteration_nodes += tie_nodes;
                    if (status == 1) {
                        timed_out = 1;
                        break;
                    }
                    if (status != 0) {
                        tt_free(&tie_ctx.tt);
                        tt_free(&ctx.tt);
                        root_search_thread_pool_destroy(&pool);
                        return -1;
                    }
                    if (tie_value > tie_best_score ||
                            (tie_value == tie_best_score &&
                                prefer_by_tie_break(ctx.tie_break_lexicographic, &ordered_root[move_index], &tie_best_move))) {
                        tie_best_score = tie_value;
                        tie_best_move = ordered_root[move_index];
                    }
                }

                tt_free(&tie_ctx.tt);
                if (timed_out) {
                    total_nodes += iteration_nodes;
                    if (current_best_score != -INFINITY) {
                        best_score = current_best_score;
                    }
                    best_move = current_best_move;
                    break;
                }
                if (move_has_value(&tie_best_move)) {
                    current_best_move = tie_best_move;
                    current_best_score = tie_best_score;
                    alpha = current_best_score;
                }
            }
        }

        total_nodes += iteration_nodes;

        if (timed_out) {
            if (current_best_score != -INFINITY) {
                best_score = current_best_score;
            }
            best_move = current_best_move;
            saved_candidate_count = depth_candidate_count;
            if (saved_candidate_count > 0) {
                memcpy(saved_candidates, depth_candidates, sizeof(RootCandidate) * depth_candidate_count);
            }
            break;
        }

        if (!tt_store(
                &ctx.tt,
                tt_key_for_state(board, player, TT_MODE_FULL, root_remaining_game_moves),
                idx,
                current_best_score,
                EXACT,
                &current_best_move)) {
            tt_free(&ctx.tt);
            root_search_thread_pool_destroy(&pool);
            return -1;
        }

        best_move = current_best_move;
        best_score = current_best_score;
        completed_depth = idx;
        saved_candidate_count = depth_candidate_count;
        if (saved_candidate_count > 0) {
            memcpy(saved_candidates, depth_candidates, sizeof(RootCandidate) * depth_candidate_count);
        }
    }

    out_result->move = best_move;
    out_result->score = best_score;
    out_result->nodes = total_nodes;
    out_result->completed_depth = completed_depth;
    out_result->timed_out = timed_out;
    out_result->avoidance_applied = avoidance_applied;
    out_result->root_candidate_count = saved_candidate_count;
    if (saved_candidate_count > 0) {
        memcpy(out_result->root_candidates, saved_candidates, sizeof(RootCandidate) * saved_candidate_count);
    }

    tt_free(&ctx.tt);
    root_search_thread_pool_destroy(&pool);
    return 0;
}

/* Runs iterative deepening search and returns the best move plus diagnostics. */
int
search_weighted_native(
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
)
{
    return search_weighted_native_threaded(
        board,
        player,
        weights,
        requested_depth,
        max_quiescence_depth,
        has_deadline,
        time_budget_ms,
        has_remaining_game_moves,
        remaining_game_moves,
        tie_break_lexicographic,
        avoid_move,
        root_candidate_limit,
        out_result
    );
}

/* Exposes the native serial reference search for parity tests and local benchmarking. */
int
search_weighted_native_serial_for_testing(
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
)
{
    return search_weighted_native_serial(
        board,
        player,
        weights,
        requested_depth,
        max_quiescence_depth,
        has_deadline,
        time_budget_ms,
        has_remaining_game_moves,
        remaining_game_moves,
        tie_break_lexicographic,
        avoid_move,
        root_candidate_limit,
        out_result
    );
}
