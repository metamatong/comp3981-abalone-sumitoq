/* Applies moves and runs the native alpha-beta search with iterative deepening. */
#include "common.h"

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
static TTEntry *
tt_lookup(TTTable *table, uint64_t key)
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
    NativeMove *out_move
)
{
    double initial_alpha = alpha;
    double initial_beta = beta;
    uint64_t key;
    TTEntry *entry;
    NativeMove tt_move;
    NativeMove best_move;
    double stand_pat;
    int maximizing;
    double best_value;
    NativeMove legal_moves[MAX_MOVES];
    NativeMove tactical_moves[MAX_MOVES];
    int legal_count;
    int tactical_count = 0;
    int move_index;
    int opponent;

    if (deadline_reached(ctx)) {
        return 1;
    }
    *nodes += 1;

    key = tt_key_for_state(board, to_move, TT_MODE_QUIESCENCE, remaining_game_moves);
    entry = tt_lookup(&ctx->tt, key);
    if (entry != NULL && entry->depth >= remaining_depth) {
        if (entry->flag == EXACT) {
            *out_value = entry->value;
            *out_move = entry->move;
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
            return 0;
        }
        tt_move = entry->move;
    } else {
        move_clear(&tt_move);
    }

    stand_pat = evaluate_weighted_native(board, root_player, ctx->weights);
    if (remaining_depth <= 0 || board_terminal(board) || remaining_game_moves <= 0) {
        move_clear(out_move);
        if (!tt_store(&ctx->tt, key, remaining_depth, stand_pat, EXACT, out_move)) {
            return -1;
        }
        *out_value = stand_pat;
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
        return 0;
    }

    move_clear(&best_move);
    order_moves(board, to_move, tactical_moves, tactical_count, &tt_move, &best_move);
    opponent = to_move == BLACK ? WHITE : BLACK;

    for (move_index = 0; move_index < tactical_count; ++move_index) {
        BoardState child = *board;
        double child_value = 0.0;
        NativeMove ignored;
        int child_remaining_game_moves = remaining_game_moves < 0 ? -1 : remaining_game_moves - 1;
        int status;

        if (deadline_reached(ctx)) {
            return 1;
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
            &ignored
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

    {
        uint8_t flag = EXACT;
        if (best_value <= initial_alpha) {
            flag = UPPERBOUND;
        } else if (best_value >= initial_beta) {
            flag = LOWERBOUND;
        }
        if (!tt_store(&ctx->tt, key, remaining_depth, best_value, flag, &best_move)) {
            return -1;
        }
    }

    *out_value = best_value;
    *out_move = best_move;
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
    NativeMove *out_move
)
{
    double initial_alpha = alpha;
    double initial_beta = beta;
    uint64_t key;
    TTEntry *entry;
    NativeMove tt_move;
    NativeMove best_move;
    NativeMove legal_moves[MAX_MOVES];
    int legal_count;
    int move_index;
    int maximizing;
    int opponent;
    double best_value;

    if (deadline_reached(ctx)) {
        return 1;
    }
    if (remaining_game_moves <= 0) {
        *out_value = evaluate_weighted_native(board, root_player, ctx->weights);
        move_clear(out_move);
        return 0;
    }
    if (depth == 0 && !board_terminal(board) && max_quiescence_depth > 0) {
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
            out_move
        );
    }
    *nodes += 1;

    key = tt_key_for_state(board, to_move, TT_MODE_FULL, remaining_game_moves);
    entry = tt_lookup(&ctx->tt, key);
    if (entry != NULL && entry->depth >= depth) {
        if (entry->flag == EXACT) {
            *out_value = entry->value;
            *out_move = entry->move;
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
            return 0;
        }
        tt_move = entry->move;
    } else {
        move_clear(&tt_move);
    }

    if (depth == 0 || board_terminal(board)) {
        *out_value = evaluate_weighted_native(board, root_player, ctx->weights);
        move_clear(out_move);
        return 0;
    }

    legal_count = generate_legal_moves_native(board, to_move, legal_moves);
    if (legal_count < 0) {
        return -1;
    }
    if (legal_count == 0) {
        *out_value = evaluate_weighted_native(board, root_player, ctx->weights);
        move_clear(out_move);
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
        int child_remaining_game_moves = remaining_game_moves < 0 ? -1 : remaining_game_moves - 1;
        int status;

        if (deadline_reached(ctx)) {
            return 1;
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
            &child_best_reply
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

    {
        uint8_t flag = EXACT;
        if (best_value <= initial_alpha) {
            flag = UPPERBOUND;
        } else if (best_value >= initial_beta) {
            flag = LOWERBOUND;
        }
        if (!tt_store(&ctx->tt, key, depth, best_value, flag, &best_move)) {
            return -1;
        }
    }

    *out_value = best_value;
    *out_move = best_move;
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

    memset(&ctx, 0, sizeof(ctx));
    memcpy(ctx.weights, weights, sizeof(double) * FEATURE_COUNT);
    ctx.tie_break_lexicographic = tie_break_lexicographic;
    ctx.deadline_at = has_deadline ? monotonic_seconds() + ((double) time_budget_ms / 1000.0) : -1.0;
    if (!tt_init(&ctx.tt, 1024U)) {
        return -1;
    }

    legal_count = generate_legal_moves_native(board, player, legal_moves);
    if (legal_count < 0) {
        tt_free(&ctx.tt);
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
        TTEntry *root_entry;
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
            BoardState child = *board;
            double child_value = 0.0;
            NativeMove ignored;
            int child_remaining_game_moves = root_remaining_game_moves < 0 ? -1 : root_remaining_game_moves - 1;
            int status;

            if (deadline_reached(&ctx)) {
                timed_out = 1;
                break;
            }

            apply_move_native(&child, &ordered_root[move_index], player);
            iteration_nodes += 1;
            status = minimax_native(
                &child,
                opponent,
                player,
                idx - 1,
                child_remaining_game_moves,
                max_quiescence_depth,
                alpha,
                beta,
                &ctx,
                &iteration_nodes,
                &child_value,
                &ignored
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
