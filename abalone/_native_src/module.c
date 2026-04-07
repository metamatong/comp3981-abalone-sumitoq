/* Exposes the native move generator, evaluator, and search engine to Python. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "common.h"

/* Parses a Python weight sequence into the fixed-size native weight buffer. */
static int
parse_weights(PyObject *weights_obj, double *weights_out)
{
    PyObject *weight_seq = PySequence_Fast(weights_obj, "weights must be a sequence");
    Py_ssize_t weight_count;
    Py_ssize_t idx;
    if (weight_seq == NULL) {
        return 0;
    }
    weight_count = PySequence_Fast_GET_SIZE(weight_seq);
    if (weight_count != FEATURE_COUNT) {
        Py_DECREF(weight_seq);
        PyErr_Format(PyExc_ValueError, "weights must contain exactly %d items", FEATURE_COUNT);
        return 0;
    }
    for (idx = 0; idx < weight_count; ++idx) {
        weights_out[idx] = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(weight_seq, idx));
        if (PyErr_Occurred()) {
            Py_DECREF(weight_seq);
            return 0;
        }
    }
    Py_DECREF(weight_seq);
    return 1;
}

/* Parses an optional Python move payload into the native move representation. */
static int
parse_optional_move_payload(PyObject *move_obj, NativeMove *out_move)
{
    PyObject *marbles_obj;
    PyObject *direction_obj;
    PyObject *marble_seq;
    Py_ssize_t count;
    Py_ssize_t idx;
    long dir_idx;
    uint8_t marbles[3];

    move_clear(out_move);
    if (move_obj == Py_None) {
        return 1;
    }
    if (!PyTuple_Check(move_obj) || PyTuple_Size(move_obj) != 2) {
        PyErr_SetString(PyExc_TypeError, "move payload must be a (marbles, direction) tuple");
        return 0;
    }

    marbles_obj = PyTuple_GET_ITEM(move_obj, 0);
    direction_obj = PyTuple_GET_ITEM(move_obj, 1);
    marble_seq = PySequence_Fast(marbles_obj, "marbles must be a sequence");
    if (marble_seq == NULL) {
        return 0;
    }
    count = PySequence_Fast_GET_SIZE(marble_seq);
    if (count < 1 || count > 3) {
        Py_DECREF(marble_seq);
        PyErr_SetString(PyExc_ValueError, "move payload must contain 1 to 3 marbles");
        return 0;
    }

    for (idx = 0; idx < count; ++idx) {
        long value = PyLong_AsLong(PySequence_Fast_GET_ITEM(marble_seq, idx));
        if (value < 0 || value >= CELL_COUNT || PyErr_Occurred()) {
            Py_DECREF(marble_seq);
            if (!PyErr_Occurred()) {
                PyErr_SetString(PyExc_ValueError, "invalid marble index");
            }
            return 0;
        }
        marbles[idx] = (uint8_t) value;
    }
    Py_DECREF(marble_seq);

    dir_idx = PyLong_AsLong(direction_obj);
    if (dir_idx < 0 || dir_idx >= DIR_COUNT || PyErr_Occurred()) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_ValueError, "invalid direction index");
        }
        return 0;
    }

    canonicalize_indices(marbles, (int) count);
    build_move(marbles, (int) count, (uint8_t) dir_idx, out_move);
    return 1;
}

/* Converts a native move into the Python tuple payload used by the adapter layer. */
static PyObject *
create_move_payload(const NativeMove *move)
{
    PyObject *marbles_payload;
    PyObject *payload;
    PyObject *dir_payload;
    int idx;

    if (!move_has_value(move)) {
        Py_RETURN_NONE;
    }

    marbles_payload = PyTuple_New(move->count);
    if (marbles_payload == NULL) {
        return NULL;
    }
    for (idx = 0; idx < move->count; ++idx) {
        PyObject *value = PyLong_FromLong(move->marbles[idx]);
        if (value == NULL) {
            Py_DECREF(marbles_payload);
            return NULL;
        }
        PyTuple_SET_ITEM(marbles_payload, idx, value);
    }

    payload = PyTuple_New(2);
    if (payload == NULL) {
        Py_DECREF(marbles_payload);
        return NULL;
    }
    dir_payload = PyLong_FromLong(move->dir_idx);
    if (dir_payload == NULL) {
        Py_DECREF(marbles_payload);
        Py_DECREF(payload);
        return NULL;
    }
    PyTuple_SET_ITEM(payload, 0, marbles_payload);
    PyTuple_SET_ITEM(payload, 1, dir_payload);
    return payload;
}

/* Converts one root-candidate record into the Python diagnostics payload. */
static PyObject *
create_candidate_payload(const RootCandidate *candidate)
{
    PyObject *payload = PyDict_New();
    PyObject *move_payload = create_move_payload(&candidate->move);
    PyObject *score_payload;
    PyObject *depth_payload;

    if (payload == NULL || move_payload == NULL) {
        Py_XDECREF(payload);
        Py_XDECREF(move_payload);
        return NULL;
    }

    score_payload = PyFloat_FromDouble(candidate->score);
    depth_payload = PyLong_FromLong(candidate->depth);
    if (score_payload == NULL || depth_payload == NULL) {
        Py_DECREF(payload);
        Py_DECREF(move_payload);
        Py_XDECREF(score_payload);
        Py_XDECREF(depth_payload);
        return NULL;
    }

    if (PyDict_SetItemString(payload, "move", move_payload) < 0 ||
            PyDict_SetItemString(payload, "score", score_payload) < 0 ||
            PyDict_SetItemString(payload, "depth", depth_payload) < 0) {
        Py_DECREF(payload);
        Py_DECREF(move_payload);
        Py_DECREF(score_payload);
        Py_DECREF(depth_payload);
        return NULL;
    }
    Py_DECREF(move_payload);
    Py_DECREF(score_payload);
    Py_DECREF(depth_payload);
    return payload;
}

/* Python wrapper for native legal-move generation. */
static PyObject *
py_generate_legal_moves(PyObject *self, PyObject *args)
{
    Py_buffer cells_buffer;
    int player;
    BoardState board;
    NativeMove moves[MAX_MOVES];
    int move_count;
    PyObject *result;
    int idx;

    (void) self;
    init_tables();

    if (!PyArg_ParseTuple(args, "y*i", &cells_buffer, &player)) {
        return NULL;
    }
    if (cells_buffer.len != CELL_COUNT) {
        PyBuffer_Release(&cells_buffer);
        PyErr_Format(PyExc_ValueError, "cells payload must be %d bytes", CELL_COUNT);
        return NULL;
    }
    if (!board_init(&board, (const uint8_t *) cells_buffer.buf, 0, 0)) {
        PyBuffer_Release(&cells_buffer);
        PyErr_SetString(PyExc_ValueError, "invalid board cells");
        return NULL;
    }
    PyBuffer_Release(&cells_buffer);

    move_count = generate_legal_moves_native(&board, player, moves);
    if (move_count < 0) {
        PyErr_SetString(PyExc_RuntimeError, "native move buffer overflow");
        return NULL;
    }

    result = PyList_New(move_count);
    if (result == NULL) {
        return NULL;
    }
    for (idx = 0; idx < move_count; ++idx) {
        PyObject *payload = create_move_payload(&moves[idx]);
        if (payload == NULL) {
            Py_DECREF(result);
            return NULL;
        }
        PyList_SET_ITEM(result, idx, payload);
    }
    return result;
}

/* Python wrapper for native weighted board evaluation. */
static PyObject *
py_evaluate_weighted(PyObject *self, PyObject *args)
{
    Py_buffer cells_buffer;
    int black_score;
    int white_score;
    int player;
    PyObject *weights_obj;
    BoardState board;
    double weights[FEATURE_COUNT];

    (void) self;
    init_tables();

    if (!PyArg_ParseTuple(args, "y*iiiO", &cells_buffer, &black_score, &white_score, &player, &weights_obj)) {
        return NULL;
    }
    if (cells_buffer.len != CELL_COUNT) {
        PyBuffer_Release(&cells_buffer);
        PyErr_Format(PyExc_ValueError, "cells payload must be %d bytes", CELL_COUNT);
        return NULL;
    }
    if (!parse_weights(weights_obj, weights)) {
        PyBuffer_Release(&cells_buffer);
        return NULL;
    }
    if (!board_init(&board, (const uint8_t *) cells_buffer.buf, black_score, white_score)) {
        PyBuffer_Release(&cells_buffer);
        PyErr_SetString(PyExc_ValueError, "invalid board cells");
        return NULL;
    }
    PyBuffer_Release(&cells_buffer);
    return PyFloat_FromDouble(evaluate_weighted_native(&board, player, weights));
}

/* Python wrapper for native iterative-deepening weighted search. */
static PyObject *
py_search_weighted(PyObject *self, PyObject *args)
{
    Py_buffer cells_buffer;
    int black_score;
    int white_score;
    int player;
    PyObject *weights_obj;
    int depth;
    int max_quiescence_depth;
    PyObject *time_budget_obj;
    PyObject *remaining_game_moves_obj;
    PyObject *tie_break_obj;
    PyObject *avoid_move_obj;
    int root_candidate_limit;
    BoardState board;
    double weights[FEATURE_COUNT];
    NativeMove avoid_move;
    SearchResultNative result;
    PyObject *payload;
    PyObject *move_payload;
    PyObject *candidate_list;
    PyObject *score_payload;
    PyObject *nodes_payload;
    PyObject *completed_depth_payload;
    PyObject *timed_out_payload;
    PyObject *avoidance_payload;
    int has_deadline = 0;
    int time_budget_ms = 0;
    int has_remaining_game_moves = 0;
    int remaining_game_moves = 0;
    int tie_break_lexicographic;
    int status;
    int idx;
    const char *tie_break;

    (void) self;
    init_tables();

    if (!PyArg_ParseTuple(
            args,
            "y*iiiOiiOOOOi",
            &cells_buffer,
            &black_score,
            &white_score,
            &player,
            &weights_obj,
            &depth,
            &max_quiescence_depth,
            &time_budget_obj,
            &remaining_game_moves_obj,
            &tie_break_obj,
            &avoid_move_obj,
            &root_candidate_limit)) {
        return NULL;
    }
    if (cells_buffer.len != CELL_COUNT) {
        PyBuffer_Release(&cells_buffer);
        PyErr_Format(PyExc_ValueError, "cells payload must be %d bytes", CELL_COUNT);
        return NULL;
    }
    if (!parse_weights(weights_obj, weights)) {
        PyBuffer_Release(&cells_buffer);
        return NULL;
    }
    if (!board_init(&board, (const uint8_t *) cells_buffer.buf, black_score, white_score)) {
        PyBuffer_Release(&cells_buffer);
        PyErr_SetString(PyExc_ValueError, "invalid board cells");
        return NULL;
    }
    PyBuffer_Release(&cells_buffer);

    if (time_budget_obj != Py_None) {
        long value = PyLong_AsLong(time_budget_obj);
        if (value < 0 || PyErr_Occurred()) {
            if (!PyErr_Occurred()) {
                PyErr_SetString(PyExc_ValueError, "time_budget_ms must be non-negative");
            }
            return NULL;
        }
        has_deadline = 1;
        time_budget_ms = (int) value;
    }
    if (remaining_game_moves_obj != Py_None) {
        long value = PyLong_AsLong(remaining_game_moves_obj);
        if (value < 0 || PyErr_Occurred()) {
            if (!PyErr_Occurred()) {
                PyErr_SetString(PyExc_ValueError, "remaining_game_moves must be non-negative");
            }
            return NULL;
        }
        has_remaining_game_moves = 1;
        remaining_game_moves = (int) value;
    }

    if (!parse_optional_move_payload(avoid_move_obj, &avoid_move)) {
        return NULL;
    }

    tie_break = PyUnicode_AsUTF8(tie_break_obj);
    if (tie_break == NULL) {
        return NULL;
    }
    tie_break_lexicographic = strcmp(tie_break, "lexicographic") == 0;

    memset(&result, 0, sizeof(result));
    status = search_weighted_native(
        &board,
        player,
        weights,
        depth,
        max_quiescence_depth,
        has_deadline,
        time_budget_ms,
        has_remaining_game_moves,
        remaining_game_moves,
        tie_break_lexicographic,
        &avoid_move,
        root_candidate_limit,
        &result
    );
    if (status < 0) {
        PyErr_SetString(PyExc_RuntimeError, "native weighted search failed");
        return NULL;
    }

    payload = PyDict_New();
    if (payload == NULL) {
        return NULL;
    }

    move_payload = create_move_payload(&result.move);
    candidate_list = PyList_New(result.root_candidate_count);
    score_payload = PyFloat_FromDouble(result.score);
    nodes_payload = PyLong_FromUnsignedLongLong(result.nodes);
    completed_depth_payload = PyLong_FromLong(result.completed_depth);
    timed_out_payload = result.timed_out ? Py_True : Py_False;
    avoidance_payload = result.avoidance_applied ? Py_True : Py_False;
    Py_INCREF(timed_out_payload);
    Py_INCREF(avoidance_payload);
    if (move_payload == NULL || candidate_list == NULL || score_payload == NULL ||
            nodes_payload == NULL || completed_depth_payload == NULL) {
        Py_DECREF(payload);
        Py_XDECREF(move_payload);
        Py_XDECREF(candidate_list);
        Py_XDECREF(score_payload);
        Py_XDECREF(nodes_payload);
        Py_XDECREF(completed_depth_payload);
        Py_XDECREF(timed_out_payload);
        Py_XDECREF(avoidance_payload);
        return NULL;
    }

    for (idx = 0; idx < result.root_candidate_count; ++idx) {
        PyObject *candidate_payload = create_candidate_payload(&result.root_candidates[idx]);
        if (candidate_payload == NULL) {
            Py_DECREF(payload);
            Py_DECREF(move_payload);
            Py_DECREF(candidate_list);
            return NULL;
        }
        PyList_SET_ITEM(candidate_list, idx, candidate_payload);
    }

    if (PyDict_SetItemString(payload, "move", move_payload) < 0 ||
            PyDict_SetItemString(payload, "score", score_payload) < 0 ||
            PyDict_SetItemString(payload, "nodes", nodes_payload) < 0 ||
            PyDict_SetItemString(payload, "completed_depth", completed_depth_payload) < 0 ||
            PyDict_SetItemString(payload, "timed_out", timed_out_payload) < 0 ||
            PyDict_SetItemString(payload, "avoidance_applied", avoidance_payload) < 0 ||
            PyDict_SetItemString(payload, "root_candidates", candidate_list) < 0) {
        Py_DECREF(payload);
        Py_DECREF(move_payload);
        Py_DECREF(candidate_list);
        Py_DECREF(score_payload);
        Py_DECREF(nodes_payload);
        Py_DECREF(completed_depth_payload);
        Py_DECREF(timed_out_payload);
        Py_DECREF(avoidance_payload);
        return NULL;
    }

    Py_DECREF(move_payload);
    Py_DECREF(candidate_list);
    Py_DECREF(score_payload);
    Py_DECREF(nodes_payload);
    Py_DECREF(completed_depth_payload);
    Py_DECREF(timed_out_payload);
    Py_DECREF(avoidance_payload);
    return payload;
}

static PyMethodDef module_methods[] = {
    {"generate_legal_moves", py_generate_legal_moves, METH_VARARGS, "Generate legal moves from a compact board payload."},
    {"evaluate_weighted", py_evaluate_weighted, METH_VARARGS, "Evaluate a board using shared heuristic weights."},
    {"search_weighted", py_search_weighted, METH_VARARGS, "Run the native weighted minimax search."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "_native",
    "Native Abalone search helpers.",
    -1,
    module_methods,
};

/* Initializes the Python extension module and shared native lookup tables. */
PyMODINIT_FUNC
PyInit__native(void)
{
    init_tables();
    return PyModule_Create(&module_def);
}
