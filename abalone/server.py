"""Abalone web server — serves the HTML UI and a JSON API for game logic."""

import json
import os
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from .board import (
    Board, Move, BLACK, WHITE, EMPTY,
    DIRECTIONS, DIRECTION_NAMES, VALID_POSITIONS,
    pos_to_str, str_to_pos, neighbor, is_valid, ROW_LETTERS,
)
from .state_space import generate_legal_moves

# ── Global game state ────────────────────────────────────────────────────────

board = Board()
board.setup_standard()
current_player = BLACK
move_history = []          # list of {move, result, snapshot, player}
INITIAL_TIME_MS = 30 * 60 * 1000
time_left_ms = {BLACK: INITIAL_TIME_MS, WHITE: INITIAL_TIME_MS}
last_clock_update_ms = int(time.time() * 1000)


def _now_ms():
    return int(time.time() * 1000)


def _game_status():
    """Return terminal status including winner and end reason."""
    if board.score[BLACK] >= 6:
        return {
            'game_over': True,
            'winner': BLACK,
            'game_over_reason': 'score',
            'timeout_player': None,
        }
    if board.score[WHITE] >= 6:
        return {
            'game_over': True,
            'winner': WHITE,
            'game_over_reason': 'score',
            'timeout_player': None,
        }
    if time_left_ms[BLACK] <= 0:
        return {
            'game_over': True,
            'winner': WHITE,
            'game_over_reason': 'timeout',
            'timeout_player': BLACK,
        }
    if time_left_ms[WHITE] <= 0:
        return {
            'game_over': True,
            'winner': BLACK,
            'game_over_reason': 'timeout',
            'timeout_player': WHITE,
        }
    return {
        'game_over': False,
        'winner': None,
        'game_over_reason': None,
        'timeout_player': None,
    }


def _tick_clock():
    """Deduct elapsed wall-clock time from the current player's timer."""
    global last_clock_update_ms
    now = _now_ms()

    # Stop decrementing clocks once the game is already over.
    if _game_status()['game_over']:
        last_clock_update_ms = now
        return

    elapsed = max(0, now - last_clock_update_ms)
    if elapsed > 0:
        time_left_ms[current_player] = max(0, time_left_ms[current_player] - elapsed)
    last_clock_update_ms = now


def _state_json():
    """Serialize current game state to a JSON-friendly dict."""
    _tick_clock()
    status = _game_status()

    cells = {}
    for pos, val in board.cells.items():
        cells[pos_to_str(pos)] = val

    legal = generate_legal_moves(board, current_player)
    legal_list = []
    for m in legal:
        marble_strs = [pos_to_str(p) for p in m.marbles]
        dr, dc = m.direction
        legal_list.append({
            'marbles': marble_strs,
            'direction': [dr, dc],
            'notation': m.to_notation(),
            'is_inline': m.is_inline,
        })

    history = []
    for entry in move_history:
        history.append({
            'notation': entry['move'].to_notation(pushed=bool(entry['result']['pushed'])),
            'player': entry['player'],
            'pushoff': entry['result']['pushoff'],
        })

    return {
        'cells': cells,
        'current_player': current_player,
        'score': board.score,
        'game_over': status['game_over'],
        'winner': status['winner'],
        'game_over_reason': status['game_over_reason'],
        'timeout_player': status['timeout_player'],
        'legal_moves': legal_list,
        'history': history,
        'marble_counts': {
            BLACK: board.marble_count(BLACK),
            WHITE: board.marble_count(WHITE),
        },
        'time_left_ms': {
            BLACK: time_left_ms[BLACK],
            WHITE: time_left_ms[WHITE],
        },
        'initial_time_ms': INITIAL_TIME_MS,
    }


def _apply_move(data):
    global current_player, last_clock_update_ms
    _tick_clock()
    if _game_status()['game_over']:
        return {'error': 'Game is over'}

    marbles = tuple(str_to_pos(s) for s in data['marbles'])
    direction = tuple(data['direction'])
    move = Move(marbles=marbles, direction=direction)

    if not board.is_legal_move(move, current_player):
        return {'error': 'Illegal move'}

    snapshot = board.copy()
    clock_snapshot = dict(time_left_ms)
    result = board.apply_move(move, current_player)
    move_history.append({
        'move': move,
        'result': result,
        'snapshot': snapshot,
        'clock_snapshot': clock_snapshot,
        'player': current_player,
    })
    current_player = WHITE if current_player == BLACK else BLACK
    last_clock_update_ms = _now_ms()
    return {'ok': True, 'result': result}


def _undo():
    global current_player, board, time_left_ms, last_clock_update_ms
    if not move_history:
        return {'error': 'Nothing to undo'}
    entry = move_history.pop()
    board = entry['snapshot']
    current_player = entry['player']
    time_left_ms = dict(entry['clock_snapshot'])
    last_clock_update_ms = _now_ms()
    return {'ok': True}


def _reset():
    global current_player, board, move_history, time_left_ms, last_clock_update_ms
    board = Board()
    board.setup_standard()
    current_player = BLACK
    move_history = []
    time_left_ms = {BLACK: INITIAL_TIME_MS, WHITE: INITIAL_TIME_MS}
    last_clock_update_ms = _now_ms()
    return {'ok': True}


# ── HTTP handler ─────────────────────────────────────────────────────────────

STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self._serve_file('index.html', 'text/html')
        elif self.path == '/api/state':
            self._json_response(_state_json())
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        if self.path == '/api/move':
            self._json_response(_apply_move(body))
        elif self.path == '/api/undo':
            self._json_response(_undo())
        elif self.path == '/api/reset':
            self._json_response(_reset())
        else:
            self.send_error(404)

    def _json_response(self, data):
        payload = json.dumps(data).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(payload))
        self.end_headers()
        self.wfile.write(payload)

    def _serve_file(self, name, mime):
        path = os.path.join(STATIC_DIR, name)
        with open(path, 'rb') as f:
            data = f.read()
        self.send_response(200)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', len(data))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        pass  # silence request logs


def run(port=9000):
    import socket
    import webbrowser
    # Find a free port if the requested one is taken
    for p in [port] + list(range(port + 1, port + 20)):
        try:
            s = socket.socket()
            s.bind(('', p))
            s.close()
            port = p
            break
        except OSError:
            continue
    server = HTTPServer(('', port), Handler)
    url = f'http://localhost:{port}'
    print(f'Abalone running at  {url}')
    webbrowser.open(url)
    server.serve_forever()
