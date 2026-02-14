const BLACK = 1, WHITE = 2, EMPTY = 0;
const ROWS = 'abcdefghi';
const DIRECTIONS = [[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [-1, -1]];
const CONTROLLER_HUMAN = 'human';
const CONTROLLER_AI = 'ai';

/* Build set of valid positions */
const VALID = [];
for (let r = 0; r < 9; r++) {
    const cmin = r <= 4 ? 1 : r - 3;
    const cmax = r <= 4 ? 5 + r : 9;
    for (let c = cmin; c <= cmax; c++) VALID.push([r, c]);
}

/* ── Hex geometry ──────────────────────────────────────── */
const R = 31;                // marble radius
const H_SP = R * 2.2;          // horizontal cell spacing
const V_SP = R * 1.9;          // vertical row spacing
const CX = 385;              // SVG center x
const CY = 325;              // SVG center y

function hexPx(r, c) {
    return {
        x: CX + H_SP * (c - r / 2 - 3),
        y: CY - V_SP * (r - 4),
    };
}

function posKey(r, c) { return ROWS[r] + c; }
function parsePos(s) { return [ROWS.indexOf(s[0]), parseInt(s[1])]; }
function getController(player) {
    return state?.controllers?.[String(player)] || CONTROLLER_HUMAN;
}
function controllerLabel(player) {
    return getController(player) === CONTROLLER_AI ? 'AI' : 'Human';
}

/* ── State ─────────────────────────────────────────────── */
let state = null;
let selected = [];
let stateFetchedAt = 0;
let fetchInFlight = false;
let agentMoveInFlight = false;
let gameStarted = false;
let hasPlayedGame = false;
let wasPausedBeforeModal = false;
let lastHistoryKey = '';

/* ── API ───────────────────────────────────────────────── */
async function fetchState(force = false) {
    if (fetchInFlight && !force) return;
    fetchInFlight = true;
    try {
        state = await (await fetch('/api/state')).json();
        stateFetchedAt = Date.now();
        render();
        maybeAutoAgentTurn();
    } finally {
        fetchInFlight = false;
    }
}
async function doMove(m) {
    const d = await (await fetch('/api/move', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(m),
    })).json();
    if (d.error) { await fetchState(true); return; }
    selected = [];
    await fetchState(true);
}
async function doUndo() { await fetch('/api/undo', { method: 'POST' }); selected = []; await fetchState(true); }
async function doReset() {
    document.getElementById('game-over').classList.remove('show');
    closeGameConfigModal();
    await fetch('/api/reset', { method: 'POST' }); selected = []; await fetchState(true);
}
async function doPause() {
    await fetch('/api/pause', { method: 'POST' });
    await fetchState(true);
}

function maybeAutoAgentTurn() {
    if (!gameStarted) return;
    if (!state) return;
    if (state.game_over || state.paused) return;
    if (state.current_controller !== CONTROLLER_AI) return;
    if (agentMoveInFlight) return;
    agentMoveInFlight = true;
    fetch('/api/agent-move', { method: 'POST' })
        .then(() => { selected = []; return fetchState(true); })
        .finally(() => { agentMoveInFlight = false; });
}

/* ── Game Config Modal ─────────────────────────────────── */
function isConfigModalOpen() {
    return document.getElementById('game-config-modal').classList.contains('show');
}
async function openGameConfigModal() {
    const closeBtn = document.getElementById('config-close-btn');
    // Show × only if a game has already been played (not first visit)
    closeBtn.style.display = hasPlayedGame ? 'block' : 'none';
    // Auto-pause the running game
    if (gameStarted && state && !state.paused && !state.game_over) {
        wasPausedBeforeModal = false;
        await fetch('/api/pause', { method: 'POST' });
        await fetchState(true);
    } else {
        wasPausedBeforeModal = state?.paused || false;
    }
    gameStarted = false;
    showConfigPage(1);
    document.getElementById('game-config-modal').classList.add('show');
}
async function closeGameConfigModal() {
    document.getElementById('game-config-modal').classList.remove('show');
    // Resume if we auto-paused (and user didn't manually pause before)
    if (hasPlayedGame && !wasPausedBeforeModal && state?.paused && !state?.game_over) {
        await fetch('/api/pause', { method: 'POST' });
        await fetchState(true);
    }
    gameStarted = hasPlayedGame;
}
function showConfigPage(page) {
    document.getElementById('config-page-1').style.display = page === 1 ? 'block' : 'none';
    document.getElementById('config-page-2').style.display = page === 2 ? 'block' : 'none';
}

async function startGame() {
    const mode = document.querySelector('input[name="cfg-mode"]:checked').value;
    const colorVal = document.querySelector('input[name="cfg-color"]:checked').value;
    const human_side = colorVal === 'black' ? 'black' : 'white';
    const board_layout = document.getElementById('cfg-layout').value;
    const sameTime = document.getElementById('cfg-same-time').checked;
    const p1Min = Number(document.getElementById('cfg-p1-time').value) || 0;
    const p2Min = sameTime ? p1Min : (Number(document.getElementById('cfg-p2-time').value) || 0);
    const max_moves = Number(document.getElementById('cfg-max-moves').value) || 0;
    const time_limit_per_move_s = Number(document.getElementById('cfg-move-limit').value) || 0;

    const payload = {
        mode,
        human_side,
        board_layout,
        ai_depth: Number(state?.ai_depth || 2),
        player1_time_ms: p1Min * 60 * 1000,
        player2_time_ms: p2Min * 60 * 1000,
        max_moves,
        time_limit_per_move_s,
    };

    await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });
    await fetch('/api/reset', { method: 'POST' });
    selected = [];
    document.getElementById('game-over').classList.remove('show');
    gameStarted = true;
    hasPlayedGame = true;
    wasPausedBeforeModal = true;  // skip resume logic — reset already unpaused
    closeGameConfigModal();
    await fetchState(true);
}

/* Sync "same time" checkbox */
document.getElementById('cfg-same-time').addEventListener('change', function () {
    const p2Input = document.getElementById('cfg-p2-time');
    if (this.checked) {
        p2Input.value = document.getElementById('cfg-p1-time').value;
        p2Input.disabled = true;
    } else {
        p2Input.disabled = false;
    }
});
// Initialize P2 input state
document.getElementById('cfg-p2-time').disabled = true;

/* ── Resign ────────────────────────────────────────────── */
async function doResign() {
    if (!state || state.game_over) return;
    await fetch('/api/resign', { method: 'POST' });
    await fetchState(true);
}

/* ── Game Over ─────────────────────────────────────────── */
function closeGameOver() {
    document.getElementById('game-over').classList.remove('show');
}

/* ── Selection ─────────────────────────────────────────── */
function toggleSelect(ps) {
    if (!gameStarted || isConfigModalOpen()) return;
    if (state.game_over) return;
    if (state.paused) return;
    if (state.current_controller !== CONTROLLER_HUMAN) return;
    const val = state.cells[ps];

    /* Clicked opponent or empty → try as destination */
    if (val !== state.current_player) {
        const dests = getValidDestinations();
        if (dests[ps]) { doMove(dests[ps]); }
        else { selected = []; render(); }
        return;
    }

    /* Toggle own marble */
    const idx = selected.indexOf(ps);
    if (idx >= 0) { selected.splice(idx, 1); }
    else {
        if (selected.length >= 3) return;
        selected.push(ps);
        if (selected.length > 1 && !isLine(selected)) selected.pop();
    }
    render();
}

function isLine(sel) {
    if (sel.length <= 1) return true;
    const pts = sel.map(parsePos).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const d = [pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]];
    if (!DIRECTIONS.some(dd => dd[0] === d[0] && dd[1] === d[1])) return false;
    for (let i = 2; i < pts.length; i++) {
        if (pts[i][0] !== pts[0][0] + i * d[0] || pts[i][1] !== pts[0][1] + i * d[1]) return false;
    }
    return true;
}

/* ── Destination map ───────────────────────────────────── */
function getValidDestinations() {
    if (!selected.length || !state) return {};
    const dests = {};
    for (const lm of state.legal_moves) {
        const mset = new Set(lm.marbles);
        if (mset.size !== selected.length) continue;
        if (!selected.every(s => mset.has(s))) continue;

        const [dr, dc] = lm.direction;

        /* For broadside: every destination cell that is NOT already in the group */
        const destCells = lm.marbles.map(m => {
            const [r, c] = parsePos(m);
            return posKey(r + dr, c + dc);
        });
        for (const d of destCells) {
            if (!mset.has(d)) dests[d] = lm;
        }

        /* For inline: also mark the goal cell (where leading marble ends up) */
        if (lm.is_inline) {
            let best = lm.marbles[0], bestDot = -Infinity;
            for (const m of lm.marbles) {
                const [r, c] = parsePos(m);
                const dot = r * dr + c * dc;
                if (dot > bestDot) { bestDot = dot; best = m; }
            }
            const [lr, lc] = parsePos(best);
            dests[posKey(lr + dr, lc + dc)] = lm;
        }
    }
    return dests;
}

/* ── Render ────────────────────────────────────────────── */
function render() {
    if (!state) return;
    renderClocks();
    renderMoveTimers();
    renderScore();
    renderControllers();
    renderTurn();
    renderEndBanner();
    renderPause();
    renderBoard();
    renderHistory();
    if (state.game_over) showGameOver();
}

function winnerPlayer() {
    if (state.winner === BLACK || state.winner === WHITE) return state.winner;
    return state.score['1'] >= 6 ? BLACK : WHITE;
}

function gameOverText() {
    const reason = state.game_over_reason;

    if (reason === 'resign') {
        const winner = winnerPlayer();
        const winnerName = winner === BLACK ? 'Black' : 'White';
        const loserName = winner === BLACK ? 'White' : 'Black';
        return {
            title: `${winnerName} Wins!`,
            reason: `${loserName} resigned.`,
            banner: `${winnerName} wins: ${loserName} resigned.`,
            cls: 'resign',
        };
    }

    if (reason === 'max_moves') {
        if (state.winner == null) {
            return {
                title: 'Draw!',
                reason: `Max moves reached. Score tied.`,
                banner: `Draw: max moves reached with tied score.`,
                cls: 'max_moves',
            };
        }
        const winner = winnerPlayer();
        const winnerName = winner === BLACK ? 'Black' : 'White';
        return {
            title: `${winnerName} Wins!`,
            reason: `Max moves reached. ${winnerName} captured more.`,
            banner: `${winnerName} wins: max moves reached.`,
            cls: 'max_moves',
        };
    }

    if (reason === 'timeout') {
        const winner = winnerPlayer();
        const winnerName = winner === BLACK ? 'Black' : 'White';
        const loserName = winner === BLACK ? 'White' : 'Black';
        return {
            title: `${winnerName} Wins on Time!`,
            reason: `${loserName} ran out of time (00:00).`,
            banner: `${winnerName} wins: ${loserName}'s clock reached 00:00.`,
            cls: 'timeout',
        };
    }

    // Default: score
    const winner = winnerPlayer();
    const winnerName = winner === BLACK ? 'Black' : 'White';
    return {
        title: `${winnerName} Wins!`,
        reason: `${winnerName} pushed 6 marbles off the board.`,
        banner: `${winnerName} wins by capture.`,
        cls: 'score',
    };
}

function getClockMs(player) {
    const key = String(player);
    let ms = state.time_left_ms?.[key] ?? 0;
    if (gameStarted && !state.game_over && !state.paused && state.current_player === player) {
        ms -= (Date.now() - stateFetchedAt);
    }
    return Math.max(0, ms);
}

function formatClock(ms) {
    if (ms <= 0) return '00:00';
    const total = Math.ceil(ms / 1000);
    const mins = Math.floor(total / 60);
    const secs = total % 60;
    return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
}

function renderClocks() {
    document.getElementById('p1-timer').textContent = formatClock(getClockMs(WHITE));
    document.getElementById('p2-timer').textContent = formatClock(getClockMs(BLACK));
}

function getMoveTimeLeftMs() {
    if (!state || !gameStarted || state.game_over || state.paused) return null;
    if (!state.time_limit_per_move_s || state.time_limit_per_move_s <= 0) return null;
    const limitMs = state.time_limit_per_move_s * 1000;
    const elapsed = Date.now() - stateFetchedAt + (stateFetchedAt - state.turn_start_ms);
    return Math.max(0, limitMs - elapsed);
}

function renderMoveTimers() {
    const p1El = document.getElementById('p1-move-timer');
    const p2El = document.getElementById('p2-move-timer');
    const moveMs = getMoveTimeLeftMs();

    if (moveMs == null) {
        p1El.textContent = '';
        p1El.className = 'move-timer';
        p2El.textContent = '';
        p2El.className = 'move-timer';
        return;
    }

    const secs = Math.ceil(moveMs / 1000);
    const text = `${secs}`;
    const warn = secs <= 5;

    if (state.current_player === WHITE) {
        p1El.textContent = text;
        p1El.className = warn ? 'move-timer warning' : 'move-timer';
        p2El.textContent = '';
        p2El.className = 'move-timer';
    } else {
        p2El.textContent = text;
        p2El.className = warn ? 'move-timer warning' : 'move-timer';
        p1El.textContent = '';
        p1El.className = 'move-timer';
    }
}

function renderScore() {
    document.getElementById('p1-score').textContent = state.score['2'];
    document.getElementById('p2-score').textContent = state.score['1'];
}

function renderControllers() {
    document.getElementById('p1-controller').textContent = controllerLabel(WHITE);
    document.getElementById('p2-controller').textContent = controllerLabel(BLACK);
}

function renderTurn() {
    const p1 = document.getElementById('player1-card');
    const p2 = document.getElementById('player2-card');
    if (state.game_over) {
        p1.classList.remove('active');
        p2.classList.remove('active');
        return;
    }
    if (state.current_player === WHITE) {
        p1.classList.add('active');
        p2.classList.remove('active');
    } else {
        p1.classList.remove('active');
        p2.classList.add('active');
    }
}

function renderPause() {
    const btn = document.getElementById('pause-btn');
    const label = document.getElementById('pause-label');
    const overlay = document.getElementById('pause-overlay');
    if (state.paused) {
        btn.classList.add('active-pause');
        label.textContent = 'Resume';
        overlay.classList.add('show');
    } else {
        btn.classList.remove('active-pause');
        label.textContent = 'Pause';
        overlay.classList.remove('show');
    }
}

function renderEndBanner() {
    const el = document.getElementById('end-banner');
    if (!state.game_over) {
        el.className = '';
        el.textContent = '';
        return;
    }
    const msg = gameOverText();
    el.className = `show ${msg.cls}`;
    el.textContent = msg.banner;
}

/* ── Board SVG ─────────────────────────────────────────── */
function renderBoard() {
    const svg = document.getElementById('board');
    const dests = getValidDestinations();
    let h = '';

    /* ── Defs: shared gradients ── */
    h += `<defs>
    <radialGradient id="gBlack" cx="38%" cy="32%" r="55%">
      <stop offset="0%" stop-color="#606060"/>
      <stop offset="100%" stop-color="#0a0a0a"/>
    </radialGradient>
    <radialGradient id="gWhite" cx="38%" cy="32%" r="55%">
      <stop offset="0%" stop-color="#ffffff"/>
      <stop offset="100%" stop-color="#9999a0"/>
    </radialGradient>
    <radialGradient id="gPit" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#1e3040"/>
      <stop offset="100%" stop-color="#162230"/>
    </radialGradient>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="1" stdDeviation="2" flood-color="#000" flood-opacity=".5"/>
    </filter>
  </defs>`;

    /* ── Board background hex ── */
    h += boardHex();

    /* ── Cells ── */
    for (const [r, c] of VALID) {
        const ps = posKey(r, c);
        const { x, y } = hexPx(r, c);
        const val = state.cells[ps] ?? EMPTY;
        const sel = selected.includes(ps);
        const dst = ps in dests;
        h += cell(x, y, r, c, ps, val, sel, dst);
    }

    /* ── Edge labels: row letters on both sides ── */
    for (let r = 0; r < 9; r++) {
        const cmin = r <= 4 ? 1 : r - 3;
        const cmax = r <= 4 ? 5 + r : 9;
        const letter = ROWS[r].toUpperCase();
        const left = hexPx(r, cmin);
        const right = hexPx(r, cmax);
        h += `<text x="${left.x - R - 16}" y="${left.y + 5}" text-anchor="middle"
      font-size="14" fill="#5a8a9e" font-weight="700" font-family="monospace">${letter}</text>`;
        h += `<text x="${right.x + R + 16}" y="${right.y + 5}" text-anchor="middle"
      font-size="14" fill="#5a8a9e" font-weight="700" font-family="monospace">${letter}</text>`;
    }

    /* ── Column numbers along bottom-left diagonal (row a) ── */
    for (let c = 1; c <= 5; c++) {
        const { x, y } = hexPx(0, c);
        h += `<text x="${x}" y="${y + R + 18}" text-anchor="middle"
      font-size="13" fill="#5a8a9e" font-weight="700" font-family="monospace">${c}</text>`;
    }
    /* ── Column numbers along top-right diagonal (row i) ── */
    for (let c = 5; c <= 9; c++) {
        const { x, y } = hexPx(8, c);
        h += `<text x="${x}" y="${y - R - 10}" text-anchor="middle"
      font-size="13" fill="#5a8a9e" font-weight="700" font-family="monospace">${c}</text>`;
    }

    svg.innerHTML = h;
}

function boardHex() {
    const pad = R + 12;
    const corners = [
        hexPx(8, 5),  // i5
        hexPx(8, 9),  // i9
        hexPx(4, 9),  // e9
        hexPx(0, 5),  // a5
        hexPx(0, 1),  // a1
        hexPx(4, 1),  // e1
    ];
    const pts = corners.map(c => {
        const dx = c.x - CX, dy = c.y - CY;
        const len = Math.hypot(dx, dy);
        return [c.x + dx / len * pad, c.y + dy / len * pad];
    });
    let d = `M${pts[0][0].toFixed(1)},${pts[0][1].toFixed(1)}`;
    for (let i = 1; i < 6; i++) d += `L${pts[i][0].toFixed(1)},${pts[i][1].toFixed(1)}`;
    d += 'Z';
    return `<path d="${d}" fill="#1a2d3d" stroke="#2a4050" stroke-width="2.5"/>`;
}

function cell(x, y, r, c, ps, val, sel, dst) {
    let h = '';
    const click = `onclick="toggleSelect('${ps}')"`;

    /* Pit (indentation) */
    h += `<circle cx="${x}" cy="${y}" r="${R}" fill="url(#gPit)"
    stroke="#1a2a36" stroke-width="1" ${click} style="cursor:pointer"/>`;

    if (val === BLACK || val === WHITE) {
        const grad = val === BLACK ? 'url(#gBlack)' : 'url(#gWhite)';
        const strokeNorm = val === BLACK ? '#000' : '#888';
        h += `<circle cx="${x}" cy="${y}" r="${R - 3}" fill="${grad}"
      stroke="${sel ? '#4fc3f7' : strokeNorm}" stroke-width="${sel ? 3 : 1.2}"
      filter="url(#shadow)" ${click} style="cursor:pointer"/>`;
        if (sel) {
            h += `<circle cx="${x}" cy="${y}" r="${R + 1}" fill="none"
        stroke="#4fc3f7" stroke-width="2" opacity=".55" style="pointer-events:none"/>`;
        }
    } else {
        /* Empty — show coordinate */
        h += `<text x="${x}" y="${y + 4}" text-anchor="middle"
      font-size="11" fill="#3e6070" font-family="monospace" font-weight="600"
      ${click} style="cursor:pointer;pointer-events:all">${ps}</text>`;
    }

    /* Destination glow */
    if (dst) {
        h += `<circle cx="${x}" cy="${y}" r="${R - 1}" fill="#4fc3f7" opacity=".18"
      stroke="#4fc3f7" stroke-width="2.5" stroke-dasharray="${val !== EMPTY ? '0' : '4 3'}"
      ${click} style="cursor:pointer"/>`;
        h += `<circle cx="${x}" cy="${y}" r="5" fill="#4fc3f7" opacity=".9"
      ${click} style="cursor:pointer"/>`;
    }

    return h;
}

/* ── History ───────────────────────────────────────────── */
function formatDuration(ms) {
    if (ms == null || ms <= 0) return '0.0s';
    const totalSec = ms / 1000;
    if (totalSec < 60) return totalSec.toFixed(1) + 's';
    const mins = Math.floor(totalSec / 60);
    const secs = Math.floor(totalSec % 60);
    return `${mins}:${String(secs).padStart(2, '0')}`;
}

function renderHistory() {
    const el = document.getElementById('history-list');
    const countEl = document.getElementById('move-count');
    countEl.textContent = `${state.history.length} moves`;

    /* Build a key from length + last move notation to detect changes */
    const last = state.history.length ? state.history[state.history.length - 1].notation : '';
    const key = `${state.history.length}:${last}`;
    if (key === lastHistoryKey) return;
    const prevLen = parseInt(lastHistoryKey) || 0;
    lastHistoryKey = key;

    if (!state.history.length) {
        el.innerHTML = '';
        return;
    }
    let h = '';
    for (let i = 0; i < state.history.length; i++) {
        const e = state.history[i];
        const sym = e.player === BLACK ? '●' : '○';
        const symColor = e.player === BLACK ? '#555' : '#eee';
        const src = e.source === CONTROLLER_AI ? 'AI' : 'H';
        const dur = formatDuration(e.duration_ms);
        h += `<div class="history-entry">
      <span class="history-num">${i + 1}.</span>
      <span class="history-player" style="color:${symColor}">${sym}</span>
      <span class="history-source">${src}</span>
      <span class="history-move">${e.notation}</span>
      ${e.pushoff ? '<span class="history-push">pushed off!</span>' : ''}
      <span class="history-time">${dur}</span>
    </div>`;
    }
    el.innerHTML = h;
    if (state.history.length > prevLen) {
        el.scrollTop = el.scrollHeight;
    }
}

/* ── Game over ─────────────────────────────────────────── */
function showGameOver() {
    const msg = gameOverText();
    document.getElementById('go-title').textContent = msg.title;
    document.getElementById('go-reason').textContent = msg.reason;
    document.getElementById('go-sub').textContent = `Score: ${state.score['1']} – ${state.score['2']}`;
    document.getElementById('game-over').classList.add('show');
}

/* ── Init ──────────────────────────────────────────────── */
fetchState();
openGameConfigModal();
setInterval(() => { if (state) { renderClocks(); renderMoveTimers(); } }, 250);
setInterval(() => { fetchState(); }, 1000);
setInterval(() => { maybeAutoAgentTurn(); }, 300);
