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
let modeSelected = false;

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
    await fetch('/api/reset', { method: 'POST' }); selected = []; await fetchState(true);
}
async function doPause() {
    await fetch('/api/pause', { method: 'POST' });
    await fetchState(true);
}
async function doAgentMove(force = false) {
    if (!state || agentMoveInFlight) return;
    if (!force) {
        if (state.game_over || state.paused) return;
        if (state.current_controller !== CONTROLLER_AI) return;
    }

    agentMoveInFlight = true;
    try {
        await fetch('/api/agent-move', { method: 'POST' });
        selected = [];
        await fetchState(true);
    } finally {
        agentMoveInFlight = false;
    }
}
function maybeAutoAgentTurn() {
    if (!modeSelected) return;
    if (!state) return;
    if (state.game_over || state.paused) return;
    if (state.current_controller !== CONTROLLER_AI) return;
    doAgentMove();
}
function isModeModalOpen() {
    return document.getElementById('mode-modal').classList.contains('show');
}
function openModeModal() {
    modeSelected = false;
    document.getElementById('mode-modal').classList.add('show');
}
function closeModeModal() {
    document.getElementById('mode-modal').classList.remove('show');
}
async function selectMode(mode) {
    const payload = {
        mode: mode,
        ai_depth: Number(state?.ai_depth || 2),
    };
    if (mode === 'hva') payload.human_side = 'black';

    await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });
    await fetch('/api/reset', { method: 'POST' });
    selected = [];
    document.getElementById('game-over').classList.remove('show');
    modeSelected = true;
    closeModeModal();
    await fetchState(true);
}
function doResign() {
    /* Resign = new game for now */
    doReset();
}

/* ── Selection ─────────────────────────────────────────── */
function toggleSelect(ps) {
    if (!modeSelected || isModeModalOpen()) return;
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
    renderScore();
    renderControllers();
    renderTurn();
    renderAgentControls();
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
    const winner = winnerPlayer();
    const winnerName = winner === BLACK ? 'Black' : 'White';
    const loserName = winner === BLACK ? 'White' : 'Black';
    if (state.game_over_reason === 'timeout') {
        return {
            title: `${winnerName} Wins on Time!`,
            reason: `${loserName} ran out of time (00:00).`,
            banner: `${winnerName} wins: ${loserName}'s clock reached 00:00.`,
            cls: 'timeout',
        };
    }
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
    if (!state.game_over && !state.paused && state.current_player === player) {
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

function renderScore() {
    document.getElementById('p1-score').textContent = state.score['2'];
    document.getElementById('p2-score').textContent = state.score['1'];
}

function renderControllers() {
    document.getElementById('p1-controller').textContent = controllerLabel(WHITE);
    document.getElementById('p2-controller').textContent = controllerLabel(BLACK);
}

function renderAgentControls() {
    const btn = document.getElementById('agent-move-btn');
    const canStep = (
        !state.game_over
        && !state.paused
        && state.current_controller === CONTROLLER_AI
    );
    btn.style.opacity = canStep ? '1' : '0.45';
    btn.style.pointerEvents = canStep ? 'auto' : 'none';
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
function renderHistory() {
    const el = document.getElementById('history-list');
    const countEl = document.getElementById('move-count');
    countEl.textContent = `${state.history.length} moves`;

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
        h += `<div class="history-entry">
      <span class="history-num">${i + 1}.</span>
      <span class="history-player" style="color:${symColor}">${sym}</span>
      <span class="history-source">${src}</span>
      <span class="history-move">${e.notation}</span>
      ${e.pushoff ? '<span class="history-push">pushed off!</span>' : ''}
    </div>`;
    }
    el.innerHTML = h;
    el.scrollTop = el.scrollHeight;
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
openModeModal();
setInterval(() => { if (state) renderClocks(); }, 250);
setInterval(() => { fetchState(); }, 1000);
setInterval(() => { maybeAutoAgentTurn(); }, 300);
