// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let overview = null;
let fps = 25;
let totalFrames = 0;
let teamNames = {};
let currentChart = 'compactness';
let lastRenderedFrame = -1;
const frameCache = {};

const PITCH = { L: 105, W: 68, PEN_D: 16.5, PEN_W: 40.32, GA_D: 5.5, GA_W: 18.32, CR: 9.15 };
const TEAM_COLORS = { 0: '#ff6432', 1: '#3264ff', '-1': '#aaaaaa' };

const video = document.getElementById('match-video');
const canvas = document.getElementById('pitch-canvas');
const ctx = canvas.getContext('2d');

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
async function init() {
    const resp = await fetch(`/api/match/${MATCH_ID}/overview`);
    overview = await resp.json();
    fps = overview.fps;
    totalFrames = overview.total_frames;
    teamNames = overview.team_names;

    renderSummary();
    renderEvents();
    renderChart();
    setupChartTabs();
    setupVideoSync();
    setupExport();

    document.getElementById('loading').style.display = 'none';
    document.getElementById('summary-section').style.display = '';
    document.getElementById('events-section').style.display = '';
    document.getElementById('charts-section').style.display = '';

    // Load first frame
    const frameData = await getFrameData(0);
    drawPitchWithPlayers(frameData);
}

// ---------------------------------------------------------------------------
// Data fetching
// ---------------------------------------------------------------------------
async function getFrameData(frameNum) {
    if (frameNum < 0 || frameNum >= totalFrames) return null;
    if (frameCache[frameNum]) return frameCache[frameNum];
    const resp = await fetch(`/api/match/${MATCH_ID}/frame/${frameNum}`);
    const data = await resp.json();
    frameCache[frameNum] = data;
    return data;
}

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------
function renderSummary() {
    const s = overview.summary;
    const container = document.getElementById('summary-content');
    let html = '';

    for (const half of ['1ère MT', '2ème MT']) {
        html += `<div class="summary-half"><div class="summary-half-label">${half}</div>`;
        for (const team of ['team_a', 'team_b']) {
            const key = `${half}_${team}`;
            if (s[key]) {
                html += `<div class="summary-line">${s[key].text}</div>`;
            }
        }
        html += `</div>`;
    }
    container.innerHTML = html;
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------
function renderEvents() {
    const events = overview.events;
    document.getElementById('event-count').textContent = `${events.length} événement${events.length > 1 ? 's' : ''}`;
    const list = document.getElementById('events-list');

    if (events.length === 0) {
        list.innerHTML = '<div style="padding:1rem;color:var(--text-secondary)">Aucun moment clé détecté</div>';
        return;
    }

    list.innerHTML = events.map(ev => {
        const min = Math.floor(ev.time / 60);
        const sec = Math.floor(ev.time % 60);
        const timeStr = `${min}:${String(sec).padStart(2, '0')}`;
        const teamClass = ev.team === 'team_a' ? 'team-a' : 'team-b';
        const teamLabel = teamNames[ev.team] || ev.team;
        return `
            <div class="event-item" onclick="seekTo(${ev.time})">
                <span class="event-time">${timeStr}</span>
                <span class="event-team ${teamClass}">${teamLabel}</span>
                <span class="event-desc">${ev.description}</span>
            </div>`;
    }).join('');
}

function seekTo(time) {
    video.currentTime = time;
    video.play();
}

// ---------------------------------------------------------------------------
// Charts
// ---------------------------------------------------------------------------
const CHART_CONFIG = {
    compactness: { title: 'Compacité (m²)', key: 'compactness', unit: 'm²' },
    block_height: { title: 'Hauteur du bloc (m)', key: 'block_height', unit: 'm' },
    width: { title: 'Largeur (m)', key: 'width', unit: 'm' },
    length: { title: 'Longueur (m)', key: 'length', unit: 'm' },
};

function renderChart() {
    const cfg = CHART_CONFIG[currentChart];
    const traces = [];

    for (const teamKey of ['team_a', 'team_b']) {
        const data = overview.metrics[teamKey];
        traces.push({
            x: data.times,
            y: data[cfg.key],
            mode: 'lines',
            name: teamNames[teamKey] || teamKey,
            line: { color: teamKey === 'team_a' ? '#ff6432' : '#3264ff', width: 2 },
            connectgaps: false,
        });
    }

    const layout = {
        template: 'plotly_dark',
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        margin: { l: 50, r: 20, t: 10, b: 40 },
        xaxis: {
            title: 'Temps (s)',
            gridcolor: '#30363d',
            tickformat: ',d',
        },
        yaxis: {
            title: cfg.title,
            gridcolor: '#30363d',
        },
        legend: { orientation: 'h', y: 1.12 },
        hovermode: 'x unified',
    };

    Plotly.newPlot('metrics-chart', traces, layout, { responsive: true });

    // Click on chart to seek video
    document.getElementById('metrics-chart').on('plotly_click', (data) => {
        if (data.points.length > 0) {
            const time = data.points[0].x;
            seekTo(time);
        }
    });
}

function setupChartTabs() {
    document.querySelectorAll('.chart-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.chart-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            currentChart = tab.dataset.chart;
            renderChart();
        });
    });
}

// ---------------------------------------------------------------------------
// Video sync
// ---------------------------------------------------------------------------
function setupVideoSync() {
    let rafId = null;

    function onFrame() {
        const frame = Math.floor(video.currentTime * fps);
        updateTimeDisplay(video.currentTime);

        if (frame !== lastRenderedFrame && frame >= 0 && frame < totalFrames) {
            lastRenderedFrame = frame;
            getFrameData(frame).then(data => {
                if (data) drawPitchWithPlayers(data);
            });
        }
        rafId = requestAnimationFrame(onFrame);
    }

    video.addEventListener('play', () => { rafId = requestAnimationFrame(onFrame); });
    video.addEventListener('pause', () => { cancelAnimationFrame(rafId); });
    video.addEventListener('seeked', () => {
        const frame = Math.floor(video.currentTime * fps);
        updateTimeDisplay(video.currentTime);
        if (frame !== lastRenderedFrame) {
            lastRenderedFrame = frame;
            getFrameData(frame).then(data => {
                if (data) drawPitchWithPlayers(data);
            });
        }
    });
}

function updateTimeDisplay(seconds) {
    const min = Math.floor(seconds / 60);
    const sec = Math.floor(seconds % 60);
    document.getElementById('time-display').textContent = `${min}:${String(sec).padStart(2, '0')}`;
}

// ---------------------------------------------------------------------------
// Pitch drawing (Canvas)
// ---------------------------------------------------------------------------
function getPitchTransform() {
    const cw = canvas.width;
    const ch = canvas.height;
    const marginX = 30;
    const marginY = 30;
    const scaleX = (cw - 2 * marginX) / PITCH.L;
    const scaleY = (ch - 2 * marginY) / PITCH.W;
    const scale = Math.min(scaleX, scaleY);
    const ox = (cw - PITCH.L * scale) / 2;
    const oy = (ch - PITCH.W * scale) / 2;
    return { scale, ox, oy };
}

function m2px(mx, my) {
    const { scale, ox, oy } = getPitchTransform();
    return [ox + mx * scale, oy + my * scale];
}

function drawPitch() {
    const { scale, ox, oy } = getPitchTransform();
    const cw = canvas.width;
    const ch = canvas.height;

    // Background
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, cw, ch);

    // Pitch surface
    const [px0, py0] = m2px(0, 0);
    ctx.fillStyle = '#228b22';
    ctx.fillRect(px0, py0, PITCH.L * scale, PITCH.W * scale);

    ctx.strokeStyle = 'rgba(255,255,255,0.8)';
    ctx.lineWidth = 1.5;

    // Outline
    ctx.strokeRect(px0, py0, PITCH.L * scale, PITCH.W * scale);

    // Halfway
    const [hx, hy0] = m2px(PITCH.L / 2, 0);
    const [, hy1] = m2px(PITCH.L / 2, PITCH.W);
    ctx.beginPath(); ctx.moveTo(hx, hy0); ctx.lineTo(hx, hy1); ctx.stroke();

    // Center circle
    const [cx, cy] = m2px(PITCH.L / 2, PITCH.W / 2);
    ctx.beginPath(); ctx.arc(cx, cy, PITCH.CR * scale, 0, Math.PI * 2); ctx.stroke();

    // Center spot
    ctx.beginPath(); ctx.arc(cx, cy, 3, 0, Math.PI * 2);
    ctx.fillStyle = 'white'; ctx.fill();

    // Penalty areas
    const py1 = (PITCH.W - PITCH.PEN_W) / 2;
    const py2 = (PITCH.W + PITCH.PEN_W) / 2;
    for (const [x0, w] of [[0, PITCH.PEN_D], [PITCH.L - PITCH.PEN_D, PITCH.PEN_D]]) {
        const [rx, ry] = m2px(x0, py1);
        ctx.strokeRect(rx, ry, w * scale, PITCH.PEN_W * scale);
    }

    // Goal areas
    const gy1 = (PITCH.W - PITCH.GA_W) / 2;
    for (const [x0, w] of [[0, PITCH.GA_D], [PITCH.L - PITCH.GA_D, PITCH.GA_D]]) {
        const [rx, ry] = m2px(x0, gy1);
        ctx.strokeRect(rx, ry, w * scale, PITCH.GA_W * scale);
    }
}

function drawPitchWithPlayers(frameData) {
    drawPitch();
    if (!frameData) return;

    const { scale } = getPitchTransform();

    // Separate teams
    const teams = { 0: [], 1: [] };
    for (const p of frameData.players) {
        if (teams[p.team]) teams[p.team].push(p);
    }

    // Draw convex hulls
    for (const teamId of [0, 1]) {
        const players = teams[teamId];
        if (players.length < 3) continue;
        const hull = convexHull(players.map(p => [p.pitch_x, p.pitch_y]));
        if (hull.length < 3) continue;

        ctx.beginPath();
        const [hx0, hy0] = m2px(hull[0][0], hull[0][1]);
        ctx.moveTo(hx0, hy0);
        for (let i = 1; i < hull.length; i++) {
            const [hx, hy] = m2px(hull[i][0], hull[i][1]);
            ctx.lineTo(hx, hy);
        }
        ctx.closePath();
        ctx.fillStyle = teamId === 0 ? 'rgba(255,100,50,0.12)' : 'rgba(50,100,255,0.12)';
        ctx.fill();
        ctx.strokeStyle = teamId === 0 ? 'rgba(255,100,50,0.5)' : 'rgba(50,100,255,0.5)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
        ctx.stroke();
        ctx.setLineDash([]);
    }

    // Draw players
    for (const p of frameData.players) {
        const [px, py] = m2px(p.pitch_x, p.pitch_y);
        const color = TEAM_COLORS[p.team] || TEAM_COLORS['-1'];

        ctx.beginPath();
        ctx.arc(px, py, 7, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Label
        if (p.cls === 'goalkeeper') {
            ctx.fillStyle = 'white';
            ctx.font = '9px sans-serif';
            ctx.fillText('GK', px + 9, py + 3);
        }
    }

    // Draw ball
    if (frameData.ball) {
        const [bx, by] = m2px(frameData.ball.pitch_x, frameData.ball.pitch_y);
        ctx.beginPath();
        ctx.arc(bx, by, 5, 0, Math.PI * 2);
        ctx.fillStyle = '#ffff00';
        ctx.fill();
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 1;
        ctx.stroke();
    }

    // Update pitch metrics below canvas
    updatePitchMetrics(frameData);
}

function updatePitchMetrics(frameData) {
    const container = document.getElementById('pitch-metrics');
    const metricsHtml = [];

    for (const [teamId, teamKey] of [[0, 'team_a'], [1, 'team_b']]) {
        const players = frameData.players.filter(p => p.team === teamId);
        const name = teamNames[teamKey] || teamKey;
        const color = teamId === 0 ? 'var(--team-a)' : 'var(--team-b)';
        metricsHtml.push(`<div class="pitch-metric"><span class="pitch-metric-label" style="color:${color}">${name}</span>: <span class="pitch-metric-value">${players.length}j</span></div>`);
    }

    container.innerHTML = metricsHtml.join('');
}

// ---------------------------------------------------------------------------
// Convex Hull (Graham scan)
// ---------------------------------------------------------------------------
function convexHull(points) {
    if (points.length < 3) return points;
    const pts = points.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);

    function cross(O, A, B) {
        return (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0]);
    }

    const lower = [];
    for (const p of pts) {
        while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0)
            lower.pop();
        lower.push(p);
    }
    const upper = [];
    for (const p of pts.reverse()) {
        while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0)
            upper.pop();
        upper.push(p);
    }
    upper.pop();
    lower.pop();
    return lower.concat(upper);
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------
function setupExport() {
    document.getElementById('export-pitch-btn').addEventListener('click', () => {
        const frame = Math.floor(video.currentTime * fps);
        const url = `/api/match/${MATCH_ID}/export/frame/${frame}`;
        const a = document.createElement('a');
        a.href = url;
        a.download = '';
        a.click();
    });
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
init();
