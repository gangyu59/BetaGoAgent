const canvas = document.getElementById("go-board");
const ctx = canvas.getContext("2d");
const statusLog = document.getElementById("status-log");

function log(msg) {
    statusLog.innerText = msg;
    console.log(msg);
}

log("Connecting to WebSocket...");
const ws = new WebSocket("ws://" + window.location.host + "/ws");

const BOARD_SIZE = 9;
const PADDING = 30;
const CELL_SIZE = (canvas.width - 2 * PADDING) / (BOARD_SIZE - 1);

let gameState = {
    board: Array(BOARD_SIZE).fill().map(() => Array(BOARD_SIZE).fill(0)),
    turn: 1,
    lastMove: null,
    analysis: null,
    score: null,
    moveCount: 0
};

let showHeatmap = true;

// --- Rendering ---

function drawBoard() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw grid
    ctx.strokeStyle = "#000";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i < BOARD_SIZE; i++) {
        ctx.moveTo(PADDING + i * CELL_SIZE, PADDING);
        ctx.lineTo(PADDING + i * CELL_SIZE, canvas.height - PADDING);
        ctx.moveTo(PADDING, PADDING + i * CELL_SIZE);
        ctx.lineTo(canvas.width - PADDING, PADDING + i * CELL_SIZE);
    }
    ctx.stroke();

    // Draw stars (hoshi) for 9x9
    const stars = [[2, 2], [6, 2], [4, 4], [2, 6], [6, 6]];
    ctx.fillStyle = "#000";
    for (let [x, y] of stars) {
        ctx.beginPath();
        ctx.arc(PADDING + x * CELL_SIZE, PADDING + y * CELL_SIZE, 3, 0, 2 * Math.PI);
        ctx.fill();
    }

    // Draw heatmap BEFORE stones so stones appear on top
    if (showHeatmap && gameState.analysis && gameState.analysis.policy) {
        drawHeatmap(gameState.analysis.policy);
    }

    // Draw stones
    for (let row = 0; row < BOARD_SIZE; row++) {
        for (let col = 0; col < BOARD_SIZE; col++) {
            const stone = gameState.board[row][col];
            if (stone !== 0) {
                drawStone(col, row, stone);
            }
        }
    }

    // Draw last move marker
    if (gameState.lastMove) {
        const [lx, ly] = gameState.lastMove;
        ctx.strokeStyle = "#ff0000";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(PADDING + ly * CELL_SIZE, PADDING + lx * CELL_SIZE, CELL_SIZE * 0.2, 0, 2 * Math.PI);
        ctx.stroke();
    }
}

function drawStone(col, row, color) {
    const cx = PADDING + col * CELL_SIZE;
    const cy = PADDING + row * CELL_SIZE;
    const r = CELL_SIZE * 0.45;

    // Reset shadow before drawing
    ctx.shadowColor = "transparent";
    ctx.shadowBlur = 0;
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 0;

    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, 2 * Math.PI);

    const grad = ctx.createRadialGradient(cx - r/3, cy - r/3, r/5, cx, cy, r);
    if (color === 1) {
        grad.addColorStop(0, "#444");
        grad.addColorStop(1, "#000");
    } else {
        grad.addColorStop(0, "#fff");
        grad.addColorStop(1, "#ddd");
    }

    ctx.fillStyle = grad;
    ctx.fill();

    // Shadow
    ctx.shadowColor = "rgba(0,0,0,0.5)";
    ctx.shadowBlur = 5;
    ctx.shadowOffsetX = 2;
    ctx.shadowOffsetY = 2;
    ctx.fill();
    ctx.shadowColor = "transparent";
    ctx.shadowBlur = 0;
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 0;

    // White stone outline
    if (color === 2) {
        ctx.strokeStyle = "#aaa";
        ctx.lineWidth = 1;
        ctx.stroke();
    }
}

function drawHeatmap(policy) {
    const maxProb = Math.max(...policy);
    if (maxProb < 0.001) return;

    for (let i = 0; i < policy.length; i++) {
        const prob = policy[i];
        if (prob < 0.01) continue;

        // policy index i = row * 9 + col
        const row = Math.floor(i / BOARD_SIZE);
        const col = i % BOARD_SIZE;

        const cx = PADDING + col * CELL_SIZE;
        const cy = PADDING + row * CELL_SIZE;

        const alpha = (prob / maxProb) * 0.5;
        ctx.fillStyle = `rgba(46, 204, 113, ${alpha})`;
        ctx.fillRect(cx - CELL_SIZE/2, cy - CELL_SIZE/2, CELL_SIZE, CELL_SIZE);

        if (prob > 0.05) {
            ctx.fillStyle = "#000";
            ctx.font = "10px Arial";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText((prob * 100).toFixed(0), cx, cy);
        }
    }
}

// --- Interaction ---

canvas.addEventListener("click", (e) => {
    if (gameState.turn !== 1) return; // Only allow clicks on human's turn

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const clickX = (e.clientX - rect.left) * scaleX;
    const clickY = (e.clientY - rect.top) * scaleY;

    // Canvas X -> col, Canvas Y -> row
    const col = Math.round((clickX - PADDING) / CELL_SIZE);
    const row = Math.round((clickY - PADDING) / CELL_SIZE);

    if (col >= 0 && col < BOARD_SIZE && row >= 0 && row < BOARD_SIZE) {
        log(`Move: (${row}, ${col})`);
        // Server expects (x, y) = (row, col) for board[x][y]
        ws.send(JSON.stringify({
            type: "move",
            x: row,
            y: col
        }));
    }
});

// --- UI Updates ---

function updateStats(turn, analysis) {
    const turnText = turn === 1 ? "Your Turn (Black)" : "AI Thinking (White)...";
    const turnEl = document.getElementById("turn-indicator");
    turnEl.innerText = turnText;

    if (turn === 1) {
        turnEl.style.color = "#ecf0f1";
        turnEl.style.fontWeight = "bold";
    } else {
        turnEl.style.color = "#e67e22";
        turnEl.style.fontWeight = "normal";
    }

    if (analysis) {
        const winRate = (analysis.value + 1) / 2 * 100;

        const fill = document.getElementById("win-rate-fill");
        const text = document.getElementById("win-rate-text");

        fill.style.width = `${winRate}%`;

        if (analysis.value > 0) {
            fill.style.backgroundColor = "#2c3e50";
            text.innerText = `Black ${winRate.toFixed(1)}%`;
            text.style.color = "#ecf0f1";
        } else {
            fill.style.backgroundColor = "#e74c3c";
            text.innerText = `White ${(100 - winRate).toFixed(1)}%`;
            text.style.color = "#2c3e50";
        }
    }
}

function updateScore(score) {
    if (!score) return;
    const blackEl = document.getElementById("score-black");
    const whiteEl = document.getElementById("score-white");
    if (blackEl) blackEl.innerText = score.black_total || 0;
    if (whiteEl) whiteEl.innerText = score.white_total || 7.5;
}

function toggleHeatmap() {
    showHeatmap = !showHeatmap;
    drawBoard();
}

function playerPass() {
    ws.send(JSON.stringify({ type: "pass" }));
}

function newGame() {
    const btn = document.getElementById("btn-new-game");
    if (btn) {
        btn.disabled = true;
        btn.innerText = "Starting...";
    }
    const resultEl = document.getElementById("game-result");
    if (resultEl) resultEl.style.display = "none";

    gameState.board = Array(BOARD_SIZE).fill().map(() => Array(BOARD_SIZE).fill(0));
    gameState.turn = 1;
    gameState.lastMove = null;
    gameState.analysis = null;
    gameState.moveCount = 0;
    drawBoard();
    updateStats(gameState.turn, gameState.analysis);

    ws.send(JSON.stringify({ type: "start_game" }));
    setTimeout(() => {
        if (btn) {
            btn.disabled = false;
            btn.innerText = "New Game";
        }
    }, 1500);
}

function toggleTraining() {
    const btn = document.getElementById("btn-train");
    if (btn) {
        btn.disabled = true;
        btn.innerText = "Processing...";
        btn.style.backgroundColor = "#f1c40f";
    }
    ws.send(JSON.stringify({ type: "toggle_training" }));
    setTimeout(() => {
        if (btn) btn.disabled = false;
    }, 2000);
}

// --- WebSocket ---

ws.onopen = () => {
    log("Connected to server");
};

ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);

    if (msg.type === "init" || msg.type === "update") {
        gameState.board = msg.board;
        gameState.turn = msg.turn;
        if (msg.last_move !== undefined) gameState.lastMove = msg.last_move;
        if (msg.analysis) gameState.analysis = msg.analysis;
        if (msg.score) {
            gameState.score = msg.score;
            updateScore(msg.score);
        }
        gameState.moveCount++;

        const moveEl = document.getElementById("move-count");
        if (moveEl) moveEl.innerText = gameState.moveCount;

        drawBoard();
        updateStats(gameState.turn, gameState.analysis);
    } else if (msg.type === "game_over") {
        gameState.board = msg.board;
        if (msg.score) {
            gameState.score = msg.score;
            updateScore(msg.score);
        }
        drawBoard();
        const resultEl = document.getElementById("game-result");
        if (resultEl) {
            const winner = msg.winner === 1 ? "Black" : "White";
            const s = msg.score;
            resultEl.innerText = `Game Over! ${winner} wins (B:${s.black_total} vs W:${s.white_total})`;
            resultEl.style.display = "block";
        }
    } else if (msg.type === "training_update") {
        updateTrainingChart(msg.stats);
        updateTrainingButton(msg.running);
    } else if (msg.type === "ai_thinking") {
        const hourglass = document.getElementById("ai-hourglass");
        if (hourglass) {
            hourglass.style.display = msg.on ? "block" : "none";
        }
    }
};

function updateTrainingButton(running) {
    const btn = document.getElementById("btn-train");
    if (running) {
        btn.innerText = "Stop Training";
        btn.style.backgroundColor = "#e74c3c";
        btn.disabled = false;
    } else {
        btn.innerText = "Start Training";
        btn.style.backgroundColor = "#2ecc71";
        btn.disabled = false;
    }
}

ws.onerror = (err) => {
    log("WebSocket Error: " + err);
};

// Initial draw
drawBoard();

// --- Chart.js Setup ---
const ctxChart = document.getElementById("lossChart").getContext("2d");
const lossChart = new Chart(ctxChart, {
    type: "line",
    data: {
        datasets: [{
            label: "Policy Loss",
            data: [],
            borderColor: "#3498db",
            tension: 0.1,
            yAxisID: "yPolicy",
            pointRadius: 1
        }, {
            label: "Value Loss",
            data: [],
            borderColor: "#e67e22",
            tension: 0.1,
            yAxisID: "yValue",
            pointRadius: 1
        }, {
            label: "Accuracy",
            data: [],
            borderColor: "#9b59b6",
            tension: 0.1,
            yAxisID: "yWinRate",
            pointRadius: 1
        }]
    },
    options: {
        responsive: true,
        parsing: false,
        animation: false,
        plugins: {
            decimation: {
                enabled: true,
                algorithm: "lttb",
                samples: 500
            }
        },
        scales: {
            x: {
                type: "linear",
                title: { display: true, text: "Step" }
            },
            yPolicy: {
                position: "left",
                beginAtZero: true,
                title: { display: true, text: "Policy Loss" }
            },
            yValue: {
                position: "right",
                beginAtZero: true,
                title: { display: true, text: "Value Loss" }
            },
            yWinRate: {
                position: "right",
                beginAtZero: true,
                max: 1.0,
                grid: { drawOnChartArea: false },
                ticks: {
                    callback: (v) => (v * 100).toFixed(0) + "%"
                }
            }
        }
    }
});

const ctxChartLarge = document.getElementById("lossChartLarge").getContext("2d");
const lossChartLarge = new Chart(ctxChartLarge, {
    type: "line",
    data: lossChart.data,
    options: {
        responsive: true,
        parsing: false,
        animation: false,
        plugins: {
            decimation: {
                enabled: true,
                algorithm: "lttb",
                samples: 2000
            }
        },
        scales: {
            x: {
                type: "linear",
                title: { display: true, text: "Step" }
            },
            yPolicy: {
                position: "left",
                beginAtZero: true
            },
            yValue: {
                position: "right",
                beginAtZero: true
            },
            yWinRate: {
                position: "right",
                beginAtZero: true,
                max: 1.0,
                grid: { drawOnChartArea: false },
                ticks: {
                    callback: (v) => (v * 100).toFixed(0) + "%"
                }
            }
        }
    }
});

const trainingOverlay = document.getElementById("training-overlay");
if (trainingOverlay) trainingOverlay.style.display = "block";

function updateTrainingChart(stats) {
    if (!stats.step) return;

    lossChart.data.datasets[0].data.push({ x: stats.step, y: stats.policy_loss });
    lossChart.data.datasets[1].data.push({ x: stats.step, y: stats.value_loss });
    lossChart.data.datasets[2].data.push({ x: stats.step, y: stats.win_rate });

    lossChart.update('none');
    lossChartLarge.update('none');

    const winRate = (stats.win_rate * 100).toFixed(1);
    const wrSpan = document.getElementById("training-win-rate");
    if (wrSpan) wrSpan.innerText = winRate;

    const lrSpan = document.getElementById("lr-val");
    if (lrSpan && stats.lr) lrSpan.innerText = stats.lr.toFixed(6);

    document.getElementById("p-loss-val").innerText = stats.policy_loss.toFixed(4);
    document.getElementById("v-loss-val").innerText = stats.value_loss.toFixed(4);

    if (stats.games !== undefined) {
        document.getElementById("game-count").innerText = stats.games;
    }
    if (stats.buffer !== undefined) {
        statusLog.innerText = `Buffer: ${stats.buffer}/${stats.min_buffer ?? 512}  |  Step: ${stats.step}`;
        const overlayEl = document.getElementById("training-overlay");
        const fillEl = document.getElementById("buffer-fill");
        if (fillEl) {
            const minBuf = stats.min_buffer ?? 512;
            const pct = Math.max(0, Math.min(100, (stats.buffer / minBuf) * 100));
            fillEl.style.width = `${pct}%`;
        }
        if (overlayEl) {
            overlayEl.style.display = (stats.step > 0) ? "none" : "block";
            const msgEl = overlayEl.querySelector(".overlay-msg");
            if (msgEl) {
                msgEl.innerText = (stats.step > 0) ? "Training running..." : "Waiting for data to start training...";
            }
        }
    }
}
