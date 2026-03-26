
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
    turn: 1, // 1=Black, 2=White
    lastMove: null,
    analysis: null
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
        // Vertical
        ctx.moveTo(PADDING + i * CELL_SIZE, PADDING);
        ctx.lineTo(PADDING + i * CELL_SIZE, canvas.height - PADDING);
        // Horizontal
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

    // Draw stones
    for (let y = 0; y < BOARD_SIZE; y++) {
        for (let x = 0; x < BOARD_SIZE; x++) {
            const stone = gameState.board[x][y];
            if (stone !== 0) {
                drawStone(x, y, stone);
            }
        }
    }

    // Draw last move marker
    if (gameState.lastMove) {
        const [lx, ly] = gameState.lastMove;
        ctx.strokeStyle = "#ff0000";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(PADDING + lx * CELL_SIZE, PADDING + ly * CELL_SIZE, CELL_SIZE * 0.2, 0, 2 * Math.PI);
        ctx.stroke();
    }

    // Draw heatmap
    if (showHeatmap && gameState.analysis && gameState.analysis.policy) {
        drawHeatmap(gameState.analysis.policy);
    }
}

function drawStone(x, y, color) {
    const cx = PADDING + x * CELL_SIZE;
    const cy = PADDING + y * CELL_SIZE;
    const r = CELL_SIZE * 0.45;

    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, 2 * Math.PI);
    
    // Gradient for 3D effect
    const grad = ctx.createRadialGradient(cx - r/3, cy - r/3, r/5, cx, cy, r);
    if (color === 1) { // Black
        grad.addColorStop(0, "#444");
        grad.addColorStop(1, "#000");
    } else { // White
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
    ctx.shadowColor = "transparent"; // Reset
}

function drawHeatmap(policy) {
    // policy is a flat array of 81 floats
    const maxProb = Math.max(...policy);
    
    for (let i = 0; i < policy.length; i++) {
        const prob = policy[i];
        if (prob < 0.01) continue; // Skip low prob
        
        const x = Math.floor(i / BOARD_SIZE);
        const y = i % BOARD_SIZE;
        
        const cx = PADDING + x * CELL_SIZE;
        const cy = PADDING + y * CELL_SIZE;
        
        // Green square with alpha based on probability
        const alpha = (prob / maxProb) * 0.6;
        ctx.fillStyle = `rgba(46, 204, 113, ${alpha})`;
        ctx.fillRect(cx - CELL_SIZE/2, cy - CELL_SIZE/2, CELL_SIZE, CELL_SIZE);
        
        // Draw prob text if high enough
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
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    const clickX = (e.clientX - rect.left) * scaleX;
    const clickY = (e.clientY - rect.top) * scaleY;
    
    // Convert to grid coordinates
    // x = (clickX - PADDING) / CELL_SIZE
    const x = Math.round((clickX - PADDING) / CELL_SIZE);
    const y = Math.round((clickY - PADDING) / CELL_SIZE);
    
    if (x >= 0 && x < BOARD_SIZE && y >= 0 && y < BOARD_SIZE) {
        log(`Click at ${x}, ${y}`);
        // Send move to server
        ws.send(JSON.stringify({
            type: "move",
            x: x,
            y: y
        }));
    }
});

// --- UI Updates ---

function updateStats(turn, analysis) {
    const turnText = turn === 1 ? "Your Turn (Black)" : "AI Thinking (White)...";
    const turnEl = document.getElementById("turn-indicator");
    turnEl.innerText = turnText;
    
    // Highlight turn style
    if (turn === 1) {
        turnEl.style.color = "#2c3e50";
        turnEl.style.fontWeight = "bold";
    } else {
        turnEl.style.color = "#e67e22";
        turnEl.style.fontWeight = "normal";
    }
    
    if (analysis) {
        // Value is from -1 (White wins) to 1 (Black wins)
        const winRate = (analysis.value + 1) / 2 * 100; // 0 to 100% for Black
        
        const fill = document.getElementById("win-rate-fill");
        const text = document.getElementById("win-rate-text");
        
        fill.style.width = `${winRate}%`;
        
        if (analysis.value > 0) {
            fill.style.backgroundColor = "#2c3e50"; // Black color
            text.innerText = `Black ${winRate.toFixed(1)}%`;
            text.style.color = "#ecf0f1";
        } else {
            fill.style.backgroundColor = "#e74c3c"; // Red/White color
            text.innerText = `White ${(100 - winRate).toFixed(1)}%`;
            text.style.color = "#2c3e50";
        }
    }
}

function toggleHeatmap() {
    showHeatmap = !showHeatmap;
    drawBoard();
}

function resetGame() {
    ws.send(JSON.stringify({ type: "reset" }));
}

function startGame() {
    ws.send(JSON.stringify({ type: "start_game" }));
}

function newGame() {
    const btn = document.getElementById("btn-new-game");
    if (btn) {
        btn.disabled = true;
        btn.innerText = "Starting...";
    }
    // Optimistic UI: clear board immediately
    gameState.board = Array(BOARD_SIZE).fill().map(() => Array(BOARD_SIZE).fill(0));
    gameState.turn = 1;
    gameState.lastMove = null;
    gameState.analysis = null;
    drawBoard();
    updateStats(gameState.turn, gameState.analysis);
    // Send to server
    ws.send(JSON.stringify({ type: "start_game" }));
    // Fallback re-enable
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
        btn.style.backgroundColor = "#f1c40f"; // Yellow
    }
    ws.send(JSON.stringify({ type: "toggle_training" }));
    // Fallback: re-enable after 2s if no server update arrives
    setTimeout(() => {
        if (btn) {
            btn.disabled = false;
        }
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
        if (msg.last_move) gameState.lastMove = msg.last_move;
        if (msg.analysis) gameState.analysis = msg.analysis;
        
        drawBoard();
        updateStats(gameState.turn, gameState.analysis);
        log("Board updated");
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
        btn.style.backgroundColor = "#e74c3c"; // Red
        btn.disabled = false;
    } else {
        btn.innerText = "Start Training";
        btn.style.backgroundColor = "#2ecc71"; // Green
        btn.disabled = false;
    }
}

ws.onerror = (err) => {
    log("WebSocket Error: " + err);
};

// Initial draw (empty board)
drawBoard();

// --- Chart.js Setup (Mock for now) ---
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
            pointRadius: 2
        }, {
            label: "Value Loss",
            data: [],
            borderColor: "#e67e22",
            tension: 0.1,
            yAxisID: "yValue",
            pointRadius: 2
        }, {
            label: "Accuracy",
            data: [],
            borderColor: "#9b59b6", // Purple
            tension: 0.1,
            yAxisID: "yWinRate",
            pointRadius: 2
        }, {
            label: "Policy Loss Target",
            data: [],
            borderColor: "#95a5a6",
            borderDash: [6, 4],
            tension: 0,
            pointRadius: 0,
            yAxisID: "yPolicy"
        }, {
            label: "Value Loss Target",
            data: [],
            borderColor: "#7f8c8d",
            borderDash: [6, 4],
            tension: 0,
            pointRadius: 0,
            yAxisID: "yValue"
        }]
    },
    options: {
        responsive: true,
        parsing: false,
        plugins: {
            decimation: {
                enabled: true,
                algorithm: "lttb",
                samples: 1000
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
                beginAtZero: true,
                suggestedMax: 0.005,
                ticks: {
                    callback: (v) => v.toFixed(4)
                }
            },
            yWinRate: {
                position: "right",
                beginAtZero: true,
                max: 1.0,
                grid: {
                    drawOnChartArea: false // Don't clutter grid
                },
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
                beginAtZero: true,
                suggestedMax: 0.005,
                ticks: {
                    callback: (v) => v.toFixed(4)
                }
            }
        }
    }
});

// Seed chart with initial points and show overlay immediately
lossChart.data.datasets[2].data.push({ x: 0, y: 0.5 }, { x: 1, y: 0.5 });
lossChart.data.datasets[3].data.push({ x: 0, y: 0.2 }, { x: 1, y: 0.2 });
lossChart.update();
lossChartLarge.update();
const trainingOverlay = document.getElementById("training-overlay");
if (trainingOverlay) trainingOverlay.style.display = "block";
// Simulate live chart updates
// setInterval(() => {
//    if (lossChart.data.labels.length > 20) {
//        lossChart.data.labels.shift();
//        lossChart.data.datasets[0].data.shift();
//        lossChart.data.datasets[1].data.shift();
//    }
//    const step = lossChart.data.labels.length + 1;
//    lossChart.data.labels.push(step);
//    lossChart.data.datasets[0].data.push(Math.random() * 2 + 1); // Mock loss
//    lossChart.data.datasets[1].data.push(Math.random() * 1 + 0.5);
//    lossChart.update();
// }, 2000);

function updateTrainingChart(stats) {
    lossChart.data.datasets[0].data.push({ x: stats.step, y: stats.policy_loss });
    lossChart.data.datasets[1].data.push({ x: stats.step, y: stats.value_loss });
    // Dataset 2 is Accuracy/WinRate
    lossChart.data.datasets[2].data.push({ x: stats.step, y: stats.win_rate }); 
    // Targets are now 3 and 4
    lossChart.data.datasets[3].data.push({ x: stats.step, y: 0.5 });
    lossChart.data.datasets[4].data.push({ x: stats.step, y: 0.2 });
    
    // Dynamically adjust value axis to show fine details
    const curVal = stats.value_loss;
    const newMax = Math.max(0.001, curVal * 3);
    lossChart.options.scales.yValue.max = newMax;
    lossChartLarge.options.scales.yValue.max = newMax;
    lossChart.update();
    lossChartLarge.update();

    const winRate = (stats.win_rate * 100).toFixed(1);
    const wrSpan = document.getElementById("training-win-rate");
    if (wrSpan) wrSpan.innerText = winRate;

    document.getElementById("p-loss-val").innerText = stats.policy_loss.toFixed(4);
    document.getElementById("v-loss-val").innerText = stats.value_loss.toFixed(4);
    
    if (stats.games !== undefined) {
        document.getElementById("game-count").innerText = stats.games;
    }
    if (stats.buffer !== undefined) {
        statusLog.innerText = `Buffer: ${stats.buffer}/${stats.min_buffer ?? 1000}  |  Step: ${stats.step}`;
        const overlayEl = document.getElementById("training-overlay");
        const fillEl = document.getElementById("buffer-fill");
        if (fillEl) {
            const minBuf = stats.min_buffer ?? 1000;
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
