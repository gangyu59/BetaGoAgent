from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import json
import asyncio
import sys
import os
import numpy as np
import torch
import subprocess

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from core.go_engine import GoEngine
from brain.network import BetaGoNet
from brain.feature_encoder import FeatureEncoder
from core.mcts import MCTS
import threading
import time
import concurrent.futures

app = FastAPI()

# Paths
BASE_DIR = os.path.join(os.path.dirname(__file__), '../../')
MODEL_PATH = os.path.join(BASE_DIR, 'model_latest.pth')
STATS_PATH = os.path.join(BASE_DIR, 'training_stats.json')
CMD_PATH = os.path.join(BASE_DIR, 'training_cmd.json')

# KataGo paths (external engine)
KATAGO_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'KataGo'))
KATAGO_EXE = os.path.join(KATAGO_DIR, 'katago.exe')
KATAGO_CONFIG = os.path.join(KATAGO_DIR, 'default_gtp.cfg')
KATAGO_MODEL = os.path.join(KATAGO_DIR, 'model.bin.gz')

# Thread Pool for CPU-bound MCTS
executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

# Global Locks
network_lock = threading.Lock()
game_lock = asyncio.Lock()

# GTP coordinate helpers for KataGo
GTP_COLUMNS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"


def coords_to_gtp(x: int, y: int, board_size: int = 9) -> str:
    """
    Convert internal (x, y) coordinates (0-based, x=row from top, y=col from left)
    to GTP coordinates like 'D4' (columns A,B,C,... skipping I; rows from bottom).
    """
    col = GTP_COLUMNS[y]
    row = board_size - x
    return f"{col}{row}"


def gtp_to_coords(coord: str, board_size: int = 9):
    """
    Convert GTP coordinate like 'D4' back to internal (x, y).
    """
    coord = coord.strip()
    if coord.lower() == "pass":
        return None
    col_char = coord[0].upper()
    row = int(coord[1:])
    y = GTP_COLUMNS.index(col_char)
    x = board_size - row
    return x, y


# KataGo process (shared for all KataGo WebSocket sessions)
katago_proc = None
katago_proc_lock = asyncio.Lock()


def katago_available() -> bool:
    return os.path.exists(KATAGO_EXE) and os.path.exists(KATAGO_CONFIG) and os.path.exists(KATAGO_MODEL)


def _start_katago_if_needed():
    global katago_proc
    if katago_proc is not None and katago_proc.poll() is None:
        return

    if not katago_available():
        print("[KataGo] Missing exe/model/config; cannot start KataGo engine.")
        katago_proc = None
        return

    print(f"[KataGo] Starting engine from {KATAGO_EXE}")
    katago_proc = subprocess.Popen(
        [KATAGO_EXE, "gtp", "-model", KATAGO_MODEL, "-config", KATAGO_CONFIG],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=KATAGO_DIR,  # Must run from KataGo dir for DLLs on Windows
    )


def _send_gtp_command_sync(cmd: str) -> str:
    """
    Send a single GTP command and return the response (without leading '=' and trailing empty line).
    This function is blocking and must be called in a thread/executor.
    """
    global katago_proc
    if katago_proc is None or katago_proc.stdin is None or katago_proc.stdout is None:
        return ""

    try:
        katago_proc.stdin.write(cmd + "\n")
        katago_proc.stdin.flush()
    except Exception as e:
        print(f"[KataGo] Failed to write command '{cmd}': {e}")
        return ""

    result_lines = []
    first = True
    while True:
        line = katago_proc.stdout.readline()
        if not line:
            break
        line = line.rstrip("\r\n")

        if first:
            first = False
            if line.startswith("="):
                content = line[1:].strip()
                if content:
                    result_lines.append(content)
            else:
                # error or other output
                result_lines.append(line)
        else:
            if line == "":
                break
            result_lines.append(line)

        if line == "":
            break

    return "\n".join(result_lines).strip()


async def _send_gtp_command(cmd: str) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _send_gtp_command_sync, cmd)


async def katago_genmove_for_moves(moves, board_size: int, color: str = "W"):
    """
    Ask KataGo for a move given a full history of moves.
    moves: list of tuples (color_str, (x, y) or None), where color_str is 'B' or 'W'.
    color: 'B' or 'W' - which side to generate move for.
    Returns: ( (x, y) or None, error_str or None )
    """
    async with katago_proc_lock:
        _start_katago_if_needed()
        if katago_proc is None:
            return None, "unavailable"

        await _send_gtp_command(f"boardsize {board_size}")
        await _send_gtp_command("clear_board")
        await _send_gtp_command("komi 7.5")

        for c, move in moves:
            if move is None:
                await _send_gtp_command(f"play {c} pass")
            else:
                mx, my = move
                coord = coords_to_gtp(mx, my, board_size=board_size)
                await _send_gtp_command(f"play {c} {coord}")

        resp = await _send_gtp_command(f"genmove {color}")
        if not resp:
            return None, "no_response"

        token = resp.split()[0]
        if token.lower() == "pass":
            return None, None

        try:
            xy = gtp_to_coords(token, board_size=board_size)
            if xy is None:
                return None, None
            return xy, None
        except Exception as e:
            print(f"[KataGo] Failed to parse GTP coord '{token}': {e}")
            return None, "parse_error"


def _katago_analyze_sync(moves, board_size: int, player: str = "B", max_moves: int = 3):
    """
    Set position, run kata-analyze for player, collect top max_moves with winrate.
    Must be called with katago_proc_lock held (run in executor).
    Returns: list of {"x": row, "y": col, "win_rate": float in [0,1]} or error message.
    """
    global katago_proc
    if katago_proc is None or katago_proc.stdin is None or katago_proc.stdout is None:
        return None, "unavailable"
    try:
        _start_katago_if_needed()
        if katago_proc is None:
            return None, "unavailable"
    except Exception:
        return None, "unavailable"

    # Set board
    _send_gtp_command_sync(f"boardsize {board_size}")
    _send_gtp_command_sync("clear_board")
    _send_gtp_command_sync("komi 7.5")
    for color, move in moves:
        if move is None:
            _send_gtp_command_sync(f"play {color} pass")
        else:
            mx, my = move
            coord = coords_to_gtp(mx, my, board_size=board_size)
            _send_gtp_command_sync(f"play {color} {coord}")

    # Start analyze with interval so KataGo outputs periodically (interval in centiseconds; 10 = 100ms)
    # Without interval it may not print until much later and we block forever on readline()
    cmd = f"kata-analyze {player} interval 10 maxmoves {max_moves}\n"
    katago_proc.stdin.write(cmd)
    katago_proc.stdin.flush()

    # Index by KataGo's "order" (0=best, 1=second, 2=third) so we get exactly 3 distinct moves
    top_moves_by_order = [None] * max_moves
    timeout_sec = 20
    deadline = time.time() + timeout_sec
    while True:
        line = katago_proc.stdout.readline()
        if not line:
            break
        line = line.rstrip("\r\n")
        if line == "":
            break
        if line.startswith("="):
            line = line[1:].strip()
            if not line:
                continue
        # Parse: info move E4 visits 487 winrate 0.480018 order 0 ...
        if line.startswith("info "):
            parts = line.split()
            move_gtp = None
            winrate = None
            order = None
            i = 0
            while i < len(parts) - 1:
                if parts[i] == "move":
                    move_gtp = parts[i + 1]
                    i += 2
                    continue
                if parts[i] == "winrate":
                    try:
                        v = float(parts[i + 1])
                        # kata-analyze: float in [0,1]; lz-analyze style: int in [0,10000]
                        if v > 1.0:
                            v = v / 10000.0
                        winrate = v
                        i += 2
                        continue
                    except (ValueError, IndexError):
                        pass
                if parts[i] == "order":
                    try:
                        order = int(parts[i + 1])
                        i += 2
                        continue
                    except (ValueError, IndexError):
                        pass
                i += 1
            if move_gtp and move_gtp.lower() != "pass" and winrate is not None and order is not None and 0 <= order < max_moves:
                try:
                    xy = gtp_to_coords(move_gtp, board_size=board_size)
                    if xy is not None:
                        x, y = xy
                        top_moves_by_order[order] = {"x": x, "y": y, "win_rate": round(winrate, 4)}
                        if all(top_moves_by_order[j] is not None for j in range(max_moves)):
                            break
                except Exception:
                    pass
        if all(top_moves_by_order[j] is not None for j in range(max_moves)):
            break
        if time.time() > deadline:
            break

    top_moves = [m for m in top_moves_by_order if m is not None]

    # KataGo often reports win rate from White's perspective. When we analyze for Black (human),
    # convert to current player's (Black's) win rate so "48% White" becomes "52% Black".
    if player == "B" and top_moves:
        for m in top_moves:
            m["win_rate"] = round(1.0 - m["win_rate"], 4)

    # Terminate analyze (send newline so KataGo stops searching)
    try:
        katago_proc.stdin.write("\n")
        katago_proc.stdin.flush()
    except Exception:
        pass
    # Consume rest of response until empty line
    while True:
        line = katago_proc.stdout.readline()
        if not line:
            break
        if line.rstrip("\r\n") == "":
            break

    return top_moves, None


async def katago_analyze_for_moves(moves, board_size: int, player: str = "B", max_moves: int = 3):
    """Get top moves and win rates for current position. Returns (list of dicts, error_str or None)."""
    async with katago_proc_lock:
        loop = asyncio.get_running_loop()
        result, err = await loop.run_in_executor(
            None, lambda: _katago_analyze_sync(moves, board_size, player, max_moves)
        )
        return result, err


# Game & AI Setup
game = GoEngine(board_size=9)
network = BetaGoNet(board_size=9)
network.lock = network_lock
encoder = FeatureEncoder(board_size=9)

# Try to load weights if exist
if os.path.exists(MODEL_PATH):
    try:
        network.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        print("[Server] Loaded initial model weights.")
    except Exception as e:
        print(f"[Server] Starting with fresh model (incompatible checkpoint: {e})")
network.eval()

# Serve frontend
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), '../frontend')
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.on_event("startup")
async def startup_event():
    print("[Server] Started. Monitoring training process...")
    if katago_available():
        print(f"[KataGo] Ready at {KATAGO_DIR}")
    else:
        print(f"[KataGo] NOT FOUND. Expecting: {KATAGO_EXE}")
    asyncio.create_task(monitor_model_updates())

async def monitor_model_updates():
    last_mtime = 0
    while True:
        await asyncio.sleep(5)
        try:
            if os.path.exists(MODEL_PATH):
                mtime = os.path.getmtime(MODEL_PATH)
                if mtime > last_mtime:
                    await asyncio.sleep(1)
                    with network_lock:
                        try:
                            state = torch.load(MODEL_PATH, map_location='cpu')
                            network.load_state_dict(state)
                            network.eval()
                            print("[Server] Model reloaded from disk.")
                            last_mtime = mtime
                        except Exception as e:
                            print(f"[Server] Model reload failed (retrying later): {e}")
        except Exception as e:
            print(f"[Server] Error checking model update: {e}")

@app.get("/")
async def get():
    return FileResponse(os.path.join(FRONTEND_DIR, 'index.html'))


@app.get("/katago")
async def get_katago():
    """Serve the KataGo play page."""
    return FileResponse(os.path.join(FRONTEND_DIR, 'katago.html'))


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in list(self.active_connections):
            try:
                await connection.send_text(message)
            except Exception:
                pass

manager = ConnectionManager()

# MCTS for play: no noise, stronger play with more sims
mcts_player = MCTS(network, encoder, num_simulations=400, add_noise=False, c_puct=2.0)

async def get_ai_move_async(game_state):
    """Run MCTS in thread pool to avoid blocking the event loop."""
    loop = asyncio.get_running_loop()

    def run_mcts():
        policy_probs, value = mcts_player.get_action_probs(game_state, temp=0.1)
        return policy_probs, value

    policy_probs, value = await loop.run_in_executor(executor, run_mcts)

    p = policy_probs.tolist()
    board_probs = p[:81]

    return board_probs, p, value


async def push_training_stats(websocket: WebSocket):
    try:
        while True:
            await asyncio.sleep(1.0)

            stats = {}
            running = False

            if os.path.exists(STATS_PATH):
                try:
                    with open(STATS_PATH, 'r') as f:
                        stats = json.load(f)
                        running = stats.get('running', False)
                except Exception:
                    pass

            if 'ts' not in stats:
                stats['ts'] = time.time()

            try:
                await websocket.send_json({
                    "type": "training_update",
                    "stats": stats,
                    "running": running
                })
            except Exception:
                break

    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"[Server] Error pushing stats: {e}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)

    # Send initial state with score
    async with game_lock:
        score = game.get_score()
        init_payload = {
            "type": "init",
            "board": game.board.tolist(),
            "turn": game.current_player,
            "score": score
        }
    await websocket.send_json(init_payload)

    stats_task = asyncio.create_task(push_training_stats(websocket))

    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            msg_type = msg.get("type")

            if msg_type == "move":
                x = msg.get("x")
                y = msg.get("y")
                if not isinstance(x, int) or not isinstance(y, int):
                    await websocket.send_json({"type": "error", "message": "Invalid move coordinates"})
                    continue

                try:
                    async with game_lock:
                        if game.is_valid_move(x, y):
                            game.step((x, y))
                            score = game.get_score()
                            await manager.broadcast(json.dumps({
                                "type": "update",
                                "board": game.board.tolist(),
                                "turn": game.current_player,
                                "last_move": [x, y],
                                "score": score,
                                "analysis": None
                            }))

                            # AI Auto-Reply when it's White's turn
                            if game.current_player == 2:
                                try:
                                    await manager.broadcast(json.dumps({"type": "ai_thinking", "on": True}))
                                except Exception:
                                    pass

                                board_probs, full_policy, ai_value = await get_ai_move_async(game)

                                # Prefer non-pass moves early in the game
                                probs = np.array(full_policy, dtype=np.float64)
                                min_moves_before_pass = 30
                                if game.move_count < min_moves_before_pass:
                                    probs[81] = 0.0
                                if probs.sum() <= 1e-12:
                                    probs = np.array(full_policy, dtype=np.float64)
                                best_idx = int(np.argmax(probs))

                                ai_moved = False
                                if best_idx == 81:
                                    # AI passes
                                    game.step(None)
                                else:
                                    ai_x, ai_y = best_idx // 9, best_idx % 9
                                    if game.is_valid_move(ai_x, ai_y):
                                        game.step((ai_x, ai_y))
                                        ai_moved = True

                                score = game.get_score()
                                await manager.broadcast(json.dumps({
                                    "type": "update",
                                    "board": game.board.tolist(),
                                    "turn": game.current_player,
                                    "last_move": None if not ai_moved else [ai_x, ai_y],
                                    "score": score,
                                    "analysis": {
                                        "policy": board_probs,
                                        "value": ai_value
                                    } if ai_moved else None
                                }))

                                try:
                                    await manager.broadcast(json.dumps({"type": "ai_thinking", "on": False}))
                                except Exception:
                                    pass

                                # Check game over after AI move
                                if game.passes >= 2:
                                    score = game.get_score()
                                    await manager.broadcast(json.dumps({
                                        "type": "game_over",
                                        "board": game.board.tolist(),
                                        "score": score,
                                        "winner": game.get_winner()
                                    }))

                except Exception as e:
                    print(f"[Server] Error processing move: {e}")
                    import traceback
                    traceback.print_exc()

            elif msg_type == "pass":
                async with game_lock:
                    game.step(None)
                    score = game.get_score()
                    await manager.broadcast(json.dumps({
                        "type": "update",
                        "board": game.board.tolist(),
                        "turn": game.current_player,
                        "last_move": None,
                        "score": score,
                        "analysis": None
                    }))
                    if game.passes >= 2:
                        await manager.broadcast(json.dumps({
                            "type": "game_over",
                            "board": game.board.tolist(),
                            "score": score,
                            "winner": game.get_winner()
                        }))

            elif msg_type == "reset" or msg_type == "start_game":
                async with game_lock:
                    game.reset()
                    score = game.get_score()
                    await manager.broadcast(json.dumps({
                        "type": "update",
                        "board": game.board.tolist(),
                        "turn": game.current_player,
                        "last_move": None,
                        "score": score,
                        "analysis": None
                    }))

            elif msg_type == "toggle_training":
                current_running = False
                if os.path.exists(STATS_PATH):
                    try:
                        with open(STATS_PATH, 'r') as f:
                            s = json.load(f)
                            current_running = s.get('running', False)
                    except:
                        pass

                new_command = 'stop' if current_running else 'start'

                try:
                    with open(CMD_PATH, 'w') as f:
                        json.dump({"command": new_command}, f)
                    print(f"[Server] Sent command: {new_command}")
                except Exception as e:
                    print(f"[Server] Failed to write command: {e}")

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    finally:
        stats_task.cancel()


@app.websocket("/ws_katago")
async def websocket_katago(websocket: WebSocket):
    """
    Separate WebSocket endpoint for playing against a strong KataGo engine.
    Uses a local KataGo process (if available) and its own GoEngine state.
    Human is Black, KataGo is White on 9x9 board.
    """
    await websocket.accept()

    if not katago_available():
        await websocket.send_json({
            "type": "error",
            "message": "KataGo engine is not fully configured on the server. "
                       "Please ensure katago.exe, default_gtp.cfg and model.bin.gz exist in the 'KataGo' folder."
        })
        await websocket.close()
        return

    board_size = 9
    game_k = GoEngine(board_size=board_size)
    moves = []  # list of (color: 'B' or 'W', move: (x, y) or None)
    ai_vs_ai_running = False

    async def run_ai_vs_ai():
        nonlocal ai_vs_ai_running, game_k, moves, board_size
        ai_vs_ai_running = True
        g = GoEngine(board_size=board_size)
        mv = []
        move_history = []  # list of {"color":"B"|"W", "x", "y"} or {"color", "pass": True}
        try:
            await websocket.send_json({"type": "ai_vs_ai_started", "board_size": board_size})
            while True:
                color = "B" if g.current_player == 1 else "W"
                await websocket.send_json({"type": "ai_thinking", "on": True})
                move, err = await katago_genmove_for_moves(mv, board_size, color=color)
                await websocket.send_json({"type": "ai_thinking", "on": False})
                if err is not None and move is None:
                    await websocket.send_json({"type": "error", "message": f"KataGo: {err}"})
                    break
                if move is None:
                    _, _, done, _ = g.step(None)
                    mv.append((color, None))
                    move_history.append({"color": color, "pass": True})
                    last_move = None
                else:
                    mx, my = move
                    if g.is_valid_move(mx, my):
                        _, _, done, _ = g.step((mx, my))
                        mv.append((color, (mx, my)))
                        move_history.append({"color": color, "x": mx, "y": my})
                        last_move = [mx, my]
                    else:
                        _, _, done, _ = g.step(None)
                        mv.append((color, None))
                        move_history.append({"color": color, "pass": True})
                        last_move = None
                score = g.get_score()
                await websocket.send_json({
                    "type": "update",
                    "board": g.board.tolist(),
                    "turn": g.current_player,
                    "last_move": last_move,
                    "score": score,
                    "move_count": len(mv),
                })
                # End game: double pass (done) or one side has no stones (complete capture, after some moves)
                if len(mv) > 4:
                    has_black = int(1 in g.board)
                    has_white = int(2 in g.board)
                    if not has_black or not has_white:
                        done = True
                # Safety: max moves to avoid endless fill (e.g. 9x9 ~200, 19x19 ~500)
                max_moves_limit = g.board_size * g.board_size * 2 + 50
                if len(mv) >= max_moves_limit:
                    done = True
                if done:
                    await websocket.send_json({
                        "type": "game_over",
                        "board": g.board.tolist(),
                        "score": score,
                        "winner": g.get_winner(),
                        "move_history": move_history,
                    })
                    break
        except Exception as e:
            await websocket.send_json({"type": "error", "message": str(e)})
        finally:
            ai_vs_ai_running = False

    # Send initial board state
    score = game_k.get_score()
    await websocket.send_json({
        "type": "init",
        "board": game_k.board.tolist(),
        "turn": game_k.current_player,
        "score": score,
    })

    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            msg_type = msg.get("type")

            if msg_type == "start_ai_vs_ai":
                if ai_vs_ai_running:
                    await websocket.send_json({"type": "error", "message": "AI对局已在进行中"})
                else:
                    asyncio.create_task(run_ai_vs_ai())
                continue

            if ai_vs_ai_running and msg_type in ("move", "pass"):
                continue

            if msg_type == "move":
                x = msg.get("x")
                y = msg.get("y")
                if not isinstance(x, int) or not isinstance(y, int):
                    await websocket.send_json({"type": "error", "message": "Invalid move coordinates"})
                    continue

                # Human is always Black in this mode
                if not game_k.is_valid_move(x, y):
                    await websocket.send_json({"type": "error", "message": "Illegal move"})
                    continue

                # Apply human move
                _, _, done, _ = game_k.step((x, y))
                moves.append(("B", (x, y)))
                score = game_k.get_score()
                await websocket.send_json({
                    "type": "update",
                    "board": game_k.board.tolist(),
                    "turn": game_k.current_player,
                    "last_move": [x, y],
                    "score": score,
                    "analysis": None,
                })

                if done:
                    await websocket.send_json({
                        "type": "game_over",
                        "board": game_k.board.tolist(),
                        "score": score,
                        "winner": game_k.get_winner(),
                    })
                    continue

                # Ask KataGo (White) to respond
                await websocket.send_json({"type": "ai_thinking", "on": True})
                ai_move, err = await katago_genmove_for_moves(moves, game_k.board_size)
                if err is not None and ai_move is None:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"KataGo error: {err}"
                    })
                    await websocket.send_json({"type": "ai_thinking", "on": False})
                    continue

                if ai_move is None:
                    # KataGo chooses to pass
                    _, _, done2, _ = game_k.step(None)
                    moves.append(("W", None))
                    last_move = None
                else:
                    ax, ay = ai_move
                    if game_k.is_valid_move(ax, ay):
                        _, _, done2, _ = game_k.step((ax, ay))
                        moves.append(("W", (ax, ay)))
                        last_move = [ax, ay]
                    else:
                        # Fallback to pass if move is illegal for some reason
                        _, _, done2, _ = game_k.step(None)
                        moves.append(("W", None))
                        last_move = None

                score = game_k.get_score()
                await websocket.send_json({
                    "type": "update",
                    "board": game_k.board.tolist(),
                    "turn": game_k.current_player,
                    "last_move": last_move,
                    "score": score,
                    "analysis": None,
                })
                await websocket.send_json({"type": "ai_thinking", "on": False})

                if done2:
                    await websocket.send_json({
                        "type": "game_over",
                        "board": game_k.board.tolist(),
                        "score": score,
                        "winner": game_k.get_winner(),
                    })

            elif msg_type == "pass":
                # Human pass, then KataGo replies
                _, _, done, _ = game_k.step(None)
                moves.append(("B", None))
                score = game_k.get_score()
                await websocket.send_json({
                    "type": "update",
                    "board": game_k.board.tolist(),
                    "turn": game_k.current_player,
                    "last_move": None,
                    "score": score,
                    "analysis": None,
                })

                if done:
                    await websocket.send_json({
                        "type": "game_over",
                        "board": game_k.board.tolist(),
                        "score": score,
                        "winner": game_k.get_winner(),
                    })
                    continue

                # KataGo move (White)
                await websocket.send_json({"type": "ai_thinking", "on": True})
                ai_move, err = await katago_genmove_for_moves(moves, game_k.board_size)
                if err is not None and ai_move is None:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"KataGo error: {err}"
                    })
                    await websocket.send_json({"type": "ai_thinking", "on": False})
                    continue

                if ai_move is None:
                    _, _, done2, _ = game_k.step(None)
                    moves.append(("W", None))
                    last_move = None
                else:
                    ax, ay = ai_move
                    if game_k.is_valid_move(ax, ay):
                        _, _, done2, _ = game_k.step((ax, ay))
                        moves.append(("W", (ax, ay)))
                        last_move = [ax, ay]
                    else:
                        _, _, done2, _ = game_k.step(None)
                        moves.append(("W", None))
                        last_move = None

                score = game_k.get_score()
                await websocket.send_json({
                    "type": "update",
                    "board": game_k.board.tolist(),
                    "turn": game_k.current_player,
                    "last_move": last_move,
                    "score": score,
                    "analysis": None,
                })
                await websocket.send_json({"type": "ai_thinking", "on": False})

                if done2:
                    await websocket.send_json({
                        "type": "game_over",
                        "board": game_k.board.tolist(),
                        "score": score,
                        "winner": game_k.get_winner(),
                    })

            elif msg_type == "set_board_size":
                size = msg.get("size")
                if size not in (9, 13, 19):
                    continue
                board_size = size
                game_k = GoEngine(board_size=board_size)
                moves = []
                score = game_k.get_score()
                await websocket.send_json({
                    "type": "update",
                    "board": game_k.board.tolist(),
                    "turn": game_k.current_player,
                    "last_move": None,
                    "score": score,
                    "analysis": None,
                })

            elif msg_type == "get_analysis":
                # turn 1 = Black (B), 2 = White (W)
                side = "B" if game_k.current_player == 1 else "W"
                top_moves, err = await katago_analyze_for_moves(moves, board_size, player=side, max_moves=3)
                if err is not None:
                    await websocket.send_json({"type": "error", "message": f"Analysis: {err}"})
                else:
                    await websocket.send_json({"type": "analysis", "top_moves": top_moves or []})

            elif msg_type == "reset" or msg_type == "start_game":
                game_k = GoEngine(board_size=board_size)
                moves = []
                score = game_k.get_score()
                await websocket.send_json({
                    "type": "update",
                    "board": game_k.board.tolist(),
                    "turn": game_k.current_player,
                    "last_move": None,
                    "score": score,
                    "analysis": None,
                })

    except WebSocketDisconnect:
        return

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8050)
