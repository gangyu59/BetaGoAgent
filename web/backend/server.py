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

# Thread Pool for CPU-bound MCTS
executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

# Global Locks
network_lock = threading.Lock()

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
    score = game.get_score()
    await websocket.send_json({
        "type": "init",
        "board": game.board.tolist(),
        "turn": game.current_player,
        "score": score
    })

    stats_task = asyncio.create_task(push_training_stats(websocket))

    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            msg = json.loads(data)

            if msg["type"] == "move":
                x, y = msg["x"], msg["y"]
                try:
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

                            # Pick best move (greedy)
                            best_idx = int(np.argmax(full_policy))
                            ai_moved = False

                            if best_idx == 81:
                                # AI passes
                                pass
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
                                        "last_move": [ai_x, ai_y],
                                        "score": score,
                                        "analysis": {
                                            "policy": board_probs,
                                            "value": ai_value
                                        }
                                    }))

                            if not ai_moved:
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

            elif msg["type"] == "pass":
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

            elif msg["type"] == "reset" or msg["type"] == "start_game":
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

            elif msg["type"] == "toggle_training":
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

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8050)
