
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import json
import asyncio
import sys
import os
import random
import numpy as np
import torch

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from core.go_engine import GoEngine
from brain.network import BetaGoNet
from brain.feature_encoder import FeatureEncoder
# from learning.trainer import Trainer      <-- Removed
# from learning.self_play import SelfPlayWorker <-- Removed
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

# --- Thread Pool for CPU-bound MCTS ---
# We use a thread pool to offload the heavy MCTS 'get_action_probs' calls
# so they don't block the main asyncio event loop (which handles WebSockets).
executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

# --- Global Locks ---
network_lock = threading.Lock()

# --- Game & AI Setup ---
game = GoEngine(board_size=9)
network = BetaGoNet(board_size=9)
# Attach lock to network for MCTS usage
network.lock = network_lock
encoder = FeatureEncoder(board_size=9)

# Try to load weights if exist
if os.path.exists(MODEL_PATH):
    try:
        network.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        print("✅ Loaded initial model weights.")
    except Exception as e:
        print(f"⚠️ Failed to load initial model: {e}")
network.eval()

# Serve frontend
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), '../frontend')
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.on_event("startup")
async def startup_event():
    print("🚀 Server Started. Monitoring training process...")
    # Start background tasks for model reloading and stats pushing
    asyncio.create_task(monitor_model_updates())

async def monitor_model_updates():
    last_mtime = 0
    while True:
        await asyncio.sleep(5)
        try:
            if os.path.exists(MODEL_PATH):
                mtime = os.path.getmtime(MODEL_PATH)
                if mtime > last_mtime:
                    # Wait a bit to ensure write is complete
                    await asyncio.sleep(1) 
                    with network_lock:
                        try:
                            # Load to temp first to verify
                            state = torch.load(MODEL_PATH, map_location='cpu')
                            network.load_state_dict(state)
                            print("🔄 Model reloaded from disk.")
                            last_mtime = mtime
                        except Exception as e:
                            print(f"⚠️ Model reload failed (retrying later): {e}")
        except Exception as e:
            print(f"Error checking model update: {e}")

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
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

mcts_player = MCTS(network, encoder, num_simulations=800)

async def get_ai_analysis_async(game_state):
    """
    Async wrapper for MCTS analysis to prevent blocking the event loop.
    """
    loop = asyncio.get_running_loop()
    
    # Define the blocking function
    def run_mcts():
        # Lower temp for stronger play
        policy_probs, value = mcts_player.get_action_probs(game_state, temp=0.2)
        return policy_probs, value
        
    # Run in thread pool
    policy_probs, value = await loop.run_in_executor(executor, run_mcts)
    
    p = policy_probs.tolist()
    # value is already float
    
    # Remove pass prob (last element) for heatmap visualization on board
    board_probs = p[:81]
    
    return board_probs, value

def get_ai_analysis(game_state):
    """
    Legacy synchronous version (avoid using in async endpoints)
    """
    policy_probs, value = mcts_player.get_action_probs(game_state, temp=0.2)
    p = policy_probs.tolist()
    board_probs = p[:81]
    return board_probs, value

async def push_training_stats(websocket: WebSocket):
    try:
        while True:
            await asyncio.sleep(1.0)
            
            stats = {}
            running = False
            
            # Read from file instead of memory objects
            if os.path.exists(STATS_PATH):
                try:
                    with open(STATS_PATH, 'r') as f:
                        stats = json.load(f)
                        running = stats.get('running', False)
                except Exception:
                    pass
            
            # Add timestamp if missing
            if 'ts' not in stats:
                stats['ts'] = time.time()
            
            try:
                await websocket.send_json({
                    "type": "training_update",
                    "stats": stats,
                    "running": running
                })
            except Exception:
                # If send fails, break the loop to allow cleanup
                break
                
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"Error pushing stats: {e}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    
    # Send initial state
    await websocket.send_json({
        "type": "init",
        "board": game.board.tolist(),
        "turn": game.current_player
    })
    
    # Use a weak reference or ensure task is cancelled properly
    stats_task = asyncio.create_task(push_training_stats(websocket))
    
    try:
        while True:
            # Wait for message with timeout to allow ping/pong or disconnect detection
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
                        # Immediately broadcast user's move without waiting for analysis
                        await manager.broadcast(json.dumps({
                            "type": "update",
                            "board": game.board.tolist(),
                            "turn": game.current_player,
                            "last_move": [x, y],
                            "analysis": None
                        }))
                        
                        # --- AI Auto-Reply Logic ---
                        if game.current_player == 2: # White (AI)
                            try:
                                await manager.broadcast(json.dumps({"type": "ai_thinking", "on": True}))
                            except Exception:
                                pass
                            # Small delay for UX
                            await asyncio.sleep(0.5)
                            
                            # Get AI prediction again (state changed)
                            ai_policy, ai_value = get_ai_analysis(game)
                            
                            # Log top moves for debugging
                            try:
                                top_indices = np.argsort(ai_policy)[-5:][::-1]
                                top_moves_info = []
                                for idx in top_indices:
                                    prob = ai_policy[idx]
                                    if idx == 81:
                                        move_str = "PASS"
                                    else:
                                        move_str = f"({idx // 9}, {idx % 9})"
                                    top_moves_info.append(f"{move_str}:{prob:.3f}")
                                print(f"🤖 AI Thoughts: {', '.join(top_moves_info)} | Value: {ai_value:.3f}")
                            except Exception as e:
                                print(f"Debug log error: {e}")

                            # Sample from MCTS distribution
                            # MCTS already masks invalid moves, so we trust the policy
                            ai_moved = False
                            
                            # Renormalize just in case
                            probs = np.array(ai_policy)
                            s = np.sum(probs)
                            if s > 1e-8:
                                probs /= s
                            else:
                                probs = np.ones_like(probs) / len(probs)
                                
                            idx = np.random.choice(len(probs), p=probs)
                            
                            if idx == 81: # PASS
                                ai_moved = False # Trigger pass handling below
                            else:
                                ai_x, ai_y = idx // 9, idx % 9
                                if game.is_valid_move(ai_x, ai_y):
                                    game.step((ai_x, ai_y))
                                    ai_moved = True
                                    
                                    # Broadcast AI move
                                    final_policy, final_value = get_ai_analysis(game)
                                    await manager.broadcast(json.dumps({
                                        "type": "update",
                                        "board": game.board.tolist(),
                                        "turn": game.current_player,
                                        "last_move": [ai_x, ai_y],
                                        "analysis": {
                                            "policy": final_policy,
                                            "value": final_value
                                        }
                                    }))
                                    try:
                                        await manager.broadcast(json.dumps({"type": "ai_thinking", "on": False}))
                                    except Exception:
                                        pass
                                else:
                                    print(f"⚠️ AI attempted invalid move ({ai_x}, {ai_y}) with prob {probs[idx]:.3f}")
                                    ai_moved = False

                            if not ai_moved:
                                print("🤖 AI decides to Pass.")
                                game.step(None)
                                await manager.broadcast(json.dumps({
                                    "type": "update",
                                    "board": game.board.tolist(),
                                    "turn": game.current_player,
                                    "last_move": None,
                                    "analysis": None
                                }))
                                try:
                                    await manager.broadcast(json.dumps({"type": "ai_thinking", "on": False}))
                                except Exception:
                                    pass
                                
                    else:
                        # Invalid move
                        pass 
                except Exception as e:
                    print(f"Error processing move: {e}")
            
            elif msg["type"] == "reset":
                game.reset()
                await manager.broadcast(json.dumps({
                    "type": "update",
                    "board": game.board.tolist(),
                    "turn": game.current_player,
                    "last_move": None,
                    "analysis": None
                }))

            elif msg["type"] == "start_game":
                game.reset()
                # Immediate update without analysis for responsiveness
                await manager.broadcast(json.dumps({
                    "type": "update",
                    "board": game.board.tolist(),
                    "turn": game.current_player,
                    "last_move": None,
                    "analysis": None
                }))
                # Then compute analysis and send again
                # Use async version
                policy, value = await get_ai_analysis_async(game)
                await manager.broadcast(json.dumps({
                    "type": "update",
                    "board": game.board.tolist(),
                    "turn": game.current_player,
                    "last_move": None,
                    "analysis": {
                        "policy": policy,
                        "value": value
                    }
                }))

            elif msg["type"] == "toggle_training":
                # Write command to file
                cmd = {}
                current_running = False
                if os.path.exists(STATS_PATH):
                    try:
                        with open(STATS_PATH, 'r') as f:
                            s = json.load(f)
                            current_running = s.get('running', False)
                    except:
                        pass
                
                # If running, we want to stop (pause). If not running, we want to start.
                new_command = 'stop' if current_running else 'start'
                
                try:
                    with open(CMD_PATH, 'w') as f:
                        json.dump({"command": new_command}, f)
                    print(f"📝 Sent command: {new_command}")
                except Exception as e:
                    print(f"❌ Failed to write command: {e}")
                
                # Optimistic response to UI?
                # The UI waits for stats update, but we can send a quick ack if needed.
                pass

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    finally:
        stats_task.cancel()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8050)

