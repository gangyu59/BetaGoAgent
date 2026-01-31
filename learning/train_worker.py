import sys
import os
import time
import json
import torch
import threading
import signal

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from core.go_engine import GoEngine
from brain.network import BetaGoNet
from brain.feature_encoder import FeatureEncoder
from learning.trainer import Trainer
from learning.self_play import SelfPlayWorker

# Paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../model_latest.pth')
STATS_PATH = os.path.join(os.path.dirname(__file__), '../training_stats.json')
CMD_PATH = os.path.join(os.path.dirname(__file__), '../training_cmd.json')

def safe_replace(src, dst, retries=5, delay=0.1):
    """
    Safely replace a file on Windows, retrying if PermissionError occurs
    (common if another process like the web server is reading the file).
    """
    for i in range(retries):
        try:
            if os.path.exists(dst):
                os.remove(dst)
            os.rename(src, dst)
            return True
        except PermissionError:
            time.sleep(delay)
        except Exception as e:
            print(f"[WARN] File replace error: {e}")
            break
    
    # Fallback: if replace fails, try to just write (not atomic but better than crash)
    # Or just ignore this update
    try:
        if os.path.exists(src):
            os.remove(src)
    except:
        pass
    return False

def main():
    # Force UTF-8 for stdout to avoid encoding errors on Windows
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    print("[START] Training Worker Process Started (PID: {})".format(os.getpid()))
    
    # Setup
    # Note: We do NOT need a lock for network in this process if it's single-threaded for training steps,
    # BUT SelfPlayWorker runs in a thread. So we DO need a lock.
    network_lock = threading.Lock()
    
    network = BetaGoNet(board_size=9)
    network.lock = network_lock
    
    # Try load existing model
    if os.path.exists(MODEL_PATH):
        try:
            network.load_state_dict(torch.load(MODEL_PATH))
            print("[INFO] Loaded existing model.")
        except Exception as e:
            print(f"[WARN] Could not load model: {e}")
            
    encoder = FeatureEncoder(board_size=9)
    
    trainer = Trainer(network, min_buffer_size=64) # Low buffer for fast startup feedback
    trainer.set_network_lock(network_lock)
    
    # Self-Play Worker (Thread inside this process)
    # Use fewer simulations for self-play to generate data faster (AlphaZero Zero used 800, but we need speed now)
    self_play_worker = SelfPlayWorker(network, encoder, trainer, num_simulations=50) 
    
    trainer.start()
    self_play_worker.start()
    
    print("[INFO] Training Loop Active...")
    
    last_save_time = time.time()
    is_paused = False
    
    try:
        while True:
            # Check for commands
            if os.path.exists(CMD_PATH):
                try:
                    with open(CMD_PATH, 'r') as f:
                        cmd_data = json.load(f)
                    
                    if cmd_data.get('command') == 'stop':
                        # We treat 'stop' as pause in this context, or we could exit.
                        # User usually expects 'stop' to pause training so they can resume.
                        is_paused = True
                    elif cmd_data.get('command') == 'start':
                        is_paused = False
                        
                    # Optional: remove command file after processing? 
                    # Better to keep it as state, or just read it. 
                    # If we delete it, server needs to write it again.
                    # Let's keep it simple: Server writes, we read.
                except Exception:
                    pass

            if is_paused:
                time.sleep(1.0)
                # Update stats to show paused
                stats = {
                    'step': trainer.training_stats.get('step', 0),
                    'policy_loss': 0,
                    'value_loss': 0,
                    'win_rate': 0,
                    'games': self_play_worker.game_count,
                    'buffer': len(trainer.buffer),
                    'min_buffer': trainer.min_buffer_size,
                    'ts': time.time(),
                    'running': False,
                    'status': 'Paused'
                }
                # Atomic write
                temp_stats_path = STATS_PATH + ".tmp"
                with open(temp_stats_path, 'w') as f:
                    json.dump(stats, f)
                safe_replace(temp_stats_path, STATS_PATH)
                continue

            # Train step
            result = trainer.train_step()
            
            if result:
                # Save stats to file for Server to read
                stats = dict(trainer.training_stats)
                stats['games'] = self_play_worker.game_count
                stats['buffer'] = len(trainer.buffer)
                stats['ts'] = time.time()
                stats['running'] = True
                
                # Atomic write to avoid read errors
                temp_stats_path = STATS_PATH + ".tmp"
                with open(temp_stats_path, 'w') as f:
                    json.dump(stats, f)
                safe_replace(temp_stats_path, STATS_PATH)
                
                # Periodically save model (every 10 seconds or so)
                if time.time() - last_save_time > 10:
                    temp_model_path = MODEL_PATH + ".tmp"
                    torch.save(network.state_dict(), temp_model_path)
                    safe_replace(temp_model_path, MODEL_PATH)
                    last_save_time = time.time()
                    # print("💾 Model Checkpoint Saved")
                    
            else:
                # Idle wait
                time.sleep(1.0)
                # Still update stats to show we are alive but waiting
                stats = {
                    'step': trainer.training_stats['step'],
                    'policy_loss': 0,
                    'value_loss': 0,
                    'win_rate': 0,
                    'games': self_play_worker.game_count,
                    'buffer': len(trainer.buffer),
                    'min_buffer': trainer.min_buffer_size,
                    'ts': time.time(),
                    'running': True,
                    'status': 'Waiting for data...'
                }
                temp_stats_path = STATS_PATH + ".tmp"
                with open(temp_stats_path, 'w') as f:
                    json.dump(stats, f)
                safe_replace(temp_stats_path, STATS_PATH)

    except KeyboardInterrupt:
        print("[STOP] Training Worker Stopping...")
        trainer.stop()
        self_play_worker.stop()

if __name__ == "__main__":
    main()
