import sys
import os
import time
import json
import torch
import threading

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
    try:
        if os.path.exists(src):
            os.remove(src)
    except:
        pass
    return False

def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    print("[START] Training Worker Process Started (PID: {})".format(os.getpid()))

    network_lock = threading.Lock()

    # Fresh network (new architecture won't match old weights)
    network = BetaGoNet(board_size=9)
    network.lock = network_lock

    # Try load existing model (only if compatible)
    if os.path.exists(MODEL_PATH):
        try:
            state = torch.load(MODEL_PATH, map_location='cpu')
            network.load_state_dict(state)
            print("[INFO] Loaded existing model.")
        except Exception as e:
            print(f"[INFO] Starting fresh (incompatible checkpoint: {e})")

    encoder = FeatureEncoder(board_size=9)

    trainer = Trainer(
        network,
        buffer_size=50000,
        batch_size=128,
        lr=0.001,
        min_buffer_size=512,
        value_loss_weight=1.0,
    )
    trainer.set_network_lock(network_lock)

    # Self-play with 100 simulations for better quality data
    self_play_worker = SelfPlayWorker(network, encoder, trainer, num_simulations=100)

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
                        is_paused = True
                    elif cmd_data.get('command') == 'start':
                        is_paused = False
                except Exception:
                    pass

            if is_paused:
                time.sleep(1.0)
                stats = {
                    'step': trainer.training_stats.get('step', 0),
                    'policy_loss': 0,
                    'value_loss': 0,
                    'total_loss': 0,
                    'win_rate': 0,
                    'games': self_play_worker.game_count,
                    'buffer': len(trainer.buffer),
                    'min_buffer': trainer.min_buffer_size,
                    'ts': time.time(),
                    'running': False,
                    'status': 'Paused'
                }
                temp_stats_path = STATS_PATH + ".tmp"
                with open(temp_stats_path, 'w') as f:
                    json.dump(stats, f)
                safe_replace(temp_stats_path, STATS_PATH)
                continue

            result = trainer.train_step()

            if result:
                stats = dict(trainer.training_stats)
                stats['games'] = self_play_worker.game_count
                stats['buffer'] = len(trainer.buffer)
                stats['ts'] = time.time()
                stats['running'] = True

                temp_stats_path = STATS_PATH + ".tmp"
                with open(temp_stats_path, 'w') as f:
                    json.dump(stats, f)
                safe_replace(temp_stats_path, STATS_PATH)

                # Save model every 15 seconds
                if time.time() - last_save_time > 15:
                    temp_model_path = MODEL_PATH + ".tmp"
                    torch.save(network.state_dict(), temp_model_path)
                    safe_replace(temp_model_path, MODEL_PATH)
                    last_save_time = time.time()
                    print("[Checkpoint] Model saved.")

            else:
                time.sleep(0.5)
                stats = {
                    'step': trainer.training_stats['step'],
                    'policy_loss': 0,
                    'value_loss': 0,
                    'total_loss': 0,
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
