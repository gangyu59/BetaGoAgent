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

    # Trainer: for 9x9 + CPU，更注重训练步数速度
    trainer = Trainer(
        network,
        buffer_size=50000,
        batch_size=128,
        lr=0.001,
        # 较小的 min_buffer_size 让训练更早、更频繁地进行
        min_buffer_size=512,
        value_loss_weight=0.5,
    )
    trainer.set_network_lock(network_lock)

    # Self-play: 用更少的 MCTS 模拟数来换取游戏速度（原来是 100）
    # 这会显著提升每小时自对弈局数和训练 step 数
    self_play_worker = SelfPlayWorker(network, encoder, trainer, num_simulations=32)

    trainer.start()
    self_play_worker.start()

    # Clear any stale command file from上一次运行，避免残留的 stop 干扰本次会话。
    if os.path.exists(CMD_PATH):
        try:
            os.remove(CMD_PATH)
            print("[INFO] Cleared stale training_cmd.json.")
        except Exception:
            pass

    def write_stats(stats_dict):
        temp_path = STATS_PATH + f".{os.getpid()}.tmp"
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(stats_dict, f)
            if os.path.exists(STATS_PATH):
                os.remove(STATS_PATH)
            os.rename(temp_path, STATS_PATH)
        except Exception as e:
            print(f"[WARN] Failed to write training_stats: {e}")

    # 初始状态：训练默认「未开始」，等待用户在前端点击 Start Training。
    write_stats({
        'step': 0,
        'policy_loss': 0.0,
        'value_loss': 0.0,
        'total_loss': 0.0,
        'win_rate': 0.0,
        'games': 0,
        'buffer': 0,
        'min_buffer': trainer.min_buffer_size,
        'ts': time.time(),
        'running': False,
        'status': 'Waiting for start...',
        'lr': trainer.training_stats['lr'],
    })

    print("[INFO] Training Loop Active...")

    last_save_time = time.time()
    # 默认暂停，等待前端通过 training_cmd.json 写入 {"command": "start"} 后再真正开始训练。
    is_paused = True
    last_game_count = 0
    steps_this_cycle = 0

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
                temp_stats_path = STATS_PATH + f".{os.getpid()}.tmp"
                try:
                    with open(temp_stats_path, 'w', encoding='utf-8') as f:
                        json.dump(stats, f)
                except PermissionError as e:
                    print(f"[WARN] Failed to open temp stats file (paused): {e}")
                else:
                    if not safe_replace(temp_stats_path, STATS_PATH):
                        print("[WARN] Failed to update training_stats.json (paused).")
                continue

            # --- Throttle: limit training to ~1 epoch per new game ---
            current_games = self_play_worker.game_count
            if current_games > last_game_count:
                last_game_count = current_games
                steps_this_cycle = 0  # New data arrived, allow more training

            buffer_len = len(trainer.buffer)
            max_steps_per_cycle = max(1, buffer_len // trainer.batch_size)

            if steps_this_cycle >= max_steps_per_cycle:
                # Already trained 1 epoch on current data, wait for more
                time.sleep(1.0)
                stats = {
                    'step': trainer.training_stats['step'],
                    'policy_loss': trainer.training_stats.get('policy_loss', 0),
                    'value_loss': trainer.training_stats.get('value_loss', 0),
                    'total_loss': trainer.training_stats.get('total_loss', 0),
                    'win_rate': trainer.training_stats.get('win_rate', 0),
                    'games': self_play_worker.game_count,
                    'buffer': buffer_len,
                    'min_buffer': trainer.min_buffer_size,
                    'ts': time.time(),
                    'running': True,
                    'status': 'Waiting for new game data...'
                }
                temp_stats_path = STATS_PATH + f".{os.getpid()}.tmp"
                try:
                    with open(temp_stats_path, 'w', encoding='utf-8') as f:
                        json.dump(stats, f)
                except PermissionError as e:
                    print(f"[WARN] Failed to open temp stats file (waiting for new games): {e}")
                else:
                    if not safe_replace(temp_stats_path, STATS_PATH):
                        print("[WARN] Failed to update training_stats.json (waiting for new games).")
                continue

            result = trainer.train_step()

            if result:
                steps_this_cycle += 1
                stats = dict(trainer.training_stats)
                stats['games'] = self_play_worker.game_count
                stats['buffer'] = len(trainer.buffer)
                stats['ts'] = time.time()
                stats['running'] = True

                temp_stats_path = STATS_PATH + f".{os.getpid()}.tmp"
                try:
                    with open(temp_stats_path, 'w', encoding='utf-8') as f:
                        json.dump(stats, f)
                except PermissionError as e:
                    print(f"[WARN] Failed to open temp stats file (training): {e}")
                else:
                    if not safe_replace(temp_stats_path, STATS_PATH):
                        print("[WARN] Failed to update training_stats.json (training).")

                # Save model every 15 seconds
                if time.time() - last_save_time > 15:
                    temp_model_path = MODEL_PATH + f".{os.getpid()}.tmp"
                    torch.save(network.state_dict(), temp_model_path)
                    if not safe_replace(temp_model_path, MODEL_PATH):
                        print("[WARN] Failed to safely replace model_latest.pth")
                    else:
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
                temp_stats_path = STATS_PATH + f".{os.getpid()}.tmp"
                try:
                    with open(temp_stats_path, 'w', encoding='utf-8') as f:
                        json.dump(stats, f)
                except PermissionError as e:
                    print(f"[WARN] Failed to open temp stats file (waiting for data): {e}")
                else:
                    if not safe_replace(temp_stats_path, STATS_PATH):
                        print("[WARN] Failed to update training_stats.json (waiting for data).")

    except KeyboardInterrupt:
        print("[STOP] Training Worker Stopping...")
        trainer.stop()
        self_play_worker.stop()

if __name__ == "__main__":
    main()
