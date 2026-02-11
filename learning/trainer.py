import torch
import torch.optim as optim
import time
import numpy as np
from collections import deque
import threading

class Trainer:
    def __init__(self, network, buffer_size=50000, batch_size=128, lr=0.001,
                 min_buffer_size=512, sleep_throttle=0.01, value_loss_weight=1.0):
        self.network = network
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10000, gamma=0.5)
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.running = False
        self.training_stats = {
            'step': 0,
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'total_loss': 0.0,
            'win_rate': 0.5,
            'lr': lr,
        }
        self.lock = threading.Lock()
        self.network_lock = None
        self.min_buffer_size = min_buffer_size
        self.sleep_throttle = sleep_throttle
        self.value_loss_weight = value_loss_weight

    def set_network_lock(self, lock):
        self.network_lock = lock

    def save_checkpoint(self, path):
        torch.save(self.network.state_dict(), path)

    def load_checkpoint(self, path):
        if hasattr(self.network, 'load_state_dict'):
            try:
                self.network.load_state_dict(torch.load(path, map_location='cpu'))
                print(f"[Trainer] Loaded checkpoint from {path}")
                return True
            except Exception as e:
                print(f"[Trainer] Failed to load checkpoint: {e}")
                return False
        return False

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def add_data(self, state, policy_target, value_target):
        with self.lock:
            self.buffer.append((state, policy_target, value_target))

    def train_step(self):
        if not self.running:
            return None

        with self.lock:
            current_buffer_size = len(self.buffer)

        if current_buffer_size < self.min_buffer_size:
            if self.training_stats['step'] % 50 == 0:
                print(f"[Trainer] Waiting for data... ({current_buffer_size}/{self.min_buffer_size})")
            return None

        time.sleep(self.sleep_throttle)

        with self.lock:
            batch = [self.buffer[i] for i in np.random.choice(
                len(self.buffer), self.batch_size, replace=False)]

        states = torch.FloatTensor(np.array([b[0] for b in batch]))
        policy_targets = torch.FloatTensor(np.array([b[1] for b in batch]))
        value_targets = torch.FloatTensor(np.array([b[2] for b in batch])).view(-1, 1)

        if self.network_lock:
            self.network_lock.acquire()
        try:
            self.network.train()
            p_logits, v_pred = self.network(states)

            # Policy loss: cross-entropy with soft targets
            p_log_probs = torch.log_softmax(p_logits, dim=1)
            policy_loss = -torch.mean(torch.sum(policy_targets * p_log_probs, dim=1))

            # Value loss: MSE
            value_loss = torch.nn.functional.mse_loss(v_pred, value_targets)

            # Accuracy
            with torch.no_grad():
                pred_actions = torch.argmax(p_logits, dim=1)
                target_actions = torch.argmax(policy_targets, dim=1)
                accuracy = (pred_actions == target_actions).float().mean().item()

            loss = policy_loss + self.value_loss_weight * value_loss

            self.optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
        finally:
            if self.network_lock:
                self.network_lock.release()

        self.training_stats['step'] += 1
        self.training_stats['policy_loss'] = policy_loss.item()
        self.training_stats['value_loss'] = value_loss.item()
        self.training_stats['total_loss'] = loss.item()
        self.training_stats['win_rate'] = accuracy
        self.training_stats['lr'] = self.optimizer.param_groups[0]['lr']

        if self.training_stats['step'] % 50 == 0:
            print(f"[Train] Step {self.training_stats['step']}: "
                  f"P={policy_loss.item():.4f} V={value_loss.item():.4f} "
                  f"Acc={accuracy:.3f} LR={self.training_stats['lr']:.6f} "
                  f"Buf={current_buffer_size}")

        return self.training_stats
