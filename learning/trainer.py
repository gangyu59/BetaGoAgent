import torch
import torch.optim as optim
import time
import numpy as np
from collections import deque
import threading

class Trainer:
    def __init__(self, network, buffer_size=20000, batch_size=32, lr=0.0001, min_buffer_size=1000, sleep_throttle=0.05, value_loss_weight=0.5):
        self.network = network
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, weight_decay=1e-4)
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.running = False
        self.training_stats = {
            'step': 0,
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'win_rate': 0.5
        }
        self.lock = threading.Lock()
        self.network_lock = None
        self.min_buffer_size = min_buffer_size
        self.sleep_throttle = sleep_throttle
        self.value_loss_weight = value_loss_weight

    def set_network_lock(self, lock):
        self.network_lock = lock

    def save_checkpoint(self, path):
        """Save model weights and training stats"""
        # We save stats as a separate JSON or inside the PTH? 
        # For simplicity, just save weights to PTH. Stats can be JSON.
        torch.save(self.network.state_dict(), path)

    def load_checkpoint(self, path):
        if hasattr(self.network, 'load_state_dict'):
            try:
                self.network.load_state_dict(torch.load(path, map_location='cpu'))
                print(f"✅ Loaded checkpoint from {path}")
                return True
            except Exception as e:
                print(f"⚠️ Failed to load checkpoint: {e}")
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

        # Wait for enough data
        with self.lock:
            current_buffer_size = len(self.buffer)
        
        if current_buffer_size < self.min_buffer_size:
            if self.training_stats['step'] % 100 == 0:
                 print(f"⏳ Trainer waiting for data... ({current_buffer_size}/{self.min_buffer_size})")
            return None
        
        time.sleep(self.sleep_throttle)

        with self.lock:
            # Sample batch
            # Use bigger batch size for stability if possible, but our batch_size is fixed at init.
            # But we can change it dynamically if needed.
            pass

        if len(self.buffer) > self.min_buffer_size:
             # If we have plenty of data, increase batch size dynamically? No, keep it simple.
             pass

        with self.lock:
            batch = [self.buffer[i] for i in np.random.choice(len(self.buffer), self.batch_size, replace=False)]
        
        states = torch.FloatTensor(np.array([b[0] for b in batch]))
        policy_targets = torch.FloatTensor(np.array([b[1] for b in batch]))
        value_targets = torch.FloatTensor(np.array([b[2] for b in batch])).view(-1, 1)

        # Forward
        if self.network_lock:
            self.network_lock.acquire()
        try:
            self.network.train()
            p_logits, v_pred = self.network(states)
            
            # Loss
            p_log_probs = torch.log_softmax(p_logits, dim=1)
            policy_loss = -torch.mean(torch.sum(policy_targets * p_log_probs, dim=1))
            value_loss = torch.nn.functional.mse_loss(v_pred, value_targets)
            
            # Debugging Value Loss
            if self.training_stats['step'] % 10 == 0:
                print(f"Step {self.training_stats['step']}: Value Target {value_targets[0].item():.4f}, Pred {v_pred[0].item():.4f}, Loss {value_loss.item():.6f}")

            # Calculate Accuracy (Top-1 Match)
            # MCTS target is soft, but we can check if max probability matches
            with torch.no_grad():
                pred_actions = torch.argmax(p_logits, dim=1)
                target_actions = torch.argmax(policy_targets, dim=1)
                accuracy = (pred_actions == target_actions).float().mean().item()

            loss = policy_loss + self.value_loss_weight * value_loss
            
            # SANITY CHECK: If loss is suspiciously low, it means we are overfitting the buffer.
            # Action: Stop training automatically to let buffer fill up more.
            # Lower threshold to 0.1 because 0.2 was still triggering too often.
            if loss.item() < 0.1: 
                print(f"⚠️ WARNING: Overfitting detected (Loss {loss.item():.6f}). Pausing training to accumulate more data...")
                # We don't stop self-play, just training updates
                time.sleep(10) # Wait 10 seconds (shorter wait, but more frequent checks)
                return None
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        finally:
            if self.network_lock:
                self.network_lock.release()
        
        self.training_stats['step'] += 1
        self.training_stats['policy_loss'] = policy_loss.item()
        self.training_stats['value_loss'] = value_loss.item()
        # Real Metric: Policy Accuracy
        self.training_stats['win_rate'] = accuracy # Reuse 'win_rate' key for now to avoid breaking backend contract too much
        
        return self.training_stats
