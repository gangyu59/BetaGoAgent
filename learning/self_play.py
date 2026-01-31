import threading
import time
import numpy as np
from core.mcts import MCTS
from core.go_engine import GoEngine

class SelfPlayWorker:
    def __init__(self, network, encoder, trainer, num_simulations=50):
        self.network = network
        self.encoder = encoder
        self.trainer = trainer
        self.running = False
        self.mcts = MCTS(network, encoder, num_simulations=num_simulations)
        self.game_count = 0
        self._thread = None

    def start(self):
        if self.running:
            return
            
        self.running = True
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self):
        self.running = False

    def _loop(self):
        print("🎮 Self-Play Worker Started")
        while True:
            if not self.running:
                time.sleep(1)
                continue
                
            self._play_one_game()
            self.game_count += 1
            # print(f"🎮 Self-Play Game {self.game_count} finished.")

    def _play_one_game(self):
        game = GoEngine(board_size=9)
        states = []
        policy_targets = []
        # Value target will be determined at end of game
        
        move_num = 0
        
        while True:
            # Temperature decay
            # AlphaZero uses temp=1 for first 30 moves.
            # For 9x9, 30 moves is basically the whole opening + midgame.
            # Let's keep it 1.0 longer to encourage diversity, otherwise it collapses to same game too fast.
            
            if move_num < 20:
                temp = 1.0
            else:
                temp = 0.5 # Gradual decay instead of hard 0.05
            
            # Let's inject lock into SelfPlayWorker
            # We will attach the lock to the network object itself in server.py for convenience
            # so MCTS can find it.
            
            policy, value = self.mcts.get_action_probs(game, temp=temp)

            # Store data
            # Input state tensor
            state_tensor = self.encoder.encode(game).squeeze(0).numpy() # (7, 9, 9)
            
            # Data Augmentation (8 symmetries)
            # We store all 8 forms to buffer immediately or later?
            # Storing later in Trainer is better for memory, but Trainer is simple.
            # Let's do it here.
            
            # Policy is 82. 0-80 map to board. 81 is pass.
            # We need to reshape policy to 9x9 to rotate it.
            p_board = policy[:81].reshape(9, 9)
            p_pass = policy[81]
            
            sym_states = []
            sym_policies = []
            
            # Generate 8 symmetries
            for k in range(4): # Rotate 0, 90, 180, 270
                # Rotate state: state has shape (7, 9, 9)
                rot_state = np.rot90(state_tensor, k, axes=(1, 2)).copy()
                rot_p_board = np.rot90(p_board, k).copy()
                
                sym_states.append(rot_state)
                sym_policies.append(np.append(rot_p_board.flatten(), p_pass))
                
                # Flip (Horizontal flip after rotation)
                flip_state = np.flip(rot_state, axis=2).copy()
                flip_p_board = np.flip(rot_p_board, axis=1).copy()
                
                sym_states.append(flip_state)
                sym_policies.append(np.append(flip_p_board.flatten(), p_pass))

            # We append lists of symmetries
            states.append(sym_states)
            policy_targets.append(sym_policies)
            
            # Sample action from policy (already temperature adjusted)
            # If temp is very small, this is effectively argmax
            action_idx = np.random.choice(82, p=policy)
            
            if action_idx == 81:
                action = None
            else:
                action = (action_idx // 9, action_idx % 9)
                
            # Step
            _, _, done, info = game.step(action)
            move_num += 1
            
            if move_num % 10 == 0:
                 print(f"   [Self-Play] Game {self.game_count+1} Move {move_num}...")

            if done:
                winner = game.get_winner() # 1 or 2
                break
                
        # Game over, generate value targets
        # If winner is 1 (Black), then for all states where current_player was 1, z=+1.
        # For states where current_player was 2, z=-1.
        
        # We need to reconstruct whose turn it was.
        # GoEngine starts with Black (1).
        
        current_player = 1
        for i in range(len(states)):
            if winner == current_player:
                z = 1.0
            else:
                z = -1.0
            
            # Add all 8 symmetries
            for j in range(8):
                self.trainer.add_data(states[i][j], policy_targets[i][j], z)
            
            # Switch for next state
            current_player = 3 - current_player
