import threading
import time
import numpy as np
from core.mcts import MCTS
from core.go_engine import GoEngine

class SelfPlayWorker:
    def __init__(self, network, encoder, trainer, num_simulations=100):
        self.network = network
        self.encoder = encoder
        self.trainer = trainer
        self.running = False
        self.mcts = MCTS(network, encoder, num_simulations=num_simulations,
                         add_noise=True, c_puct=2.0)
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
        print("[Self-Play] Worker Started")
        while True:
            if not self.running:
                time.sleep(1)
                continue

            try:
                self._play_one_game()
                self.game_count += 1
                print(f"[Self-Play] Game {self.game_count} finished.")
            except Exception as e:
                print(f"[Self-Play] Game error (skipping): {e}")
                import traceback
                traceback.print_exc()

    def _play_one_game(self):
        game = GoEngine(board_size=9)
        states = []
        policy_targets = []
        players = []

        move_num = 0
        max_game_moves = 9 * 9 * 2  # reasonable limit for 9x9

        while True:
            # Temperature: high early for exploration, low later for exploitation
            if move_num < 16:
                temp = 1.0
            elif move_num < 30:
                temp = 0.5
            else:
                temp = 0.1

            policy, value = self.mcts.get_action_probs(game, temp=temp)

            # Store state and policy
            state_tensor = self.encoder.encode(game).squeeze(0).numpy()  # (11, 9, 9)

            # Data augmentation: generate 8 symmetries
            p_board = policy[:81].reshape(9, 9)
            p_pass = policy[81]

            sym_states = []
            sym_policies = []

            for k in range(4):
                rot_state = np.rot90(state_tensor, k, axes=(1, 2)).copy()
                rot_p_board = np.rot90(p_board, k).copy()

                sym_states.append(rot_state)
                sym_policies.append(np.append(rot_p_board.flatten(), p_pass))

                flip_state = np.flip(rot_state, axis=2).copy()
                flip_p_board = np.flip(rot_p_board, axis=1).copy()

                sym_states.append(flip_state)
                sym_policies.append(np.append(flip_p_board.flatten(), p_pass))

            states.append(sym_states)
            policy_targets.append(sym_policies)
            players.append(game.current_player)

            # Sample action (discourage early pass during self-play)
            probs = np.array(policy, dtype=np.float64)
            min_moves_before_pass = 30  # before this many moves, do not pass
            if move_num < min_moves_before_pass:
                probs[81] = 0.0

            if probs.sum() <= 1e-12:
                probs = np.array(policy, dtype=np.float64)

            probs = probs / probs.sum()
            action_idx = np.random.choice(82, p=probs)

            if action_idx == 81:
                action = None
            else:
                action = (action_idx // 9, action_idx % 9)

            # Ensure we only play legal moves (policy can have numerical noise)
            if action is not None and not game.is_valid_move(action[0], action[1]):
                legal = game.get_legal_moves()
                # Prefer a board move over pass if any
                action = None
                for m in legal:
                    if m is not None:
                        action = m
                        break

            _, _, done, info = game.step(action)
            move_num += 1

            if done or move_num >= max_game_moves:
                if not done:
                    # Force end by double pass
                    game.step(None)
                    game.step(None)
                winner = game.get_winner()
                break

        # Assign value targets based on game outcome
        for i in range(len(states)):
            if winner == 0:
                z = 0.0
            elif winner == players[i]:
                z = 1.0
            else:
                z = -1.0

            for j in range(8):
                self.trainer.add_data(states[i][j], policy_targets[i][j], z)
