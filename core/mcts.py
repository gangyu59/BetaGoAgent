import numpy as np
import math
import torch

class MCTSNode:
    def __init__(self, parent=None, prior_prob=1.0):
        self.parent = parent
        self.children = {}  # action -> MCTSNode
        self.visit_count = 0
        self.value_sum = 0
        self.prior_prob = prior_prob
        self.is_expanded = False

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def ucb_score(self, c_puct=2.0):
        u = c_puct * self.prior_prob * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return self.value() + u

class MCTS:
    def __init__(self, network, encoder, c_puct=2.0, num_simulations=100,
                 add_noise=True):
        self.network = network
        self.encoder = encoder
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.add_noise = add_noise

    def search(self, game_state):
        root = MCTSNode()

        # Expand root
        self._expand(root, game_state)

        # Add Dirichlet noise to root for exploration (only during self-play)
        if self.add_noise and len(root.children) > 0:
            actions = list(root.children.keys())
            noise = np.random.dirichlet([0.03 * 82] * len(actions))
            for i, action in enumerate(actions):
                root.children[action].prior_prob = (
                    0.75 * root.children[action].prior_prob + 0.25 * noise[i]
                )

        for _ in range(self.num_simulations):
            node = root
            simulation_game = game_state.clone()
            search_path = [node]

            # 1. Selection - descend until we find an unexpanded node
            while node.is_expanded and len(node.children) > 0:
                action, node = self._select_child(node)
                simulation_game.step(action)
                search_path.append(node)

            # 2. Expansion & Evaluation
            value = 0
            if simulation_game.passes >= 2 or simulation_game.move_count >= simulation_game.board_size ** 2 * 3:
                # Terminal node - use actual game result
                winner = simulation_game.get_winner()
                # Value from perspective of the player whose turn it is
                if winner == simulation_game.current_player:
                    value = 1.0
                else:
                    value = -1.0
            else:
                # Expand and get neural network value estimate
                value = self._expand(node, simulation_game)

            # 3. Backpropagate
            # value is from perspective of simulation_game.current_player (the player to move at the leaf).
            # As we go up, we alternate signs because parent nodes represent the opponent's perspective.
            self._backpropagate(node, value)

        return root

    def get_action_probs(self, game_state, temp=1.0):
        root = self.search(game_state)

        counts = []
        actions = []
        for action, child in root.children.items():
            actions.append(action)
            counts.append(child.visit_count)

        counts = np.array(counts, dtype=np.float64)

        if temp == 0 or np.sum(counts) == 0:
            if np.sum(counts) == 0:
                probs = np.ones_like(counts) / len(counts)
            else:
                best_idx = np.argmax(counts)
                probs = np.zeros_like(counts, dtype=float)
                probs[best_idx] = 1.0
        else:
            log_counts = np.log(counts + 1e-10) / temp
            log_counts -= np.max(log_counts)  # numerical stability
            probs = np.exp(log_counts)
            probs /= np.sum(probs)

        # Map to 82-dim policy vector
        policy_target = np.zeros(82)
        for i, action in enumerate(actions):
            if action is None:
                idx = 81
            else:
                idx = action[0] * 9 + action[1]
            policy_target[idx] = probs[i]

        return policy_target, root.value()

    def _select_child(self, node):
        best_score = -float('inf')
        best_action = None
        best_child = None

        for action, child in node.children.items():
            score = child.ucb_score(self.c_puct)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def _expand(self, node, game_state):
        tensor = self.encoder.encode(game_state)

        if hasattr(self.network, 'lock') and self.network.lock:
            with self.network.lock:
                self.network.eval()
                with torch.no_grad():
                    p_logits, v = self.network(tensor)
        else:
            self.network.eval()
            with torch.no_grad():
                p_logits, v = self.network(tensor)

        p_probs = torch.softmax(p_logits, dim=1).squeeze(0).numpy()
        value = v.item()

        legal_moves = game_state.get_legal_moves()

        legal_probs_sum = 0
        valid_children = {}

        for move in legal_moves:
            if move is None:
                idx = 81
            else:
                idx = move[0] * 9 + move[1]

            prob = p_probs[idx]
            legal_probs_sum += prob

            child = MCTSNode(parent=node, prior_prob=prob)
            valid_children[move] = child

        # Re-normalize priors over legal moves
        if legal_probs_sum > 1e-8:
            for child in valid_children.values():
                child.prior_prob /= legal_probs_sum
        else:
            for child in valid_children.values():
                child.prior_prob = 1.0 / len(valid_children)

        node.children = valid_children
        node.is_expanded = True

        return value

    def _backpropagate(self, node, value):
        # value is from the perspective of the player-to-move at the leaf node.
        # The leaf node itself stores value from the perspective of the player-to-move there.
        # Its parent stores value from the opponent's perspective, so we negate.
        current = node
        current_val = value

        while current is not None:
            current.visit_count += 1
            current.value_sum += current_val
            current = current.parent
            current_val = -current_val
