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

    def ucb_score(self, c_puct=1.0):
        # Upper Confidence Bound for Trees
        u = c_puct * self.prior_prob * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return self.value() + u

class MCTS:
    def __init__(self, network, encoder, c_puct=1.0, num_simulations=100):
        self.network = network
        self.encoder = encoder
        self.c_puct = c_puct
        self.num_simulations = num_simulations

    def search(self, game_state):
        root = MCTSNode()
        
        # Add Dirichlet noise to root for exploration (self-play only, but we use it generally here for now)
        # We will expand root first
        self._expand(root, game_state)
        
        # Add noise
        if len(root.children) > 0:
            actions = list(root.children.keys())
            noise = np.random.dirichlet([0.3] * len(actions)) # Higher noise for 9x9
            for i, action in enumerate(actions):
                root.children[action].prior_prob = 0.75 * root.children[action].prior_prob + 0.25 * noise[i]

        for _ in range(self.num_simulations):
            node = root
            simulation_game = game_state.clone()
            
            # 1. Selection
            while node.is_expanded and len(node.children) > 0:
                action, node = self._select_child(node)
                simulation_game.step(action)
            
            # 2. Expansion & Evaluation
            # Check if game ended
            # (Simple check: if passes >= 2 or max moves)
            # But simulation_game.step returns done.
            # We need to know if the last step was done.
            # Let's assume step() handles logic, but we need to check if game is over.
            
            # Since step returns (board, reward, done, info), we can track it, but MCTS usually descends until leaf.
            # If node is terminal, value is actual reward.
            
            value = 0
            if simulation_game.passes >= 2 or simulation_game.move_count > simulation_game.board_size**2 * 2:
                # Game over
                winner = simulation_game.get_winner() # 1 or 2
                # Value for current player: if winner == current_player, +1, else -1
                # Note: 'value' in MCTS usually represents "value for the player who just moved" or "current player to move".
                # AlphaZero standard: Value is from perspective of current_player (who is about to move).
                # If it is black's turn, value +1 means black wins.
                
                if winner == simulation_game.current_player:
                    value = 1.0
                else:
                    value = -1.0
            else:
                # Expand
                value = self._expand(node, simulation_game)
            
            # 3. Backup
            self._backpropagate(node, value)
            
        return root

    def get_action_probs(self, game_state, temp=1.0):
        root = self.search(game_state)
        
        counts = []
        actions = []
        for action, child in root.children.items():
            actions.append(action)
            counts.append(child.visit_count)
            
        counts = np.array(counts)
        
        if temp == 0:
            best_idx = np.argmax(counts)
            probs = np.zeros_like(counts, dtype=float)
            probs[best_idx] = 1.0
        else:
            # probs = counts ** (1.0 / temp)
            # probs /= np.sum(probs)
            # Use safer softmax-like approach if needed, but standard is power
            probs = (counts ** (1.0 / temp)) / np.sum(counts ** (1.0 / temp))
            
        # Map back to 82-dim array for training target
        # Action is (x, y) or None.
        # Flat index: x * 9 + y. Pass is 81.
        policy_target = np.zeros(82)
        for i, action in enumerate(actions):
            if action is None:
                idx = 81
            else:
                idx = action[0] * 9 + action[1]
            policy_target[idx] = probs[i]
            
        return policy_target, root.value() # Value of root state

    def _select_child(self, node):
        # Select child with highest UCB score
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
        # Get prediction from network
        tensor = self.encoder.encode(game_state)
        # Need network on CPU for MCTS threads usually, or batched. For simple implementation:
        
        # We assume self.network_lock is set on the network instance or handled externally
        # But here we just call it. If global lock is needed, the caller should handle it?
        # Actually MCTS search loop does many inference calls.
        # It's better if we pass a thread-safe predict_fn instead of raw network.
        
        # For this simple version, let's assume we can call it.
        # But wait, trainer updates weights. Concurrent read/write is bad.
        # Let's add a lock parameter to MCTS or use the one on network if exists.
        
        if hasattr(self.network, 'lock') and self.network.lock:
             with self.network.lock:
                 with torch.no_grad():
                    p_logits, v = self.network(tensor)
        else:
            with torch.no_grad():
                p_logits, v = self.network(tensor)
            
        # Policy
        p_probs = torch.softmax(p_logits, dim=1).squeeze(0).numpy() # (82,)
        value = v.item() # Scalar [-1, 1] from perspective of current player
        
        # Mask invalid moves
        legal_moves = game_state.get_legal_moves()
        
        # Normalize probabilities over legal moves
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
            
        # Re-normalize priors
        if legal_probs_sum > 0:
            for child in valid_children.values():
                child.prior_prob /= legal_probs_sum
        else:
            # Fallback if network gives 0 prob to all legal moves (rare)
            for child in valid_children.values():
                child.prior_prob = 1.0 / len(valid_children)

        # FPU (First Play Urgency)
        # Use a simpler FPU: if unvisited, assume it has value of parent (minus small epsilon)
        # or just 0 (draw). 0 is safe.
        
        # Add noise to priors in root is handled by caller.
        
        node.children = valid_children
        node.is_expanded = True
        
        return value

    def _backpropagate(self, node, value):
        # value is from perspective of leaf node's player.
        # We need to alternate signs as we go up the tree (Minimax style).
        # Actually, in AlphaZero:
        # v is value for current player state.
        # If child node state value is V, then for parent (previous player), value is -V.
        
        current = node
        current_val = value
        
        while current is not None:
            current.visit_count += 1
            current.value_sum += current_val
            current = current.parent
            current_val = -current_val # Flip for opponent
