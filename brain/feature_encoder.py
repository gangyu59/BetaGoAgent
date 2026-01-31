import numpy as np
import torch

class FeatureEncoder:
    def __init__(self, board_size=9):
        self.board_size = board_size
        # 7 planes:
        # 0: Current Player Stones
        # 1: Opponent Stones
        # 2: History -1 Current Player
        # 3: History -1 Opponent
        # 4: History -2 Current Player
        # 5: History -2 Opponent
        # 6: Color Plane (All 1 if Black, All 0 if White)
        self.num_planes = 7

    def encode(self, game_state):
        """
        Encode the game state into a tensor.
        game_state: GoEngine instance
        """
        board = game_state.board
        current_player = game_state.current_player
        opponent = 3 - current_player
        
        features = np.zeros((self.num_planes, self.board_size, self.board_size), dtype=np.float32)
        
        # Plane 0: Current Player
        features[0] = (board == current_player)
        
        # Plane 1: Opponent
        features[1] = (board == opponent)
        
        # History
        history = game_state.history
        # History -1
        if len(history) >= 1:
            h1 = history[-1]
            features[2] = (h1 == current_player)
            features[3] = (h1 == opponent)
        
        # History -2
        if len(history) >= 2:
            h2 = history[-2]
            features[4] = (h2 == current_player)
            features[5] = (h2 == opponent)
            
        # Plane 6: Color
        if current_player == 1: # Black
            features[6] = 1.0
        else: # White
            features[6] = 0.0
            
        return torch.FloatTensor(features).unsqueeze(0) # (1, 7, H, W)
