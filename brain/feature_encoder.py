import numpy as np
import torch

class FeatureEncoder:
    def __init__(self, board_size=9):
        self.board_size = board_size
        # 11 planes:
        # 0: Current Player Stones
        # 1: Opponent Stones
        # 2: Empty points
        # 3: History -1 Current Player
        # 4: History -1 Opponent
        # 5: History -2 Current Player
        # 6: History -2 Opponent
        # 7: Current player liberty count (per-stone, normalized)
        # 8: Opponent liberty count (per-stone, normalized)
        # 9: Legal moves for current player
        # 10: Color Plane (All 1 if Black, All 0 if White)
        self.num_planes = 11

    def encode(self, game_state):
        """Encode the game state into a tensor."""
        board = game_state.board
        current_player = game_state.current_player
        opponent = 3 - current_player
        size = self.board_size

        features = np.zeros((self.num_planes, size, size), dtype=np.float32)

        # Plane 0: Current Player stones
        features[0] = (board == current_player)

        # Plane 1: Opponent stones
        features[1] = (board == opponent)

        # Plane 2: Empty points
        features[2] = (board == 0)

        # History -1
        history = game_state.history
        if len(history) >= 1:
            h1 = history[-1]
            features[3] = (h1 == current_player)
            features[4] = (h1 == opponent)

        # History -2
        if len(history) >= 2:
            h2 = history[-2]
            features[5] = (h2 == current_player)
            features[6] = (h2 == opponent)

        # Liberty counts (per stone, normalized by max_liberties=8)
        max_lib = 8.0
        for x in range(size):
            for y in range(size):
                if board[x, y] == current_player:
                    _, libs = game_state._get_group_and_liberties(board, x, y)
                    features[7][x, y] = min(libs, max_lib) / max_lib
                elif board[x, y] == opponent:
                    _, libs = game_state._get_group_and_liberties(board, x, y)
                    features[8][x, y] = min(libs, max_lib) / max_lib

        # Legal moves
        for x in range(size):
            for y in range(size):
                if game_state.is_valid_move(x, y):
                    features[9][x, y] = 1.0

        # Color plane
        if current_player == 1:
            features[10] = 1.0
        else:
            features[10] = 0.0

        return torch.FloatTensor(features).unsqueeze(0)  # (1, 11, H, W)
