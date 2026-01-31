# -*- coding: utf-8 -*-
import numpy as np

class GoEngine:
    """
    A high-performance pure Python (NumPy) Go rules engine.
    Implements: capture, ko, suicide rule, scoring.
    """
    def __init__(self, board_size=9, komi=7.5):
        self.board_size = board_size
        self.komi = komi
        self.reset()

    def reset(self):
        """Reset the game board"""
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1  # 1=Black, 2=White
        self.ko = None  # Ko position (x, y)
        self.history = []  # History for superko check (simplified for now)
        self.move_count = 0
        self.passes = 0
        self.dead_stones = {1: 0, 2: 0} # Captured stones count
        return self.get_state()

    def get_state(self):
        """Return a copy of the current board state"""
        return self.board.copy()

    def switch_player(self):
        self.current_player = 3 - self.current_player

    def is_eyeish(self, x, y, owner):
        """Check if point is an eye (simplified) for heuristic pruning"""
        if x < 0 or x >= self.board_size or y < 0 or y >= self.board_size:
            return False
        if self.board[x, y] != 0:
            return False
            
        # Check 4 neighbors: must be owner or edge
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                if self.board[nx, ny] != owner:
                    return False
        
        return True

    def is_valid_move(self, x, y):
        """Check if a move is valid"""
        if x < 0 or x >= self.board_size or y < 0 or y >= self.board_size:
            return False
        if self.board[x, y] != 0:
            return False
        
        # Temporary move for testing
        test_board = self.board.copy()
        test_board[x, y] = self.current_player
        
        # 1. Check Ko
        if self.ko and (x, y) == self.ko:
            return False

        # 2. Check Liberties (Suicide rule)
        opponent = 3 - self.current_player
        captured = self._find_captured(test_board, x, y, opponent)
        
        # If captured opponent stones, it has liberties, so valid
        if len(captured) > 0:
            return True
            
        # If no capture, check self liberties
        if not self._has_liberty(test_board, x, y):
            return False # Suicide
            
        # 3. Heuristic: Don't fill own eye (unless it captures or saves group)
        # This is not a rule, but a strong heuristic to speed up learning in early phase
        if self.is_eyeish(x, y, self.current_player):
            # Only allow if it captures something (snapback) or saves own stones?
            # Simplified: don't fill simple eyes
            if len(captured) == 0:
                 return False

        return True

    def step(self, action):
        """
        Execute a move
        action: (x, y) or None (Pass)
        Returns: (board, reward, done, info)
        """
        if action is None:
            self.passes += 1
            self.switch_player()
            self.move_count += 1
            # Double Pass ends game
            if self.passes >= 2:
                return self.board.copy(), self._calculate_reward(), True, {"result": "double_pass"}
            return self.board.copy(), 0, False, {}

        x, y = action
        if not self.is_valid_move(x, y):
            raise ValueError(f"Invalid move: {action} for player {self.current_player}")

        self.passes = 0 # Reset pass count
        
        # Place stone
        self.board[x, y] = self.current_player
        opponent = 3 - self.current_player
        
        # Capture logic
        captured_stones = self._find_captured(self.board, x, y, opponent)
        for cx, cy in captured_stones:
            self.board[cx, cy] = 0
        self.dead_stones[self.current_player] += len(captured_stones)

        # Update Ko state
        # If one stone is captured and the new stone has only 1 liberty, it's a Ko
        if len(captured_stones) == 1 and self._count_liberties(self.board, x, y) == 1:
            self.ko = captured_stones[0]
        else:
            self.ko = None

        self.history.append(self.board.copy())
        self.move_count += 1
        self.switch_player()
        
        # Max moves limit
        if self.move_count > self.board_size ** 2 * 2:
            return self.board.copy(), 0, True, {"result": "max_moves"}

        return self.board.copy(), 0, False, {}

    def _find_captured(self, board, x, y, opponent):
        """Find captured stones after placing at (x,y)"""
        captured = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                if board[nx, ny] == opponent:
                    group, liberties = self._get_group_and_liberties(board, nx, ny)
                    if liberties == 0:
                        captured.extend(group)
        return list(set(captured))

    def _has_liberty(self, board, x, y):
        _, liberties = self._get_group_and_liberties(board, x, y)
        return liberties > 0
        
    def _count_liberties(self, board, x, y):
        _, liberties = self._get_group_and_liberties(board, x, y)
        return liberties

    def _get_group_and_liberties(self, board, start_x, start_y):
        """BFS to find group and count liberties"""
        color = board[start_x, start_y]
        group = []
        visited = set()
        queue = [(start_x, start_y)]
        visited.add((start_x, start_y))
        
        liberty_points = set()

        while queue:
            cx, cy = queue.pop(0)
            group.append((cx, cy))
            
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                    if board[nx, ny] == 0:
                        liberty_points.add((nx, ny))
                    elif board[nx, ny] == color and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        
        return group, len(liberty_points)

    def _calculate_reward(self):
        """
        Simple Area Scoring (Stone count + Komi).
        Does not implement full territory counting yet (TODO).
        """
        black_score = np.sum(self.board == 1)
        white_score = np.sum(self.board == 2) + self.komi
        
        if black_score > white_score:
            return 1 # Black wins
        else:
            return -1 # White wins

    def clone(self):
        """Create a deep copy of the game engine"""
        new_game = GoEngine(self.board_size, self.komi)
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.ko = self.ko
        new_game.history = [h.copy() for h in self.history]
        new_game.move_count = self.move_count
        new_game.passes = self.passes
        new_game.dead_stones = self.dead_stones.copy()
        return new_game

    def get_legal_moves(self):
        """Return list of valid moves including pass (None)"""
        moves = []
        for x in range(self.board_size):
            for y in range(self.board_size):
                if self.is_valid_move(x, y):
                    moves.append((x, y))
        moves.append(None) # Pass is always an option (usually)
        return moves

    def get_winner(self):
        if self._calculate_reward() > 0:
            return 1
        return 2

    def render(self):
        """Print board to console"""
        print("   " + " ".join([str(i) for i in range(self.board_size)]))
        for i in range(self.board_size):
            row_str = f"{i} "
            for j in range(self.board_size):
                if self.board[i, j] == 0:
                    row_str += " ."
                elif self.board[i, j] == 1:
                    row_str += " X"
                else:
                    row_str += " O"
            print(row_str)
        print(f"Ko: {self.ko}, Move: {self.move_count}, Dead: B{self.dead_stones[1]}/W{self.dead_stones[2]}")
