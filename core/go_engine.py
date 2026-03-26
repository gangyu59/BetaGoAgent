# -*- coding: utf-8 -*-
import numpy as np

class GoEngine:
    """
    Go rules engine for 9x9 board.
    Implements: capture, ko, suicide rule, Chinese-rules territory scoring.
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
        self.history = []  # Board state history
        self.move_count = 0
        self.passes = 0
        self.dead_stones = {1: 0, 2: 0}
        return self.get_state()

    def get_state(self):
        """Return a copy of the current board state"""
        return self.board.copy()

    def switch_player(self):
        self.current_player = 3 - self.current_player

    def is_valid_move(self, x, y):
        """Check if a move is valid (pure rules, no heuristic pruning)"""
        if x < 0 or x >= self.board_size or y < 0 or y >= self.board_size:
            return False
        if self.board[x, y] != 0:
            return False

        # 1. Check Ko
        if self.ko and (x, y) == self.ko:
            return False

        # Temporary move for testing
        test_board = self.board.copy()
        test_board[x, y] = self.current_player

        # 2. Check captures
        opponent = 3 - self.current_player
        captured = self._find_captured(test_board, x, y, opponent)

        # If captured opponent stones, move is always valid
        if len(captured) > 0:
            return True

        # 3. Check self liberties (suicide rule)
        if not self._has_liberty(test_board, x, y):
            return False

        return True

    def step(self, action):
        """
        Execute a move.
        action: (x, y) or None (Pass)
        Returns: (board, reward, done, info)
        """
        if action is None:
            self.passes += 1
            self.history.append(self.board.copy())
            self.switch_player()
            self.move_count += 1
            if self.passes >= 2:
                winner = self.get_winner()
                if winner == 0:
                    reward = 0
                else:
                    reward = 1 if winner == 1 else -1
                return self.board.copy(), reward, True, {"result": "double_pass", "winner": winner}
            return self.board.copy(), 0, False, {}

        x, y = action
        if not self.is_valid_move(x, y):
            raise ValueError(f"Invalid move: {action} for player {self.current_player}")

        self.passes = 0

        # Place stone
        self.board[x, y] = self.current_player
        opponent = 3 - self.current_player

        # Capture logic
        captured_stones = self._find_captured(self.board, x, y, opponent)
        for cx, cy in captured_stones:
            self.board[cx, cy] = 0
        self.dead_stones[self.current_player] += len(captured_stones)

        # Update Ko state
        if len(captured_stones) == 1 and self._count_liberties(self.board, x, y) == 1:
            self.ko = captured_stones[0]
        else:
            self.ko = None

        self.history.append(self.board.copy())
        self.move_count += 1
        self.switch_player()

        max_moves = self.board_size ** 2 * 3
        if self.move_count >= max_moves:
            winner = self.get_winner()
            if winner == 0:
                reward = 0
            else:
                reward = 1 if winner == 1 else -1
            return self.board.copy(), reward, True, {"result": "max_moves", "winner": winner}

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
        if color == 0:
            return [], 0
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

    def _flood_fill_territory(self, board):
        """
        Compute territory using flood-fill.
        Empty points surrounded entirely by one color belong to that color.
        """
        size = self.board_size
        visited = np.zeros((size, size), dtype=bool)
        black_territory = 0
        white_territory = 0

        for x in range(size):
            for y in range(size):
                if board[x, y] == 0 and not visited[x, y]:
                    region = []
                    queue = [(x, y)]
                    visited[x, y] = True
                    borders = set()

                    while queue:
                        cx, cy = queue.pop(0)
                        region.append((cx, cy))
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nx, ny = cx + dx, cy + dy
                            if 0 <= nx < size and 0 <= ny < size:
                                if board[nx, ny] == 0 and not visited[nx, ny]:
                                    visited[nx, ny] = True
                                    queue.append((nx, ny))
                                elif board[nx, ny] != 0:
                                    borders.add(int(board[nx, ny]))

                    if borders == {1}:
                        black_territory += len(region)
                    elif borders == {2}:
                        white_territory += len(region)

        return black_territory, white_territory

    def _calculate_score(self):
        """Chinese rules scoring: stones on board + territory"""
        black_stones = int(np.sum(self.board == 1))
        white_stones = int(np.sum(self.board == 2))
        black_territory, white_territory = self._flood_fill_territory(self.board)
        black_score = black_stones + black_territory
        white_score = white_stones + white_territory + self.komi
        return black_score, white_score

    def get_winner(self):
        """
        Return winner: 1 = Black, 2 = White, 0 = draw.
        With komi=7.5 on 9x9, draws are extremely rare but handled for completeness.
        """
        black_score, white_score = self._calculate_score()
        if black_score > white_score:
            return 1
        elif white_score > black_score:
            return 2
        else:
            return 0

    def get_score(self):
        """Return detailed scoring info"""
        black_stones = int(np.sum(self.board == 1))
        white_stones = int(np.sum(self.board == 2))
        black_territory, white_territory = self._flood_fill_territory(self.board)
        return {
            'black_stones': black_stones,
            'white_stones': white_stones,
            'black_territory': black_territory,
            'white_territory': white_territory,
            'black_total': black_stones + black_territory,
            'white_total': white_stones + white_territory + self.komi,
            'komi': self.komi,
        }

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
        moves.append(None)
        return moves

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
        bs, ws = self._calculate_score()
        print(f"Ko: {self.ko}, Move: {self.move_count}, Score: B{bs}/W{ws}")
