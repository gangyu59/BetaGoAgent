# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.go_engine import GoEngine
import unittest

class TestGoEngine(unittest.TestCase):
    def setUp(self):
        self.game = GoEngine(board_size=9)

    def test_capture(self):
        """Test capture logic"""
        # Black (0,1)
        self.game.step((0, 1))
        # White (0,0) - about to be captured
        self.game.step((0, 0))
        # Black (1,0) - captures White (0,0)
        self.game.step((1, 0))

        self.assertEqual(self.game.board[0, 0], 0, "Position (0,0) should be empty")
        self.assertEqual(self.game.dead_stones[1], 1, "Black should have captured 1 stone")

    def test_suicide(self):
        """Test suicide rule"""
        game = GoEngine(9)
        game.step((0, 1))  # B -> W
        game.step(None)    # W pass -> B
        game.step((1, 0))  # B -> W

        self.assertEqual(game.current_player, 2, "It should be White's turn")
        is_valid = game.is_valid_move(0, 0)
        self.assertFalse(is_valid, "Suicide move should be invalid")

    def test_no_eye_blocking(self):
        """Verify that filling your own eye is a LEGAL move (not blocked by heuristic)."""
        game = GoEngine(9)
        # Create a simple eye at (1,1) surrounded by Black
        game.board[0, 1] = 1
        game.board[1, 0] = 1
        game.board[1, 2] = 1
        game.board[2, 1] = 1
        game.current_player = 1
        # (1,1) is a simple eye - it must be a legal move
        self.assertTrue(game.is_valid_move(1, 1),
                       "Filling own eye must be legal (no heuristic blocking)")

    def test_territory_scoring(self):
        """Verify territory scoring counts empty regions, not just stones."""
        game = GoEngine(9)
        # Place Black stones along row 4 to divide the board
        for col in range(9):
            game.board[4, col] = 1

        # Top half is empty surrounded by Black (rows 0-3)
        # Bottom half is empty, open (rows 5-8, not surrounded by one color)
        # Actually, row 4 is all Black. Rows 0-3 border only Black -> Black territory
        # Rows 5-8 border only Black from the top, but also the edge.
        # Wait - edges don't count as borders. Empty regions only check stone colors.
        # So rows 5-8 also border only Black -> also Black territory.

        black_score, white_score = game._calculate_score()

        # 9 Black stones + all 72 empty points as Black territory = 81
        # White = 0 + 7.5 komi
        self.assertEqual(black_score, 81, "Black should have 9 stones + 72 territory = 81")
        self.assertAlmostEqual(white_score, 7.5, places=1)

    def test_territory_scoring_mixed(self):
        """Test territory scoring with both colors having territory."""
        game = GoEngine(9)
        # Black controls top, White controls bottom
        for col in range(9):
            game.board[3, col] = 1
            game.board[5, col] = 2

        # Row 4 is empty, bordered by both colors -> neutral
        # Rows 0-2 (27 points) bordered by Black only -> Black territory
        # Rows 6-8 (27 points) bordered by White only -> White territory

        black_score, white_score = game._calculate_score()
        # Black: 9 stones + 27 territory = 36
        # White: 9 stones + 27 territory + 7.5 komi = 43.5
        self.assertEqual(black_score, 36)
        self.assertAlmostEqual(white_score, 43.5, places=1)

    def test_pass_recorded_in_history(self):
        """Verify that pass moves are recorded in history."""
        game = GoEngine(9)
        game.step((0, 0))  # Black plays
        self.assertEqual(len(game.history), 1)
        game.step(None)    # White passes
        self.assertEqual(len(game.history), 2, "Pass should be recorded in history")

    def test_double_pass_ends_game(self):
        """Test that double pass ends the game."""
        game = GoEngine(9)
        game.step(None)  # Black passes
        _, _, done, info = game.step(None)  # White passes
        self.assertTrue(done, "Double pass should end the game")
        self.assertEqual(info.get("result"), "double_pass")

    def test_clone(self):
        """Test that clone creates independent copy."""
        game = GoEngine(9)
        game.step((4, 4))
        clone = game.clone()
        clone.step((0, 0))
        self.assertEqual(game.board[0, 0], 0, "Original should be unaffected by clone")
        self.assertNotEqual(clone.board[0, 0], 0, "Clone should have the new stone")

if __name__ == '__main__':
    unittest.main()
