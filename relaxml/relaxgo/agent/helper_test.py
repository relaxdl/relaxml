import unittest
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
from relaxgo.agent.helpers import is_point_an_eye
from relaxgo.goboard_slow import Board
from relaxgo.gotypes import Player, Point


class EyeTest(unittest.TestCase):

    def test_corner(self):
        board = Board(19, 19)
        #  5  .  .  .  .  .
        #  4  .  .  .  .  .
        #  3  .  .  .  .  .
        #  2  x  x  .  .  .
        #  1  .  x  .  .  .
        #     A  B  C  D  E
        board.place_stone(Player.black, Point(1, 2))
        board.place_stone(Player.black, Point(2, 2))
        board.place_stone(Player.black, Point(2, 1))
        self.assertTrue(is_point_an_eye(board, Point(1, 1), Player.black))
        self.assertFalse(is_point_an_eye(board, Point(1, 1), Player.white))

    def test_corner_false_eye(self):
        board = Board(19, 19)
        #  5  .  .  .  .  .
        #  4  .  .  .  .  .
        #  3  .  .  .  .  .
        #  2  x  .  .  .  .
        #  1  .  x  .  .  .
        #     A  B  C  D  E
        board.place_stone(Player.black, Point(1, 2))
        board.place_stone(Player.black, Point(2, 1))
        self.assertFalse(is_point_an_eye(board, Point(1, 1), Player.black))
        #  5  .  .  .  .  .
        #  4  .  .  .  .  .
        #  3  .  .  .  .  .
        #  2  x  o  .  .  .
        #  1  .  x  .  .  .
        #     A  B  C  D  E
        board.place_stone(Player.white, Point(2, 2))
        self.assertFalse(is_point_an_eye(board, Point(1, 1), Player.black))

    def test_middle(self):
        board = Board(19, 19)
        #  5  .  .  .  .  .
        #  4  .  x  x  x  .
        #  3  .  x  .  x  .
        #  2  .  x  x  x  .
        #  1  .  .  .  .  .
        #     A  B  C  D  E
        board.place_stone(Player.black, Point(2, 2))
        board.place_stone(Player.black, Point(3, 2))
        board.place_stone(Player.black, Point(4, 2))
        board.place_stone(Player.black, Point(4, 3))
        board.place_stone(Player.white, Point(4, 4))
        board.place_stone(Player.black, Point(3, 4))
        board.place_stone(Player.black, Point(2, 4))
        board.place_stone(Player.black, Point(2, 3))
        self.assertTrue(is_point_an_eye(board, Point(3, 3), Player.black))


if __name__ == '__main__':
    unittest.main()
