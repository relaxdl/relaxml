#! /usr/bin/env python3
# -*- coding:utf-8 -*-
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relaxgo.goboard import Board
from relaxgo.gotypes import (Player, Point)
from relaxgo import scoring


class ScoringTest(unittest.TestCase):

    def test_scoring(self):
        #  5  .  o  .  o  o
        #  4  o  o  o  o  .
        #  3  x  x  x  o  o
        #  2  .  x  x  x  x
        #  1  .  x  .  x  .
        #     A  B  C  D  E
        board = Board(5, 5)
        board.place_stone(Player.black, Point(1, 2))
        board.place_stone(Player.black, Point(1, 4))
        board.place_stone(Player.black, Point(2, 2))
        board.place_stone(Player.black, Point(2, 3))
        board.place_stone(Player.black, Point(2, 4))
        board.place_stone(Player.black, Point(2, 5))
        board.place_stone(Player.black, Point(3, 1))
        board.place_stone(Player.black, Point(3, 2))
        board.place_stone(Player.black, Point(3, 3))
        board.place_stone(Player.white, Point(3, 4))
        board.place_stone(Player.white, Point(3, 5))
        board.place_stone(Player.white, Point(4, 1))
        board.place_stone(Player.white, Point(4, 2))
        board.place_stone(Player.white, Point(4, 3))
        board.place_stone(Player.white, Point(4, 4))
        board.place_stone(Player.white, Point(5, 2))
        board.place_stone(Player.white, Point(5, 4))
        board.place_stone(Player.white, Point(5, 5))
        territory = scoring.evaluate_territory(board)
        self.assertEqual(9, territory.num_black_stones)
        self.assertEqual(4, territory.num_black_territory)
        self.assertEqual(9, territory.num_white_stones)
        self.assertEqual(3, territory.num_white_territory)
        self.assertEqual(0, territory.num_dame)


if __name__ == '__main__':
    unittest.main()