# -*- coding:utf-8 -*-
import platform
import subprocess
import numpy as np
from .gotypes import Player, Point
"""
围棋坐标可以用多种方式来指定, 但在欧洲最常见得方法是从A开始的字母表示列,
用1开始的数字表示行. 在这个坐标系中, 标准19x19棋盘左下角为A1, 有上角为T19. 
注意: 依照惯例, 我们忽略字母I, 避免与数字1混淆

用`.`表示空点
用`x`表示黑棋
用`o`表示白棋
"""

COLS = 'ABCDEFGHJKLMNOPQRST'
STONE_TO_CHAR = {
    None: ' . ',  # 空点
    Player.black: ' x ',
    Player.white: ' o ',
}


def print_move(player, move):
    if move.is_pass:
        move_str = 'passes'
    elif move.is_resign:
        move_str = 'resigns'
    else:
        move_str = '%s%d' % (COLS[move.point.col - 1], move.point.row)
    print('%s %s' % (player, move_str))


def print_board(board):
    for row in range(board.num_rows, 0, -1):
        bump = " " if row <= 9 else ""  # 让1位和2位的数字对齐
        line = []
        for col in range(1, board.num_cols + 1):
            stone = board.get(Point(row=row, col=col))
            line.append(STONE_TO_CHAR[stone])
        print('%s%d %s' % (bump, row, ''.join(line)))
    print('    ' + '  '.join(COLS[:board.num_cols]))


def point_from_coords(coords: str) -> Point:
    """
    coords转换成point

    例如:
    C3 -> Point
    E11 -> Point
    """
    col = COLS.index(coords[0]) + 1
    row = int(coords[1:])
    return Point(row=row, col=col)


def coords_from_point(point: Point) -> str:
    """
    point转换成coords

    例如:
    Point -> C3
    Point -> E11
    """
    return f'{COLS[point.row-1]}{point.row}'


def clear_screen():
    # see https://stackoverflow.com/a/23075152/323316
    if platform.system() == 'Windows':
        subprocess.Popen('cls', shell=True).communicate()
    else:  # Linux and Mac
        print(chr(27) + '[2J')


class MoveAge:

    def __init__(self, board):
        self.move_ages = -np.ones((board.num_rows, board.num_cols))

    def get(self, row, col):
        return self.move_ages[row, col]

    def reset_age(self, point):
        self.move_ages[point.row - 1, point.col - 1] = -1

    def add(self, point):
        self.move_ages[point.row - 1, point.col - 1] = 0

    def increment_all(self):
        self.move_ages[self.move_ages > -1] += 1