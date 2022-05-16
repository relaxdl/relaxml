# -*- coding:utf-8 -*-
from typing import Any, List
from enum import Enum, unique
from collections import namedtuple
from typing import List

from pyparsing import And

__all__ = [
    'Player',
    'Point',
]


@unique
class Player(Enum):
    black = 1
    white = 2

    @property
    def other(self):
        """
        >>> assert Player.white.other == Player.black
        >>> assert Player.black.other == Player.white
        """
        return Player.black if self == Player.white else Player.white


class Point(namedtuple('Point', 'row col')):
    """
    棋盘交叉点的坐标

    >>> point = Point(row=1, col=2)
    >>> point.row
        1
    >>> point.col
        2

    # 直接比较
    >>> assert Point(row=1, col=2) == Point(row=1, col=2)
    >>> assert hash(Point(row=1, col=2)) == hash(Point(row=1, col=2))
    """

    def neighbors(self) -> List[Any]:
        return [
            Point(self.row - 1, self.col),
            Point(self.row + 1, self.col),
            Point(self.row, self.col - 1),
            Point(self.row, self.col + 1),
        ]

    def __deepcopy__(self, memodict={}):
        """
        Point一旦创建, 是不可变的, 直接返回自己
        """
        return self
