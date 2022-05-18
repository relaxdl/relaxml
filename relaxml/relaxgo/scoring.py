# -*- coding:utf-8 -*-
from collections import namedtuple
from typing import Dict, Union, Tuple, List
from typing import Any

from .gotypes import Player, Point
"""
在自己的回合中, 双方都可以选择不落子, 从而跳过当前回合. 如果双方接连跳过回合, 比赛就结束了

在计算评分时, `死棋`的处理方式与吃子完全相同.

围棋比赛的目标是`比对方控制棋盘上更大的底盘`. 计算得分有两种不同的方法, 但是得出的结果通常是相同的

1. 数目法: 棋盘上每一颗被己方棋子完全包围的交叉点都记做一分, 称为`一目`. 己方吃掉的每颗棋子也被算作一分,
   加起来谁的总分高谁就获胜
2. 数子法: 在数子法中, 每一目算一分, 己方在棋盘上剩下的每颗棋子也算一分

除及特殊情况外, 这两种方法得到的胜负结果一般是相同的. 如果没有过早结束棋局的话, 双方提子数的差别与双方棋盘上
剩下棋子的差别往往是一样的. 数目法在休闲棋局中更常见, 但对计算机而言, 数子法更方便

此外, 执白子的一方还要得到额外的分数, 以补偿后手的劣势. 这种补偿称为`贴子(komi)`. 在数目法中一般贴6.5子, 在数子法中
一般贴7.5子. 这里额外的0.5子用来确保不会出现平局 
"""


class Territory:
    """
    一个`territory_map`将棋盘分为三类:
    1. tones: 直接被黑子或者白子占领的point
    2. territory: 被黑子或者白子完全包围的point
    3. dame: 中立
    """

    def __init__(self, territory_map: Dict[Point, Union[Player, str]]) -> None:
        self.num_black_territory = 0
        self.num_white_territory = 0
        self.num_black_stones = 0
        self.num_white_stones = 0
        self.num_dame = 0
        self.dame_points = []  # List[Point]
        for point, status in territory_map.items():
            if status == Player.black:
                self.num_black_stones += 1  # black stone
            elif status == Player.white:
                self.num_white_stones += 1  # white stone
            elif status == 'territory_b':
                self.num_black_territory += 1  # black territory
            elif status == 'territory_w':
                self.num_white_territory += 1  # white territory
            elif status == 'dame':
                self.num_dame += 1  # dame
                self.dame_points.append(point)


class GameResult(namedtuple('GameResult', 'b w komi')):
    """
    游戏结果
    """

    @property
    def winner(self) -> Player:
        """
        获胜的一方
        """
        if self.b > self.w + self.komi:
            return Player.black
        return Player.white

    @property
    def winning_margin(self) -> int:
        """
        |w + komi(贴目) - b|
        """
        w = self.w + self.komi
        return abs(self.b - w)

    def __str__(self):
        w = self.w + self.komi
        if self.b > w:
            return 'B+%.1f' % (self.b - w, )
        return 'W+%.1f' % (w - self.b, )


def evaluate_territory(board: Any) -> Territory:
    """
    将board映射成5种类型:
    white
    black
    territory_b
    territory_w
    dame

    1. 如果point已经访问过了则直接跳过
    2. 如果point已经有棋子了, 直接写入到status
    3. 如果point没有棋子:
       a. 如果point完全被一种颜色包围(它的边界只有一种颜色), 则这个point属于`territory_*`
       b. 如果point没有被一种颜色包围, 则这个point属于`dame`
    """
    status = {}  # Dict[Point, Union[Player, str]]
    for r in range(1, board.num_rows + 1):
        for c in range(1, board.num_cols + 1):
            p = Point(row=r, col=c)
            # 如果point已经访问过了则直接跳过
            if p in status:
                continue
            stone = board.get(p)
            if stone is not None:
                # 如果point已经有棋子了, 直接写入到status
                status[p] = board.get(p)
            else:
                # 如果point没有棋子
                group, neighbors = _collect_region(p, board)
                if len(neighbors) == 1:
                    # 如果point完全被一种颜色包围(它的边界只有一种颜色),
                    # 则这个point属于`territory_*`
                    neighbor_stone = neighbors.pop()
                    stone_str = 'b' if neighbor_stone == Player.black else 'w'
                    fill_with = 'territory_' + stone_str
                else:
                    # 如果point没有被一种颜色包围, 则这个point属于`dame`
                    fill_with = 'dame'
                for pos in group:
                    status[pos] = fill_with
    return Territory(status)


def _collect_region(
        start_pos: Point,
        board: Any,
        visited: Union[Dict[Point, bool],
                       None] = None) -> Tuple[List[Point], set]:
    """
    从start_pos搜寻棋盘上连续的部分(all_points), 以及这个部分所有的边界(all_borders)

    判断出边界border, 就可以知道连续的部分是否被对方全部围住

    返回:
    all_points: 连续的部分(同色棋子)
    all_borders: 边界, 可能的元素: Player.white, Player.black, None
    """
    if visited is None:
        visited = {}  # Dict[Point, bool]
    if start_pos in visited:
        return [], set()
    all_points = [start_pos]  # List[Point], 连续的部分(同色棋子)
    all_borders = set()  # 边界, 可能的元素: Player.white, Player.black, None
    visited[start_pos] = True
    here = board.get(start_pos)
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    # 遍历和sart_pos相邻的上下左右四个Point
    for delta_r, delta_c in deltas:
        next_p = Point(row=start_pos.row + delta_r,
                       col=start_pos.col + delta_c)
        if not board.is_on_grid(next_p):
            continue
        neighbor = board.get(next_p)
        if neighbor == here:
            # 同色
            points, borders = _collect_region(next_p, board, visited)  # 递归
            all_points += points
            all_borders |= borders
        else:
            # 不同色/没有棋子
            all_borders.add(neighbor)
    return all_points, all_borders


def compute_game_result(game_state: Any) -> GameResult:
    territory = evaluate_territory(game_state.board)
    return GameResult(
        territory.num_black_territory + territory.num_black_stones,
        territory.num_white_territory + territory.num_white_stones,
        komi=7.5)
