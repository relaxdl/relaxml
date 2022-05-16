# -*- coding:utf-8 -*-
from xmlrpc.client import Boolean
from ..gotypes import Point, Player

__all__ = [
    'is_point_an_eye',
]


def is_point_an_eye(board, point: Point, color: Player) -> Boolean:
    """
    判断棋盘上的某个Point是否为`眼`

    在机器人自我对弈(self play)中, 我们要保证自我对弈的棋局要正常结束. 在人对人的棋局中
    如果双方都无法通过下一步动作获得更多的优势, 一局比赛就结束了. 即使对人类来说这也是一个
    很复杂的概念, 初学者往往在终盘阶段还无所适从的在对方地盘里落子, 或者眼睁睁看着对方侵入
    自己认为已经稳固的地盘. 而对计算机来说这就更加困难, 如果机器人的逻辑是只要还有合法的动
    作就一直继续下去的话, 那么最终它只会把自己的气都填满, 从而丢掉所有棋子

    我们可以想出几个启发式的规则来帮助机器人更合理的结束棋局, 例如:
    a. 不要在完全被同色棋子所包围的区域落子
    b. 不要选择落子后会导致只剩一口气的动作
    c. 如果对方棋子只剩一口气, 总是吃掉它
    不幸的是, 这几条规则过于严格了, 如果机器人遵循这几条规则, 那么对手就可以利用这些弱点吃
    掉本来可以救活的大龙, 或者拯救快要因气尽而被提走的棋链等. 总的来说我们制定的规则应该`尽
    可能的减少机器人选择空间的限制`, 以便于未来能够用更复杂的算法学习高级别的策略

    要解决这个问题, 可以参考围棋发展的历史, 在古代规则很简单: 棋盘上棋子多得一方就是获胜方. 
    双方都会尽量填满所有可以填满的空点, 只留下眼. 这样会让棋局终盘阶段拖很长时间, 因此人们慢
    慢想出了加速的办法: 如果黑方明显控制棋盘的一块区域(若白方在这个区域落子, 最终肯定会被吃掉),
    那么就不需要黑子填满这个区域, 而只要双方都同意将该区域视为黑方地盘即可, 这就是`地盘`概念的
    来源. 随着时代发展和规则的演变, 最终变成了明确的终盘统计指标
    
    这种评分规则避免了判断哪里是谁的地盘的问题, 但是我们还得防止机器人`自吃`. 我们需要增加一条
    规则, `禁止机器人自己填补自己的眼`, 而且要用最严格的定义, 这里眼的定义是: 一个空点, 它所有
    相邻交叉点以及四个对角相邻点中有3个以上都是己方的棋子(这样定义眼的话, 在某些情况下可能会错过
    有效的眼, 不过为了保持逻辑实现的简单, 我们暂时接受这个错误)

    下面是具体的实现规则(接近`真眼`的定义):
    1. 眼必须是一个空点
    2. 所有相邻的点必须是己方棋子
    3. 如果这个空点位于棋盘内部, 己方棋子至少得控制4个对角相邻点中的3个; 如果空点在边缘, 
       则必须控制所有的对角相邻点
    """
    # 眼必须是一个空点
    if board.get(point) is not None:
        return False

    # 所有相邻的点必须是己方棋子
    for neighbor in point.neighbors():
        if board.is_on_grid(neighbor):
            neighbor_color = board.get(neighbor)
            if neighbor_color != color:
                return False

    # 如果这个空点位于棋盘内部, 己方棋子至少得控制4个对角相邻点中的3个
    # 如果空点在边缘, 则必须控制所有的对角相邻点
    friendly_corners = 0
    off_board_corners = 0
    corners = [
        Point(point.row - 1, point.col - 1),
        Point(point.row - 1, point.col + 1),
        Point(point.row + 1, point.col - 1),
        Point(point.row + 1, point.col + 1),
    ]
    for corner in corners:
        if board.is_on_grid(corner):
            corner_color = board.get(corner)
            if corner_color == color:
                friendly_corners += 1
        else:
            off_board_corners += 1
    # 空点在边缘或者角落
    if off_board_corners > 0:
        return off_board_corners + friendly_corners == 4

    # 空点在棋盘内部
    return friendly_corners >= 3
