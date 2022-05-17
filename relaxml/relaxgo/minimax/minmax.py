# -*- coding:utf-8 -*-
from enum import Enum
import random
from typing import Any

from ..agent.base import Agent

__all__ = [
    'MinimaxAgent',
]
"""
游戏树: 一个棋局指向多个可能的后续棋局, 这种结构称为游戏树

极大极小搜索算法:
每一回合, `它会在双方间切换视角`, 我们期望自己的得分最大化, 
对手则希望我们的得分最小化(假设对手和你一样聪明)
"""


class GameResult(Enum):
    LOSS = 1
    DRAW = 2  # 平局
    WIN = 3


def reverse_game_result(game_result: GameResult) -> GameResult:
    if game_result == GameResult.LOSS:
        return game_result.WIN
    if game_result == GameResult.WIN:
        return game_result.LOSS
    return GameResult.DRAW


def best_result(game_state: Any) -> GameResult:
    """
    评估当前的game_state, 计算出对于game_state.next_player,
    也就是下一回合的执子方来说最好的结果

    递归调用`best_result`就是`绘制游戏树`的过程:
    X回合:                         [状态11]
                         /           |
    O回合:            [状态21]      [状态22]   ...
                /       |       /     |
    X回合: [状态31] [状态32] [状态33] [状态34]   ...
    """
    if game_state.is_over():
        # 游戏结束(游戏树的叶子节点)
        if game_state.winner() == game_state.next_player:
            return GameResult.WIN
        elif game_state.winner() is None:
            return GameResult.DRAW
        else:
            return GameResult.LOSS
    # 1. 首先循环遍历所有可能动作, 并计算下一个游戏状态
    # 2. 接着假设`对方会尽力反击的假象动作`. 对这个新棋局调用best_result, 得到对方
    #    从这个新棋局所能够获得的最佳结果, 对方结果的reverse就是自己方结果
    # 3. 最后在遍历完所有动作之后, 选择能给己方带来最佳结果的那个动作
    best_result_so_far = GameResult.LOSS
    # 遍历所有合法的动作
    for candidate_move in game_state.legal_moves():
        # 计算如果选择这个动作, 会导致什么游戏状态
        next_state = game_state.apply_move(candidate_move)
        # 找到对方的最佳结果
        opponent_best_result = best_result(next_state)  # 递归调用
        # 对方结果的reverse就是自己方结果
        our_result = reverse_game_result(opponent_best_result)
        if our_result.value > best_result_so_far.value:
            best_result_so_far = our_result
    return best_result_so_far


class MiniMaxAgent(Agent):

    def select_move(self, game_state):
        winning_moves = []
        draw_moves = []
        losing_moves = []
        # 遍历所有合法的动作
        for possible_move in game_state.legal_moves():
            # 计算如果选择这个动作, 会导致什么游戏状态
            next_state = game_state.apply_move(possible_move)
            # 由于下一回合对方执子, 需要找到对方可能获得的最佳结果
            opponent_best_outcome = best_result(next_state)
            # 对方结果的reverse就是自己方结果
            our_best_outcome = reverse_game_result(opponent_best_outcome)

            # 根据动作导致的最终结果为动作分类
            if our_best_outcome == GameResult.WIN:
                winning_moves.append(possible_move)
            elif our_best_outcome == GameResult.DRAW:
                draw_moves.append(possible_move)
            else:
                losing_moves.append(possible_move)

        # 挑选最佳结果
        if winning_moves:
            return random.choice(winning_moves)
        if draw_moves:
            return random.choice(draw_moves)
        return random.choice(losing_moves)
