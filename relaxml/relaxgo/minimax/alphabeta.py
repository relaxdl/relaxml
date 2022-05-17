# -*- coding:utf-8 -*-
import random
from typing import Any, Callable
from ..gotypes import Player, Point
from ..agent.base import Agent

__all__ = [
    'AlphaBetaAgent',
]

MAX_SCORE = 999999
MIN_SCORE = -999999


def capture_diff(game_state: Any) -> int:
    """
    站在下一回合的执子方, 也就是game_state.next_player的角度, 评估当前game_state的函数(eval_fn)
    计算棋盘上黑子和白子的数量差, 这和计算双方提子数量差是一致的, 除非某一方提前跳过回合

    返回:
    diff: 提子的数量差
    """
    black_stones = 0
    white_stones = 0
    for r in range(1, game_state.board.num_rows + 1):
        for c in range(1, game_state.board.num_cols + 1):
            p = Point(row=r, col=c)
            color = game_state.board.get(p)
            if color == Player.black:
                black_stones += 1
            elif color == Player.white:
                white_stones += 1
    # 如果是黑方落子的回合, 那么返回`黑子数量-白子数量`
    # 如果是白子落子的回合, 那么返回`白子数量-黑子数量`
    diff = black_stones - white_stones
    if game_state.next_player == Player.black:
        return diff
    return -1 * diff


def alpha_beta_result(game_state: Any, max_depth: int, best_black: int,
                      best_white: int, eval_fn: Callable) -> int:
    """
    对于下一回合的执子方, 也就是game_state.next_player来说最好的结果

    alpha-beta剪枝: 在搜索的过程中跟踪记录到目前为止双方的最佳结果, 将这两个结果值称为alpha和beta
    alpha-beta剪枝可以`减少搜索宽度`, 一旦发现能让对方变的更坏的分支就直接返回, 不用再继续搜索同一层
    的其它分支了

    game_state.next_player=黑, 需要评估黑方的动作:
    <1> 完全评估黑方的第一个可能动作, 可以得到一个黑方到目前为止的: best_black
    <2> 对黑方的第二个可能动作进行评估, 这时候, 白方有一个回应动作, 如果白方的这个回应动作可以让黑方变
        的更坏. 就可以停止遍历白方的回应动作直接返回了, 不需要继续评估其它的白方动作了. 因为我们已经知道
        黑方的这个动作比之前的动作更坏, 黑方不应该继续尝试这个分支了. 即使白方可能还有其它更好的回应, 但
        我们已经知道黑方选择这个分支不合理了

    下图: 每一条线表示一个动作, []表示一个棋局状态
                                   [黑]                               game_state.next_player=黑
                 /(黑方动作1)        |(黑方动作2)        \(黑方动作3) 
               [白]                [白]                [白]            game_state.next_player=白
          /     |    \       /      |     \       /     |     \ 
        [黑]   [黑]   [黑]   [黑]   [黑]   [黑]   [黑]   [黑]   [黑]     game_state.next_player=黑
      /  |  \   .      .
    [白][白][白] ...   ...                                             game_state.next_player=白
    """
    # 如果游戏结束了(游戏树的叶子节点), 就可以立即得到哪一方获胜
    if game_state.is_over():
        if game_state.winner() == game_state.next_player:
            return MAX_SCORE
        else:
            return MIN_SCORE

    # 已经达到最大搜索深度, 评估当前棋盘状态
    if max_depth == 0:
        return eval_fn(game_state)

    best_so_far = MIN_SCORE
    # 遍历所有合法动作
    for candidate_move in game_state.legal_moves():
        # 计算如果选择这个动作, 会导致什么游戏状态
        next_state = game_state.apply_move(candidate_move)
        # 从next_state开始, 找到对方的最佳结果
        opponent_best_result = alpha_beta_result(next_state, max_depth - 1,
                                                 best_black, best_white,
                                                 eval_fn)
        # 对方结果的reverse就是自己方结果
        our_result = -1 * opponent_best_result
        if our_result > best_so_far:
            best_so_far = our_result

        # 下面这部分代码的作用是提前停止搜索, 提前返回. 一旦发现能让对方变的更坏的
        # 分支就直接返回, 不用再继续搜索同一层的其它分支了, 不需要一直搜索到最佳评分
        #
        # (如果去掉下面这段停止搜索提前返回的代码, 就变成了DepthPrune算法)
        # --提前返回-开始--
        if game_state.next_player == Player.white:
            # 为白方选择一个动作, 如果白方可以将黑方限制在更低的分数(也就是让黑方的
            # 局势变得更坏), 就可以直接返回, 不用再继续搜索了
            if best_so_far > best_white:
                best_white = best_so_far
            outcome_for_black = -1 * best_so_far  # 白方对手(黑方)的最佳分数
            if outcome_for_black < best_black:
                return best_so_far
        elif game_state.next_player == Player.black:
            # 为黑方选择一个动作, 如果黑方可以将白方限制在更低的分数(也就是让白方的
            # 局势变得更坏), 就可以直接返回, 不用再继续搜索了
            if best_so_far > best_black:
                best_black = best_so_far
            outcome_for_white = -1 * best_so_far  # 黑方对手(白方)的最佳分数
            if outcome_for_white < best_white:
                return best_so_far
        # --提前返回-结束--
    return best_so_far


class AlphaBetaAgent(Agent):

    def __init__(self, max_depth: int, eval_fn: Callable) -> None:
        Agent.__init__(self)
        self.max_depth = max_depth
        self.eval_fn = eval_fn

    def select_move(self, game_state):
        best_moves = []
        best_score = None
        best_black = MIN_SCORE
        best_white = MIN_SCORE
        # 遍历所有合法的动作
        for possible_move in game_state.legal_moves():
            # 计算如果选择这个动作, 会导致什么游戏状态
            next_state = game_state.apply_move(possible_move)
            # 由于下一回合对方执子, 需要找到对方可能获得的最佳结果
            opponent_best_outcome = alpha_beta_result(next_state,
                                                      self.max_depth,
                                                      best_black, best_white,
                                                      self.eval_fn)
             # 对方结果的reverse就是自己方结果
            our_best_outcome = -1 * opponent_best_outcome
            if (not best_moves) or our_best_outcome > best_score:
                # 找到了更好的best_score
                best_moves = [possible_move]
                best_score = our_best_outcome

                # 更新best_black或者best_white
                if game_state.next_player == Player.black:
                    best_black = best_score
                elif game_state.next_player == Player.white:
                    best_white = best_score
            elif our_best_outcome == best_score:
                # 找到了同分数的best_score
                best_moves.append(possible_move)
        return random.choice(best_moves)