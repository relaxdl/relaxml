# -*- coding:utf-8 -*-
from enum import Enum
import random
from typing import Any, Callable

from ..gotypes import Player, Point
from ..agent.base import Agent

__all__ = [
    'DepthPrunedAgent',
]

MAX_SCORE = 999999
MIN_SCORE = -999999
"""
棋类的几个特点:
1. 任务需要做出一系列决策
2. 早期的决策可能会影响未来的决策
3. 在动作序列结束时, 可以评估目标完成的程度
4. 决策序列的可能组合数量可能非常庞大

搜索树是一类搜索策略, 它们循环遍历许多可能的决策序列, 并找到那些可能产生最佳结果的决策
要在复杂的棋类游戏中采用树搜索, 需要一种策略来消除树的部分, 这种寻找树中可以忽略部分的
过程称为`剪枝`

游戏树是二维的, 它具有`宽度`和`深度`. 宽度是指某个给定棋局下可能动作的数量. 而深度是指
某个棋局到可能的最终游戏状态的回合数. 对于某个特定棋类游戏, 我们通常会考虑它的典型宽度和
典型深度, 并以此来估计游戏树的尺寸. 游戏树中棋局的数量大致由公式W^d给出(W的d次方), 其中
W是平均宽度, d是平均深度. 例如国际象棋中, 每一回合玩家通常有大约30种选择, 并且一局棋需要
大约80回合结束, 因此游戏树的尺寸可以估算为30^80大约为10^118. 而通常围棋每一回合有250种
合法动作, 一场比赛大约需要150回合, 这样游戏树的尺寸是250^150大约为10^359

W^d这个公式是呈指数增长的一个例子, 当增加搜索深度时, 需要考虑的棋局数会迅速增加

我们假设一下平均游戏宽度和平均游戏深度大约为10的游戏, 可供搜索的游戏树包含10^10(大约100亿)
个棋局. 如果我们有办法在每个回合中快速忽略两个可能的动作, 将搜索树的宽度减少为8, 其次我们
发现只需要预测9步而不是10步, 就能找到最好的游戏结果, 这样我们只需要搜索8^9(约1.3亿)个棋局了, 
与完整的搜索空间相比, 可以节省超过98%的计算量, 所以`剪枝`的关键点是`即使只微弱的缩减搜索的宽
度或者深度, 也能够大大减少动作选择需要的时间`

最常用的有两种`剪枝`技术:一种是用于`减少搜索深度的棋局评估函数`; 另一种是用于`减少搜索宽度的
alpha-beta剪枝`. 这两项技术共同构成了经典的棋盘游戏AI的支柱

这里实现的是: `减少搜索深度的棋局评估函数`

如果遍历游戏树直至结束, 就可以计算出获胜者. 但是我们如何在棋局的早期就做到这一点呢? 人类棋手
往往在盘中就对哪一方领先有所感知,如果可以让计算机程序捕获到这种感觉, 就能够大大减少搜索所需要的
深度. 用来模仿这种感觉, 去判断哪一方领先以及领先多少的函数, 就叫做`棋局评估函数`.

在许多棋类游戏中, 我们可以利用对规则的了解来手工制作棋局评估函数, 下面是两个高度简化的评估函数:
* 西洋跳棋: 棋盘上每个常规棋子算作1分, 再加上每个国王算作2分, 计算己方棋子的总分, 并减去对方的分数
* 国际象棋: 每个兵计1分, 每个象或者马计3分, 每个车计5分, 皇后计9分. 计算己方棋子的总分, 并减去对方
           的分数

在围棋中, 可以做出一个与这两种棋相似的启发式规则: 将吃掉的棋子相加, 然后减掉对方吃掉棋子的数量. 等效
的, 也可以计算棋盘上留存棋子的数量差(在实际中, 这个启发式的规则并不是一个有效的评估函数, 在围棋中吃子
的威胁往往比实际提子更为重要, 一盘棋持续到一百多个回合才第一次提子的情况也是很常见的). 
"""


def capture_diff(game_state) -> int:
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


def best_result(game_state: Any, max_depth: int, eval_fn: Callable) -> int:
    """
    对于下一回合的执子方, 也就是game_state.next_player来说最好的结果

    这里的实现和MiniMax算法很相似, 但是有如下区别:
    1. MiniMax算法会评估到游戏树的叶子节点, DepthPrune算法评估max_depth
       层就返回.
    2. MiniMax算法返回的是一个表示获胜, 失败或者平局的枚举值, 而这里返回的是
       一个数字, 用来表示评估函数的值. 传统上, `我们从下一回合的执子方视角来计
       算得分`, 得分越高意味着下一回合执子方更有希望获胜, 当我们从对方的视角来
       评估棋局之后, 只要将得分乘以-1, 就可以算回自己的视角



    递归调用`best_result`就是`绘制游戏树`的过程:
    黑方回合:                             [状态11]
                             /             |
    白方回合:            [状态21]         [状态22]   ...
                   /      |          /     |
    黑方回合:  [状态31]  [状态32]  [状态33] [状态34]   ...

    参数:
    game_state: 游戏状态
    max_depth: 参数决定要提前搜索的步数(层数), 每经过一个回合, 这个值减1
               max_depth=0时, 就可以停止搜索, 并调用棋局评估函数
    eval_fn: 游戏状态的评估函数

    返回:
    score: 当前游戏状态的分数
    """
    # 如果游戏结束了(游戏树的叶子节点), 就可以立即得到哪一方获胜
    if game_state.is_over():
        # 游戏结束
        if game_state.winner() == game_state.next_player:
            return MAX_SCORE
        else:
            return MIN_SCORE

    # 已经达到最大搜索深度, 评估当前棋盘状态
    if max_depth == 0:
        return eval_fn(game_state)

    best_so_far = MIN_SCORE
    # 遍历所有合法的动作
    for candidate_move in game_state.legal_moves():
        # 计算如果选择这个动作, 会导致什么游戏状态
        next_state = game_state.apply_move(candidate_move)
        # 从next_state开始, 找到对方的最佳结果
        opponent_best_result = best_result(next_state, max_depth - 1,
                                           eval_fn)  # 递归调用
        # 对方结果的reverse就是自己方结果
        out_result = -1 * opponent_best_result
        if out_result > best_so_far:
            best_so_far = out_result
    return best_so_far


class DepthPrunedAgent(Agent):

    def __init__(self, max_depth: int, eval_fn: Callable) -> None:
        Agent.__init__(self)
        self.max_depth = max_depth
        self.eval_fn = eval_fn

    def select_move(self, game_state):
        best_moves = []
        best_score = None
        # 遍历所有合法的动作
        for possible_move in game_state.legal_moves():
            # 计算如果选择这个动作, 会导致什么游戏状态
            next_state = game_state.apply_move(possible_move)
            # 由于下一回合对方执子, 需要找到对方可能获得的最佳结果
            opponent_best_outcome = best_result(next_state, self.max_depth,
                                                self.eval_fn)
            # 对方结果的reverse就是自己方结果
            our_best_outcome = -1 * opponent_best_outcome

            if (not best_moves) or our_best_outcome > best_score:
                # 找到了更好的best_score
                best_moves = [possible_move]
                best_score = our_best_outcome
            elif our_best_outcome == best_score:
                # 找到了同分数的best_score
                best_moves.append(possible_move)

        # 挑选最佳结果
        return random.choice(best_moves)