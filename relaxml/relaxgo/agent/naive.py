# -*- coding:utf-8 -*-
import random
from .base import Agent
from .helpers import is_point_an_eye
from ..goboard_slow import Move, GameState
from ..gotypes import Point

__all__ = ['RandomBot']


class RandomBot(Agent):

    def select_move(self, game_state: GameState) -> Move:
        """
        随机的选择一个合法的动作, 只要不填上自己的眼就可以了

        如果找不到合法动作, 就会跳过回合
        """
        candidates = []  # 候选动作
        for r in range(1, game_state.board.num_rows + 1):
            for c in range(1, game_state.board.num_cols + 1):
                candidate = Point(row=r, col=c)
                if game_state.is_valid_move(Move.play(candidate)) and \
                        not is_point_an_eye(game_state.board,
                                            candidate,
                                            game_state.next_player):
                    candidates.append(candidate)
        if not candidates:
            return Move.pass_turn()
        return Move.play(random.choice(candidates))
