# -*- coding:utf-8 -*-
from typing import Tuple
import numpy as np
from .helper_fast import is_point_an_eye
from ..goboard_fast import Move, GameState
from ..gotypes import Point
from .base import Agent

__all__ = ['FastRandomBot']


class FastRandomBot(Agent):

    def __init__(self):
        Agent.__init__(self)
        self.dim = None
        self.point_cache = []

    def _update_cache(self, dim: Tuple[int, int]) -> None:
        self.dim = dim
        rows, cols = dim
        self.point_cache = []
        for r in range(1, rows + 1):
            for c in range(1, cols + 1):
                self.point_cache.append(Point(row=r, col=c))

    def select_move(self, game_state: GameState) -> Move:
        """
        随机的选择一个合法的动作, 只要不填上自己的眼就可以了

        如果找不到合法动作, 就会跳过回合
        """
        dim = (game_state.board.num_rows, game_state.board.num_cols)
        if dim != self.dim:
            self._update_cache(dim)

        idx = np.arange(len(self.point_cache))
        np.random.shuffle(idx)
        for i in idx:
            p = self.point_cache[i]
            if game_state.is_valid_move(Move.play(p)) and \
                    not is_point_an_eye(game_state.board,
                                        p,
                                        game_state.next_player):
                return Move.play(p)
        return Move.pass_turn()