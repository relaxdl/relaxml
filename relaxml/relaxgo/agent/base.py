# -*- coding:utf-8 -*-

__all__ = [
    'Agent',
]


class Agent:
    """
    机器人Agent, Agent只需要一个核心功能, 告诉它当前的game_state, 
    它可以返回当前应该做的动作
    """

    def __init__(self):
        pass

    def select_move(self, game_state):
        """
        根据当前的游戏状态选择一个动作

        参数:
        game_state: GameState

        返回:
        move: Move
        """
        raise NotImplementedError()

    def diagnostics(self):
        return {}
