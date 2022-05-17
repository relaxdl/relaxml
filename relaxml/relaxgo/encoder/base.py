# -*- coding:utf-8 -*-
from typing import Any, Union, Tuple
import importlib

__all__ = [
    'Encoder',
    'get_encoder_by_name',
]
"""
棋局当前的状态编码成训练特征, 而棋盘的下一个动作, 作为训练标签使用
"""


class Encoder:

    def name(self) -> str:
        """
        编码器的名字(可以输出到日志或者存储下来)
        """
        raise NotImplementedError()

    def encode(self, game_state):
        """
        编码棋盘数据
        """
        raise NotImplementedError()

    def encode_point(self, point: Any) -> int:
        """
        将围棋棋盘上的交叉点(point)转换为整数索引

        point -> index
        """
        raise NotImplementedError()

    def decode_point_index(self, index: int) -> Any:
        """
        将整数索引转换为围棋棋盘上的交叉点(point)

        index -> point
        """
        raise NotImplementedError()

    def num_points(self) -> int:
        """
        棋盘上交叉点的总数, 也就是棋盘的宽度x高度
        """
        raise NotImplementedError()

    def shape(self) -> Tuple:
        """
        棋盘编码后的shape
        """
        raise NotImplementedError()


def get_encoder_by_name(name, board_size: Union[Tuple[int, int],
                                                int]) -> Encoder:
    """
    根据名字来创建编码器实例
    """
    if isinstance(board_size, int):
        board_size = (board_size, board_size)
    module = importlib.import_module('relaxgo.encoder.' + name)

    # 每个编码器实现文件中必须提供一个create()函数来创建新实例
    constructor = getattr(module, 'create')
    return constructor(board_size)