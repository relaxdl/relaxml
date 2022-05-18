from typing import Any, Union, Tuple
import numpy as np
from ..encoder.base import Encoder
from ..goboard import Move
from ..gotypes import Player, Point


"""
如果没有见过以往落子序列的情况,仅仅根据当前棋盘的状态是无法判断`劫争`的. 如果我们使用
`OnePlaneEncoder`来编码棋盘, 只使用一个平面, 将黑子编码为-1, 白子编码为1, 空白编码为0. 
是无法判断`劫争`的情况的.

SimpleEncoder使用了11个平面, 除了对`劫争`进行显式的编码, 还对棋子剩余的气数进行了建模和编码,
并区分黑子和白子. 只剩下一口气的棋子由于下一回合就可能被吃掉, 因此具有额外的战术意义. 由于新的模型
可以直接`看到`这个属性, 就能更容易的了解它对棋局的影响. 为劫争和气数单独创造特征平面, 实际上相当于
给模型增加了提示, 强调了这些概念的重要性
"""

class SimpleEncoder(Encoder):
    """
    0 - 剩1口气的黑棋编码为1, 其它棋子编码为0
    1 - 有2口气的黑棋编码为1, 其它棋子编码为0
    2 - 有3口气的黑棋编码为1, 其它棋子编码为0
    3 - 有4口气(及以上)的黑棋编码为1, 其它棋子编码为0
    4 - 剩1口气的白棋编码为1, 其它棋子编码为0
    5 - 有2口气的白棋编码为1, 其它棋子编码为0
    6 - 有3口气的白棋编码为1, 其它棋子编码为0
    7 - 有4口气(及以上)的白棋编码为1, 其它棋子编码为0
    8 - 如果是黑方回合, 这个平面设置为1
    9 - 如果是白方回合, 这个平面设置为1
    10 - 由于劫争而不能落子的点
    """

    def __init__(self, board_size: Union[Tuple[int, int], int]) -> None:
        self.board_width, self.board_height = board_size
        self.num_planes = 11

    def name(self) -> str:
        return 'simple'

    def encode(self, game_state: Any) -> np.ndarray:
        """
        参数:
        game_state: 游戏状态

        返回:
        board_matrix: [num_planes=11, board_height, board_width]
        """
        board_tensor = np.zeros(self.shape())
        if game_state.next_player == Player.black:
            # 8 - 如果是黑方回合, 这个平面设置为1
            board_tensor[8] = 1
        else:
            # 9 - 如果是白方回合, 这个平面设置为1
            board_tensor[9] = 1
        # 遍历棋盘的每个位置
        for r in range(self.board_height):
            for c in range(self.board_width):
                p = Point(row=r + 1, col=c + 1)
                go_string = game_state.board.get_go_string(p)

                if go_string is None:
                    # 10 - 由于劫争而不能落子的点
                    if game_state.does_move_violate_ko(game_state.next_player,
                                                       Move.play(p)):
                        board_tensor[10][r][c] = 1
                else:
                    liberty_plane = min(4, go_string.num_liberties) - 1
                    if go_string.color == Player.white:
                        liberty_plane += 4
                    board_tensor[liberty_plane][r][c] = 1

        return board_tensor

    def encode_point(self, point: Point) -> int:
        """
        将围棋棋盘上的交叉点(point)转换为整数索引

        point -> index
        """
        return self.board_width * (point.row - 1) + (point.col - 1)

    def decode_point_index(self, index: int) -> Point:
        """
        将整数索引转换为围棋棋盘上的交叉点(point)

        index -> point
        """
        row = index // self.board_width
        col = index % self.board_width
        return Point(row=row + 1, col=col + 1)

    def num_points(self) -> int:
        """
        棋盘上交叉点的总数, 也就是棋盘的宽度x高度
        """
        return self.board_width * self.board_height

    def shape(self) -> Tuple[int, int, int]:
        """
        棋盘编码后的shape

        返回:
        [num_planes=11, board_height, board_width]
        """
        return self.num_planes, self.board_height, self.board_width


def create(board_size: Union[Tuple[int, int], int]) -> SimpleEncoder:
    """
    创建SimpleEncoder
    """
    return SimpleEncoder(board_size)