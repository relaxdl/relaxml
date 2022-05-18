from typing import Any, Union, Tuple
import numpy as np
from ..encoder.base import Encoder
from ..goboard import Move, Point


class SevenPlaneEncoder(Encoder):
    """
    0 - 下一回合执子方仅剩1口气的棋子编码为1, 其它棋子编码为0
    1 - 下一回合执子方有2口气的棋子编码为1, 其它棋子编码为0
    2 - 下一回合执子方有3口气(及以上)的棋子编码为1, 其它棋子编码为0
    3 - 对手仅剩1口气的棋子编码为1, 其它棋子编码为0
    4 - 对手有2口气的棋子编码为1, 其它棋子编码为0
    5 - 对手有3口气(及以上)的棋子编码为1, 其它棋子编码为0
    6 - 由于劫争而不能落子的点
    """

    def __init__(self, board_size: Union[Tuple[int, int], int]) -> None:
        self.board_width, self.board_height = board_size
        self.num_planes = 7

    def name(self) -> str:
        return 'sevenplane'

    def encode(self, game_state: Any) -> np.ndarray:
        """
        参数:
        game_state: 游戏状态

        返回:
        board_matrix: [num_planes=7, board_height, board_width]
        """
        board_tensor = np.zeros(self.shape())
        base_plane = {
            game_state.next_player: 0,  # 下一回合执子方: 0,1,2
            game_state.next_player.other: 3  # 对手: 3,4,5
        }
        # 遍历棋盘的每个位置
        for row in range(self.board_height):
            for col in range(self.board_width):
                p = Point(row=row + 1, col=col + 1)
                go_string = game_state.board.get_go_string(p)
                if go_string is None:
                    # 6 - 由于劫争而不能落子的点
                    if game_state.does_move_violate_ko(game_state.next_player,
                                                       Move.play(p)):
                        board_tensor[6][row][col] = 1
                else:
                    liberty_plane = min(3, go_string.num_liberties) - 1
                    liberty_plane += base_plane[go_string.color]
                    board_tensor[liberty_plane][row][col] = 1
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
        [num_planes=7, board_height, board_width]
        """
        return self.num_planes, self.board_height, self.board_width


def create(board_size: Union[Tuple[int, int], int]) -> SevenPlaneEncoder:
    """
    创建SevenPlaneEncoder
    """
    return SevenPlaneEncoder(board_size)
