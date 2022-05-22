from typing import Tuple, Union
from ..encoder.base import Encoder
from ..utils import is_ladder_escape, is_ladder_capture
from ..gotypes import Point, Player
from ..goboard_fast import Move
from ..agent.helper_fast import is_point_an_eye
import numpy as np
"""
Feature name   num of planes   Description
执子颜色            3           三个特征平面分别代表: 执子方(stone)/对手方(opponent stone)/空点(empty)
一                 1           全部是1
零                 1           全部是0
明智度              1           一个动作如果合法, 且不会填补当前棋手的眼, 则会在平面上填入1, 否则填入0
动作回合数          8            这个集合有8个二元平面, 代表一个落子动作离现在有多少回合
气数               8            当前动作所在棋链的气数
动作后气数           8           如果这个动作执行后, 还会剩多少口气
吃子数              8           这个动作会吃掉多少颗对方的棋子
自劫争数            8            如果这个动作执行后, 有多少自己方的棋子会陷入劫争, 可能下一回合被对方提走
征子提子            1            这颗棋子是否会通过征子吃掉
引征               1            这颗棋子是否能够逃出一个可能的征子局面
当前执子方          1            如果当前执子方为黑子, 则平面填入1; 如果是白子则填入0
"""
"""
1. 策略网络所用的棋盘拜纳姆有48个特征平面; 价值网络有49个特征平面
2. 这48个特征平面有11种概念, AlphaGo更多的利用了围棋专有的定式, 最典型的就是引入了`征子`和`引征`
3. 棋盘编码器都采用了二元特征(binary features), 例如在捕获气的概念上, 不是只用一个特征平面来表示棋盘上
   每颗子的气数, 而是用多个二元特征平面来表示一颗子有1口气, 2口气还是3口气等
4. AlphaGo所有特征都是针对下一回合执子方的. 例如在特征集`吃子数`用来记录一个动作能吃掉的棋子数目, 只记录当前
   执子方能吃掉的棋子数量, 不论它是黑方还是白方
"""

FEATURE_OFFSETS = {
    "stone_color": 0,
    "ones": 3,
    "zeros": 4,
    "sensibleness": 5,
    "turns_since": 6,
    "liberties": 14,
    "liberties_after": 22,
    "capture_size": 30,
    "self_atari_size": 38,
    "ladder_capture": 46,
    "ladder_escape": 47,
    "current_player_color": 48
}


def offset(feature: str) -> int:
    return FEATURE_OFFSETS[feature]


class AlphaGoEncoder(Encoder):

    def __init__(self,
                 board_size: Tuple[int, int] = (19, 19),
                 use_player_plane: bool = True) -> None:
        """
        参数:
        board_size: 棋盘尺寸
        use_player_plane: 训练价值网络的时候使用第49个平面
        """
        if isinstance(board_size, int):
            board_size = (board_size, board_size)
        self.board_width, self.board_height = board_size
        self.use_player_plane = use_player_plane
        self.num_planes = 48 + use_player_plane  # 48/49

    def name(self) -> str:
        return 'alphago'

    def encode(self, game_state):
        """
        TODO
        """
        board_tensor = np.zeros(
            (self.num_planes, self.board_height, self.board_width))
        for r in range(self.board_height):
            for c in range(self.board_width):
                point = Point(row=r + 1, col=c + 1)

                go_string = game_state.board.get_go_string(point)
                if go_string and go_string.color == game_state.next_player:
                    board_tensor[offset("stone_color")][r][c] = 1
                elif go_string and go_string.color == game_state.next_player.other:
                    board_tensor[offset("stone_color") + 1][r][c] = 1
                else:
                    board_tensor[offset("stone_color") + 2][r][c] = 1

                board_tensor[offset("ones")] = self.ones()
                board_tensor[offset("zeros")] = self.zeros()

                if not is_point_an_eye(game_state.board, point,
                                       game_state.next_player):
                    board_tensor[offset("sensibleness")][r][c] = 1

                ages = min(game_state.board.move_ages.get(r, c), 8)
                if ages > 0:
                    board_tensor[offset("turns_since") + ages][r][c] = 1

                if game_state.board.get_go_string(point):
                    liberties = min(
                        game_state.board.get_go_string(point).num_liberties, 8)
                    board_tensor[offset("liberties") + liberties][r][c] = 1

                move = Move(point)
                if game_state.is_valid_move(move):
                    new_state = game_state.apply_move(move)
                    liberties = min(
                        new_state.board.get_go_string(point).num_liberties, 8)
                    board_tensor[offset("liberties_after") +
                                 liberties][r][c] = 1

                    adjacent_strings = [
                        game_state.board.get_go_string(nb)
                        for nb in point.neighbors()
                    ]
                    capture_count = 0
                    for go_string in adjacent_strings:
                        other_player = game_state.next_player.other
                        if go_string and go_string.num_liberties == 1 and go_string.color == other_player:
                            capture_count += len(go_string.stones)
                    capture_count = min(capture_count, 8)
                    board_tensor[offset("capture_size") +
                                 capture_count][r][c] = 1

                if go_string and go_string.num_liberties == 1:
                    go_string = game_state.board.get_go_string(point)
                    if go_string:
                        num_atari_stones = min(len(go_string.stones), 8)
                        board_tensor[offset("self_atari_size") +
                                     num_atari_stones][r][c] = 1

                if is_ladder_capture(game_state, point):
                    board_tensor[offset("ladder_capture")][r][c] = 1

                if is_ladder_escape(game_state, point):
                    board_tensor[offset("ladder_escape")][r][c] = 1

                if self.use_player_plane:
                    if game_state.next_player == Player.black:
                        board_tensor[offset("ones")] = self.ones()
                    else:
                        board_tensor[offset("zeros")] = self.zeros()

        return board_tensor

    def ones(self) -> np.ndarray:
        """
        返回全是1的平面
        [1, board_height, board_width]
        """
        return np.ones((1, self.board_height, self.board_width))

    def zeros(self) -> np.ndarray:
        """
        返回全是0的平面
        [1, board_height, board_width]
        """
        return np.zeros((1, self.board_height, self.board_width))

    def capture_size(self, game_state, num_planes=8):
        pass

    def encode_point(self, point: Point) -> int:
        """
        Point -> index
        """
        return self.board_width * (point.row - 1) + (point.col - 1)

    def decode_point_index(self, index: int) -> Point:
        """
        index -> Point
        """
        row = index // self.board_width
        col = index % self.board_width
        return Point(row=row + 1, col=col + 1)

    def num_points(self) -> int:
        return self.board_width * self.board_height

    def shape(self) -> Tuple[int, int, int]:
        return self.num_planes, self.board_height, self.board_width


def create(board_size: Union[Tuple[int, int], int]) -> AlphaGoEncoder:
    """
    创建AlphaGoEncoder
    """
    return AlphaGoEncoder(board_size)