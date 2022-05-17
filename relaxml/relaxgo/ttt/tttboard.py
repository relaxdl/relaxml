import copy
from typing import Any, List, Union
from .ttttypes import Player, Point

__all__ = [
    'Board',
    'GameState',
    'Move',
]


class IllegalMoveError(Exception):
    pass


# 左上角: Point(1, 1)
# 右下角: Point(3, 3)
BOARD_SIZE = 3
ROWS = tuple(range(1, BOARD_SIZE + 1))
COLS = tuple(range(1, BOARD_SIZE + 1))

# 左上角->右下角
DIAG_1 = (Point(1, 1), Point(2, 2), Point(3, 3))
# 右上角 -> 左下角
DIAG_2 = (Point(1, 3), Point(2, 2), Point(3, 1))


class Move:

    def __init__(self, point: Point) -> None:
        self.point = point


class Board:
    """
    棋盘
    """

    def __init__(self) -> None:
        self._grid = {}  # Dict[Point, Player]

    def place(self, player: Player, point: Point) -> None:
        """
        落子
        """
        assert self.is_on_grid(point)
        assert self._grid.get(point) is None
        self._grid[point] = player

    @staticmethod
    def is_on_grid(point: Point) -> bool:
        """
        检查point是否在棋盘内
        """
        return 1 <= point.row <= BOARD_SIZE and \
            1 <= point.col <= BOARD_SIZE

    def get(self, point: Point) -> Union[Player, None]:
        """
        返回point位置的player
        """
        return self._grid.get(point)


class GameState:

    def __init__(self, board: Board, next_player: Player, move: Move) -> None:
        self.board = board
        self.next_player = next_player  # 下一回合的执子方
        self.last_move = move  # 上一步动作, 刚创建的游戏时, 这个字段为None

    def apply_move(self, move: Move) -> Any:
        """
        执行move, 返回新的GameState对象
        """
        next_board = copy.deepcopy(self.board)
        next_board.place(self.next_player, move.point)
        return GameState(next_board, self.next_player.other, move)

    @classmethod
    def new_game(cls) -> Any:
        """
        创建一盘游戏, 返回初始的GameState对象
        """
        board = Board()
        return GameState(board, Player.x, None)

    def is_valid_move(self, move: Move) -> bool:
        """
        判断动作是否合法
        """
        return (self.board.get(move.point) is None and not self.is_over())

    def legal_moves(self) -> List[Move]:
        """
        返回当前合法的moves
        """
        moves = []
        for row in ROWS:
            for col in COLS:
                move = Move(Point(row, col))
                if self.is_valid_move(move):
                    moves.append(move)
        return moves

    def is_over(self) -> bool:
        """
        判断棋局是否结束
        """
        if self._has_3_in_a_row(Player.x):
            return True
        if self._has_3_in_a_row(Player.o):
            return True

        if all(
                self.board.get(Point(row, col)) is not None for row in ROWS
                for col in COLS):
            # 棋盘都下满了但是没有获胜方(平局)
            return True
        return False

    def _has_3_in_a_row(self, player: Player) -> bool:
        """
        判断player是否有3连
        """
        # 竖着判断列
        for col in COLS:
            if all(self.board.get(Point(row, col)) == player for row in ROWS):
                return True
        # 横着判断行
        for row in ROWS:
            if all(self.board.get(Point(row, col)) == player for col in COLS):
                return True
        # 左上角到右下角
        if self.board.get(Point(1, 1)) == player and \
                self.board.get(Point(2, 2)) == player and \
                self.board.get(Point(3, 3)) == player:
            return True
        # 右下角到左上角
        if self.board.get(Point(1, 3)) == player and \
                self.board.get(Point(2, 2)) == player and \
                self.board.get(Point(3, 1)) == player:
            return True
        return False

    def winner(self) -> Union[Player, None]:
        """
        返回游戏获胜一方
        """
        if self._has_3_in_a_row(Player.x):
            return Player.x
        if self._has_3_in_a_row(Player.o):
            return Player.o
        return None
