# -*- coding:utf-8 -*-
from typing import Any, Dict, List, Tuple, Union
import copy
from .gotypes import Player, Point
from .utils import MoveAge
from . import zobrist
from .scoring import compute_game_result

__all__ = ['Move', 'GoString', 'Board', 'GameState']

neighbor_tables = {}
corner_tables = {}


def init_neighbor_table(
        dim: Tuple[int,
                   int]) -> Dict[Tuple[int, int], Dict[Point, List[Point]]]:
    """
    初始化全局的neighbor_table
    """
    rows, cols = dim
    new_table = {}
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            p = Point(row=r, col=c)
            full_neighbors = p.neighbors()
            true_neighbors = [
                n for n in full_neighbors
                if 1 <= n.row <= rows and 1 <= n.col <= cols
            ]
            new_table[p] = true_neighbors
    neighbor_tables[dim] = new_table


def init_corner_table(
        dim: Tuple[int,
                   int]) -> Dict[Tuple[int, int], Dict[Point, List[Point]]]:
    """
    初始化全局的corner_table
    """
    rows, cols = dim
    new_table = {}
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            p = Point(row=r, col=c)
            full_neighbors = [
                Point(row=p.row - 1, col=p.col - 1),
                Point(row=p.row - 1, col=p.col + 1),
                Point(row=p.row + 1, col=p.col - 1),
                Point(row=p.row + 1, col=p.col + 1)
            ]
            true_neighbors = [
                n for n in full_neighbors
                if 1 <= n.row <= rows and 1 <= n.col <= cols
            ]
            new_table[p] = true_neighbors
    corner_tables[dim] = new_table


class IllegalMoveError(Exception):
    pass


class Move:
    """
    表示一个回合中可能采取的动作. 有三种动作:
    1. 在棋盘上落下一颗棋子(play)
    2. 跳过回合(pass)
    3. 认输(resign)

    遵循美国围棋协会(AGA)的惯例, 我们使用术语动作(move)来表示这三种行动中的一个. 
    在实际棋局中, 需要传递一个Point对象指定落子的位置

    在使用中我们通常调用:Move.play(), Move.pass_turn(), Move.resign()来构造
    一个动作, 而不是直接调用Move的构造函数
    """

    def __init__(self,
                 point: Point = None,
                 is_pass=False,
                 is_resign=False) -> None:
        self.point = point
        self.is_play = (point is not None)
        self.is_pass = is_pass
        self.is_resign = is_resign

    @classmethod
    def play(cls, point: Point):
        """
        在棋盘上落下一颗棋子
        """
        return Move(point)

    @classmethod
    def pass_turn(cls):
        """
        跳过回合
        """
        return Move(is_pass=True)

    @classmethod
    def resign(cls):
        """
        认输
        """
        return Move(is_resign=True)

    def __str__(self):
        if self.is_pass:
            return 'pass'
        if self.is_resign:
            return 'resign'
        return f'(r {self.point.row}, c {self.point.col})'

    def __hash__(self):
        return hash((self.is_play, self.is_pass, self.is_resign, self.point))

    def __eq__(self, other):
        return (self.is_play, self.is_pass, self.is_resign,
                self.point) == (other.is_play, other.is_pass, other.is_resign,
                                other.point)


class GoString:
    """
    棋链(string of stones)

    1. 独立检查棋子会增加计算的复杂度, 可以将同色的棋子组成一个整体, 同时跟踪整体的
    状态以及它们的气. 这样在实现棋盘逻辑时效率更高
    2. GoString一旦创建, 是不可变的. 我们使用frozenset来实现. frozenset没有添加
    和删除元素, 因此需要更新的时候, 不能修改现有集合, 只能创建新的集合
    """

    def __init__(self, color: Player, stones: Union[List[Point], set],
                 liberties: Union[List[Point], set]) -> None:
        """
        参数:
        color: 颜色
        stones: 棋子的Point
        liberties: 气的Point
        """
        self.color = color
        self.stones = frozenset(stones)  # 棋子的Point
        self.liberties = frozenset(liberties)  # 气的Point

    def without_liberty(self, point: Point) -> Any:
        """
        删除一口气

        当对方在这条棋链相邻的地方落子时, 棋链的气通常会减少
        """
        new_liberties = self.liberties - set([point])
        return GoString(self.color, self.stones, new_liberties)

    def with_liberty(self, point: Point) -> Any:
        """
        增加一口气

        当这条棋链或者己方其它棋链吃掉对方棋子的时候, 棋链的气会增加
        """
        new_liberties = self.liberties | set([point])
        return GoString(self.color, self.stones, new_liberties)

    def merge_with(self, go_string: Any) -> Any:
        """
        当一次落子把两颗棋子连接起来的时候, 需要merge两个GoString
        """
        assert go_string.color == self.color
        combined_stones = self.stones | go_string.stones
        return GoString(self.color, combined_stones,
                        (self.liberties | go_string.liberties) -
                        combined_stones)

    @property
    def num_liberties(self) -> int:
        """
        GoString的气数
        """
        return len(self.liberties)

    def __eq__(self, other):
        return isinstance(other, GoString) and \
            self.color == other.color and \
            self.stones == other.stones and \
            self.liberties == other.liberties

    def __deepcopy__(self, memodict={}):
        return GoString(self.color, self.stones, copy.deepcopy(self.liberties))


class Board:
    """
    棋盘

    最主要的功能是可以`落子`, 落子后自动更新棋盘的状态
    """

    def __init__(self, num_rows: int, num_cols: int) -> None:
        self.num_rows = num_rows
        self.num_cols = num_cols
        self._grid = {}  # Dict[Point, GoString] 用来跟踪棋盘的状态
        self._hash = zobrist.EMPTY_BOARD  # 使用zobrist哈希来跟踪棋盘状态

        global neighbor_tables, corner_tables
        dim = (num_rows, num_cols)
        if dim not in neighbor_tables:
            init_neighbor_table(dim)
        if dim not in corner_tables:
            init_corner_table(dim)
        self.neighbor_table = neighbor_tables[dim]
        self.corner_table = corner_tables[dim]
        self.move_ages = MoveAge(self)

    def neighbors(self, point: Point) -> List[Point]:
        """
        返回neighbors
        """
        return self.neighbor_table[point]

    def corners(self, point: Point) -> List[Point]:
        """
        返回corners
        """
        return self.corner_table[point]

    def place_stone(self, player: Player, point: Point) -> None:
        """
        落子

        1. 检测point是否在棋盘内以及point是否已经有棋子
        2. 合并任何同色且相邻的棋链
        3. 针对player & point应用哈希来更新棋盘状态
        4. 减少对方所有相邻棋链的气
        5. 如果对方某条棋链没有气了, 要提走它

        注意:
        Board落子规则并不会检测`自吃`, 自吃的检测放到GameState中, GameState中
        会禁止`自吃`
        """
        # 检测point是否在棋盘内以及point是否已经有棋子
        assert self.is_on_grid(point)
        assert self._grid.get(point) is None

        adjacent_same_color = []  # List[GoString] point周围(neighbor)同色棋链
        adjacent_opposite_color = []  # List[GoString] point周围(neighbor)对方棋链
        liberties = []  # List[Point] point周围(neighbor)的气
        self.move_ages.increment_all()
        self.move_ages.add(point)

        # 遍历point的neighbors
        for neighbor in self.neighbor_table[point]:
            neighbor_string = self._grid.get(neighbor)
            if neighbor_string is None:
                # 气
                liberties.append(neighbor)
            elif neighbor_string.color == player:
                # 同色棋链
                if neighbor_string not in adjacent_same_color:
                    adjacent_same_color.append(neighbor_string)
            else:
                # 对方棋链
                if neighbor_string not in adjacent_opposite_color:
                    adjacent_opposite_color.append(neighbor_string)
        new_string = GoString(player, [point], liberties)

        # 合并任何同色且相邻的棋链
        for same_color_string in adjacent_same_color:
            new_string = new_string.merge_with(same_color_string)
        for new_string_point in new_string.stones:
            self._grid[new_string_point] = new_string

        # 针对player & point应用哈希来更新棋盘状态
        self._hash ^= zobrist.HASH_CODE[point, None]
        self._hash ^= zobrist.HASH_CODE[point, player]

        for other_color_string in adjacent_opposite_color:
            # 减少对方所有相邻棋链的气
            replacement = other_color_string.without_liberty(point)
            if replacement.num_liberties:
                self._replace_string(replacement)
            else:
                # 如果对方某条棋链没有气了, 要提走它
                self._remove_string(other_color_string)

    def _replace_string(self, new_string: GoString) -> None:
        """
        更新围棋棋盘grid
        """
        for point in new_string.stones:
            self._grid[point] = new_string

    def _remove_string(self, string: GoString) -> None:
        """
        提走一条棋链的所有棋子

        在提走一条棋链的时候, 其它棋链会增加气数
        """
        for point in string.stones:
            self.move_ages.reset_age(point)
            for neighbor in self.neighbor_table[point]:
                neighbor_string = self._grid.get(neighbor)
                if neighbor_string is None:
                    continue
                if neighbor_string is not string:
                    self._replace_string(neighbor_string.with_liberty(point))
            self._grid[point] = None

            # 在zobrist哈希中, 需要通过逆应用这步动作的哈希值来实现提子
            self._hash ^= zobrist.HASH_CODE[point, string.color]
            self._hash ^= zobrist.HASH_CODE[point, None]

    def is_self_capture(self, player: Player, point: Point) -> bool:
        """
        判断是否是`自吃`
        """
        friendly_strings = []
        for neighbor in self.neighbor_table[point]:
            neighbor_string = self._grid.get(neighbor)
            if neighbor_string is None:
                # point是有气的, 不是`自吃`
                return False
            elif neighbor_string.color == player:
                friendly_strings.append(neighbor_string)
            else:
                if neighbor_string.num_liberties == 1:
                    # 这是吃对方的子, 并不是`自吃`
                    return False

        if all(neighbor.num_liberties == 1 for neighbor in friendly_strings):
            return True
        return False

    def is_capture(self, player: Player, point: Point) -> bool:
        """
        判断是否能吃掉对方的子
        """
        for neighbor in self.neighbor_table[point]:
            neighbor_string = self._grid.get(neighbor)
            if neighbor_string is None:
                continue
            elif neighbor_string.color == player:
                continue
            else:
                if neighbor_string.num_liberties == 1:
                    # 对方的棋链只有一口气, 可以吃掉对方
                    return True
        return False

    def is_on_grid(self, point: Point) -> bool:
        """
        检查point是否在棋盘内
        """
        return 1 <= point.row <= self.num_rows and \
            1 <= point.col <= self.num_cols

    def get(self, point: Point) -> Union[Player, None]:
        """
        返回point位置的color
        """
        string = self._grid.get(point)
        if string is None:
            return None
        return string.color

    def get_go_string(self, point: Point) -> Union[GoString, None]:
        """
        返回point位置的棋链
        """
        string = self._grid.get(point)
        if string is None:
            return None
        return string

    def __eq__(self, other):
        return isinstance(other, Board) and \
            self.num_rows == other.num_rows and \
            self.num_cols == other.num_cols and \
            self._hash == other._hash

    def __deepcopy__(self, memodict={}):
        copied = Board(self.num_rows, self.num_cols)
        copied._grid = copy.copy(self._grid)
        copied._hash = self._hash
        return copied

    def zobrist_hash(self):
        return self._hash


class GameState:
    """
    游戏状态
    1. 棋子的布局
    2. 下一回合的执子方
    3. 上一回合的游戏状态(整个游戏状态就构成了一条链表)
    4. 上一步动作

    功能:
    1. 跟踪游戏状态
    2. 执行落子等动作
    3. 检查动作合法性(在副本上检测)
    """

    def __init__(self, board: Board, next_player: Player, previous: Any,
                 move: Union[Move, None]) -> None:
        self.board = board
        self.next_player = next_player  # 下一回合的执子方
        self.previous_state = previous  # 上一回合的游戏状态, 刚创建的游戏时, 这个这个字段为None
        # previous_states 所有历史状态的set:
        # set of (next_player, board.zobrist_hash)
        if self.previous_state is None:
            self.previous_states = frozenset()
        else:
            # 使用frozenset()来存储了所有的游戏状态, 可以`高效的判断劫争`
            self.previous_states = frozenset(previous.previous_states | {(
                previous.next_player, previous.board.zobrist_hash())})
        self.last_move = move  # 上一步动作, 刚创建的游戏时, 这个字段为None

    def apply_move(self, move: Move) -> Any:
        """
        执行move, 返回新的GameState对象
        """
        if move.is_play:
            next_board = copy.deepcopy(self.board)
            next_board.place_stone(self.next_player, move.point)
        else:
            next_board = self.board

        return GameState(next_board, self.next_player.other, self, move)

    @classmethod
    def new_game(cls, board_size: int) -> Any:
        """
        创建一盘游戏, 返回初始的GameState对象
        """
        if isinstance(board_size, int):
            board_size = (board_size, board_size)
        board = Board(*board_size)
        return GameState(board, Player.black, None, None)

    def is_move_self_capture(self, player: Player, move: Move) -> bool:
        """
        判断是否是`自吃(self capture)`

        当棋链只剩下一口气时, 如果在这口气所对应的Point上落子, 导致己方气尽而被提走, 我们称
        为`自吃`. 大多数围棋规则都禁止这样的动作(也有例外, 四年一度的应氏杯世界职业围棋锦标赛
        Ing Cup就允许这种自吃的落子规则, 它是国际围棋比赛中奖项最大的比赛之一)

        我们的代码中是`禁止自吃的`, 这样做和现行规则保持一致, 同时也减少机器人需要考虑的动作数
        目. 要构造出一个选择自吃是最佳解决方案的场景也不是不可能, 但在正式比赛中这种情况基本上闻
        所未闻

        在检查新落棋子是否气尽之前, 一般应当先判断是否能吃掉对方的棋子, 在所有规则中, 这一步动作
        都是有效的吃子而不是自吃

        Board类实际上是允许自吃动作的, 但是在GameState类中, 我们可以在Board的一个副本上执行落
        子动作, 然后检查副本上的气数并检查这个规则
        """
        if not move.is_play:
            return False

        return self.board.is_self_capture(player, move.point)

    @property
    def situation(self) -> Tuple[Player, int]:
        """
        游戏状态包括: 棋盘上所有棋子的位置zobrist哈希, 以及下一回合的执子方
        """
        return (self.next_player, self.board.zobrist_hash())

    def does_move_violate_ko(self, player: Player, move: Move) -> bool:
        """
        判断是否是`劫争(ko)`

        为了保证棋局最终能够结束, 那些会让棋局回到之前某个状态的落子动作是禁止的. 在实现劫
        争规则的时候要注意`反提(snapback)`. 劫争规则一般归纳为`不能立即重新吃掉对方棋子`
        """
        if not move.is_play:
            return False
        next_board = copy.deepcopy(self.board)
        next_board.place_stone(player, move.point)
        next_situation = (player.other, next_board.zobrist_hash())
        return next_situation in self.previous_states

    def is_valid_move(self, move: Move) -> bool:
        """
        判断动作是否合法

        1. 确认落子的Point是空的(Board.place_stone会做这个检测)
        2. 检查落子是自吃
        3. 检查落子不违反劫争规则
        """
        if self.is_over():
            return False
        if move.is_pass or move.is_resign:
            return True
        return (self.board.get(move.point) is None
                and not self.is_move_self_capture(self.next_player, move)
                and not self.does_move_violate_ko(self.next_player, move))

    def is_over(self) -> bool:
        """
        判断棋局是否结束

        1. 如果一方直接认输(resign)则棋局结束
        2. 如果双方接连跳过回合(pass)则棋局结束
        """
        if self.last_move is None:
            return False

        # 如果一方直接认输(resign)则棋局结束
        if self.last_move.is_resign:
            return True

        # 如果双方接连跳过回合(pass)则棋局结束
        second_last_move = self.previous_state.last_move
        if second_last_move is None:
            return False
        return self.last_move.is_pass and second_last_move.is_pass

    def legal_moves(self) -> List[Move]:
        """
        返回当前合法的moves
        """
        moves = []
        for row in range(1, self.board.num_rows + 1):
            for col in range(1, self.board.num_cols + 1):
                move = Move.play(Point(row=row, col=col))
                if self.is_valid_move(move):
                    moves.append(move)
        moves.append(Move.pass_turn())
        moves.append(Move.resign())
        return moves

    def winner(self) -> Union[Player, None]:
        """
        返回游戏获胜一方
        """
        if not self.is_over():
            return None
        if self.last_move.is_resign:
            return self.next_player
        game_result = compute_game_result(self)
        return game_result.winner
