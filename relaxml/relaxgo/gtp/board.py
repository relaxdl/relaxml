from ..gotypes import Point
from ..goboard_fast import Move

__all__ = [
    'coords_to_gtp_position',
    'gtp_position_to_coords',
]

COLS = 'ABCDEFGHJKLMNOPQRST'


def coords_to_gtp_position(move: Move) -> str:
    """
    Move(r, c) -> GTP Board location

    >>> move = Move.play(Point(1,1))
    >>> coords_to_gtp_position(move)
        'A1'
    """
    point = move.point
    return COLS[point.col - 1] + str(point.row)


def gtp_position_to_coords(gtp_position: str) -> Move:
    """
    GTP Board location -> Move(r, c)

    >>> gtp_position_to_coords('A1')
        (r 1, c 1)
    """
    col_str, row_str = gtp_position[0], gtp_position[1:]
    point = Point(int(row_str), COLS.find(col_str.upper()) + 1)
    return Move(point)
