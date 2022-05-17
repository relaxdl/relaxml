#! /usr/bin/env python3
# -*- coding:utf-8 -*-
import sys
import os
import click

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relaxgo.ttt.ttttypes import Player, Point
from relaxgo.ttt.tttboard import GameState, Move, Board
from relaxgo.minimax.minimax import MiniMaxAgent
"""
井字棋人机对战
"""

COL_NAMES = 'ABC'


def print_board(board: Board) -> None:
    """
    e.g.
       A   B   C
    1  X | X | O
    2  O | O | X
    3  X | X | O
    """
    print('   A   B   C')
    for row in (1, 2, 3):
        pieces = []
        for col in (1, 2, 3):
            piece = board.get(Point(row, col))
            if piece == Player.x:
                pieces.append('X')
            elif piece == Player.o:
                pieces.append('O')
            else:
                pieces.append(' ')
        print('%d  %s' % (row, ' | '.join(pieces)))


def point_from_coords(text: str) -> Point:
    """
    e.g.
    A1 -> Point
    """
    col_name = text[0]
    row = int(text[1])
    return Point(row, COL_NAMES.index(col_name) + 1)


@click.group()
def cli():
    pass


@cli.command()
def run():
    """
    ./play_ttt.py run
    """
    game = GameState.new_game()
    human_player = Player.x
    bot = MiniMaxAgent()
    while not game.is_over():
        print_board(game.board)
        if game.next_player == human_player:
            human_move = input('--')
            point = point_from_coords(human_move.strip())
            move = Move(point)
        else:
            move = bot.select_move(game)
        game = game.apply_move(move)
    print_board(game.board)
    winner = game.winner()
    if winner is None:
        print('It is a draw.')
    else:
        print(f'Winner: {winner}')


if __name__ == "__main__":
    cli()