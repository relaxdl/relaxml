#! /usr/bin/env python3
# -*- coding:utf-8 -*-
import sys
import os
import click

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relaxgo.agent.naive import RandomBot
from relaxgo import goboard_slow

from relaxgo.gotypes import Player
from relaxgo.utils import print_board, print_move, clear_screen, point_from_coords
"""
人机对战(机器人是随机机器人)
"""


@click.group()
def cli():
    pass


@cli.command()
@click.option('--board',
              default='slow',
              type=click.Choice(['slow']),
              help='board模式')
def run(board):
    """
    ./human_v_bot.py run --board=slow
    """
    board_size = 9
    if board == 'slow':
        game = goboard_slow.GameState.new_game(board_size)
    # 构造一个随机机器人
    bot = RandomBot()
    while not game.is_over():
        clear_screen()
        print_board(game.board)
        if game.next_player == Player.black:
            human_move = input('--')
            point = point_from_coords(human_move)
            move = goboard_slow.Move.play(point)
        else:
            move = bot.select_move(game)
        print_move(game.next_player, move)
        game = game.apply_move(move)


if __name__ == "__main__":
    cli()