#! /usr/bin/env python3
# -*- coding:utf-8 -*-
import sys
import os
import time
import click

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relaxgo.agent.naive import RandomBot
from relaxgo.agent.naive_fast import FastRandomBot
from relaxgo import goboard_slow
from relaxgo import goboard
from relaxgo import goboard_fast
from relaxgo.gotypes import Player
from relaxgo.utils import print_board, print_move, clear_screen
"""
`随机机器人 VS 随机机器人`, 目的只是为了测试游戏流程
"""


@click.group()
def cli():
    pass


@cli.command()
@click.option('--board',
              default='slow',
              type=click.Choice(['slow', 'normal', 'fast']),
              help='board模式')
def run(board):
    """
    ./bot_v_bot.py run --board=fast
    """
    board_size = 9
    if board == 'slow':
        game = goboard_slow.GameState.new_game(board_size)
    elif board == 'normal':
        game = goboard.GameState.new_game(board_size)
    elif board == 'fast':
        game = goboard_fast.GameState.new_game(board_size)

    # 构造两个随机机器人
    if board == 'slow' or board == 'normal':
        bots = {Player.black: RandomBot(), Player.white: RandomBot()}
    else:
        bots = {Player.black: FastRandomBot(), Player.white: FastRandomBot()}
    while not game.is_over():
        time.sleep(0.3)
        clear_screen()
        print_board(game.board)
        bot_move = bots[game.next_player].select_move(game)
        print_move(game.next_player, bot_move)
        game = game.apply_move(bot_move)


if __name__ == "__main__":
    cli()