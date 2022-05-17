#! /usr/bin/env python3
# -*- coding:utf-8 -*-
import sys
import os
import click

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relaxgo.utils import print_board, print_move, point_from_coords
from relaxgo.goboard_fast import GameState, Move
from relaxgo.gotypes import Player
from relaxgo.mcts.mcts import MCTSAgent

BOARD_SIZE = 5


@click.group()
def cli():
    pass


@cli.command()
def run():
    """
    ./mcts_go.py run
    """
    game = GameState.new_game(BOARD_SIZE)
    bot = MCTSAgent(10000, 1.4)

    while not game.is_over():
        print_board(game.board)
        if game.next_player == Player.black:
            human_move = input('-- ')
            point = point_from_coords(human_move.strip())
            move = Move.play(point)
        else:
            move = bot.select_move(game)
        print_move(game.next_player, move)
        game = game.apply_move(move)


if __name__ == "__main__":
    cli()