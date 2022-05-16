#! /usr/bin/env python3
# -*- coding:utf-8 -*-
import sys
import os
import random
import click

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from relaxgo.gotypes import Player, Point

MAX63 = 0x7fffffffffffffff
"""
zobrist哈希

使用zobrist哈希可以加速棋局. 为了检测`劫争`的情形, 我们需要检查棋局的全部历史记录, 
才能判断当前的棋局是否已经出现过. 这样做所需要的计算量是很大的. 为了避免这个问题, 我们
可以修改一下程序, 不用存储整个棋局的历史, 而只存储占用空间更小的`哈希值`

在zobrist哈希技术中, 我们需要为棋盘每一个可能发生的动作分配一个哈希值, 为了获得最好的结果, 
每个哈希值应该随机生成. 在围棋中, 落子动作有黑白两种可能, 因此在19x19规格的棋盘上, 完整的
zobrist哈希应当由2x19x19=722个哈希值组成. 我们可以用这722个哈希值来代表单个动作, 给最复
杂的棋盘布局编码

这个过程的有趣之处在于用一个哈希值就能够对整个棋盘的状态进行编码. 比如从空白棋盘开始, 可以把空
白棋盘的哈希值设置为0. 每个落子动作都具有特定的哈希值, 可以用`棋盘的哈希值与动作对应的哈希值进
行XOR操作来计算新的棋盘哈希值`, 我们将这个运算称为`应用该哈希值`. 按照这个逻辑, 每一个落子的动作, 
就可以将它的哈希值应用到棋盘上, 这样, 我们只用单个哈希值就可以跟踪当前的棋盘状态了

对于任何动作, 都可以再次应用它的哈希值来撤回它(这也是XOR操作特有的方便之处), 我们将这个操作称为
`逆应用该哈希值`. 这一点很重要, 有了这个特性, 就可以在提子时轻松的从棋盘上移除棋子. 例如: 如果要
吃掉棋盘上C3处的黑子, 可以应用C3的哈希值, 将它从当前棋盘状态对应的哈希值中移除. 这么做的话, 还必
须把吃掉C3处黑子的白子的哈希值也应用到棋盘上, 如果白方一次落子吃掉多颗黑子, 则需要将它们的哈希值全
部都逆应用到棋盘上
"""


def to_python(player_state):
    if player_state is None:
        return 'None'
    if player_state == Player.black:
        return Player.black
    return Player.white


@click.group()
def cli():
    pass


@cli.command()
def run():
    table = {}
    empty_board = 0
    for row in range(1, 20):
        for col in range(1, 20):
            for state in (None, Player.black, Player.white):
                code = random.randint(0, MAX63)
                table[Point(row, col), state] = code

    # 生成代码
    print('# -*- coding:utf-8 -*-')
    print('from .gotypes import (Player, Point)')
    print('')
    print("__all__ = ['HASH_CODE', 'EMPTY_BOARD']")
    print('')
    print('HASH_CODE = {')
    for (pt, state), hash_code in table.items():
        print('    (%r, %s): %r,' % (pt, to_python(state), hash_code))
    print('}')
    print('')
    print('EMPTY_BOARD = %d' % (empty_board, ))


"""
执行: ./gen_zobrist.py run
用来生成zobrist.py, 将输出的内容copy到zobrist.py中
"""
if __name__ == "__main__":
    cli()