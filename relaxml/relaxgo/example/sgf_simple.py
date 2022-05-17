# -*- coding:utf-8 -*-
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))

from relaxgo.gosgf import Sgf_game
from relaxgo.goboard_fast import GameState, Move
from relaxgo.gotypes import Point
from relaxgo.utils import print_board
"""
SGF格式说明:

https://www.red-bean.com/sgf/go.html

演示gosgf的使用

常见的围棋数据格式: 智能游戏格式(Smart Game Format, SGF), 以前叫做(Smart Go Format), 
最早开发于80年代末, 目前它的主流版本是第四版(FF[4]), 发布于90年代末. SGF是一种基于文本的简
单格式, 但表达能力很强, 可用于表示围棋游戏, 围棋游戏的变化体, 以及其它类型的棋盘游戏

SGF的核心是棋局的元数据和落子的动作记录: 元数据的格式是两个大写字母表示一个属性的名称, 然后在中
括号里面指定它的值. 例如, 尺寸SZ[9]表示9x9的围棋盘上进行棋局. 围棋落子动作的编码方式是, 落子棋
盘的第3行第3列上的白子记录为W[cc], 棋盘的左上角是aa. 要表示跳过回合, 可以记录成空的动作W[], B[]

SGF文件由多个节点(node)组成, 节点之间用分号分隔. 第一个节点包含棋局相关的元数据信息, 棋盘大小, 
规则集, 棋局结果以及其它背景信息. 后续每个节点代表棋局中的一步动作.文件中的空白是完全无关的

例如:
GM[1] - 游戏编号为1
SZ[9] - 棋盘尺寸9x9
HA[0] - 让0子
KM[6.5] - 贴6目半
RU[Japenese] - 游戏规则使用日式规则
RE[W+9.5] - 白方胜9目半

注意, SGF格式(棋盘左上角是aa, 先描述col, 再描述row)
B[ab] - 中的的a表示从左上角开始第1列; b表示从左上角开始数第2行【从棋盘左上角开始数, 先列后行】
B[ba] - 中的的b表示从左上角开始第2列; a表示从左上角开始数第1行【从棋盘左上角开始数, 先列后行】

SGF格式 -> Point(棋盘坐下角是row=1, col=1, 先描述row, 再描述col):
SZ[19];B[ab]会被转化为 - b 17 0 -> Point(row=17+1, col=0+1)
SZ[19];W[ba]会被转化为 - w 18 1 -> Point(row=18+1, col=1+1)
SZ[9];B[ab]会被转化为 - b 7 0 -> Point(row=7+1, col=0+1)
SZ[9];W[ba]会被转化为 - w 8 1 -> Point(row=8+1, col=1+1)

模块go/gosgf负责处理SGF文件的所有逻辑(这个模块改编自Gomill Python库), 
是一个外部库, 我们直接拿来使用, 下面的例子演示:
1. 创建一个Sgf_game实例
2. 使用Sgf_game加载SGF游戏信息
3. 逐个读出信息, 并把每一回合的动作执行到一个GameState对象
"""


def run():
    # sgf_content = '(;GM[1]FF[4]SZ[9];B[ee];W[ef];B[ff];W[df];B[fe];W[fc];B[ec];W[gd];B[fb])'
    sgf_content = '(;GM[1]FF[4]SZ[9];B[ga];)'

    # 创建Sgf_game实例
    sgf_game = Sgf_game.from_string(sgf_content)
    game_state = GameState.new_game(9)  # 9
    for item in sgf_game.main_sequence_iter():
        color, move_tuple = item.get_move()
        if color is not None and move_tuple is not None:
            row, col = move_tuple
            print(color, row, col)
            point = Point(row=row + 1, col=col + 1)
            move = Move.play(point)
            game_state = game_state.apply_move(move)
            print_board(game_state.board)


if __name__ == "__main__":
    run()
    # 最后一个move会输出:
    # b 7 5
    #  9  .  .  .  .  .  .  .  .  .
    #  8  .  .  .  .  .  x  .  .  .
    #  7  .  .  .  .  x  o  .  .  .
    #  6  .  .  .  .  .  .  o  .  .
    #  5  .  .  .  .  x  x  .  .  .
    #  4  .  .  .  o  o  x  .  .  .
    #  3  .  .  .  .  .  .  .  .  .
    #  2  .  .  .  .  .  .  .  .  .
    #  1  .  .  .  .  .  .  .  .  .
    #     A  B  C  D  E  F  G  H  J