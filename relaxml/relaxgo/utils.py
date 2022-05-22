# -*- coding:utf-8 -*-
import os
import requests
import hashlib
import zipfile
import tarfile
import platform
import subprocess
import numpy as np
import tempfile
import h5py
import tensorflow.keras as keras
from .gotypes import Player, Point
from .goboard import Move
"""
围棋坐标可以用多种方式来指定, 但在欧洲最常见得方法是从A开始的字母表示列,
用1开始的数字表示行. 在这个坐标系中, 标准19x19棋盘左下角为A1, 有上角为T19. 
注意: 依照惯例, 我们忽略字母I, 避免与数字1混淆

用`.`表示空点
用`x`表示黑棋
用`o`表示白棋
"""

COLS = 'ABCDEFGHJKLMNOPQRST'
STONE_TO_CHAR = {
    None: ' . ',  # 空点
    Player.black: ' x ',
    Player.white: ' o ',
}

_DATA_HUB = dict()
_DATA_HUB['features-40k'] = (
    'f3f0bdb3dd8a5cc663ef56ca4c8e06032034531d',
    'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/go/features-40k.npy')
_DATA_HUB['labels-40k'] = (
    'd959a562ef5189413a0d1b3525972e9c9dd2b598',
    'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/go/labels-40k.npy')
_DATA_HUB['kgs'] = (
    '49fb4f6366e650efb446679586d22c3a3b3bd875',
    'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/go/kgs.zip')
_DATA_HUB['kgs.json'] = (
    'fb838e1c844a4a67af2be1fa8e256483c7b2f967',
    'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/go/kgs.json')


def download(name: str, cache_dir: str = '../data') -> str:
    """
    下载数据
    """
    sha1_hash, url = _DATA_HUB[name]
    fname = os.path.join(cache_dir, url.split('/ml/go/')[-1])
    fdir = os.path.dirname(fname)
    os.makedirs(fdir, exist_ok=True)
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname
    print(f'download {url} -> {fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    print(f'download {fname} success!')
    # e.g. ../data/file.zip
    return fname


def download_extract(name: str, cache_dir: str = '../data') -> str:
    """
    下载数据 & 解压
    """
    # 下载数据集
    fname = download(name, cache_dir)

    # 解压
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, 'Only zip/tar files can be extracted.'
    fp.extractall(base_dir)
    # e.g. ../data/file
    return data_dir


def print_move(player, move):
    if move.is_pass:
        move_str = 'passes'
    elif move.is_resign:
        move_str = 'resigns'
    else:
        move_str = '%s%d' % (COLS[move.point.col - 1], move.point.row)
    print('%s %s' % (player, move_str))


def print_board(board):
    for row in range(board.num_rows, 0, -1):
        bump = " " if row <= 9 else ""  # 让1位和2位的数字对齐
        line = []
        for col in range(1, board.num_cols + 1):
            stone = board.get(Point(row=row, col=col))
            line.append(STONE_TO_CHAR[stone])
        print('%s%d %s' % (bump, row, ''.join(line)))
    print('    ' + '  '.join(COLS[:board.num_cols]))


def point_from_coords(coords: str) -> Point:
    """
    coords转换成point

    例如:
    C3 -> Point
    E11 -> Point
    """
    col = COLS.index(coords[0]) + 1
    row = int(coords[1:])
    return Point(row=row, col=col)


def coords_from_point(point: Point) -> str:
    """
    point转换成coords

    例如:
    Point -> C3
    Point -> E11
    """
    return f'{COLS[point.row-1]}{point.row}'


def clear_screen():
    # see https://stackoverflow.com/a/23075152/323316
    if platform.system() == 'Windows':
        subprocess.Popen('cls', shell=True).communicate()
    else:  # Linux and Mac
        print(chr(27) + '[2J')


class MoveAge:

    def __init__(self, board):
        self.move_ages = -np.ones((board.num_rows, board.num_cols))

    def get(self, row, col):
        return self.move_ages[row, col]

    def reset_age(self, point):
        self.move_ages[point.row - 1, point.col - 1] = -1

    def add(self, point):
        self.move_ages[point.row - 1, point.col - 1] = 0

    def increment_all(self):
        self.move_ages[self.move_ages > -1] += 1


def save_model_to_hdf5_group(model: keras.Model, h5file: h5py.Group) -> None:
    """
            h5file
        /         \
    encoder(group)  model[参数h5file]
                         |
                  kerasmodel(group)

    1. 将keras model模型保存到一个临时文件中, 保存的格式是h5
    2. 重新打开这个保存模型的h5文件, 将其内容copy到`h5file`的`kerasmodel`节点
    """
    tempfd, tempfname = tempfile.mkstemp(prefix='tmp-kerasmodel')
    try:
        os.close(tempfd)
        # 1. 将keras model模型保存到一个临时文件中, 保存的格式是h5
        keras.models.save_model(model, tempfname)

        # 2. 重新打开这个保存模型的h5文件, 将其内容copy到`h5file`的`kerasmodel`节点
        serialized_model = h5py.File(tempfname, 'r')
        root_item = serialized_model.get('/')
        serialized_model.copy(root_item, h5file, 'kerasmodel')
        serialized_model.close()
    finally:
        os.unlink(tempfname)


def load_model_from_hdf5_group(h5file: h5py.Group,
                               custom_objects=None) -> keras.Model:
    """
            h5file
        /         \
    encoder(group)  model[参数h5file]
                        |
                kerasmodel(group)

    1. 将模型的数据解压到临时文件中
       a. 创建临时文件来保存模型
       b. 将属性写入临时文件
       c. 将属性写入临时文件
    2. 从临时文件中加载模型
    """
    tempfd, tempfname = tempfile.mkstemp(prefix='tmp-kerasmodel')
    try:
        os.close(tempfd)
        # 1. 将模型的数据解压到临时文件中

        # a. 创建临时文件来保存模型
        serialized_model = h5py.File(tempfname, 'w')
        root_item = h5file.get('kerasmodel')

        # b. 将属性写入临时文件
        for attr_name, attr_value in root_item.attrs.items():
            serialized_model.attrs[attr_name] = attr_value

        # c. 遍历所有keys, 将其对应Node写入临时文件
        for k in root_item.keys():
            h5file.copy(root_item.get(k), serialized_model, k)
        serialized_model.close()

        # 2. 从临时文件加载模型
        return keras.models.load_model(tempfname,
                                       custom_objects=custom_objects)
    finally:
        os.unlink(tempfname)


def is_ladder_capture(game_state, candidate, recursion_depth=50):
    """
    TODO
    """
    return is_ladder(True, game_state, candidate, None, recursion_depth)


def is_ladder_escape(game_state, candidate, recursion_depth=50):
    """
    TODO
    """
    return is_ladder(False, game_state, candidate, None, recursion_depth)


def is_ladder(try_capture,
              game_state,
              candidate,
              ladder_stones=None,
              recursion_depth=50):
    """
    TODO
    Ladders are played out in reversed roles, one player tries to capture,
    the other to escape. We determine the ladder status by recursively calling
    is_ladder in opposite roles, providing suitable capture or escape candidates.
    Arguments:
    try_capture: boolean flag to indicate if you want to capture or escape the ladder
    game_state: current game state, instance of GameState
    candidate: a move that potentially leads to escaping the ladder or capturing it, instance of Move
    ladder_stones: the stones to escape or capture, list of Point. Will be inferred if not provided.
    recursion_depth: when to stop recursively calling this function, integer valued.
    Returns True if game state is a ladder and try_capture is true (the ladder captures)
    or if game state is not a ladder and try_capture is false (you can successfully escape)
    and False otherwise.
    """

    if not game_state.is_valid_move(Move(candidate)) or not recursion_depth:
        return False

    next_player = game_state.next_player
    capture_player = next_player if try_capture else next_player.other
    escape_player = capture_player.other

    if ladder_stones is None:
        ladder_stones = guess_ladder_stones(game_state, candidate,
                                            escape_player)

    for ladder_stone in ladder_stones:
        current_state = game_state.apply_move(candidate)

        if try_capture:
            candidates = determine_escape_candidates(game_state, ladder_stone,
                                                     capture_player)
            attempted_escapes = [  # now try to escape
                is_ladder(False, current_state, escape_candidate, ladder_stone,
                          recursion_depth - 1)
                for escape_candidate in candidates
            ]

            if not any(attempted_escapes):
                return True  # if at least one escape fails, we capture
        else:
            if count_liberties(current_state, ladder_stone) >= 3:
                return True  # successful escape
            if count_liberties(current_state, ladder_stone) == 1:
                continue  # failed escape, others might still do
            candidates = liberties(current_state, ladder_stone)
            attempted_captures = [  # now try to capture
                is_ladder(True, current_state, capture_candidate, ladder_stone,
                          recursion_depth - 1)
                for capture_candidate in candidates
            ]
            if any(attempted_captures):
                continue  # failed escape, try others
            return True  # candidate can't be caught in a ladder, escape.
    return False  # no captures / no escapes


def is_candidate(game_state, move, player):
    """
    TODO
    """
    return game_state.next_player == player and \
        count_liberties(game_state, move) == 2


def guess_ladder_stones(game_state, move, escape_player):
    """
    TODO
    """
    adjacent_strings = [
        game_state.board.get_go_string(nb) for nb in move.neighbors()
        if game_state.board.get_go_string(nb)
    ]
    if adjacent_strings:
        string = adjacent_strings[0]
        neighbors = []
        for string in adjacent_strings:
            stones = string.stones
            for stone in stones:
                neighbors.append(stone)
        return [
            Move(nb) for nb in neighbors
            if is_candidate(game_state, Move(nb), escape_player)
        ]
    else:
        return []


def determine_escape_candidates(game_state, move, capture_player):
    """
    TODO
    """
    escape_candidates = move.neighbors()
    for other_ladder_stone in game_state.board.get_go_string(move).stones:
        for neighbor in other_ladder_stone.neighbors():
            right_color = game_state.color(neighbor) == capture_player
            one_liberty = count_liberties(game_state, neighbor) == 1
            if right_color and one_liberty:
                escape_candidates.append(liberties(game_state, neighbor))
    return escape_candidates


def count_liberties(game_state, move):
    """
    TODO
    """
    if game_state.board.get_go_string(move):
        return game_state.board.get_go_string(move).num_liberties
    else:
        return 0


def liberties(game_state, move):
    """
    TODO
    """
    return list(game_state.board.get_go_string(move).liberties)