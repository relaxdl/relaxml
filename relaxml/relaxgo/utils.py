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