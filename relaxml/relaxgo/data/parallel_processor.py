import os
from typing import List, Union, Tuple
import glob
import os.path
import tarfile
import gzip
import shutil
import numpy as np
import multiprocessing
from os import sys

from ..gosgf import Sgf_game
from ..goboard_fast import Board, GameState, Move
from ..gotypes import Player, Point
from .index_processor import KGSIndex
from .sampling import Sampler
from .generator import DataGenerator
from ..encoder.base import get_encoder_by_name


def worker(jobinfo: List[Tuple]):
    """
    执行`GoDataProcessor.process_zip()`, 处理一个*.tar.gz文件及其对应的game_list
    """
    try:
        clazz, encoder, zip_file, data_file_name, game_list = jobinfo
        clazz(encoder=encoder).process_zip(zip_file, data_file_name, game_list)
    except (KeyboardInterrupt, SystemExit):
        raise Exception('>>> Exiting child process.')


class GoDataProcessor:
    """
    # 不使用generator, 数据一次性返回
    >>> processer = GoDataProcessor()
    >>> features, labels = processer.load_go_data('train', 200)
    >>> features.shape
        (24576, 1, 19, 19)
    >>> labels.shape
        (24576,)
    
    # 使用generator, 数据批量返回
    >>> processer = GoDataProcessor()
    >>> generator = processer.load_go_data('train', 200, use_generator=True)
    >>> for features, labels in generator.generate(batch_size=128):
    >>>     print(features.shape)
    >>>     print(labels.shape)
    >>>     break
        (128, 1, 19, 19)
        (128, )


    """

    def __init__(self,
                 encoder: str = 'oneplane',
                 data_directory: str = '../data/kgs') -> None:
        """
        参数:
        encoder: 编码器的名字
        data_directory: 数据存放路径
        """
        self.encoder_string = encoder
        self.encoder = get_encoder_by_name(encoder, 19)
        self.data_dir = data_directory

    def load_go_data(
        self,
        data_type: str = 'train',
        num_samples: int = 1000,
        use_generator: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], DataGenerator]:
        """    
        下载围棋棋谱, 加载, 采样, 处理, 返回
    
        参数:
        data_type: train | test
        num_samples: 棋局的数量(不是样本的数量)
        use_generator: 是否返回DataGenerator

        返回: (features, labels) | generator
        features: [n, num_planes, board_height, board_width]
        labels: [n, ]
        """
        # 下载数据
        # '../data/kgs' -> '../data'
        index = KGSIndex(data_directory=self.data_dir[:-4])
        index.download_files()

        # 采样
        sampler = Sampler(data_dir=self.data_dir)
        # data:
        # [('KGS-2004-19-12106-.tar.gz', 7883),
        #  ('KGS-2006-19-10388-.tar.gz', 10064),
        #  ('KGS-2012-19-13665-.tar.gz', 8488),
        #  ...
        #  ('KGS-2009-19-18837-.tar.gz', 1993),
        #  ('KGS-2005-19-13941-.tar.gz', 9562),
        #  ('KGS-2003-19-7582-.tar.gz', 265),
        #  ('KGS-2009-19-18837-.tar.gz', 9086),
        #  ('KGS-2005-19-13941-.tar.gz', 13444)]
        data = sampler.draw_data(data_type, num_samples)

        # 并行的执行self.process_zip()处理*.tar.gz及其对应的game_list
        self.map_to_workers(data_type, data)
        if use_generator:
            generator = DataGenerator(data_type, self.data_dir, data)
            return generator
        else:
            features_and_labels = self.consolidate_games(data_type, data)
            return features_and_labels

    def unzip_data(self, zip_file_name: str) -> str:
        """
        将*.tar.gz解压成*.tar
        
        例如:
        zip_file_name: KGS-2013-19-13783-.tar.gz -> KGS-2013-19-13783-.tar
        """
        this_gz = gzip.open(self.data_dir + '/' +
                            zip_file_name)  # 将`gz`文件解压成`tar`文件

        tar_file = zip_file_name[0:-3]  # 删除文件名结尾的*.gz
        this_tar = open(self.data_dir + '/' + tar_file, 'wb')

        shutil.copyfileobj(this_gz, this_tar)  # 将解压后的内容写入到`tar`文件
        this_tar.close()
        return tar_file

    def process_zip(self, zip_file_name: str, data_file_name: str,
                    game_list: List[int]) -> None:
        """    
        一个*.tar.gz文件中会有多个*.sgf文件, 我们要处理的是game_list中列出来的*.sgf文件,
        会在本地磁盘生成处理好的features & labels文件

        最终会生成:
        ../data/kgs/KGS-2013-19-13783-train_features_0.npy - 1024个样本
        ../data/kgs/KGS-2013-19-13783-train_features_1.npy - 1024个样本
        ../data/kgs/KGS-2013-19-13783-train_features_2.npy - 1024个样本
        ...
        ../data/kgs/KGS-2013-19-13783-train_labels_0.npy  - 1024个label
        ../data/kgs/KGS-2013-19-13783-train_labels_1.npy  - 1024个label
        ../data/kgs/KGS-2013-19-13783-train_labels_2.npy  - 1024个label
        ...

        数据格式:
        features: [1024, num_planes, board_height, board_width]
        labels: [1024, ]

        内部实现逻辑:
        1. 调用unzip_data解压当前文件
        2. 初始化一个Encoder实例来编码SGF棋谱(直接使用self.encoder)
        3. 初始化合理形状的特征和标签NumPy数组
        4. 迭代遍历棋局列表, 并逐个处理棋局数据
           a. 每一局开始前处理让子的逻辑self.get_handicap
           b. 将每一回合的下一步动作编码为label
           c. 将每一回合的当前棋盘布局状态编码为feature
           d. 把下一步动作执行到棋盘上并继续
        5. 在本地文件系统分块存储特征和标签

        之所以要分块存储, 因为NumPy数组会迅速增大, 而数据存储在较小文件中可以保留更多灵活性, 
        例如: 我们可以把分块文件合并起来, 也可以根据需要将每个文件单独加载到内存. 我们在实现
        while循环中分块的最后一部分数据可能会丢掉, 但是影响不大

        参数:
        zip_file_name: 包含多个*.sgf的压缩文件
            e.g. KGS-2013-19-13783-.tar.gz
        data_file_name: 是生成的目标文件的前缀
            e.g. KGS-2013-19-13783-train
            KGS-2013-19-13783-train_features_*.npy
            KGS-2013-19-13783-train_labels_*.npy
            ...
        game_list: 需要处理的*.sgf索引列表
            e.g. [5696, 3746, 8279, ... ]
        """
        print(f'process zip: {zip_file_name}')
        # 调用unzip_data解压当前文件
        tar_file = self.unzip_data(zip_file_name)
        zip_file = tarfile.open(self.data_dir + '/' + tar_file)
        name_list = zip_file.getnames()
        # game_list对应的*.sgf中的样本总数
        total_examples = self.num_total_examples(zip_file, game_list,
                                                 name_list)
        # 初始化合理形状的特征和标签NumPy数组
        # shape [num_planes, board_height, board_width]
        shape = self.encoder.shape()
        # feature_shape [total_examples, num_planes, board_height, board_width]
        feature_shape = np.insert(shape, 0, np.asarray([total_examples]))
        # features.shape [total_examples, num_planes, board_height, board_width]
        features = np.zeros(feature_shape)
        # labels.shape [total_examples, ]
        labels = np.zeros((total_examples, ))

        counter = 0
        # 迭代遍历game_list对应的棋局(*.sgf)列表, 并逐个处理棋局数据
        for index in game_list:
            name = name_list[index + 1]  # 跳过第一个, 这个文件不是*.sgf
            if not name.endswith('.sgf'):
                raise ValueError(name + ' is not a valid sgf')
            sgf_content = zip_file.extractfile(name).read()
            sgf = Sgf_game.from_string(sgf_content)
            # 每一局开始前处理让子的逻辑, 得到开盘的游戏状态
            game_state, first_move_done = self.get_handicap(sgf)

            # 遍历SGF文件的所有动作
            for item in sgf.main_sequence_iter():
                color, move_tuple = item.get_move()
                point = None
                if color is not None:
                    if move_tuple is not None:
                        row, col = move_tuple
                        point = Point(row + 1, col + 1)
                        move = Move.play(point)
                    else:
                        move = Move.pass_turn()
                    # 编码一个样本
                    # 将每一回合的下一步动作编码为label
                    # 将每一回合的当前棋盘布局状态编码为feature
                    if first_move_done and point is not None:
                        features[counter] = self.encoder.encode(game_state)
                        labels[counter] = self.encoder.encode_point(point)
                        counter += 1
                    # 把下一步动作执行到棋盘上并继续
                    game_state = game_state.apply_move(move)
                    first_move_done = True
        # 在本地文件系统分块存储特征和标签
        feature_file_base = self.data_dir + '/' + data_file_name + '_features_%d'
        label_file_base = self.data_dir + '/' + data_file_name + '_labels_%d'

        chunk = 0
        chunksize = 1024
        while features.shape[0] >= chunksize:
            feature_file = feature_file_base % chunk
            label_file = label_file_base % chunk
            chunk += 1
            current_features, features = features[:chunksize], features[
                chunksize:]
            current_labels, labels = labels[:chunksize], labels[chunksize:]
            np.save(feature_file, current_features)
            np.save(label_file, current_labels)

    def consolidate_games(
            self, data_type: str,
            samples: List[Tuple[str, int]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        将分块的数据合并成一个目标文件:
        features_train.npy
        labels_train.npy
        或者:
        features_test.npy
        labels_test.npy

        数据尺寸:
        使用oneplane encoder, 当num_samples=1000的情况下(有1000个sgf文件), 
        样本大约是233M, 标签大约是1.3M. 这样推算如果有10000个*.sgf文件, 样本大
        约是2GB, 标签大约是13M

        参数:
        data_type: train | test
        samples: 每个元素对应一个*.sgf文件
            e.g.
            [('KGS-2004-19-12106-.tar.gz', 7883),
             ('KGS-2006-19-10388-.tar.gz', 10064),
             ('KGS-2012-19-13665-.tar.gz', 8488),
             ...
             ('KGS-2009-19-18837-.tar.gz', 1993),
             ('KGS-2005-19-13941-.tar.gz', 9562),
             ('KGS-2003-19-7582-.tar.gz', 265),
             ('KGS-2009-19-18837-.tar.gz', 9086),
             ('KGS-2005-19-13941-.tar.gz', 13444)]
        
        返回: (features, labels)
        features: [num_examples, num_planes, board_height, board_width]
        labels: [num_examples, ]
        """
        files_needed = set(file_name for file_name, index in samples)
        # file_names:
        # ['KGS-2004-19-12106-',
        #  'KGS-2006-19-10388-',
        #  'KGS-2012-19-13665-',
        #  ...]
        file_names = []
        for zip_file_name in files_needed:
            file_name = zip_file_name.replace('.tar.gz', '') + data_type
            file_names.append(file_name)

        feature_list = []
        label_list = []
        for file_name in file_names:
            file_prefix = file_name.replace('.tar.gz', '')
            base = self.data_dir + '/' + file_prefix + '_features_*.npy'
            for feature_file in glob.glob(base):
                label_file = feature_file.replace('features', 'labels')
                # x.shape: [n, num_planes, board_height, board_width]
                x = np.load(feature_file)
                # y.shape: [n, ]
                y = np.load(label_file)
                x = x.astype('float32')
                feature_list.append(x)
                label_list.append(y)
        features = np.concatenate(feature_list, axis=0)
        labels = np.concatenate(label_list, axis=0)
        # 保存为文件
        np.save('{}/features_{}.npy'.format(self.data_dir, data_type),
                features)
        np.save('{}/labels_{}.npy'.format(self.data_dir, data_type), labels)
        # 返回
        # features.shape [num_examples, num_planes, board_height, board_width]
        # labels.shape [num_examples, ]
        return features, labels

    @staticmethod
    def get_handicap(sgf: Sgf_game) -> Tuple[GameState, bool]:
        """
        获取让子信息, 并将它们布置在空白棋盘上
        
        例如:
        HA[5]...AB[dd][pd][jj][dp][pp]

        sgf.get_handicap() - HA(让子)
        sgf.get_root().get_setup_stones() - AB(Add Black), AW(Add White), AE(Add Empty)

        参数:
        sgf: Sgf_game

        返回: (game_state, first_move_done)
        game_state: 最新的Game State
        first_move_done: 如果有让子为True; 否则为False
        """
        go_board = Board(19, 19)
        first_move_done = False  # 如果有让子, 设置为True
        move = None
        game_state = GameState.new_game(19)
        if sgf.get_handicap() is not None and sgf.get_handicap() != 0:
            for setup in sgf.get_root().get_setup_stones():
                for move in setup:
                    row, col = move
                    go_board.place_stone(Player.black, Point(row + 1, col + 1))
            first_move_done = True
            game_state = GameState(go_board, Player.white, None, move)
        return game_state, first_move_done

    def map_to_workers(self, data_type: str, samples: List[Tuple[str, int]]):
        """
        并行的处理samepls文件(并行执行self.process_zip)

        参数:
        data_type: train | test
        samples:
            [('KGS-2004-19-12106-.tar.gz', 7883),
            ('KGS-2006-19-10388-.tar.gz', 10064),
            ('KGS-2012-19-13665-.tar.gz', 8488),
            ...
            ('KGS-2009-19-18837-.tar.gz', 1993),
            ('KGS-2005-19-13941-.tar.gz', 9562),
            ('KGS-2003-19-7582-.tar.gz', 265),
            ('KGS-2009-19-18837-.tar.gz', 9086),
            ('KGS-2005-19-13941-.tar.gz', 13444)]
        """
        zip_names = set()  # 保存samples中所有的文件名*.tar.gz
        # 按压缩文件名对所有采样出来的*.sgf文件索引进行分组
        # indices_by_zip_name:
        # {
        #   'KGS-2013-19-13783-.tar.gz': [5696, 3746, 8279, ... ],
        #   'KGS-2008-19-14002-.tar.gz': [10428, 7795, 1509, ...],
        #   ...
        # }
        indices_by_zip_name = {}  # Dict[str, List[int]]
        for filename, index in samples:
            zip_names.add(filename)
            if filename not in indices_by_zip_name:
                indices_by_zip_name[filename] = []
            indices_by_zip_name[filename].append(index)

        # 删除老的features & labels文件
        base = self.data_dir + f'/*{data_type}_features_*.npy'
        for feature_file in glob.glob(base):
            os.remove(feature_file)
        base = self.data_dir + f'/*{data_type}_labels_*.npy'
        for label_file in glob.glob(base):
            os.remove(label_file)

        # 需要并行处理的*.tar.gz文件
        zips_to_process = []  # 并行的参数
        for zip_name in zip_names:
            base_name = zip_name.replace('.tar.gz', '')
            data_file_name = base_name + data_type
            # 处理一个*.tar.gz文件中的game_lists, 生成features & labels,
            # 生成的features & labels会写入磁盘
            # e.g.
            # 'KGS-2013-19-13783-.tar.gz'
            # 'KGS-2013-19-13783-train'
            # [5696, 3746, 8279, ... ],
            if not os.path.isfile(self.data_dir + '/' + data_file_name):
                # 等价于:
                # self.process_zip(zip_name, data_file_name, indices_by_zip_name[zip_name])
                zips_to_process.append(
                    (self.__class__, self.encoder_string, zip_name,
                     data_file_name, indices_by_zip_name[zip_name]))

        cores = multiprocessing.cpu_count()
        # Bug Fix:
        # ValueError: not enough values to unpack
        if cores > len(zip_names):
            cores = len(zip_names)
        pool = multiprocessing.Pool(processes=cores)
        p = pool.map_async(worker, zips_to_process)
        try:
            _ = p.get()
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            sys.exit(-1)

    def num_total_examples(self, zip_file: tarfile.TarFile,
                           game_list: List[int], name_list: List[str]) -> int:
        """
        计算`game_list`对应的*.sgf文件中有多少个效动作总数

        注意: 对于有让子的棋局, 第一个动作就可以作为样本; 对于没有让子的棋局, 从第二个动作开始才能作为样本

        >>> tar_file = process.unzip_data('KGS-2013-19-13783-.tar.gz')
        >>> zip_file = tarfile.open(process.data_dir + '/' + tar_file)
        >>> process.num_total_examples(zip_file, game_list=[0, 1],
                                       name_list=[
                                         'kgs-19-2013',
                                         'kgs-19-2013/2013-07-30-6.sgf',
                                         'kgs-19-2013/2013-03-20-12.sgf'])
            218

        参数:
        zip_file: tar file
        game_list: 要统计的*.sgf文件索引
            e.g. [5696, 3746, 8279, ... ]
        name_list: 文件名的列表
            e.g. ['kgs-19-2013',
                  'kgs-19-2013/2013-07-30-6.sgf',
                  'kgs-19-2013/2013-03-20-12.sgf',
                  ... ]
        """
        total_examples = 0
        # 遍历game_list对应的所有棋局*.sgf
        for index in game_list:
            name = name_list[index + 1]  # 跳过第一个, 这个文件不是*.sgf
            if name.endswith('.sgf'):
                sgf_content = zip_file.extractfile(name).read()
                sgf = Sgf_game.from_string(sgf_content)
                _, first_move_done = self.get_handicap(sgf)

                num_moves = 0  # 一个*.sgf中一共有多少move
                # 遍历一局
                for item in sgf.main_sequence_iter():
                    color, move = item.get_move()
                    if color is not None:
                        # 对于有让子的棋局, 第一个动作就可以作为样本;
                        # 对于没有让子的棋局, 从第二个动作开始才能作为样本
                        if first_move_done:
                            num_moves += 1
                        first_move_done = True
                total_examples = total_examples + num_moves
            else:
                raise ValueError(name + ' is not a valid sgf')
        return total_examples