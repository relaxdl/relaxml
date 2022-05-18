import os
import random
from typing import List, Tuple, Union
from .index_processor import KGSIndex
"""
采样模块
1. 确保随机选择指定数量的棋局
2. 确保训练样本和测试样本必须是分开的, 不能出现任何重叠

样本的格式如下:
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


class Sampler:
    """
    采样:
    1. 返回测试样本
       Sampler构造完成之后, 会自动构建好测试样本, 调用直接返回就可以
    2. 返回训练样本
       a. 下载文件索引index
       b. 从文件索引中采样直到满足样本数量(会过滤掉测试样本的内容)
    """

    def __init__(self,
                 data_dir='../data/kgs',
                 num_test_games=1000,
                 cap_year=2019,
                 seed=1337):
        """
        初始化完成后, 会自动构建好测试样本

        参数:
        data_dir: 数据文件保存路径
        num_test_games: 测试样本数量
        cap_year: 只会采样满足year <= cap_year的样本
        seed: 随机种子
        """
        self.data_dir = data_dir
        self.num_test_games = num_test_games
        self.test_games = []  # 测试样本
        self.train_games = []  # 训练样本
        # 测试样本的缓存文件
        self.test_cache = os.path.join(data_dir,
                                       f'test_samples_{num_test_games}.cache')
        self.cap_year = cap_year

        random.seed(seed)
        self.compute_test_samples()  # 构造测试样本

    def draw_data(self, data_type: str,
                  num_samples: Union[int, None]) -> List[Tuple[str, int]]:
        """
        采样

        >>> sampler = Sampler()
        >>> samples = sampler.draw_data('train', 5)
        >>> print(samples)
            [('KGS-2009-19-18837-.tar.gz', 9098), 
            ('KGS-2005-19-13941-.tar.gz', 8165), 
            ('KGS-2018_03-19-833-.tar.gz', 725), 
            ('KGS-2010-19-17536-.tar.gz', 10315), 
            ('KGS-2014-19-13029-.tar.gz', 12752)]
    
        参数:
        data_type: test | train
        num_samples: 样本数量, 一局游戏为一个样本 [train]

        返回:
        samples: 样本
        """
        if data_type == 'test':
            return self.test_games
        elif data_type == 'train' and num_samples is not None:
            return self.draw_training_samples(num_samples)
        elif data_type == 'train' and num_samples is None:
            return self.draw_all_training()
        else:
            raise ValueError(
                data_type +
                " is not a valid data type, choose from 'train' or 'test'")

    def draw_samples(self, num_sample_games: int) -> List[Tuple[str, int]]:
        """
        从index中采样

        注意: 只会采样满足year <= cap_year的样本

        >>> sampler = Sampler()
        >>> samples = sampler.draw_samples(5)
        >>> print(samples)
            [('KGS-2009-19-18837-.tar.gz', 10659), 
            ('KGS-2005-19-13941-.tar.gz', 13444), 
            ('KGS-2006-19-10388-.tar.gz', 10064), 
            ('KGS-2004-19-12106-.tar.gz', 7883), 
            ('KGS-2005-19-13941-.tar.gz', 9562)]

        参数:
        num_sample_games: 样本数量, 一局游戏为一个样本
        
        返回:
        samples: 样本
        """
        available_games = []  # List[Tuple[str, int]]
        index = KGSIndex(data_directory=self.data_dir)
        index.load_index()

        # 从index中读取所有的games, 写入available_games
        for fileinfo in index.file_info:
            filename = fileinfo['filename']
            year = int(filename.split('-')[1].split('_')[0])
            # 大于cap_year的直接跳过
            if year > self.cap_year:
                continue
            num_games = fileinfo['num_games']
            for i in range(num_games):
                available_games.append((filename, i))
        print('>>> Total number of games used: ' + str(len(available_games)))

        # 从available_games采样:
        # 每次采样一个样本, 直到满足数量为止
        sample_set = set()
        while len(sample_set) < num_sample_games:
            sample = random.choice(available_games)
            if sample not in sample_set:
                sample_set.add(sample)
        print('Drawn ' + str(num_sample_games) + ' samples:')
        return list(sample_set)

    def compute_test_samples(self) -> None:
        """
        计算测试样本

        1. 如果不存在self.test_cache, 创建一个本地文件存储固定的测试样本
        2. 如果self.test_cache存在, 则直接读取其中的内容作为测试样本
        3. 将测试样本写入self.test_games
        """
        if not os.path.isfile(self.test_cache):
            test_games = self.draw_samples(self.num_test_games)
            test_sample_file = open(self.test_cache, 'w')
            for sample in test_games:
                test_sample_file.write(str(sample) + "\n")
            test_sample_file.close()

        test_sample_file = open(self.test_cache, 'r')
        sample_contents = test_sample_file.read()
        test_sample_file.close()
        for line in sample_contents.split('\n'):
            if line != "":
                (filename, index) = eval(line)
                self.test_games.append((filename, index))

    def draw_training_samples(self,
                              num_sample_games: int) -> List[Tuple[str, int]]:
        """
        采样(会过滤掉测试样本)

        参数:
        num_sample_games: 样本数量, 一局游戏为一个样本
        """
        available_games = []
        index = KGSIndex(data_directory=self.data_dir)
        index.load_index()
        for fileinfo in index.file_info:
            filename = fileinfo['filename']
            year = int(filename.split('-')[1].split('_')[0])
            if year > self.cap_year:
                continue
            num_games = fileinfo['num_games']
            for i in range(num_games):
                available_games.append((filename, i))
        print('total num games: ' + str(len(available_games)))

        # 采样: 每次采样一个样本, 直到满足数量为止
        sample_set = set()
        while len(sample_set) < num_sample_games:
            sample = random.choice(available_games)
            if sample not in self.test_games:  # 过滤掉test game
                sample_set.add(sample)
        print('Drawn ' + str(num_sample_games) + ' samples:')
        return list(sample_set)

    def draw_all_training(self) -> List[Tuple[str, int]]:
        """
        采样(会过滤掉测试样本), 返回所有训练样本
        """
        available_games = []
        index = KGSIndex(data_directory=self.data_dir)
        index.load_index()
        for fileinfo in index.file_info:
            filename = fileinfo['filename']
            year = int(filename.split('-')[1].split('_')[0])
            if year > self.cap_year:
                continue
            if 'num_games' in fileinfo.keys():
                num_games = fileinfo['num_games']
            else:
                continue
            for i in range(num_games):
                available_games.append((filename, i))
        print('total num games: ' + str(len(available_games)))

        sample_set = set()
        for sample in available_games:
            if sample not in self.test_games:
                sample_set.add(sample)
        print('Drawn all samples, ie ' + str(len(sample_set)) + ' samples:')
        return list(sample_set)

    def draw_training_games(self) -> None:
        """
        将训练样本写入self.train_games(会过滤掉self.test_games中的测试样本)
        """
        index = KGSIndex(data_directory=self.data_dir)
        index.load_index()
        for file_info in index.file_info:
            filename = file_info['filename']
            year = int(filename.split('-')[1].split('_')[0])
            if year > self.cap_year:
                continue
            num_games = file_info['num_games']
            for i in range(num_games):
                sample = (filename, i)
                if sample not in self.test_games:  # 过滤掉test game
                    self.train_games.append(sample)
        print('total num training games: ' + str(len(self.train_games)))