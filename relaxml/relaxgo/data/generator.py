from typing import List, Tuple
import glob
import numpy as np


class DataGenerator:
    """
    训练数据生成器
    """

    def __init__(self,
                 data_type: str = 'train',
                 data_directory: str = '../data/kgs',
                 samples: List[Tuple[str, int]] = []) -> None:
        """
        参数:
        data_type: train | test
        data_directory: 数据存放路径
        samples: 样本
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
        """
        self.data_type = data_type
        self.data_directory = data_directory
        self.samples = samples
        self.files = set(file_name for file_name, index in samples)
        self.num_samples = None

    def get_num_samples(self, batch_size: int = 128) -> int:
        """
        返回样本数
        """
        if self.num_samples is not None:
            return self.num_samples
        else:
            self.num_samples = 0
            for X, y in self._generate(batch_size=batch_size):
                self.num_samples += X.shape[0]
            return self.num_samples

    def _generate(self, batch_size: int = 128):
        """
        创建并返回批量数据(遍历一次就是一个epoch)

        功能和`processor.GoDataProcessor.consolidate_games()`类似, 不同的是:
        a. 前者会把数据加载到内存一次性返回(需要一个巨大的NumPy数组)
        b. `_generate()`只需要yield一个小批量数据即可

        yield的data形状:
        features: [batch_size, num_planes, board_height, board_width]
        labels: [batch_size, ]
        """
        for zip_file_name in self.files:
            file_name = zip_file_name.replace('.tar.gz', '') + self.data_type
            base = self.data_directory + '/' + file_name + '_features_*.npy'
            for feature_file in glob.glob(base):
                label_file = feature_file.replace('features', 'labels')
                x = np.load(feature_file)
                y = np.load(label_file)
                x = x.astype('float32')
                while x.shape[0] >= batch_size:
                    x_batch, x = x[:batch_size], x[batch_size:]
                    y_batch, y = y[:batch_size], y[batch_size:]
                    # x_batch.shape [batch_size, num_planes, board_height, board_width]
                    # y_batch.shape [batch_size, ]
                    yield x_batch, y_batch

    def generate(self, batch_size: int = 128):
        """
        这个会无限循环下去
        """
        while True:
            for item in self._generate(batch_size):
                yield item