import os
import requests
import hashlib
import numpy as np
from typing import List, Tuple
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


def download(cache_dir: str = '../data') -> str:
    """
    下载数据
    """
    sha1_hash = '76e5be1548fd8222e5074cf0faae75edff8cf93f'
    url = 'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/airfoil_self_noise.dat'
    fname = os.path.join(cache_dir, url.split('/ml/')[-1])
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
    # e.g. ../data/airfoil_self_noise.dat
    return fname


def load_array(data_arrays: List[Tensor],
               batch_size: int,
               is_train: bool = True) -> DataLoader:
    """
    构造一个PyTorch数据迭代器
    """
    dataset = TensorDataset(*data_arrays)
    return DataLoader(dataset, batch_size, shuffle=is_train)


def load_data_airfoil_self_noise(
        batch_size: int = 10,
        n: int = 1500,
        cache_dir: str = '../data') -> Tuple[DataLoader, int]:
    """
    >>> batch_size = 32
    >>> data_iter, feature_dim = load_data_airfoil_self_noise(32)
    >>> for X, y in data_iter:
    >>>     print(X[0])
    >>>     print(y[0])
    >>>     assert X.shape == (32, 5)
    >>>     assert y.shape == (32, )
    >>>     break
        tensor([-0.5986,  0.5270,  0.1695, -0.7234,  0.9275])
        tensor(-0.1451)
    
    参数:
    batch_size: 批量大小
    n: 返回数据集包含的样本数
    cache_dir: 数据在本地磁盘的缓存目录
    """
    data = np.genfromtxt(download(cache_dir), dtype=np.float32, delimiter='\t')
    data = torch.from_numpy((data - data.mean(axis=0)) / data.std(axis=0))
    data_iter = load_array((data[:n, :-1], data[:n, -1]),
                           batch_size,
                           is_train=True)
    return data_iter, data.shape[1] - 1


if __name__ == '__main__':
    batch_size = 32
    data_iter, feature_dim = load_data_airfoil_self_noise(32)
    for X, y in data_iter:
        print(X[0])
        print(y[0])
        assert X.shape == (32, 5)
        assert y.shape == (32, )
        break
# tensor([-0.5986,  0.5270,  0.1695, -0.7234,  0.9275])
# tensor(-0.1451)