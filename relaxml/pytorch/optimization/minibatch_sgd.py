from typing import Dict, Iterator, Tuple, List, Callable
import numpy as np
import hashlib
import requests
import os
import sys
import time
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
"""
小批量随机梯度下降

说明:
https://tech.foxrelax.com/optimization/minibatch_sgd/
"""


class Timer:

    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.tik = time.time()

    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        return sum(self.times) / len(self.times)

    def sum(self):
        return sum(self.times)

    def cumsum(self):
        return np.array(self.times).cumsum().tolist()


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


def sgd(params: List[Tensor], states: List[Tensor],
        hyperparams: Dict[str, Tensor]) -> None:
    for p in params:
        p.data.sub_(hyperparams['lr'] * p.grad)
        p.grad.zero_()


def linreg(X: Tensor, w: Tensor, b: Tensor) -> Tensor:
    """
    线性回归模型
    """
    return torch.matmul(X, w) + b


def squared_loss(y_hat: Tensor, y: Tensor) -> Tensor:
    """
    均方损失
    """
    return (y_hat - y.reshape(y_hat.shape))**2 / 2


def train_optimization(trainer_fn: Callable,
                       states: List[Tensor],
                       hyperparams: Dict[str, Tensor],
                       data_iter: Iterator,
                       feature_dim: int,
                       num_epochs: int = 2) -> None:
    # Initialization
    w = torch.normal(mean=0.0,
                     std=0.01,
                     size=(feature_dim, 1),
                     requires_grad=True)
    b = torch.zeros((1), requires_grad=True)
    net, loss = lambda X: linreg(X, w, b), squared_loss

    times = []
    history = [[], [], []]  # 记录: 训练集损失, 方便后续绘图
    num_batches = len(data_iter)
    for epoch in range(num_epochs):
        # 训练
        metric_train = [0.0] * 2  # 统计: 训练集损失之和, 训练集样本数量之和
        train_iter_tqdm = tqdm(data_iter, file=sys.stdout)
        for i, (X, y) in enumerate(train_iter_tqdm):
            t_start = time.time()
            l = loss(net(X), y).mean()  # 计算loss的均值
            l.backward()
            trainer_fn([w, b], states, hyperparams)

            with torch.no_grad():
                metric_train[0] += float(l * X.shape[0])
                metric_train[1] += float(X.shape[0])
            times.append(time.time() - t_start)
            train_loss = metric_train[0] / metric_train[1]
            history[0].append((epoch + (i + 1) / num_batches, train_loss))
            train_iter_tqdm.desc = f'epoch {epoch}, step {i+1}, train loss {train_loss:.3f}'

    print(f'train loss {train_loss:.3f}')
    print(f'{metric_train[1] * num_epochs / sum(times):.1f} examples/sec')

    # plot
    plt.figure(figsize=(6, 4))
    # 训练集损失
    plt.plot(*zip(*history[0]), '-', label='train loss')
    plt.xlabel('epoch')
    # 从epoch=1开始显示, 0-1这个范围的数据丢弃不展示,
    # 因为只有训练完成1个epochs之后, 才会有第一条test acc记录
    plt.xlim((1, num_epochs))
    plt.grid()
    plt.legend()
    plt.show()


def train_concise_optimization(trainer_fn: Callable,
                               hyperparams: Dict[str, Tensor],
                               data_iter: Iterator,
                               feature_dim: int,
                               num_epochs: int = 4) -> None:
    # Initialization
    net = nn.Sequential(nn.Linear(feature_dim, 1))

    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.normal_(m.weight, std=0.01)

    net.apply(init_weights)
    optimizer = trainer_fn(net.parameters(), **hyperparams)
    loss = nn.MSELoss(reduction='mean')

    times = []
    history = [[], [], []]  # 记录: 训练集损失, 方便后续绘图
    num_batches = len(data_iter)
    for epoch in range(num_epochs):
        # 训练
        metric_train = [0.0] * 2  # 统计: 训练集损失之和, 训练集样本数量之和
        train_iter_tqdm = tqdm(data_iter, file=sys.stdout)
        for i, (X, y) in enumerate(train_iter_tqdm):
            t_start = time.time()
            optimizer.zero_grad()
            out = net(X)
            y = y.reshape(out.shape)
            l = loss(out, y) / 2  # PyTorch的MSE Loss实现和我们的不同
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric_train[0] += float(l * X.shape[0])
                metric_train[1] += float(X.shape[0])
            times.append(time.time() - t_start)
            train_loss = metric_train[0] / metric_train[1]
            history[0].append((epoch + (i + 1) / num_batches, train_loss))
            train_iter_tqdm.desc = f'epoch {epoch}, step {i+1}, train loss {train_loss:.3f}'

    print(f'train loss {train_loss:.3f}')
    print(f'{metric_train[1] * num_epochs / sum(times):.1f} examples/sec')

    # plot
    plt.figure(figsize=(6, 4))
    # 训练集损失
    plt.plot(*zip(*history[0]), '-', label='train loss')
    plt.xlabel('epoch')
    # 从epoch=1开始显示, 0-1这个范围的数据丢弃不展示,
    # 因为只有训练完成1个epochs之后, 才会有第一条test acc记录
    plt.xlim((1, num_epochs))
    plt.grid()
    plt.legend()
    plt.show()


def train_sgd(lr, batch_size, num_epochs=2):
    data_iter, feature_dim = load_data_airfoil_self_noise(batch_size)
    train_optimization(sgd, None, {'lr': lr}, data_iter, feature_dim,
                       num_epochs)


def demo1():
    # train_sgd(1, 1500, 10)
    # train_sgd(0.005, 1)
    # train_sgd(.4, 100)
    train_sgd(.05, 10)

def demo2():
    data_iter, feature_dim = load_data_airfoil_self_noise(10)
    trainer = torch.optim.SGD
    train_concise_optimization(trainer, {'lr': 0.05}, data_iter, feature_dim)

if __name__ == '__main__':
    demo2()