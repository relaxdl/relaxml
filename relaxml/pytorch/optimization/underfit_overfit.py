from typing import Callable, List
import math
import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt
"""
模型选择、欠拟合和过拟合

说明:
https://tech.foxrelax.com/optimization/underfit_overfit/
"""


def evaluate_loss(net: nn.Module, data_iter: DataLoader,
                  loss: Callable) -> Tensor:
    """
    评估给定数据集上模型的损失
    """
    metric = [0.0] * 2  # 损失的总和, 样本数量
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric[0] += l.sum()
        metric[1] += l.numel()
    return metric[0] / metric[1]


def load_array(data_arrays: List[Tensor],
               batch_size: int,
               is_train: bool = True) -> DataLoader:
    """
    构造一个PyTorch数据迭代器
    """
    dataset = TensorDataset(*data_arrays)
    return DataLoader(dataset, batch_size, shuffle=is_train)


def train(train_features,
          test_features,
          train_labels,
          test_labels,
          num_epochs=400):
    loss = nn.MSELoss()
    input_shape = train_features.shape[-1]
    # 不设置偏置，因为我们已经在多项式特征中实现了它
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = load_array((train_features, train_labels.reshape(-1, 1)),
                            batch_size)
    test_iter = load_array((test_features, test_labels.reshape(-1, 1)),
                           batch_size,
                           is_train=False)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    history = [[], []]  # 记录: 训练集损失, 测试集的损失, 方便后续绘图
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 计算梯度并更新参数
            optimizer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
        if epoch == 0 or (epoch + 1) % 20 == 0:
            history[0].append(
                (epoch + 1, evaluate_loss(net, train_iter,
                                          loss).detach().numpy()))
            history[1].append(
                (epoch + 1, evaluate_loss(net, test_iter,
                                          loss).detach().numpy()))
    print('weight:', net[0].weight.data.numpy())

    plt.figure(figsize=(6, 4))
    # 训练集损失, 测试集的损失
    plt.plot(*zip(*history[0]), '-', label='train loss')
    plt.plot(*zip(*history[1]), 'm--', label='test loss')
    plt.xlabel('epoch')
    plt.xlim((1, num_epochs))
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    max_degree = 20  # 多项式的最大阶数
    n_train, n_test = 100, 100  # 训练和测试数据集大小
    true_w = np.zeros(max_degree)  # 分配大量的空间
    true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])
    features = np.random.normal(size=(n_train + n_test, 1))
    np.random.shuffle(features)
    # poly_features的形状: (`n_train` + `n_test`, `max_degree`)
    poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
    for i in range(max_degree):
        poly_features[:, i] /= math.gamma(i + 1)  # `gamma(n)` = (n-1)!
    # `labels`的形状: (`n_train` + `n_test`,)
    labels = np.dot(poly_features, true_w)
    labels += np.random.normal(scale=0.1, size=labels.shape)
    assert poly_features.shape == (n_train + n_test, max_degree)
    assert labels.shape == (n_train + n_test, )

    true_x, features, poly_features, labels = [
        torch.tensor(x, dtype=torch.float32)
        for x in [true_w, features, poly_features, labels]
    ]
    features[:2], poly_features[:2, :], labels[:2]

    # demo1:
    # train(poly_features[:n_train, :4], poly_features[n_train:, :4],
    #       labels[:n_train], labels[n_train:])

    # demo2:
    # train(poly_features[:n_train, :2], poly_features[n_train:, :2],
    #       labels[:n_train], labels[n_train:])

    # demo3:
    train(poly_features[:n_train, :], poly_features[n_train:, :],
          labels[:n_train], labels[n_train:])
