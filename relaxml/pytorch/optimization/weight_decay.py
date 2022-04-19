from typing import List, Tuple, Callable
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt
"""
权重衰减

说明:
https://tech.foxrelax.com/optimization/weight_decay/
"""


def synthetic_data(w: Tensor,
                   b: Tensor,
                   num_examples: int = 1000) -> Tuple[Tensor, Tensor]:
    """
    生成: y=Xw+b+噪声

    >>> num_examples = 1000
    >>> true_w = torch.tensor([2, -3.4])
    >>> true_b = 4.2
    >>> features, labels = synthetic_data(true_w, true_b, num_examples)
    >>> assert features.shape == (num_examples, 2)
    >>> assert labels.shape == (num_examples, 1)
    """
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)

    # X.shape [num_examples, num_features]
    # y.shape [num_examples, 1]
    return X, y.reshape((-1, 1))


def load_linreg_synthetic(data_arrays: List[Tensor],
                          batch_size: int,
                          is_train: bool = True) -> DataLoader:
    """
    加载线性回归数据集

    >>> num_examples = 1000
    >>> true_w = torch.tensor([2, -3.4])
    >>> true_b = 4.2
    >>> features, labels = synthetic_data(true_w, true_b, num_examples)
    >>> batch_size = 10
    >>> for X, y in load_linreg_synthetic((features, labels), batch_size):
    >>>     assert X.shape == (batch_size, 2)
    >>>     assert y.shape == (batch_size, 1)
    >>>     break
    """
    dataset = TensorDataset(*data_arrays)
    return DataLoader(dataset, batch_size, shuffle=is_train)


def init_params() -> List[Tensor]:
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]


def l2_penalty(w: Tensor) -> Tensor:
    return torch.sum(w.pow(2)) / 2


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


def sgd(params: List[Tensor], lr: float, batch_size: int) -> None:
    """
    小批量梯度下降
    """
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


def train(lambd):
    w, b = init_params()
    net, loss = lambda X: linreg(X, w, b), squared_loss
    num_epochs, lr = 100, 0.003
    history = [[], []]  # 记录: 训练集损失, 测试集的损失, 方便后续绘图
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 增加了L2范数惩罚项, 广播机制使l2_penalty(w)成为一个长度为`batch_size`的向量
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            history[0].append(
                (epoch + 1, evaluate_loss(net, train_iter,
                                          loss).detach().numpy()))
            history[1].append(
                (epoch + 1, evaluate_loss(net, test_iter,
                                          loss).detach().numpy()))
    print('w的L2范数是：', torch.norm(w).item())
    plt.figure(figsize=(6, 4))
    # 训练集损失, 测试集的损失
    plt.plot(*zip(*history[0]), '-', label='train loss')
    plt.plot(*zip(*history[1]), 'm--', label='test loss')
    plt.xlabel('epoch')
    plt.xlim((1, num_epochs))
    plt.grid()
    plt.legend()
    plt.show()


def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss()
    num_epochs, lr = 100, 0.003
    # 偏置参数没有衰减
    trainer = torch.optim.SGD([{
        'params': net[0].weight,
        'weight_decay': wd
    }, {
        'params': net[0].bias
    }],
                              lr=lr)
    history = [[], []]  # 记录: 训练集损失, 测试集的损失, 方便后续绘图
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            history[0].append(
                (epoch + 1, evaluate_loss(net, train_iter,
                                          loss).detach().numpy()))
            history[1].append(
                (epoch + 1, evaluate_loss(net, test_iter,
                                          loss).detach().numpy()))
    print('w的L2范数：', net[0].weight.norm().item())
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
    n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
    true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
    train_data = synthetic_data(true_w, true_b, n_train)
    train_iter = load_linreg_synthetic(train_data, batch_size)
    test_data = synthetic_data(true_w, true_b, n_train)
    test_iter = load_linreg_synthetic(test_data, batch_size, is_train=False)

    # demo1
    # train(lambd=0.0)

    # demo2
    # train(lambd=3.0)

    # demo3
    # train_concise(0)

    # demo4
    train_concise(3.0)
