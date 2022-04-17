from typing import List, Tuple, Union, Iterator
import random
import torch
from torch import Tensor
"""
线性回归从零实现

说明:
https://tech.foxrelax.com/basic/linear_regression/
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


def data_iter(batch_size: int, features: Tensor, labels: Tensor) -> Iterator:
    """
    >>> batch_size = 10
    >>> num_examples = 1000
    >>> true_w = torch.tensor([2, -3.4])
    >>> true_b = 4.2
    >>> features, labels = synthetic_data(true_w, true_b, num_examples)
    >>> for X, y in data_iter(batch_size, features, labels):
    >>>     assert X.shape == (batch_size, 2)
    >>>     assert y.shape == (batch_size, 1)
    >>>     break
    """
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i +
                                                   batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


def gen_params() -> Tuple[Tensor, Tensor]:
    """
    生成模型参数: [w, b]
    """
    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return w, b


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


def sgd(params: Union[Tuple, List], lr: float, batch_size: int) -> None:
    """
    小批量随机梯度下降
    """
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


def train(true_w: Tensor, true_b: Tensor) -> Tuple[Tensor, Tensor]:
    """
    训练
    """
    w, b = gen_params()
    num_epochs, lr, batch_size = 3, 0.03, 10
    net = linreg
    loss = squared_loss
    features, labels = synthetic_data(true_w, true_b, 1000)
    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y)
            l.sum().backward()
            sgd([w, b], lr, batch_size)
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
    return w.reshape(-1), b


if __name__ == '__main__':
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    w, b = train(true_w, true_b)
    print('true_w={}, true_b={}'.format(true_w.detach().numpy(), true_b))
    print('w={}, b={}'.format(w.detach().numpy(), b.detach().numpy()))
# 输出:
# epoch 1, loss 0.027572
# epoch 2, loss 0.000102
# epoch 3, loss 0.000048
# true_w=[ 2.  -3.4], true_b=4.2
# w=[ 1.9998878 -3.3999581], b=[4.1997714]