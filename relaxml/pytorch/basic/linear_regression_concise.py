from typing import List, Tuple
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
"""
线性回归简洁实现

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


def load_array(data_arrays: List[Tensor],
               batch_size: int,
               is_train: bool = True) -> DataLoader:
    """
    构造一个PyTorch数据迭代器

    >>> num_examples = 1000
    >>> true_w = torch.tensor([2, -3.4])
    >>> true_b = 4.2
    >>> features, labels = synthetic_data(true_w, true_b, num_examples)
    >>> batch_size = 10
    >>> for X, y in load_array((features, labels), batch_size):
    >>>     assert X.shape == (batch_size, 2)
    >>>     assert y.shape == (batch_size, 1)
    >>>     break
    """
    dataset = TensorDataset(*data_arrays)
    return DataLoader(dataset, batch_size, shuffle=is_train)


def linreg() -> nn.Module:
    """
    线性回归模型
    """
    net = nn.Sequential(nn.Linear(2, 1))
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)
    return net


def train(true_w: Tensor, true_b: Tensor) -> Tuple[Tensor, Tensor]:
    """
    训练
    """
    num_epochs, lr, batch_size = 3, 0.03, 10
    net = linreg()
    loss = nn.MSELoss()
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    features, labels = synthetic_data(true_w, true_b, 1000)
    for epoch in range(num_epochs):
        for X, y in load_array((features, labels), batch_size):
            l = loss(net(X), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {l:f}')
    w = net[0].weight.data
    b = net[0].bias.data
    return w.reshape(-1), b


if __name__ == '__main__':
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    w, b = train(true_w, true_b)
    print('true_w={}, true_b={}'.format(true_w.detach().numpy(), true_b))
    print('w={}, b={}'.format(w.detach().numpy(), b.detach().numpy()))
# 输出:
# epoch 1, loss 0.000221
# epoch 2, loss 0.000090
# epoch 3, loss 0.000091
# true_w=[ 2.  -3.4], true_b=4.2
# w=[ 2.0001764 -3.400615 ], b=[4.1996145]