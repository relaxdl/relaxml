from typing import Tuple, List
import sys
import time
import torch
from torch import Tensor
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from matplotlib import pyplot as plt
from tqdm import tqdm
"""
实现LeNet

实现说明:
https://tech.foxrelax.com/classics_net/lenet/
"""


class Reshape(nn.Module):

    def forward(self, x: Tensor) -> Tensor:
        return x.view(-1, 1, 28, 28)


def lenet() -> nn.Module:
    """
    实现LeNet

    >>> x = torch.randn((256, 1, 28, 28))
    >>> net = lenet()
    >>> assert net(x).shape == (256, 10)

    输入:
    x.shape: [batch_size, 1, 28, 28]

    输出:
    output.shape: [batch_size, 10]
    """
    net = nn.Sequential(
        # output [batch_size, 1, 28, 28]
        Reshape(),
        # output [batch_size, 6, 28, 28]
        nn.Conv2d(1, 6, kernel_size=5, padding=2),
        # output [batch_size, 6, 28, 28]
        nn.Sigmoid(),
        # output [batch_size, 6, 14, 14]
        nn.AvgPool2d(kernel_size=2, stride=2),
        # output [batch_size, 16, 10, 10]
        nn.Conv2d(6, 16, kernel_size=5),
        # output [batch_size, 16, 10, 10]
        nn.Sigmoid(),
        # output [batch_size, 16, 5, 5]
        nn.AvgPool2d(kernel_size=2, stride=2),
        # output [batch_size, 400]
        nn.Flatten(),
        # output [batch_size, 120]
        nn.Linear(16 * 5 * 5, 120),
        # output [batch_size, 120]
        nn.Sigmoid(),
        # output [batch_size, 84]
        nn.Linear(120, 84),
        # output [batch_size, 84]
        nn.Sigmoid(),
        # output [batch_size, 10]
        nn.Linear(84, 10),
    )

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)

    return net


def load_data_fashion_mnist(
        batch_size: int,
        resize: int = None,
        root: str = '../data') -> Tuple[DataLoader, DataLoader]:
    """
    下载Fashion-MNIST数据集, 然后将其加载到内存中

    1. 60000张训练图像和对应Label
    2. 10000张测试图像和对应Label
    3. 10个类别
    4. 每张图像28x28x1的分辨率

    >>> train_iter, test_iter = load_data_fashion_mnist(batch_size=256)
    >>> for x, y in train_iter:
    >>>     assert x.shape == (256, 1, 28, 28)
    >>>     assert y.shape == (256, )
    >>>     break
    """
    # 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式
    # 并除以255使得所有像素的数值均在0到1之间
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=root,
                                                    train=True,
                                                    transform=trans,
                                                    download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root=root,
                                                   train=False,
                                                   transform=trans,
                                                   download=True)
    return (DataLoader(mnist_train, batch_size, shuffle=True),
            DataLoader(mnist_test, batch_size, shuffle=False))


def accuracy(y_hat: Tensor, y: Tensor) -> Tensor:
    """
    计算预测正确的数量

    参数:
    y_hat.shape: [batch_size, num_classes]
    y.shape: [batch_size,]
    """
    _, predicted = torch.max(y_hat, 1)
    cmp = predicted.type(y.dtype) == y
    return cmp.type(y.dtype).sum()


def train_gpu(net: nn.Module,
              train_iter: DataLoader,
              test_iter: DataLoader,
              num_epochs: int = 10,
              loss: nn.Module = None,
              optimizer: Optimizer = None,
              device: torch.device = None,
              verbose: bool = False,
              save_path: str = None) -> List[List[Tuple[int, float]]]:
    """
    用GPU训练模型
    """
    if device is None:
        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')

    print('training on', device)
    net.to(device)
    if loss is None:
        loss = nn.CrossEntropyLoss(reduction='mean')
    if optimizer is None:
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    times = []
    history = [[], [], []]  # 记录: 训练集损失, 训练集准确率, 测试集准确率, 方便后续绘图
    num_batches = len(train_iter)
    best_test_acc = 0.0
    for epoch in range(num_epochs):
        # 训练
        metric_train = [0.0] * 3  # 统计: 训练集损失之和, 训练集准确数量之和, 训练集样本数量之和
        net.train()
        train_iter_tqdm = tqdm(train_iter, file=sys.stdout)
        for i, (X, y) in enumerate(train_iter_tqdm):
            t_start = time.time()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric_train[0] += float(l * X.shape[0])
                metric_train[1] += float(accuracy(y_hat, y))
                metric_train[2] += float(X.shape[0])
            times.append(time.time() - t_start)
            train_loss = metric_train[0] / metric_train[2]
            train_acc = metric_train[1] / metric_train[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                history[0].append((epoch + (i + 1) / num_batches, train_loss))
                history[1].append((epoch + (i + 1) / num_batches, train_acc))
            train_iter_tqdm.desc = f'epoch {epoch}, step {i+1}, train loss {train_loss:.3f}, train acc {train_acc:.3f}'

        # 评估
        metric_test = [0.0] * 2  # 测试准确数量之和, 测试样本数量之和
        net.eval()
        with torch.no_grad():
            for X, y in test_iter:
                X, y = X.to(device), y.to(device)
                metric_test[0] += float(accuracy(net(X), y))
                metric_test[1] += float(X.shape[0])
            test_acc = metric_test[0] / metric_test[1]
            history[2].append((epoch + 1, test_acc))
            print(f'epoch {epoch}, step {i+1}, train loss {train_loss:.3f}, '
                  f'train acc {train_acc:.3f}, test acc {test_acc:.3f}')
            if test_acc > best_test_acc and save_path:
                best_test_acc = test_acc
                torch.save(net.state_dict(), save_path)

    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric_train[2] * num_epochs / sum(times):.1f} '
          f'examples/sec on {str(device)}')
    return history


def plot_history(
    history: List[List[Tuple[int, float]]], figsize: Tuple[int, int] = (6, 4)
) -> None:
    plt.figure(figsize=figsize)
    # 训练集损失, 训练集准确率, 测试集准确率
    num_epochs = len(history[2])
    plt.plot(*zip(*history[0]), '-', label='train loss')
    plt.plot(*zip(*history[1]), 'm--', label='train acc')
    plt.plot(*zip(*history[2]), 'g-.', label='test acc')
    plt.xlabel('epoch')
    # 从epoch=1开始显示, 0-1这个范围的数据丢弃不展示,
    # 因为只有训练完成1个epochs之后, 才会有第一条test acc记录
    plt.xlim((1, num_epochs))
    plt.grid()
    plt.legend()
    plt.show()


def run() -> None:
    train_iter, test_iter = load_data_fashion_mnist(batch_size=256)
    net = lenet()
    kwargs = {
        'num_epochs': 20,
        'loss': nn.CrossEntropyLoss(reduction='mean'),
        'optimizer': torch.optim.Adam(net.parameters(), lr=0.01)
    }
    history = train_gpu(net, train_iter, test_iter, **kwargs)
    plot_history(history)


if __name__ == '__main__':
    run()
# training on cuda
# epoch 0, step 235, train loss 1.294, train acc 0.503: 100%|██████████| 235/235 [00:04<00:00, 49.77it/s]
# epoch 0, step 235, train loss 1.294, train acc 0.503, test acc 0.746
# epoch 1, step 235, train loss 0.535, train acc 0.795: 100%|██████████| 235/235 [00:04<00:00, 49.47it/s]
# epoch 1, step 235, train loss 0.535, train acc 0.795, test acc 0.818
# epoch 2, step 235, train loss 0.438, train acc 0.838: 100%|██████████| 235/235 [00:04<00:00, 50.86it/s]
# epoch 2, step 235, train loss 0.438, train acc 0.838, test acc 0.842
# epoch 3, step 235, train loss 0.394, train acc 0.854: 100%|██████████| 235/235 [00:04<00:00, 50.57it/s]
# epoch 3, step 235, train loss 0.394, train acc 0.854, test acc 0.843
# epoch 4, step 235, train loss 0.366, train acc 0.864: 100%|██████████| 235/235 [00:04<00:00, 49.54it/s]
# epoch 4, step 235, train loss 0.366, train acc 0.864, test acc 0.857
# epoch 5, step 235, train loss 0.342, train acc 0.874: 100%|██████████| 235/235 [00:04<00:00, 49.36it/s]
# epoch 5, step 235, train loss 0.342, train acc 0.874, test acc 0.860
# epoch 6, step 235, train loss 0.328, train acc 0.877: 100%|██████████| 235/235 [00:04<00:00, 49.08it/s]
# epoch 6, step 235, train loss 0.328, train acc 0.877, test acc 0.867
# epoch 7, step 235, train loss 0.313, train acc 0.883: 100%|██████████| 235/235 [00:04<00:00, 48.16it/s]
# epoch 7, step 235, train loss 0.313, train acc 0.883, test acc 0.874
# epoch 8, step 235, train loss 0.304, train acc 0.887: 100%|██████████| 235/235 [00:04<00:00, 50.83it/s]
# epoch 8, step 235, train loss 0.304, train acc 0.887, test acc 0.871
# epoch 9, step 235, train loss 0.292, train acc 0.890: 100%|██████████| 235/235 [00:04<00:00, 50.52it/s]
# epoch 9, step 235, train loss 0.292, train acc 0.890, test acc 0.877
# epoch 10, step 235, train loss 0.283, train acc 0.893: 100%|██████████| 235/235 [00:04<00:00, 50.43it/s]
# epoch 10, step 235, train loss 0.283, train acc 0.893, test acc 0.874
# epoch 11, step 235, train loss 0.276, train acc 0.897: 100%|██████████| 235/235 [00:04<00:00, 50.58it/s]
# epoch 11, step 235, train loss 0.276, train acc 0.897, test acc 0.881
# epoch 12, step 235, train loss 0.266, train acc 0.900: 100%|██████████| 235/235 [00:04<00:00, 50.04it/s]
# epoch 12, step 235, train loss 0.266, train acc 0.900, test acc 0.883
# epoch 13, step 235, train loss 0.260, train acc 0.902: 100%|██████████| 235/235 [00:04<00:00, 48.90it/s]
# epoch 13, step 235, train loss 0.260, train acc 0.902, test acc 0.884
# epoch 14, step 235, train loss 0.250, train acc 0.906: 100%|██████████| 235/235 [00:04<00:00, 49.02it/s]
# epoch 14, step 235, train loss 0.250, train acc 0.906, test acc 0.888
# epoch 15, step 235, train loss 0.248, train acc 0.906: 100%|██████████| 235/235 [00:04<00:00, 48.13it/s]
# epoch 15, step 235, train loss 0.248, train acc 0.906, test acc 0.892
# epoch 16, step 235, train loss 0.241, train acc 0.909: 100%|██████████| 235/235 [00:04<00:00, 50.03it/s]
# epoch 16, step 235, train loss 0.241, train acc 0.909, test acc 0.885
# epoch 17, step 235, train loss 0.233, train acc 0.913: 100%|██████████| 235/235 [00:04<00:00, 50.49it/s]
# epoch 17, step 235, train loss 0.233, train acc 0.913, test acc 0.891
# epoch 18, step 235, train loss 0.229, train acc 0.913: 100%|██████████| 235/235 [00:04<00:00, 49.33it/s]
# epoch 18, step 235, train loss 0.229, train acc 0.913, test acc 0.891
# epoch 19, step 235, train loss 0.223, train acc 0.917: 100%|██████████| 235/235 [00:04<00:00, 50.37it/s]
# epoch 19, step 235, train loss 0.223, train acc 0.917, test acc 0.886
# train loss 0.223, train acc 0.917, test acc 0.886
# 83041.6 examples/sec on cuda