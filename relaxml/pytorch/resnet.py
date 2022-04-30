from typing import Tuple, Union, List
import time
import sys
import torch
from torch import Tensor
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from matplotlib import pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
"""
实现ResNet

实现说明:
https://tech.foxrelax.com/classics_net/resnet/
"""


class Residual(nn.Module):
    def __init__(self,
                 input_channels: int,
                 num_channels: int,
                 use_1x1conv: bool = False,
                 stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels,
                               num_channels,
                               kernel_size=3,
                               padding=1,
                               stride=stride)
        self.conv2 = nn.Conv2d(num_channels,
                               num_channels,
                               kernel_size=3,
                               padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels,
                                   num_channels,
                                   kernel_size=1,
                                   stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X: Tensor) -> Tensor:
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


def resnet_block(input_channels: int,
                 num_channels: int,
                 num_residuals: int,
                 first_block: bool = False) -> List[Residual]:
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(
                Residual(input_channels,
                         num_channels,
                         use_1x1conv=True,
                         stride=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


def resnet() -> nn.Module:
    """
    实现ResNet

    >>> x = torch.randn((256, 1, 96, 96))
    >>> net = resnet()
    >>> assert net(x).shape == (256, 10)

    输入:
    x [batch_size, 1, 96, 96)

    输出:
    output [batch_size, 10]
    """

    # input [256, 3, 96, 96]
    # output [256, 64, 24, 24]
    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, padding=3, stride=2),
                       nn.BatchNorm2d(64), nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, padding=1, stride=2))

    # output [256, 64, 24, 24]
    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))

    # output [256, 128, 12, 12]
    b3 = nn.Sequential(*resnet_block(64, 128, 2))

    # output [256, 256, 6, 6]
    b4 = nn.Sequential(*resnet_block(128, 256, 2))

    # output [256, 512, 3, 3]
    b5 = nn.Sequential(*resnet_block(256, 512, 2))

    # output [256, 512, 1, 1]  AdaptiveAvgPool2d
    #        [256, 512]        Flatten
    #        [256, 10]         Linear
    net = nn.Sequential(b1, b2, b3, b4, b5, nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(), nn.Linear(512, 10))

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)

    return net


def load_data_fashion_mnist(
        batch_size: int,
        resize: Union[int, Tuple[int, int]] = None,
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
    y_hat [batch_size, num_classes]
    y [batch_size,]
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
    train_iter, test_iter = load_data_fashion_mnist(batch_size=128,
                                                    resize=(96, 96))
    net = resnet()
    kwargs = {
        'num_epochs': 10,
        'loss': nn.CrossEntropyLoss(reduction='mean'),
        'optimizer': torch.optim.SGD(net.parameters(), lr=0.05)
    }
    history = train_gpu(net, train_iter, test_iter, **kwargs)
    plot_history(history)


if __name__ == '__main__':
    run()
# training on cuda
# epoch 0, step 469, train loss 0.424, train acc 0.849: 100%|██████████| 469/469 [01:01<00:00,  7.62it/s]
# epoch 0, step 469, train loss 0.424, train acc 0.849, test acc 0.870
# epoch 1, step 469, train loss 0.230, train acc 0.915: 100%|██████████| 469/469 [01:02<00:00,  7.45it/s]
# epoch 1, step 469, train loss 0.230, train acc 0.915, test acc 0.830
# epoch 2, step 469, train loss 0.177, train acc 0.934: 100%|██████████| 469/469 [01:03<00:00,  7.39it/s]
# epoch 2, step 469, train loss 0.177, train acc 0.934, test acc 0.882
# epoch 3, step 469, train loss 0.136, train acc 0.950: 100%|██████████| 469/469 [01:03<00:00,  7.34it/s]
# epoch 3, step 469, train loss 0.136, train acc 0.950, test acc 0.914
# epoch 4, step 469, train loss 0.100, train acc 0.964: 100%|██████████| 469/469 [01:04<00:00,  7.31it/s]
# epoch 4, step 469, train loss 0.100, train acc 0.964, test acc 0.906
# epoch 5, step 469, train loss 0.075, train acc 0.973: 100%|██████████| 469/469 [01:04<00:00,  7.30it/s]
# epoch 5, step 469, train loss 0.075, train acc 0.973, test acc 0.913
# epoch 6, step 469, train loss 0.055, train acc 0.981: 100%|██████████| 469/469 [01:04<00:00,  7.29it/s]
# epoch 6, step 469, train loss 0.055, train acc 0.981, test acc 0.920
# epoch 7, step 469, train loss 0.036, train acc 0.988: 100%|██████████| 469/469 [01:04<00:00,  7.27it/s]
# epoch 7, step 469, train loss 0.036, train acc 0.988, test acc 0.898
# epoch 8, step 469, train loss 0.028, train acc 0.991: 100%|██████████| 469/469 [01:05<00:00,  7.19it/s]
# epoch 8, step 469, train loss 0.028, train acc 0.991, test acc 0.921
# epoch 9, step 469, train loss 0.021, train acc 0.993: 100%|██████████| 469/469 [01:04<00:00,  7.25it/s]
# epoch 9, step 469, train loss 0.021, train acc 0.993, test acc 0.925
# train loss 0.021, train acc 0.993, test acc 0.925
# 1135.5 examples/sec on cuda