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
实现DenseNet

实现说明:
https://tech.foxrelax.com/classics_net/densenet/
"""


def conv_block(input_channels: int, num_channels: int) -> nn.Module:
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))


class DenseBlock(nn.Module):

    def __init__(self, num_convs: int, input_channels: int,
                 num_channels: int) -> None:
        super().__init__()
        layers = []
        for i in range(num_convs):
            layers.append(
                conv_block(num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layers)

    def forward(self, X: Tensor) -> Tensor:
        for blk in self.net:
            Y = blk(X)
            # 连接通道维度上每个块的输入和输出
            X = torch.cat((X, Y), dim=1)
        return X


def transition_block(input_channels: int, num_channels: int) -> None:
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))


def densenet() -> nn.Module:
    """
    实现DenseNet

    >>> x = torch.randn((256, 1, 96, 96))
    >>> net = densenet()
    >>> assert net(x).shape == (256, 10)

    输入:
    x.shape: [batch_size, 1, 96, 96)

    输出:
    output.shape: [batch_size, 10]
    """
    # input [256, 1, 96, 96]
    # output [256, 64, 24, 24]
    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, padding=3, stride=2),
                       nn.BatchNorm2d(64), nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, padding=1, stride=2))
    num_channels, growth_rate = 64, 32
    num_convs_in_dense_blocks = [4, 4, 4, 4]
    # output [256, 192, 24, 24]  blks[0] DenseBlock
    # output [256, 96, 12, 12]   blks[1] transition_block
    # output [256, 224, 12, 12]  blks[2] DenseBlock
    # output [256, 112, 6, 6]    blks[3] transition_block
    # output [256, 240, 6, 6]    blks[4] DenseBlock
    # output [256, 120, 3, 3]    blks[5] transition_block
    # output [256, 248, 3, 3]    blks[6] DenseBlock
    # output [256, 248, 3, 3]    blks[7] transition_block
    blks = []
    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        blks.append(DenseBlock(num_convs, num_channels, growth_rate))
        # 上一个稠密块的输出通道数
        num_channels += num_convs * growth_rate
        # 在稠密块之间添加一个转换层, 使通道数量减半
        if i != len(num_convs_in_dense_blocks) - 1:
            blks.append(transition_block(num_channels, num_channels // 2))
            num_channels = num_channels // 2

    # output [256, 248, 3, 3])   BatchNorm2d
    # output [256, 248, 1, 1])   AdaptiveMaxPool2d
    # output [256, 248])         Flatten
    # output [256, 10])          Linear
    net = nn.Sequential(b1, *blks, nn.BatchNorm2d(num_channels), nn.ReLU(),
                        nn.AdaptiveMaxPool2d((1, 1)), nn.Flatten(),
                        nn.Linear(num_channels, 10))

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
    train_iter, test_iter = load_data_fashion_mnist(batch_size=128,
                                                    resize=(96, 96))
    net = densenet()
    kwargs = {
        'num_epochs': 10,
        'loss': nn.CrossEntropyLoss(reduction='mean'),
        'optimizer': torch.optim.SGD(net.parameters(), lr=0.1)
    }
    history = train_gpu(net, train_iter, test_iter, **kwargs)
    plot_history(history)


if __name__ == '__main__':
    run()
# training on cuda
# epoch 0, step 469, train loss 0.612, train acc 0.801: 100%|██████████| 469/469 [00:48<00:00,  9.71it/s]
# epoch 0, step 469, train loss 0.612, train acc 0.801, test acc 0.703
# epoch 1, step 469, train loss 0.315, train acc 0.885: 100%|██████████| 469/469 [00:48<00:00,  9.63it/s]
# epoch 1, step 469, train loss 0.315, train acc 0.885, test acc 0.873
# epoch 2, step 469, train loss 0.264, train acc 0.902: 100%|██████████| 469/469 [00:48<00:00,  9.59it/s]
# epoch 2, step 469, train loss 0.264, train acc 0.902, test acc 0.891
# epoch 3, step 469, train loss 0.227, train acc 0.916: 100%|██████████| 469/469 [00:49<00:00,  9.54it/s]
# epoch 3, step 469, train loss 0.227, train acc 0.916, test acc 0.831
# epoch 4, step 469, train loss 0.207, train acc 0.924: 100%|██████████| 469/469 [00:49<00:00,  9.52it/s]
# epoch 4, step 469, train loss 0.207, train acc 0.924, test acc 0.879
# epoch 5, step 469, train loss 0.188, train acc 0.931: 100%|██████████| 469/469 [00:49<00:00,  9.50it/s]
# epoch 5, step 469, train loss 0.188, train acc 0.931, test acc 0.872
# epoch 6, step 469, train loss 0.172, train acc 0.937: 100%|██████████| 469/469 [00:49<00:00,  9.47it/s]
# epoch 6, step 469, train loss 0.172, train acc 0.937, test acc 0.899
# epoch 7, step 469, train loss 0.156, train acc 0.941: 100%|██████████| 469/469 [00:49<00:00,  9.48it/s]
# epoch 7, step 469, train loss 0.156, train acc 0.941, test acc 0.901
# epoch 8, step 469, train loss 0.144, train acc 0.946: 100%|██████████| 469/469 [00:49<00:00,  9.39it/s]
# epoch 8, step 469, train loss 0.144, train acc 0.946, test acc 0.924
# epoch 9, step 469, train loss 0.128, train acc 0.952: 100%|██████████| 469/469 [00:49<00:00,  9.45it/s]
# epoch 9, step 469, train loss 0.128, train acc 0.952, test acc 0.914
# train loss 0.128, train acc 0.952, test acc 0.914
# 1573.2 examples/sec on cuda