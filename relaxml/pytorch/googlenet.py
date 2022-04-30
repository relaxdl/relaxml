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
实现GoogLeNet

实现说明:
https://tech.foxrelax.com/classics_net/googlenet/
"""


class Inception(nn.Module):
    """
    实现Inception
    """

    # `c1`--`c4` 是每条路径的输出通道数
    def __init__(self, in_channels: int, c1: int, c2: Tuple[int, int],
                 c3: Tuple[int, int], c4: int, **kwargs) -> None:
        super().__init__()
        # 线路1, 单1 x 1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2, 1 x 1卷积层后接3 x 3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3, 1 x 1卷积层后接5 x 5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4, 3 x 3最大汇聚层后接1 x 1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, X: Tensor) -> Tensor:
        p1 = F.relu(self.p1_1(X))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(X))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(X))))
        p4 = F.relu(self.p4_2(self.p4_1(X)))
        # 在通道维度上连结输出
        return torch.cat((p1, p2, p3, p4), dim=1)


def googlenet() -> nn.Module:
    """
    实现GoogLeNet

    >>> x = torch.randn((256, 1, 96, 96))
    >>> net = googlenet()
    >>> assert net(x).shape == (256, 10)

    输入:
    x [batch_size, 1, 96, 96)

    输出:
    output [batch_size, 10]
    """

    # input [256, 3, 96, 96]
    # output [256, 64, 24, 24]
    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                       nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    # output [256, 192, 12, 12]
    b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(),
                       nn.Conv2d(64, 192, kernel_size=3, padding=1),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    # output [256, 480, 6, 6]
    b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                       Inception(256, 128, (128, 192), (32, 96), 64),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    # output [256, 832, 3, 3]
    b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                       Inception(512, 160, (112, 224), (24, 64), 64),
                       Inception(512, 128, (128, 256), (24, 64), 64),
                       Inception(512, 112, (144, 288), (32, 64), 64),
                       Inception(528, 256, (160, 320), (32, 128), 128),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    # output [256, 1024]
    b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                       Inception(832, 384, (192, 384), (48, 128), 128),
                       nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())

    # output [256, 10]
    net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))

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
    net = googlenet()
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
# epoch 0, step 469, train loss 1.769, train acc 0.323: 100%|██████████| 469/469 [01:00<00:00,  7.75it/s]
# epoch 0, step 469, train loss 1.769, train acc 0.323, test acc 0.696
# epoch 1, step 469, train loss 0.632, train acc 0.761: 100%|██████████| 469/469 [01:01<00:00,  7.63it/s]
# epoch 1, step 469, train loss 0.632, train acc 0.761, test acc 0.815
# epoch 2, step 469, train loss 0.445, train acc 0.831: 100%|██████████| 469/469 [01:02<00:00,  7.54it/s]
# epoch 2, step 469, train loss 0.445, train acc 0.831, test acc 0.839
# epoch 3, step 469, train loss 0.369, train acc 0.860: 100%|██████████| 469/469 [01:02<00:00,  7.49it/s]
# epoch 3, step 469, train loss 0.369, train acc 0.860, test acc 0.864
# epoch 4, step 469, train loss 0.328, train acc 0.876: 100%|██████████| 469/469 [01:02<00:00,  7.46it/s]
# epoch 4, step 469, train loss 0.328, train acc 0.876, test acc 0.871
# epoch 5, step 469, train loss 0.303, train acc 0.886: 100%|██████████| 469/469 [01:03<00:00,  7.44it/s]
# epoch 5, step 469, train loss 0.303, train acc 0.886, test acc 0.870
# epoch 6, step 469, train loss 0.281, train acc 0.894: 100%|██████████| 469/469 [01:03<00:00,  7.36it/s]
# epoch 6, step 469, train loss 0.281, train acc 0.894, test acc 0.889
# epoch 7, step 469, train loss 0.265, train acc 0.899: 100%|██████████| 469/469 [01:03<00:00,  7.40it/s]
# epoch 7, step 469, train loss 0.265, train acc 0.899, test acc 0.880
# epoch 8, step 469, train loss 0.249, train acc 0.905: 100%|██████████| 469/469 [01:03<00:00,  7.41it/s]
# epoch 8, step 469, train loss 0.249, train acc 0.905, test acc 0.863
# epoch 9, step 469, train loss 0.238, train acc 0.909: 100%|██████████| 469/469 [01:03<00:00,  7.42it/s]
# epoch 9, step 469, train loss 0.238, train acc 0.909, test acc 0.887
# train loss 0.238, train acc 0.909, test acc 0.887
# 1163.9 examples/sec on cuda