from typing import Tuple, Union, List
import torch
import time
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
"""
多GPU训练模型简洁实现

实现说明:
https://tech.foxrelax.com/performance/multiple_gpus_concise/
"""


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


def resnet18(num_classes: int = 10, in_channels: int = 1) -> nn.Module:
    """
    修改过的ResNet-18模型

    >>> x = torch.randn((256, 1, 28, 28))
    >>> net = resnet18()
    >>> assert net(x).shape == (256, 10)  # [batch_size, num_classes]
    """

    def resnet_block(in_channels,
                     out_channels,
                     num_residuals,
                     first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(
                    Residual(in_channels,
                             out_channels,
                             use_1x1conv=True,
                             stride=2))
            else:
                blk.append(Residual(out_channels, out_channels))
        return nn.Sequential(*blk)

    # 该模型使用了更小的卷积核、步长和填充, 而且删除了最大池化层
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64), nn.ReLU())
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1, 1)))
    net.add_module("fc",
                   nn.Sequential(nn.Flatten(), nn.Linear(512, num_classes)))
    return net


def try_gpu(i: int = 0) -> torch.device:
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus() -> List[torch.device]:
    devices = [
        torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())
    ]
    return devices if devices else [torch.device('cpu')]


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


def train(net: nn.Module,
          num_epochs: int = 10,
          num_gpus: int = 2,
          batch_size: int = 256,
          lr: float = 0.2) -> None:
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    devices = [try_gpu(i) for i in range(num_gpus)]

    def init_weights(m):
        if type(m) in [nn.Linear, nn.Conv2d]:
            nn.init.normal_(m.weight, std=0.01)

    net.apply(init_weights)
    # 核心: 在多个GPU上设置模型
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.SGD(net.parameters(), lr)
    loss = nn.CrossEntropyLoss()
    times = []
    history = [[]]  # 记录: 训练集损失, 方便后续绘图
    for epoch in range(num_epochs):
        net.train()
        t_start = time.time()
        for X, y in train_iter:
            trainer.zero_grad()
            X, y = X.to(devices[0]), y.to(devices[0])
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        times.append(time.time() - t_start)
        # 在GPU0上评估模型
        metric_test = [0.0] * 2  # 测试准确数量之和, 测试样本数量之和
        with torch.no_grad():
            for X, y in test_iter:
                X = X.to(devices[0])
                y = y.to(devices[0])
                metric_test[0] += float(accuracy(net(X), y))
                metric_test[1] += float(y.numel())
            test_acc = metric_test[0] / metric_test[1]
            history[0].append((epoch + 1, test_acc))
        print(f'epoch {epoch}, test acc {test_acc:.3f}')

    print(
        f'test acc {history[0][-1][1]:.2f}，{sum(times)/num_epochs:.1f} epoch/sec on {str(devices)}'
    )

    plt.figure(figsize=(6, 4))
    plt.plot(*zip(*history[0]), '-', label='test acc')
    plt.xlabel('epoch')
    # 从epoch=1开始显示, 0-1这个范围的数据丢弃不展示,
    # 因为只有训练完成1个epochs之后, 才会有第一条test acc记录
    plt.xlim((1, num_epochs))
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    net = resnet18(num_classes=10, in_channels=1)
    # train(net, num_epochs=10, num_gpus=1, batch_size=256, lr=0.1)
    # epoch 0, test acc 0.870
    # epoch 1, test acc 0.888
    # epoch 2, test acc 0.904
    # epoch 3, test acc 0.908
    # epoch 4, test acc 0.911
    # epoch 5, test acc 0.912
    # epoch 6, test acc 0.921
    # epoch 7, test acc 0.916
    # epoch 8, test acc 0.925
    # epoch 9, test acc 0.924
    # test acc 0.92，22.0 epoch/sec on [device(type='cuda', index=0)]
    train(net, num_epochs=10, num_gpus=2, batch_size=256, lr=0.1)
    # epoch 0, test acc 0.884
    # epoch 1, test acc 0.863
    # epoch 2, test acc 0.914
    # epoch 3, test acc 0.898
    # epoch 4, test acc 0.917
    # epoch 5, test acc 0.894
    # epoch 6, test acc 0.910
    # epoch 7, test acc 0.917
    # epoch 8, test acc 0.912
    # epoch 9, test acc 0.928
    # test acc 0.93，16.6 epoch/sec on [device(type='cuda', index=0), device(type='cuda', index=1)]