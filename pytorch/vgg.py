from typing import Tuple, List, Union
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
实现VGG

实现说明:
https://tech.foxrelax.com/classics_net/vgg/
"""


def vgg_block(num_convs: int, in_channels: int,
              out_channels: int) -> nn.Module:
    """
    VGG Block
    
    数据经过vgg block, 高和宽会减半

    参数:
    num_convs: 卷积层的数量
    in_channels: 输入通道数
    out_channels: 输出通道数
    """
    layers = []
    # 高宽不变
    for _ in range(num_convs):
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels

    # 高宽减半
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


def vgg(conv_arch: Tuple[Tuple[int, int]] = None, ratio: int = 1) -> nn.Module:
    """
    VGG

    VGG的默认Conv Arch为:
    ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    * 第一个数表示: `num_convs`, 也就是每个vgg block卷积层的数量
    * 第二个数表示: `out_channels`, 这个vgg block输出通道数

    由于模型参数比较多, 我们可以设置一个ratio来缩小其网络规模

    标准规模的VGG网络
    >>> x = torch.randn((256, 1, 224, 224))
    >>> net = vgg()
    >>> assert net(x).shape == (256, 10)

    Mini的VGG网络(参数少, 训练速度快)
    >>> x = torch.randn((256, 1, 224, 224))
    >>> net = vgg(ratio=4)
    >>> assert net(x).shape == (256, 10)
    """
    if conv_arch is None:
        conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

    conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]

    vgg_blks = []
    in_channels = 1
    for (num_convs, out_channels) in conv_arch:
        vgg_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    net = nn.Sequential(
        # input   [batch_size, 1, 224, 224]
        # block1  [batch_size, 64, 112, 112]
        # block2  [batch_size, 128, 56, 56]
        # block3  [batch_size, 256, 28, 28]
        # block4  [batch_size, 512, 14, 14]
        # block5  [batch_size, 512, 7, 7]
        *vgg_blks,
        # output [batch_size, 25088]
        nn.Flatten(),
        # 全连接层部分
        # output [batch_size, 4096]
        nn.Linear(out_channels * 7 * 7, 4096),
        # output [batch_size, 4096]
        nn.ReLU(),
        # output [batch_size, 4096]
        nn.Dropout(0.5),
        # output [batch_size, 4096]
        nn.Linear(4096, 4096),
        # output [batch_size, 4096]
        nn.ReLU(),
        # output [batch_size, 4096]
        nn.Dropout(0.5),
        # output [batch_size, 10]
        nn.Linear(4096, 10))

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
    train_iter, test_iter = load_data_fashion_mnist(batch_size=256,
                                                    resize=(224, 224))
    net = vgg(ratio=4)
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
# epoch 0, step 235, train loss 1.594, train acc 0.416: 100%|██████████| 235/235 [01:50<00:00,  2.12it/s]
# epoch 0, step 235, train loss 1.594, train acc 0.416, test acc 0.776
# epoch 1, step 235, train loss 0.521, train acc 0.805: 100%|██████████| 235/235 [01:50<00:00,  2.12it/s]
# epoch 1, step 235, train loss 0.521, train acc 0.805, test acc 0.802
# epoch 2, step 235, train loss 0.406, train acc 0.851: 100%|██████████| 235/235 [01:51<00:00,  2.11it/s]
# epoch 2, step 235, train loss 0.406, train acc 0.851, test acc 0.863
# epoch 3, step 235, train loss 0.350, train acc 0.871: 100%|██████████| 235/235 [01:50<00:00,  2.12it/s]
# epoch 3, step 235, train loss 0.350, train acc 0.871, test acc 0.876
# epoch 4, step 235, train loss 0.317, train acc 0.884: 100%|██████████| 235/235 [01:51<00:00,  2.11it/s]
# epoch 4, step 235, train loss 0.317, train acc 0.884, test acc 0.876
# epoch 5, step 235, train loss 0.290, train acc 0.894: 100%|██████████| 235/235 [01:51<00:00,  2.11it/s]
# epoch 5, step 235, train loss 0.290, train acc 0.894, test acc 0.896
# epoch 6, step 235, train loss 0.270, train acc 0.902: 100%|██████████| 235/235 [01:50<00:00,  2.12it/s]
# epoch 6, step 235, train loss 0.270, train acc 0.902, test acc 0.898
# epoch 7, step 235, train loss 0.256, train acc 0.907: 100%|██████████| 235/235 [01:51<00:00,  2.11it/s]
# epoch 7, step 235, train loss 0.256, train acc 0.907, test acc 0.894
# epoch 8, step 235, train loss 0.240, train acc 0.911: 100%|██████████| 235/235 [01:51<00:00,  2.12it/s]
# epoch 8, step 235, train loss 0.240, train acc 0.911, test acc 0.886
# epoch 9, step 235, train loss 0.226, train acc 0.916: 100%|██████████| 235/235 [01:50<00:00,  2.12it/s]
# epoch 9, step 235, train loss 0.226, train acc 0.916, test acc 0.887
# train loss 0.226, train acc 0.916, test acc 0.887
# 695.0 examples/sec on cuda