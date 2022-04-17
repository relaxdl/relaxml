from typing import Tuple, List, Union
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
from tqdm import tqdm
"""
实现AlexNet

实现说明:
https://tech.foxrelax.com/classics_net/alexnet/
"""


def alexnet() -> nn.Module:
    """
    实现AlexNet

    >>> x = torch.randn((256, 1, 224, 224))
    >>> net = alexnet()
    >>> assert net(x).shape == (256, 10)

    输入:
    x.shape: [batch_size, 1, 224, 224)

    输出:
    output.shape: [batch_size, 10]
    """
    net = nn.Sequential(
        # 这里, 我们使用一个11*11的更大窗口来捕捉对象
        # 同时, 步幅为4, 以减少输出的高度和宽度
        # 另外, 输出通道的数目远大于LeNet
        # output [1, 96, 54, 54]
        nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1),
        # output [1, 96, 54, 54]
        nn.ReLU(),
        # output [1, 96, 26, 26]
        nn.MaxPool2d(kernel_size=3, stride=2),
        # 减小卷积窗口, 使用填充为2来使得输入与输出的高和宽一致, 且增大输出通道数
        # output [1, 256, 26, 26]
        nn.Conv2d(96, 256, kernel_size=5, padding=2),
        # output [1, 256, 26, 26]
        nn.ReLU(),
        # output [1, 256, 12, 12]
        nn.MaxPool2d(kernel_size=3, stride=2),
        # 使用三个连续的卷积层和较小的卷积窗口
        # 除了最后的卷积层, 输出通道的数量进一步增加
        # 在前两个卷积层之后, 池化层不用于减少输入的高度和宽度
        # output [1, 384, 12, 12]
        nn.Conv2d(256, 384, kernel_size=3, padding=1),
        # output [1, 384, 12, 12]
        nn.ReLU(),
        # output [1, 384, 12, 12]
        nn.Conv2d(384, 384, kernel_size=3, padding=1),
        # output [1, 384, 12, 12]
        nn.ReLU(),
        # output [1, 256, 12, 12]
        nn.Conv2d(384, 256, kernel_size=3, padding=1),
        # output [1, 256, 12, 12]
        nn.ReLU(),
        # output [1, 256, 5, 5]
        nn.MaxPool2d(kernel_size=3, stride=2),
        # output [1, 6400]
        nn.Flatten(),
        # 这里全连接层的输出数量是LeNet中的好几倍. 使用dropout层来减轻过度拟合
        # output [1, 4096]
        nn.Linear(6400, 4096),
        # output [1, 4096]
        nn.ReLU(),
        # output [1, 4096]
        nn.Dropout(p=0.5),
        # output [1, 4096]
        nn.Linear(4096, 4096),
        # output [1, 4096]
        nn.ReLU(),
        # output [1, 4096]
        nn.Dropout(p=0.5),
        # 最后是输出层. 由于这里使用Fashion-MNIST, 所以用类别数为10, 而非论文中的1000
        # output [1, 10]
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
    net = alexnet()
    kwargs = {
        'num_epochs': 10,
        'loss': nn.CrossEntropyLoss(reduction='mean'),
        'optimizer': torch.optim.SGD(net.parameters(), lr=0.01)
    }
    history = train_gpu(net, train_iter, test_iter, **kwargs)
    plot_history(history)


if __name__ == '__main__':
    run()
# training on cuda
# epoch 0, step 235, train loss 1.783, train acc 0.350: 100%|██████████| 235/235 [00:58<00:00,  3.99it/s]
# epoch 0, step 235, train loss 1.783, train acc 0.350, test acc 0.624
# epoch 1, step 235, train loss 0.840, train acc 0.683: 100%|██████████| 235/235 [00:58<00:00,  4.01it/s]
# epoch 1, step 235, train loss 0.840, train acc 0.683, test acc 0.717
# epoch 2, step 235, train loss 0.675, train acc 0.748: 100%|██████████| 235/235 [00:58<00:00,  4.01it/s]
# epoch 2, step 235, train loss 0.675, train acc 0.748, test acc 0.783
# epoch 3, step 235, train loss 0.587, train acc 0.781: 100%|██████████| 235/235 [00:58<00:00,  4.02it/s]
# epoch 3, step 235, train loss 0.587, train acc 0.781, test acc 0.806
# epoch 4, step 235, train loss 0.535, train acc 0.801: 100%|██████████| 235/235 [00:58<00:00,  4.02it/s]
# epoch 4, step 235, train loss 0.535, train acc 0.801, test acc 0.806
# epoch 5, step 235, train loss 0.494, train acc 0.817: 100%|██████████| 235/235 [00:58<00:00,  4.03it/s]
# epoch 5, step 235, train loss 0.494, train acc 0.817, test acc 0.793
# epoch 6, step 235, train loss 0.464, train acc 0.829: 100%|██████████| 235/235 [00:58<00:00,  4.00it/s]
# epoch 6, step 235, train loss 0.464, train acc 0.829, test acc 0.841
# epoch 7, step 235, train loss 0.439, train acc 0.839: 100%|██████████| 235/235 [00:58<00:00,  4.01it/s]
# epoch 7, step 235, train loss 0.439, train acc 0.839, test acc 0.836
# epoch 8, step 235, train loss 0.419, train acc 0.846: 100%|██████████| 235/235 [00:58<00:00,  4.01it/s]
# epoch 8, step 235, train loss 0.419, train acc 0.846, test acc 0.856
# epoch 9, step 235, train loss 0.401, train acc 0.854: 100%|██████████| 235/235 [00:58<00:00,  4.01it/s]
# epoch 9, step 235, train loss 0.401, train acc 0.854, test acc 0.858
# train loss 0.401, train acc 0.854, test acc 0.858
# 1740.5 examples/sec on cuda