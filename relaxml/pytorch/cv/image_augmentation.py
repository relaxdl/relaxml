from typing import Callable, Tuple, Union, List
import os
import hashlib
import requests
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from matplotlib import pyplot as plt
from tqdm import tqdm
"""
图像增广

实现说明:
https://tech.foxrelax.com/cv/image_augmentation/
"""


def download(cache_dir: str = '../data') -> str:
    """
    下载数据
    """
    sha1_hash = 'f9c5b905d908b97eeeb64ff34a46fa8b143f88f8'
    url = 'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/img/cat1.jpg'
    fname = os.path.join(cache_dir, url.split('/ml/')[-1])
    fdir = os.path.dirname(fname)
    os.makedirs(fdir, exist_ok=True)
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname
    print(f'download {url} -> {fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    print(f'download {fname} success!')
    # e.g. ../data/cat1.jpg
    return fname


def show_images(imgs: List[Union[Tensor, np.ndarray]],
                num_rows: int,
                num_cols: int,
                titles: List[str] = None,
                scale: float = 1.5) -> plt.Axes:
    """
    Plot a list of images

    imgs需要[H, W, C]或者[H, W]这样的格式

    >>> img = plt.imread(download('cat3')) # [H, W, C]
    >>> show_images([img, img, img, img], 2, 2, 
                    titles=['t1', 't2', 't3', 't4'])
    >>> plt.show()
    """
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL Image
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


def apply(img: Union[np.ndarray, Tensor],
          aug: Callable,
          num_rows: int = 2,
          num_cols: int = 4,
          scale: float = 1.5) -> None:
    """
    大多数图像增广方法都具有一定的随机性. 为了便于观察图像增广的效果, 
    我们下面定义辅助函数apply. 此函数在输入图像img上多次运行图像增广
    方法aug并显示所有结果

    >>> img = Image.open(download())
    >>> plt.imshow(img)
    >>> apply(img, torchvision.transforms.RandomHorizontalFlip())
    >>> plt.show()
    """
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    show_images(Y, num_rows, num_cols, scale=scale)


def load_data_cifar10(batch_size: int,
                      train_transform: nn.Module = None,
                      test_transform: nn.Module = None,
                      root: str = '../data') -> Tuple[DataLoader, DataLoader]:
    """
    下载CIFAR10数据集, 然后将其加载到内存中
    
    >>> train_iter, test_iter = load_data_cifar10(batch_size=32)
    >>> for X, y in train_iter:
    >>>     assert X.shape == (32, 3, 32, 32)
    >>>     assert y.shape == (32, )
    >>>     break
    """
    if train_transform is None:
        train_transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()])
    if test_transform is None:
        test_transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()])
    cifar10_train = torchvision.datasets.CIFAR10(root=root,
                                                 train=True,
                                                 transform=train_transform,
                                                 download=True)
    cifar10_test = torchvision.datasets.CIFAR10(root=root,
                                                train=False,
                                                transform=test_transform,
                                                download=True)
    return (DataLoader(cifar10_train, batch_size, shuffle=True),
            DataLoader(cifar10_test, batch_size, shuffle=False))


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

    >>> x = torch.randn((256, 3, 32, 32))
    >>> net = resnet18(in_channels=3)
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

    def init_weights(m):
        if type(m) in [nn.Linear, nn.Conv2d]:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    return net


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
    train_augs = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor()
    ])
    test_augs = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()])
    net = resnet18(10, 3)
    kwargs = {
        'num_epochs': 10,
        'loss': nn.CrossEntropyLoss(reduction='mean'),
        'optimizer': torch.optim.Adam(net.parameters(), lr=0.001)
    }

    train_iter, test_iter = load_data_cifar10(batch_size=256,
                                              train_transform=train_augs,
                                              test_transform=test_augs)
    history = train_gpu(net, train_iter, test_iter, **kwargs)
    plot_history(history)


if __name__ == '__main__':
    run()
# training on cuda
# epoch 0, step 196, train loss 1.373, train acc 0.513: 100%|██████████| 196/196 [00:45<00:00,  4.35it/s]
# epoch 0, step 196, train loss 1.373, train acc 0.513, test acc 0.511
# epoch 1, step 196, train loss 0.820, train acc 0.708: 100%|██████████| 196/196 [00:44<00:00,  4.37it/s]
# epoch 1, step 196, train loss 0.820, train acc 0.708, test acc 0.695
# epoch 2, step 196, train loss 0.603, train acc 0.791: 100%|██████████| 196/196 [00:44<00:00,  4.38it/s]
# epoch 2, step 196, train loss 0.603, train acc 0.791, test acc 0.763
# epoch 3, step 196, train loss 0.477, train acc 0.835: 100%|██████████| 196/196 [00:44<00:00,  4.38it/s]
# epoch 3, step 196, train loss 0.477, train acc 0.835, test acc 0.744
# epoch 4, step 196, train loss 0.390, train acc 0.864: 100%|██████████| 196/196 [00:44<00:00,  4.38it/s]
# epoch 4, step 196, train loss 0.390, train acc 0.864, test acc 0.821
# epoch 5, step 196, train loss 0.331, train acc 0.883: 100%|██████████| 196/196 [00:44<00:00,  4.37it/s]
# epoch 5, step 196, train loss 0.331, train acc 0.883, test acc 0.829
# epoch 6, step 196, train loss 0.276, train acc 0.904: 100%|██████████| 196/196 [00:44<00:00,  4.38it/s]
# epoch 6, step 196, train loss 0.276, train acc 0.904, test acc 0.824
# epoch 7, step 196, train loss 0.233, train acc 0.920: 100%|██████████| 196/196 [00:44<00:00,  4.37it/s]
# epoch 7, step 196, train loss 0.233, train acc 0.920, test acc 0.825
# epoch 8, step 196, train loss 0.190, train acc 0.933: 100%|██████████| 196/196 [00:44<00:00,  4.37it/s]
# epoch 8, step 196, train loss 0.190, train acc 0.933, test acc 0.840
# epoch 9, step 196, train loss 0.166, train acc 0.943: 100%|██████████| 196/196 [00:44<00:00,  4.38it/s]
# epoch 9, step 196, train loss 0.166, train acc 0.943, test acc 0.848
# train loss 0.166, train acc 0.943, test acc 0.848
# 1332.1 examples/sec on cuda