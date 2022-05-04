import os
from typing import Tuple, List, Union
import requests
import sys
import time
import hashlib
import zipfile
import tarfile
import torch
import numpy as np
from torch import Tensor
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from matplotlib import pyplot as plt
from tqdm import tqdm
"""
微调

实现说明:
https://tech.foxrelax.com/cv/fine_tuning/
"""


def download(cache_dir: str = '../data') -> str:
    """
    下载数据
    """
    sha1_hash = 'fba480ffa8aa7e0febbb511d181409f899b9baa5'
    url = 'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/hotdog.zip'
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
    # e.g. ../data/hotdog.zip
    return fname


def download_extract(cache_dir: str = '../data') -> str:
    """
    下载数据 & 解压
    """
    # 下载数据集
    fname = download(cache_dir)

    # 解压
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, 'Only zip/tar files can be extracted.'
    fp.extractall(base_dir)
    # e.g. ../data/hotdog
    return data_dir


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


def load_data_hotdog(
        batch_size: int = 32,
        cache_dir: str = '../data') -> Tuple[DataLoader, DataLoader]:
    """
    hotdog/
        train/
            hotdog/
            not-hotdog/
        test/
            hotdog/
            not-hotdog/

    >>> train_iter, test_iter = load_data_hotdog(batch_size=32)
    >>> for x, y in train_iter:
    >>>     assert x.shape == (32, 3, 224, 224)
    >>>     assert y.shape == (32, )
    >>>     break
    """
    data_dir = download_extract(cache_dir)

    normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])

    train_augs = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(), normalize
    ])

    test_augs = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(), normalize
    ])

    train_iter = DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=train_augs),
                            batch_size=batch_size,
                            shuffle=True)
    test_iter = DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'test'), transform=test_augs),
                           batch_size=batch_size)
    return train_iter, test_iter


def resnet18(pretrained: bool = True) -> nn.Module:
    """
    参数:
    pretrained=True, 使用预训练好的模型参数
    pretrained=False, 所有模型参数随机初始化
    """
    pretrained_net = torchvision.models.resnet18(pretrained=pretrained)
    # 替换: Linear(in_features=512, out_features=1000, bias=True)
    pretrained_net.fc = nn.Linear(pretrained_net.fc.in_features, 2)
    nn.init.xavier_uniform_(pretrained_net.fc.weight)
    return pretrained_net


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


def run(pretrained: bool = True) -> None:
    """
    参数:
    pretrained=True的时候是'微调'模型, 使用预训练好的模型参数
    pretrained=False的时候是从头开始训练模型, 所有模型参数随机初始化
    """
    net = resnet18(pretrained)
    if pretrained:
        # 微调(针对不同的层使用不同的学习率)
        lr = 5e-4
        params_1x = [
            param for name, param in net.named_parameters()
            if name not in ["fc.weight", "fc.bias"]
        ]
        optimizer = torch.optim.SGD(
            [
                {
                    'params': params_1x
                },
                {
                    'params': net.fc.parameters(),
                    'lr': lr * 10  # 学习率x10
                }
            ],
            lr=lr,
            weight_decay=0.001)
    else:
        # 从头开始训练
        lr = 5e-3
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=lr,
                                    weight_decay=0.001)
    kwargs = {
        'num_epochs': 5,
        'loss': nn.CrossEntropyLoss(reduction='mean'),
        'optimizer': optimizer
    }

    train_iter, test_iter = load_data_hotdog(batch_size=128)
    history = train_gpu(net, train_iter, test_iter, **kwargs)
    plot_history(history)


if __name__ == '__main__':
    # run(True)  # 微调
    # epoch 0, step 16, train loss 0.781, train acc 0.626: 100%|██████████| 16/16 [00:11<00:00,  1.45it/s]
    # epoch 0, step 16, train loss 0.781, train acc 0.626, test acc 0.714
    # epoch 1, step 16, train loss 0.524, train acc 0.740: 100%|██████████| 16/16 [00:11<00:00,  1.45it/s]
    # epoch 1, step 16, train loss 0.524, train acc 0.740, test acc 0.797
    # epoch 2, step 16, train loss 0.416, train acc 0.805: 100%|██████████| 16/16 [00:10<00:00,  1.46it/s]
    # epoch 2, step 16, train loss 0.416, train acc 0.805, test acc 0.835
    # epoch 3, step 16, train loss 0.380, train acc 0.837: 100%|██████████| 16/16 [00:11<00:00,  1.44it/s]
    # epoch 3, step 16, train loss 0.380, train acc 0.837, test acc 0.859
    # epoch 4, step 16, train loss 0.332, train acc 0.853: 100%|██████████| 16/16 [00:10<00:00,  1.46it/s]
    # epoch 4, step 16, train loss 0.332, train acc 0.853, test acc 0.877
    # train loss 0.332, train acc 0.853, test acc 0.877
    # 649.9 examples/sec on cuda

    run(False)  # 从头开始训练
    # epoch 0, step 16, train loss 0.493, train acc 0.770: 100%|██████████| 16/16 [00:11<00:00,  1.45it/s]
    # epoch 0, step 16, train loss 0.493, train acc 0.770, test acc 0.786
    # epoch 1, step 16, train loss 0.391, train acc 0.829: 100%|██████████| 16/16 [00:11<00:00,  1.45it/s]
    # epoch 1, step 16, train loss 0.391, train acc 0.829, test acc 0.811
    # epoch 2, step 16, train loss 0.380, train acc 0.840: 100%|██████████| 16/16 [00:10<00:00,  1.46it/s]
    # epoch 2, step 16, train loss 0.380, train acc 0.840, test acc 0.807
    # epoch 3, step 16, train loss 0.360, train acc 0.844: 100%|██████████| 16/16 [00:10<00:00,  1.46it/s]
    # epoch 3, step 16, train loss 0.360, train acc 0.844, test acc 0.811
    # epoch 4, step 16, train loss 0.358, train acc 0.840: 100%|██████████| 16/16 [00:10<00:00,  1.46it/s]
    # epoch 4, step 16, train loss 0.358, train acc 0.840, test acc 0.831
    # train loss 0.358, train acc 0.840, test acc 0.831
    # 649.9 examples/sec on cuda