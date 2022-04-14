import os
import sys
import time
import requests
import hashlib
from shutil import copy, rmtree
import random
import zipfile
import tarfile
from typing import Dict, Tuple, List
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torchvision import transforms, datasets
from matplotlib import pyplot as plt
from tqdm import tqdm
"""
实现SqueezeNet

实现说明:
https://tech.foxrelax.com/lightweight/squeezenet/
"""


def load_weight(version: str = '1.0', cache_dir: str = '../data') -> str:
    """
    加载预训练权重(class=1000的ImageNet数据集上训练的)
    """
    if version == '1.0':
        sha1_hash = 'd0a99b3365c6afd7e42ee95c0cb3ccc5f0f16465'
        url = 'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/squeezenet1_0-b66bff10.pth'
    elif version == '1.1':
        sha1_hash = 'a0124d22d576026f7050008a60337a0dd2c6b473'
        url = 'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/squeezenet1_1-b8a52dc0.pth'
    else:
        raise ValueError(
            f'Unsupported SqueezeNet version {version}: 1.0 or 1.1 expected')
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
    # e.g. ../data/squeezenet1_0-b66bff10.pth
    return fname


class Fire(nn.Module):
    """
    数据流经Fire模块, 通道数会改变, 尺寸不会改变

    >>> fire = Fire(96, 16, 64, 64)
    >>> x = torch.randn(2, 96, 224, 224)
    >>> assert fire(x).shape == (2, 64 + 64, 224, 224)
    """

    def __init__(self, inplanes: int, squeeze_planes: int,
                 expand1x1_planes: int, expand3x3_planes: int) -> None:
        """
        参数:
        inplanes: 输入通道数
        squeeze_planes: 压缩之后的通道数
        expand1x1_planes: 扩展通道数e1
        expand3x3_planes: 扩展通道数e2
        """
        super().__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes,
                                   expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes,
                                   expand3x3_planes,
                                   kernel_size=3,
                                   padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        squeeze -> expand -> concat

        参数:
        x的形状: (batch_size, inplanes, H, W)

        返回:
        output的形状: (batch_size, expand3x3_planes+expand1x1_planes, H, W)
        """
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet(nn.Module):

    def __init__(self,
                 version: str = "1_0",
                 num_classes: int = 1000,
                 dropout: float = 0.5) -> None:
        super().__init__()
        self.num_classes = num_classes
        if version == "1.0":
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        elif version == "1.1":
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )

        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(nn.Dropout(p=dropout), final_conv,
                                        nn.ReLU(inplace=True),
                                        nn.AdaptiveAvgPool2d((1, 1)))

        # 模块初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)


def squeezenet(pretrained: bool = True,
               num_classes: int = 5,
               version: str = '1.0',
               dropout: float = 0.5) -> nn.Module:
    net = SqueezeNet(version, num_classes, dropout)
    if pretrained:
        model_weight_path = load_weight(version, cache_dir='../data')
        pre_weights = torch.load(model_weight_path, map_location='cpu')
        # 当新构建的网络(net)的分类器的数量和预训练权重分类器的数量不一致时, 删除分类器这一层的权重
        pre_dict = {
            k: v
            for k, v in pre_weights.items()
            if net.state_dict()[k].numel() == v.numel()  # 只保留权重数量一致的层
        }
        # 加载权重
        net.load_state_dict(pre_dict, strict=False)
        # 冻结特征提取层的权重
        for param in net.features.parameters():
            param.requires_grad = False
    return net


def download(cache_dir: str = '../data') -> str:
    """
    下载数据
    """
    sha1_hash = '26e0a7488bac8eec33de22a5e8b569cb050e1c2e'
    url = 'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/flower_photos.zip'
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
    # e.g. ../data/flower_photos.zip
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
    # e.g. ../data/flower_photos
    return data_dir


def mk_file(file_path: str) -> None:
    if os.path.exists(file_path):
        # 如果文件夹存在，则先删除原文件夹在重新创建
        rmtree(file_path)
    os.makedirs(file_path)


def process_data(data_path: str, val_rate: float = 0.1) -> Tuple[str, str]:
    """
    data_path=../data/flower_photos
    ../data/flower_photos/ (3670个样本)
                daisy/
                dandelion/
                roses/
                sunflowers/
                tulips/
    
    生成的训练集: 3306个样本
    ../data/train/
                daisy/
                dandelion/
                roses/
                sunflowers/
                tulips/
    
    生成的验证集: 364个样本
    ../data/val/
                daisy/
                dandelion/
                roses/
                sunflowers/
                tulips/
    """
    # ['roses', 'sunflowers', 'daisy', 'dandelion', 'tulips']
    flower_class = [
        cla for cla in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, cla))
    ]
    root_path = os.path.dirname(data_path)
    # 建立保存训练集的文件夹
    train_root = os.path.join(root_path, "train")
    mk_file(train_root)
    for cla in flower_class:
        # 建立每个类别对应的文件夹
        mk_file(os.path.join(train_root, cla))

    # 建立保存验证集的文件夹
    val_root = os.path.join(root_path, "val")
    mk_file(val_root)
    for cla in flower_class:
        # 建立每个类别对应的文件夹
        mk_file(os.path.join(val_root, cla))

    # 遍历所有的类
    for cla in flower_class:
        cla_path = os.path.join(data_path, cla)
        images = os.listdir(cla_path)
        num = len(images)
        # 随机采样验证集的索引
        eval_index = random.sample(images, k=int(num * val_rate))
        for index, image in enumerate(images):
            if image in eval_index:
                # 将分配至验证集中的文件复制到相应目录
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(val_root, cla)
                copy(image_path, new_path)
            else:
                # 将分配至训练集中的文件复制到相应目录
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(train_root, cla)
                copy(image_path, new_path)
            print("\r[{}] processing [{}/{}]".format(cla, index + 1, num),
                  end="")  # processing bar
        print()

    print("processing done!")
    return train_root, val_root


def load_data_flower(
    batch_size: int,
    resize: int = 224,
    root: str = '../data'
) -> Tuple[DataLoader, DataLoader, Dict[str, int], Dict[int, str]]:
    """
    加载Flower数据集

    1. 一共5个类别
    2. 训练集: 3306 images
    3. 验证集: 364 images
    4. 图片尺寸: 3x224x224 [默认处理成这个尺寸]

    >>> train_iter, val_iter, class_to_idx, idx_to_class = 
            load_data_flower(32, root='../data')
    >>> for X, y in val_iter:
    >>>     assert X.shape == (32, 3, 224, 224)
    >>>     assert y.shape == (32, )
    >>>     break
    >>> print(class_to_idx)
        {'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}
    >>> print(idx_to_class)
        {0: 'daisy', 1: 'dandelion', 2: 'roses', 3: 'sunflowers', 4: 'tulips'}
    """
    data_dir = download_extract(root)
    train_root, val_root = process_data(data_dir)

    data_transform = {
        "train":
        transforms.Compose([
            transforms.RandomResizedCrop(resize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val":
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    train_dataset = datasets.ImageFolder(train_root,
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    class_to_idx = train_dataset.class_to_idx
    idx_to_class = dict((val, key) for key, val in class_to_idx.items())

    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    validate_dataset = datasets.ImageFolder(val_root,
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    val_iter = DataLoader(validate_dataset,
                          batch_size=batch_size,
                          shuffle=False)

    print("using {} images for training, {} images for validation.".format(
        train_num, val_num))
    return train_iter, val_iter, class_to_idx, idx_to_class


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
    train_iter, test_iter, _, _ = load_data_flower(batch_size=32)
    net = squeezenet()
    kwargs = {
        'num_epochs': 10,
        'loss': nn.CrossEntropyLoss(reduction='mean'),
        'optimizer': torch.optim.Adam(net.parameters(), lr=0.0001),
        'save_path': '../data/squeezenet_v1_0.pth'
    }
    history = train_gpu(net, train_iter, test_iter, **kwargs)
    plot_history(history)


if __name__ == '__main__':
    run()
# 训练10个epochs的结果:
# training on cuda
# epoch 0, step 104, train loss 1.282, train acc 0.528: 100%|██████████| 104/104 [00:18<00:00,  5.52it/s]
# epoch 0, step 104, train loss 1.282, train acc 0.528, test acc 0.753
# epoch 1, step 104, train loss 0.791, train acc 0.770: 100%|██████████| 104/104 [00:18<00:00,  5.58it/s]
# epoch 1, step 104, train loss 0.791, train acc 0.770, test acc 0.808
# epoch 2, step 104, train loss 0.612, train acc 0.813: 100%|██████████| 104/104 [00:18<00:00,  5.55it/s]
# epoch 2, step 104, train loss 0.612, train acc 0.813, test acc 0.863
# epoch 3, step 104, train loss 0.531, train acc 0.838: 100%|██████████| 104/104 [00:18<00:00,  5.57it/s]
# epoch 3, step 104, train loss 0.531, train acc 0.838, test acc 0.863
# epoch 4, step 104, train loss 0.477, train acc 0.850: 100%|██████████| 104/104 [00:18<00:00,  5.57it/s]
# epoch 4, step 104, train loss 0.477, train acc 0.850, test acc 0.871
# epoch 5, step 104, train loss 0.463, train acc 0.851: 100%|██████████| 104/104 [00:18<00:00,  5.55it/s]
# epoch 5, step 104, train loss 0.463, train acc 0.851, test acc 0.879
# epoch 6, step 104, train loss 0.447, train acc 0.856: 100%|██████████| 104/104 [00:18<00:00,  5.53it/s]
# epoch 6, step 104, train loss 0.447, train acc 0.856, test acc 0.871
# epoch 7, step 104, train loss 0.424, train acc 0.862: 100%|██████████| 104/104 [00:18<00:00,  5.56it/s]
# epoch 7, step 104, train loss 0.424, train acc 0.862, test acc 0.882
# epoch 8, step 104, train loss 0.410, train acc 0.864: 100%|██████████| 104/104 [00:18<00:00,  5.58it/s]
# epoch 8, step 104, train loss 0.410, train acc 0.864, test acc 0.893
# epoch 9, step 104, train loss 0.398, train acc 0.869: 100%|██████████| 104/104 [00:18<00:00,  5.58it/s]
# epoch 9, step 104, train loss 0.398, train acc 0.869, test acc 0.896
# train loss 0.398, train acc 0.869, test acc 0.896
# 701.7 examples/sec on cuda