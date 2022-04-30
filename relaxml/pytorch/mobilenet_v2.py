from typing import Tuple, Dict
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
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torchvision import transforms, datasets
from matplotlib import pyplot as plt
from tqdm import tqdm
"""
实现MobileNet V2

实现说明:
https://tech.foxrelax.com/lightweight/mobilenet_v2/
"""


def _make_divisible(ch: float, divisor: int = 8, min_ch: int = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


def load_weight(cache_dir: str = '../data') -> str:
    """
    加载预训练权重(class=1000的ImageNet数据集上训练的)
    """
    sha1_hash = '9d6df55a618d1707f020679b8cd68c91d4dec003'
    url = 'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/mobilenet_v2-b0353104.pth'
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
    # e.g. ../data/mobilenet_v2-b0353104.pth
    return fname


class ConvBNReLU(nn.Sequential):
    """
    1. 当groups=1的时候是普通卷积
    2. 当groups=in_channel=out_channel时, 是Depthwise Separable卷积(DW)
    """

    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1) -> None:
        """
        参数:
        in_channel: 输入通道数
        out_channel: 输出通道数
        kernel_size: 卷积核大小
        stride: 步长
        groups: 1或者in_channel
        """
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel,
                      out_channel,
                      kernel_size,
                      stride,
                      padding,
                      groups=groups,
                      bias=False), nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True))


class InvertedResidual(nn.Module):
    """
    实现Inverted Residual Block

    注意: 
    当stride=1并且in_channel == out_channel会有残差连接, 否则没有残差连接
    因为stride不等于1等于做了降维操作, 没法直接做add; 输入通道和输出通道不一致, 
    也没法直接做add
    """

    def __init__(self, in_channel: int, out_channel: int, stride: int,
                 expand_ratio: float) -> None:
        """
        参数:
        in_channel: 输入通道数
        out_channel: 输出通道数
        stride: 步长
        expand_ratio: 隐藏层的扩展因子
        """
        super(InvertedResidual, self).__init__()
        hidden_channel = in_channel * expand_ratio
        # 判断是否有残差连接
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        if expand_ratio != 1:
            # 1x1 pointwise conv
            layers.append(ConvBNReLU(in_channel, hidden_channel,
                                     kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            ConvBNReLU(hidden_channel,
                       hidden_channel,
                       stride=stride,
                       groups=hidden_channel),
            # 1x1 pointwise conv(linear)
            # 注意: 这里是没有ReLU6激活函数的
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):

    def __init__(self,
                 num_classes: int = 1000,
                 alpha: float = 1.0,
                 round_nearest: int = 8) -> None:
        """
        参数:
        num_classes: 分类数量
        alpha: 模型缩放因子
               当alpha>1时, 相当于扩大模型的规模
               alpha<1时, 相当于缩小模型的规模
        round_nearest: 默认为8
        """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = _make_divisible(32 * alpha, round_nearest)
        last_channel = _make_divisible(1280 * alpha, round_nearest)

        # 根据MobileNet V2论文中的设置配置倒残差块
        inverted_residual_setting = [
            # t, c, n, s
            # t: 膨胀因子, 也就是每个倒残差块用1x1卷积升维之后的通道数
            # c: 输出通道数
            # n: 重复几次
            # s: 第一个倒残差快的stirde
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = []
        # conv1 layer
        features.append(ConvBNReLU(3, input_channel, stride=2))
        # 构建: Inverted Residual Blockes
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * alpha, round_nearest)
            for i in range(n):
                # stride只有在重复第一次的时候为s, 其它时候都为1,
                # 因为我们的降维操作只做一次
                stride = s if i == 0 else 1
                features.append(
                    block(input_channel,
                          output_channel,
                          stride,
                          expand_ratio=t))
                input_channel = output_channel
        features.append(ConvBNReLU(input_channel, last_channel, 1))
        self.features = nn.Sequential(*features)

        # 构建分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Dropout(0.2),
                                        nn.Linear(last_channel, num_classes))

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def mobilenet_v2(pretrained: bool = True,
                 num_classes: int = 5,
                 alpha: float = 1.0,
                 round_nearest: float = 8) -> MobileNetV2:
    net = MobileNetV2(num_classes, alpha, round_nearest)
    if pretrained:
        model_weight_path = load_weight(cache_dir='../data')
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
    train_iter, test_iter, _, _ = load_data_flower(batch_size=32)
    net = mobilenet_v2()
    kwargs = {
        'num_epochs': 10,
        'loss': nn.CrossEntropyLoss(reduction='mean'),
        'optimizer': torch.optim.Adam(net.parameters(), lr=0.0001),
        'save_path': '../data/mobilenet_v2.pth'
    }
    history = train_gpu(net, train_iter, test_iter, **kwargs)
    plot_history(history)


if __name__ == '__main__':
    run()
# training on cuda
# epoch 0, step 104, train loss 1.324, train acc 0.576: 100%|██████████| 104/104 [00:18<00:00,  5.65it/s]
# epoch 0, step 104, train loss 1.324, train acc 0.576, test acc 0.832
# epoch 1, step 104, train loss 0.963, train acc 0.769: 100%|██████████| 104/104 [00:18<00:00,  5.56it/s]
# epoch 1, step 104, train loss 0.963, train acc 0.769, test acc 0.896
# epoch 2, step 104, train loss 0.796, train acc 0.808: 100%|██████████| 104/104 [00:18<00:00,  5.76it/s]
# epoch 2, step 104, train loss 0.796, train acc 0.808, test acc 0.896
# epoch 3, step 104, train loss 0.693, train acc 0.817: 100%|██████████| 104/104 [00:18<00:00,  5.77it/s]
# epoch 3, step 104, train loss 0.693, train acc 0.817, test acc 0.904
# epoch 4, step 104, train loss 0.610, train acc 0.828: 100%|██████████| 104/104 [00:18<00:00,  5.62it/s]
# epoch 4, step 104, train loss 0.610, train acc 0.828, test acc 0.931
# epoch 5, step 104, train loss 0.581, train acc 0.835: 100%|██████████| 104/104 [00:17<00:00,  5.79it/s]
# epoch 5, step 104, train loss 0.581, train acc 0.835, test acc 0.915
# epoch 6, step 104, train loss 0.539, train acc 0.847: 100%|██████████| 104/104 [00:18<00:00,  5.75it/s]
# epoch 6, step 104, train loss 0.539, train acc 0.847, test acc 0.923
# epoch 7, step 104, train loss 0.517, train acc 0.847: 100%|██████████| 104/104 [00:18<00:00,  5.75it/s]
# epoch 7, step 104, train loss 0.517, train acc 0.847, test acc 0.920
# epoch 8, step 104, train loss 0.491, train acc 0.856: 100%|██████████| 104/104 [00:17<00:00,  5.78it/s]
# epoch 8, step 104, train loss 0.491, train acc 0.856, test acc 0.931
# epoch 9, step 104, train loss 0.477, train acc 0.853: 100%|██████████| 104/104 [00:17<00:00,  5.79it/s]
# epoch 9, step 104, train loss 0.477, train acc 0.853, test acc 0.931
# train loss 0.477, train acc 0.853, test acc 0.931
# 1352.4 examples/sec on cuda