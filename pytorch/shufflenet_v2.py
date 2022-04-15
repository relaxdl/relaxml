import os
import sys
import time
import requests
import hashlib
from typing import Dict, List, Callable, Tuple
from shutil import copy, rmtree
import random
import zipfile
import tarfile
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torchvision import transforms, datasets
from matplotlib import pyplot as plt
from tqdm import tqdm
"""
实现ShuffleNet V2

实现说明:
https://tech.foxrelax.com/lightweight/shufflenet_v2/
"""


def load_weight(arch: float = '0.5', cache_dir: str = '../data') -> str:
    """
    加载预训练权重(class=1000的ImageNet数据集上训练的)
    """
    if arch == '0.5':
        sha1_hash = '6fe102c4aa96adc3ffd18abc2ec570bc1d1fdb6e'
        url = 'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/shufflenetv2_x0.5-f707e7126e.pth'
    elif arch == '1.0':
        sha1_hash = '8e40771828d47c5beeee7a94864093340836695a'
        url = 'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/shufflenetv2_x1-5666bf0f80.pth'
    else:
        raise ValueError(
            f'Unsupported ShuffleNet V2 arch {arch}: 0.5 or 1.0 expected')
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
        print(sha1.hexdigest())
        if sha1.hexdigest() == sha1_hash:
            return fname
    print(f'download {url} -> {fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    print(f'download {fname} success!')
    # e.g. ../data/squeezenet1_0-b66bff10.pth
    return fname


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    """
    1. Reshape: 先将输入通道这个维度Reshape成两个维度: 一个是卷积的group数, 一个是每个group包含的channel数
    2. Transpose: 将扩展出去的两个维度置换
    3. Flatten: 将置换后的通道Flatten后就完成了Channel Shuffle

    参数:
    x.shape: [batch_size, channels, height, width]
    groups: group的数量

    返回:
    output.shape: [batch_size, channels, height, width]
    """
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups  # 每个group有多少个channel

    # reshape
    # x.shape: [batch_size, channels, height, width] ->
    #          [batch_size, groups, channels_per_group, height, width]
    x = x.view(batchsize, groups, channels_per_group, height, width)

    # transpose
    # x.shape: [batch_size, channels_per_group, groups, height, width]
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    # x.shape: [batch_size, channels, height, width]
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    """
    实现ShuffleNet V2基础网络块

    # stride=1, 输出通道和输入通道相等, 输入特征图的宽度和高度和输出特征图的宽度和高度一致
    >>> x = torch.randn((2, 8, 64, 64))
    >>> blk1 = InvertedResidual(8, 8, 1)
    >>> assert blk1(x).shape == (2, 8, 64, 64)

    # stride=2, 输出通道为输入通道的2倍, 输出特征图的宽度和高度相比输入特征图的宽度和高度减半
    >>> x = torch.randn((2, 8, 64, 64))
    >>> blk2 = InvertedResidual(8, 16, 2)
    >>> assert blk2(x).shape == (2, 16, 32, 32)
    """

    def __init__(self, inp: int, oup: int, stride: int) -> None:
        """
        1. stride=1的时候, inp=oup
        2. stride=2的时候, inp*2=oup
        
        参数:
        inp: 输入的通道数
        oup: 输出的通道数, 需要是2的整数倍, 因为左右两个分支对应的通道数为oup/2
        stride: 步长, 只有1或者2两个取值
        """
        super().__init__()

        if not (1 <= stride <= 3):
            raise ValueError("illegal stride value")
        self.stride = stride

        branch_features = oup // 2
        # stride=1的时候, 要满足: inp = branch_features*2 = oup
        # << 1是位操作, 相当于x2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            # stride=2时:
            # [batch_size, h, w, inp] -> [batch_size, h/2, w/2, branch_features]
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp,
                                    inp,
                                    kernel_size=3,
                                    stride=self.stride,
                                    padding=1),
                nn.BatchNorm2d(inp),
                # 通过1x1卷积将输出通道变为branch_features
                nn.Conv2d(inp,
                          branch_features,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            # stride=1时, 不做任何处理
            # [batch_size, h, w, branch_features] -> [batch_size, h, w, branch_features]
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            # stride=1时, 输入通道数=branch_features
            # stride=2时, 输入通道数=inp
            nn.Conv2d(
                inp if (self.stride > 1) else branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features,
                                branch_features,
                                kernel_size=3,
                                stride=self.stride,
                                padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features,
                      branch_features,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i: int,
                       o: int,
                       kernel_size: int,
                       stride: int = 1,
                       padding: int = 0,
                       bias: bool = False) -> nn.Conv2d:
        """
        Depthwise Conv
        """
        return nn.Conv2d(i,
                         o,
                         kernel_size,
                         stride,
                         padding,
                         bias=bias,
                         groups=i)  # group=i

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            # 将x在通道这个维度一分为二: x1, x2
            x1, x2 = x.chunk(2, dim=1)  # split
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            # stride=2
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):

    def __init__(
        self,
        stages_repeats: List[int],
        stages_out_channels: List[int],
        num_classes: int = 1000,
        inverted_residual: Callable[..., nn.Module] = InvertedResidual,
    ) -> None:
        """
        参数:
        stages_repeats: e.g. [stage2, stage3, stage4]的重复次数, 因为论文中stage1,stage5只重复了一次
                             e.g. [4, 8, 4]
        stages_out_channels: [stage1, stage2, stage3, stage4, stage5]的输出通道数, 
                             e.g. [24, 48, 96, 192, 1024]
        num_classes: 分类的类别
        inverted_residual: 自己实现的ShuffleNet V2基础网络块: InvertedResidual
        """
        super().__init__()

        if len(stages_repeats) != 3:
            raise ValueError(
                "expected stages_repeats as list of 3 positive ints")
        if len(stages_out_channels) != 5:
            raise ValueError(
                "expected stages_out_channels as list of 5 positive ints")
        self._stage_out_channels = stages_out_channels

        # stage1
        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # stage2, stage3, stage4
        stage_names = [f"stage{i}" for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            # e.g. arch=0.5时的数值:
            # stage2, 4, 48
            # stage3, 8, 96
            # stage3, 8, 192
            #
            # 每个stage的第一个block: stride=2
            seq = [inverted_residual(input_channels, output_channels, 2)]
            # 其它的block: stride=1
            for i in range(repeats - 1):
                seq.append(
                    inverted_residual(output_channels, output_channels, 1))
            # 会创建:
            # self.stage2
            # self.stage3
            # self.stage4
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Linear(output_channels, num_classes)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # [TorchScript super()]
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        # [2,3]分别对应H,W这两个维度, 经过mean方法之后, H,W这两个维度就没有了
        # 就只剩下batch,channel这两个维度了
        x = x.mean([2, 3])  # globalpool
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def shufflenet_v2(arch: str = '1.0',
                  pretrained: bool = True,
                  num_classes: int = 5) -> ShuffleNetV2:
    if arch == '0.5':
        stages_repeats = [4, 8, 4]
        stages_out_channels = [24, 48, 96, 192, 1024]
    elif arch == '1.0':
        stages_repeats = [4, 8, 4]
        stages_out_channels = [24, 116, 232, 464, 1024]
    else:
        raise ValueError(
            f'Unsupported ShuffleNet V2 arch {arch}: 0.5 or 1.0 expected')
    net = ShuffleNetV2(stages_repeats, stages_out_channels, num_classes)

    if pretrained:
        model_weight_path = load_weight(arch, cache_dir='../data')
        pre_weights = torch.load(model_weight_path, map_location='cpu')
        # 当新构建的网络(net)的分类器的数量和预训练权重分类器的数量不一致时, 删除分类器这一层的权重
        pre_dict = {
            k: v
            for k, v in pre_weights.items()
            if net.state_dict()[k].numel() == v.numel()  # 只保留权重数量一致的层
        }
        # 加载权重
        net.load_state_dict(pre_dict, strict=False)

        # 冻结权重
        for name, para in net.named_parameters():
            # 除最后的全连接层外, 其他权重全部冻结
            if 'fc' not in name:
                para.requires_grad_(False)
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
    y_hat的形状: [batch_size, num_classes]
    y的形状: [batch_size,]
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
    net = shufflenet_v2()
    kwargs = {
        'num_epochs': 10,
        'loss': nn.CrossEntropyLoss(reduction='mean'),
        'optimizer': torch.optim.Adam(net.parameters(), lr=0.001),
        'save_path': '../data/shufflenetv2_1.0.pth'
    }
    history = train_gpu(net, train_iter, test_iter, **kwargs)
    plot_history(history)


if __name__ == '__main__':
    run()
# 训练10个epochs的结果:
# training on cuda
# epoch 0, step 104, train loss 1.500, train acc 0.468: 100%|██████████| 104/104 [00:18<00:00,  5.77it/s]
# epoch 0, step 104, train loss 1.500, train acc 0.468, test acc 0.690
# epoch 1, step 104, train loss 1.306, train acc 0.700: 100%|██████████| 104/104 [00:17<00:00,  5.93it/s]
# epoch 1, step 104, train loss 1.306, train acc 0.700, test acc 0.799
# epoch 2, step 104, train loss 1.159, train acc 0.763: 100%|██████████| 104/104 [00:17<00:00,  5.91it/s]
# epoch 2, step 104, train loss 1.159, train acc 0.763, test acc 0.799
# epoch 3, step 104, train loss 1.042, train acc 0.799: 100%|██████████| 104/104 [00:17<00:00,  5.92it/s]
# epoch 3, step 104, train loss 1.042, train acc 0.799, test acc 0.824
# epoch 4, step 104, train loss 0.949, train acc 0.806: 100%|██████████| 104/104 [00:17<00:00,  5.91it/s]
# epoch 4, step 104, train loss 0.949, train acc 0.806, test acc 0.830
# epoch 5, step 104, train loss 0.885, train acc 0.811: 100%|██████████| 104/104 [00:17<00:00,  5.91it/s]
# epoch 5, step 104, train loss 0.885, train acc 0.811, test acc 0.854
# epoch 6, step 104, train loss 0.826, train acc 0.812: 100%|██████████| 104/104 [00:17<00:00,  5.94it/s]
# epoch 6, step 104, train loss 0.826, train acc 0.812, test acc 0.852
# epoch 7, step 104, train loss 0.779, train acc 0.819: 100%|██████████| 104/104 [00:17<00:00,  5.90it/s]
# epoch 7, step 104, train loss 0.779, train acc 0.819, test acc 0.857
# epoch 8, step 104, train loss 0.741, train acc 0.821: 100%|██████████| 104/104 [00:17<00:00,  5.90it/s]
# epoch 8, step 104, train loss 0.741, train acc 0.821, test acc 0.846
# epoch 9, step 104, train loss 0.707, train acc 0.828: 100%|██████████| 104/104 [00:17<00:00,  5.92it/s]
# epoch 9, step 104, train loss 0.707, train acc 0.828, test acc 0.860
# train loss 0.707, train acc 0.828, test acc 0.860
# 1713.9 examples/sec on cuda