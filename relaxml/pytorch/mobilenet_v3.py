import os
import sys
import time
from typing import Dict, Tuple, List
import requests
import hashlib
from functools import partial
from shutil import copy, rmtree
import random
import zipfile
import tarfile
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
import torch.nn.functional as F
from torchvision import transforms, datasets
from matplotlib import pyplot as plt
from tqdm import tqdm
"""
实现MobileNet V3

实现说明:
https://tech.foxrelax.com/lightweight/mobilenet_v3/
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


def load_weight(arch: str = 'large', cache_dir: str = '../data') -> None:
    """
    加载预训练权重(class=1000的ImageNet数据集上训练的)
    """
    if arch == 'large':
        sha1_hash = 'db90cc27bffc26e99f4ce8a907f0c0e565366666'
        url = 'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/mobilenet_v3_large-8738ca79.pth'
    elif arch == 'small':
        sha1_hash = 'af9828929cb043737714380ab5af08b7ef76c5b2'
        url = 'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/mobilenet_v3_small-047dcff4.pth'
    else:
        raise ValueError(
            f'Unsupported MobileNet V3 arch {arch}: large or small expected')
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
    # e.g. ../data/mobilenet_v3_large-8738ca79.pth
    return fname


class ConvBNActivation(nn.Sequential):
    """
    数据通过这一层, 通道数会改变, 宽度和高度不会改变
    """

    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: nn.Module = None,
                 activation_layer: nn.Module = None) -> None:
        """
        参数:
        in_planes: 输入的通道数
        out_planes: 输出的通道数
        kernel_size: 卷积核大小
        stride: 步长
        groups: 卷积group
        norm_layer: 默认为: nn.BatchNorm2d
        activation_layer: 默认为: nn.ReLU6
        """
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNActivation, self).__init__(
            nn.Conv2d(in_channels=in_planes,
                      out_channels=out_planes,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      groups=groups,
                      bias=False), norm_layer(out_planes),
            activation_layer(inplace=True))


class SqueezeExcitation(nn.Module):
    """
    SE通道注意力

    数据通过这一层并不改变形状, 只是增加了注意力机制
    >>> x = torch.randn((2, 32, 128, 128))
    >>> assert SqueezeExcitation(32, 4)(x).shape == (2, 32, 128, 128)
    """

    def __init__(self, input_c: int, squeeze_factor: int = 4) -> None:
        """
        参数:
        input_c: 输入通道数数
        squeeze_factor: 第一个全连接层的输出为: input_c/squeeze_factor
        """
        super(SqueezeExcitation, self).__init__()
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_c, squeeze_c, 1)
        self.fc2 = nn.Conv2d(squeeze_c, input_c, 1)

    def forward(self, x: Tensor) -> Tensor:
        # scale.shape [batch_size, 1, 1, c]
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale, inplace=True)
        # 返回的形状就是x的形状
        return scale * x


class InvertedResidualConfig:
    """
    Block配置信息(和论文中的配置一一对应)
    """

    def __init__(self, input_c: int, kernel: int, expanded_c: int, out_c: int,
                 use_se: bool, activation: str, stride: int,
                 width_multi: float) -> None:
        """
        input_c: 输入的通道数
        kernel: 卷积核大小
        expanded_c: BNeck中第一个1x1卷积层输出的特征图的通道数
        out_c: 输出通道数
        use_se: 是否使用SE通道注意力
        activation: 激活函数的类型. HS(h-swish), RE(ReLU)
        stride: 步长, 网络使用卷积stride操作进行降采样, 没有使用pooling操作
        width_multi: 调节每个卷积层Channel的倍率因子
        """
        self.input_c = self.adjust_channels(input_c, width_multi)
        self.kernel = kernel
        self.expanded_c = self.adjust_channels(expanded_c, width_multi)
        self.out_c = self.adjust_channels(out_c, width_multi)
        self.use_se = use_se
        self.use_hs = activation == "HS"  # whether using h-swish activation
        self.stride = stride

    @staticmethod
    def adjust_channels(channels: int, width_multi: float) -> int:
        return _make_divisible(channels * width_multi, 8)


class InvertedResidual(nn.Module):
    """
    倒残差块(BNeck) MobileNet V3网络的基本结构
    """

    def __init__(self, cnf: InvertedResidualConfig,
                 norm_layer: nn.Module) -> None:
        """
        参数:
        cnf: InvertedResidualConfig
        norm_layer: Norm Layer
        """
        super(InvertedResidual, self).__init__()

        if cnf.stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        # 满足下面两个条件才使用残差连接:
        # 1. stride=2
        # 2. 输入通道和输出通道相同
        self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c)

        layers = []
        # "HS" - h-swish
        # "RE" - relu
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # 步骤1 Expand
        # 第一个1x1的卷积层
        if cnf.expanded_c != cnf.input_c:
            layers.append(
                ConvBNActivation(cnf.input_c,
                                 cnf.expanded_c,
                                 kernel_size=1,
                                 norm_layer=norm_layer,
                                 activation_layer=activation_layer))

        # 步骤2 Depthwise Conv
        # 逐通道卷积(分组卷积, 分组数=输入通道数=输出通道数)
        layers.append(
            ConvBNActivation(cnf.expanded_c,
                             cnf.expanded_c,
                             kernel_size=cnf.kernel,
                             stride=cnf.stride,
                             groups=cnf.expanded_c,
                             norm_layer=norm_layer,
                             activation_layer=activation_layer))

        # 步骤3 SE通道注意力(可选)
        # 数据通过这一层并不改变形状, 只是增加了注意力机制
        if cnf.use_se:
            layers.append(SqueezeExcitation(cnf.expanded_c))

        # 步骤4 Project
        # 用一个1x1卷积层来融合不同通道的特征, 同时改变输出通道数
        layers.append(
            ConvBNActivation(cnf.expanded_c,
                             cnf.out_c,
                             kernel_size=1,
                             norm_layer=norm_layer,
                             activation_layer=nn.Identity))

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_c
        self.is_strided = cnf.stride > 1

    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)
        if self.use_res_connect:
            result += x

        return result


class MobileNetV3(nn.Module):

    def __init__(self,
                 inverted_residual_setting: List[InvertedResidualConfig],
                 last_channel: int,
                 num_classes: int = 1000,
                 block: InvertedResidual = None,
                 norm_layer: nn.Module = None) -> None:
        """
        参数:
        inverted_residual_setting: list of InvertedResidualConfig
        last_channel: 倒数第二个全连接层输出节点个数
        num_classes: 类别数
        block: 默认是我们自己实现的BNeck: InvertedResidual 
        norm_layer: Norm Layer
        """
        super(MobileNetV3, self).__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers = []

        # 构建第一层
        firstconv_output_c = inverted_residual_setting[0].input_c
        layers.append(
            ConvBNActivation(3,
                             firstconv_output_c,
                             kernel_size=3,
                             stride=2,
                             norm_layer=norm_layer,
                             activation_layer=nn.Hardswish))
        # 构建中间的BNeck
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        # 构建最后几层
        lastconv_input_c = inverted_residual_setting[-1].out_c
        lastconv_output_c = 6 * lastconv_input_c
        layers.append(
            ConvBNActivation(lastconv_input_c,
                             lastconv_output_c,
                             kernel_size=1,
                             norm_layer=norm_layer,
                             activation_layer=nn.Hardswish))
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(lastconv_output_c, last_channel),
            nn.Hardswish(inplace=True), nn.Dropout(p=0.2, inplace=True),
            nn.Linear(last_channel, num_classes))

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def mobilenet_v3_large(num_classes: int = 1000,
                       reduced_tail: bool = False) -> MobileNetV3:
    """
    参数:
    num_classes: 类别数量
    reduced_tail: 是否要缩减C4, C5的特征数量
    """
    width_multi = 1.0
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels,
                              width_multi=width_multi)

    reduce_divider = 2 if reduced_tail else 1

    inverted_residual_setting = [
        # input_c, kernel, expanded_c, out_c, use_se, activation, stride
        bneck_conf(16, 3, 16, 16, False, "RE", 1),
        bneck_conf(16, 3, 64, 24, False, "RE", 2),  # C1
        bneck_conf(24, 3, 72, 24, False, "RE", 1),
        bneck_conf(24, 5, 72, 40, True, "RE", 2),  # C2
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 3, 240, 80, False, "HS", 2),  # C3
        bneck_conf(80, 3, 200, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 480, 112, True, "HS", 1),
        bneck_conf(112, 3, 672, 112, True, "HS", 1),
        bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2),  # C4
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider,
                   160 // reduce_divider, True, "HS", 1),
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider,
                   160 // reduce_divider, True, "HS", 1),
    ]
    last_channel = adjust_channels(1280 // reduce_divider)  # C5

    return MobileNetV3(inverted_residual_setting=inverted_residual_setting,
                       last_channel=last_channel,
                       num_classes=num_classes)


def mobilenet_v3_small(num_classes: int = 1000,
                       reduced_tail: bool = False) -> MobileNetV3:
    """
    参数:
    num_classes: 类别数量
    reduced_tail: 是否要缩减C4, C5的特征数量
    """
    width_multi = 1.0
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels,
                              width_multi=width_multi)

    reduce_divider = 2 if reduced_tail else 1

    inverted_residual_setting = [
        # input_c, kernel, expanded_c, out_c, use_se, activation, stride
        bneck_conf(16, 3, 16, 16, True, "RE", 2),  # C1
        bneck_conf(16, 3, 72, 24, False, "RE", 2),  # C2
        bneck_conf(24, 3, 88, 24, False, "RE", 1),
        bneck_conf(24, 5, 96, 40, True, "HS", 2),  # C3
        bneck_conf(40, 5, 240, 40, True, "HS", 1),
        bneck_conf(40, 5, 240, 40, True, "HS", 1),
        bneck_conf(40, 5, 120, 48, True, "HS", 1),
        bneck_conf(48, 5, 144, 48, True, "HS", 1),
        bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2),  # C4
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider,
                   96 // reduce_divider, True, "HS", 1),
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider,
                   96 // reduce_divider, True, "HS", 1)
    ]
    last_channel = adjust_channels(1024 // reduce_divider)  # C5

    return MobileNetV3(inverted_residual_setting=inverted_residual_setting,
                       last_channel=last_channel,
                       num_classes=num_classes)


def mobilenet_v3(arch: str = 'large',
                 pretrained: bool = True,
                 num_classes: int = 5) -> MobileNetV3:
    if arch == 'large':
        net = mobilenet_v3_large(num_classes)
    elif arch == 'small':
        net = mobilenet_v3_small(num_classes)
    else:
        raise ValueError(
            f'Unsupported MobileNet V3 arch {arch}: large or small expected')

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

    train_iter = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=batch_size,
                                             shuffle=True)

    validate_dataset = datasets.ImageFolder(val_root,
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    val_iter = torch.utils.data.DataLoader(validate_dataset,
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
    net = mobilenet_v3()
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
# epoch 0, step 104, train loss 1.003, train acc 0.700: 100%|██████████| 104/104 [00:17<00:00,  5.81it/s]
# epoch 0, step 104, train loss 1.003, train acc 0.700, test acc 0.868
# epoch 1, step 104, train loss 0.540, train acc 0.839: 100%|██████████| 104/104 [00:17<00:00,  5.83it/s]
# epoch 1, step 104, train loss 0.540, train acc 0.839, test acc 0.885
# epoch 2, step 104, train loss 0.441, train acc 0.861: 100%|██████████| 104/104 [00:17<00:00,  5.83it/s]
# epoch 2, step 104, train loss 0.441, train acc 0.861, test acc 0.896
# epoch 3, step 104, train loss 0.404, train acc 0.857: 100%|██████████| 104/104 [00:17<00:00,  5.82it/s]
# epoch 3, step 104, train loss 0.404, train acc 0.857, test acc 0.909
# epoch 4, step 104, train loss 0.370, train acc 0.875: 100%|██████████| 104/104 [00:17<00:00,  5.82it/s]
# epoch 4, step 104, train loss 0.370, train acc 0.875, test acc 0.923
# epoch 5, step 104, train loss 0.341, train acc 0.887: 100%|██████████| 104/104 [00:17<00:00,  5.82it/s]
# epoch 5, step 104, train loss 0.341, train acc 0.887, test acc 0.907
# epoch 6, step 104, train loss 0.340, train acc 0.888: 100%|██████████| 104/104 [00:17<00:00,  5.85it/s]
# epoch 6, step 104, train loss 0.340, train acc 0.888, test acc 0.909
# epoch 7, step 104, train loss 0.344, train acc 0.884: 100%|██████████| 104/104 [00:17<00:00,  5.85it/s]
# epoch 7, step 104, train loss 0.344, train acc 0.884, test acc 0.915
# epoch 8, step 104, train loss 0.328, train acc 0.889: 100%|██████████| 104/104 [00:17<00:00,  5.83it/s]
# epoch 8, step 104, train loss 0.328, train acc 0.889, test acc 0.929
# epoch 9, step 104, train loss 0.314, train acc 0.897: 100%|██████████| 104/104 [00:17<00:00,  5.79it/s]
# epoch 9, step 104, train loss 0.314, train acc 0.897, test acc 0.912
# train loss 0.314, train acc 0.897, test acc 0.912
# 1524.0 examples/sec on cuda