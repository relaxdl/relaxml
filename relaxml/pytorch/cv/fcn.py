import os
from typing import Callable, List, Tuple, Union
import requests
import hashlib
import zipfile
import tarfile
import time
import sys
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torchvision
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch.optim.optimizer import Optimizer
from matplotlib import pyplot as plt
from tqdm import tqdm
"""
语义分割(FCN)

实现说明:
https://tech.foxrelax.com/cv/fcn/
"""
"""
数据集的VOCtrainval_11-May-2012.tar文件大约为2GB
VOCdevkit/VOC2012/
                 Annotations/
                 ImageSets/
                 JPEGImages/
                 SegmentationClass/
                 SegmentationObject/
"""


def download(cache_dir: str = '../data') -> str:
    """
    下载数据
    """
    sha1_hash = '4e443f8a2eca6b1dac8a6c57641b67dd40621a49'
    url = 'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/VOCtrainval_11-May-2012.tar'
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
    # e.g. ../data/VOCtrainval_11-May-2012.tar
    return fname


def download_extract(cache_dir: str = '../data') -> str:
    """
    下载数据 & 解压
    """
    # 下载数据集
    fname = download(cache_dir)

    # 解压
    base_dir = os.path.dirname(fname)
    _, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, 'Only zip/tar files can be extracted.'
    fp.extractall(base_dir)
    # e.g. ../data/VOCdevkit
    return os.path.join(base_dir, 'VOCdevkit')


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


def read_voc_images(
        voc_dir: str,
        is_train: bool = True) -> Tuple[List[Tensor], List[Tensor]]:
    """
    将所有输入的图像(features)和标签(labels)读入内存

    1. `./VOCdevkit/VOC2012/ImageSets/Segmentation`目录下包含用于训练和测试样本的文本文件
    2. `./VOCdevkit/VOC2012/JPEGImages` - 存储每个样本的输入图像(features)
    3. `./VOCdevkit/VOC2012/SegmentationClass` - 存储每个样本的标签(labels)
        标签也采用图像格式, 其尺寸和它所标注的输入图像的尺寸相同
    4. 标签中颜色相同的像素属于同一个语义类别

    >>> voc_dir = download_extract()
    >>> train_features, train_labels = read_voc_images(os.path.join(voc_dir, 'VOC2012'), True)
    >>> assert len(train_features) == 1464
    >>> assert len(train_labels) == 1464
    # 不同样本的尺寸是不一样的: [channels, height, width]
    >>> train_features[0].shape
        torch.Size([3, 281, 500]) 
    >>> train_features[1].shape
        torch.Size([3, 375, 500])

    n = 5
    imgs = train_features[0:n] + train_labels[0:n]
    imgs = [img.permute(1, 2, 0) for img in imgs]
    show_images(imgs, 2, n)
    plt.show()

    返回: features, labels
    features: List of [channels, height, width]
    labels: List of [channels, height, width]
    
    features和labels都是一个list, 里面的元素是3-D Tensor, 取值范围是0-255
    """
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    mode = torchvision.io.image.ImageReadMode.RGB
    # train.txt内容: 每行都是一个文件名
    # 2007_000032
    # 2007_000039
    # 2007_000063
    # 2007_000068
    #
    # features:
    # VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg
    #
    # labels:
    # VOCdevkit/VOC2012/SegmentationClass/2007_000032.png
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(
            torchvision.io.read_image(
                os.path.join(voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(
            torchvision.io.read_image(
                os.path.join(voc_dir, 'SegmentationClass', f'{fname}.png'),
                mode))
    # 返回的features和labels都是一个list, 里面的元素是3-D Tensor,
    # 取值范围是0-255
    return features, labels


# 定义的两个常量, 方便地查找标签中每个像素的类索引
# RGB颜色值
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]
# 类名
VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor'
]


def voc_colormap2label() -> Tensor:
    """
    从RGB到VOC类别索引的映射

    RGB [128, 128, 0]对应的label是bird, 在索引3的位置
    (R*256 + G)*256 + B = (128*256 + 128)*256 + 0 = 8421376

    >>> colormap2label = voc_colormap2label()
    >>> colormap2label[8421376]
        tensor(3)
    """
    colormap2label = torch.zeros(256**3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[(colormap[0] * 256 + colormap[1]) * 256 +
                       colormap[2]] = i
    return colormap2label


def voc_label_indices(colormap: Tensor, colormap2label: Tensor) -> Tensor:
    """
    将VOC标签中的`RGB值`映射到它们的`类别索引`
    也就是将[C,H,W]格式的labels转换为[H,W]格式的labels

    >>> voc_dir = download_extract()
    >>> train_features, train_labels = read_voc_images(os.path.join(voc_dir, 'VOC2012'), True)
    >>> colormap2label = voc_colormap2label()
    >>> y = voc_label_indices(train_labels[0], colormap2label)
    >>> y[105:115, 130:140]
        tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]])
    >>> VOC_CLASSES[1]
        aeroplane

    参数:
    colormap: [channels, height, width]
    colormap2label: voc_colormap2label()

    返回:
    output: [height, width]
    """
    # colormap.shape [channels, height, width]
    #             -> [height, width, channels]
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    # idx.shape [height, width]

    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256 +
           colormap[:, :, 2])
    # colormap2label[idx].shape [height, width]
    return colormap2label[idx]


def voc_rand_crop(feature: Tensor, label: Tensor, height: int,
                  width: int) -> Tuple[Tensor, Tensor]:
    """
    随机裁剪`特征`和`标签`图像(处理一张Image)

    在图像分类的时候, 我们会通过缩放图像使其符合模型的输入形状. 然而在语义分割中, 我们通常
    将图像裁剪为固定尺寸, 而不是缩放, 具体来说, 我们使用`随机裁剪的方式`, 裁剪输入图像和标签的相同区域

    补充: 在语义分割中, 如果进行图像缩放, features可以用插值的方式来处理, 但是labels无法用插值来正确
    处理. 所以我们一般采用随机裁剪的方式来处理

    >>> voc_dir = download_extract()
    >>> train_features, train_labels = read_voc_images(os.path.join(voc_dir, 'VOC2012'), True)
    >>> n = 5
    >>> imgs = []
    >>> for _ in range(n):
    >>>     img = voc_rand_crop(train_features[0], train_labels[0], 200, 300)
    >>>     assert img[0].shape == (3, 200, 300)  # feature
    >>>     assert img[1].shape == (3, 200, 300)  # label
    >>>     imgs += img
    >>> imgs = [img.permute(1, 2, 0) for img in imgs]
    >>> show_images(imgs[::2] + imgs[1::2], 2, n)
    >>> plt.show()


    参数:
    feature: [channels, raw_height, raw_width]
    label: [channels, raw_height, raw_width]
    height: 需要裁剪的高度
    width: 需要裁剪的宽度

    返回: feature, label
    feature: [channels, height, width]
    label: [channels, height, width]
    """
    rect = torchvision.transforms.RandomCrop.get_params(
        feature, (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label


class VOCSegDataset(Dataset):
    """
    一个用于加载VOC数据集的自定义数据集

    >>> voc_dir = download_extract()
    >>> voc_ds = VOCSegDataset(True, (320, 480), os.path.join(voc_dir, 'VOC2012'))
    >>> feature, label = voc_ds[0]
    >>> assert feature.shape == (3, 320, 480)
    >>> assert label.shape == (320, 480)
        read 1114 examples
    """

    def __init__(self, is_train: bool, crop_size: Tuple[int, int],
                 voc_dir: str) -> None:
        """
        参数:
        is_train: 是否是训练集
        crop_size: (height, width)
        voc_dir: `VOCdevkit/VOC2012`
        """
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        # features: List of [channels, raw_height, raw_width]
        # labels: List of [channels, raw_height, raw_width]
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [
            self.normalize_image(feature) for feature in self.filter(features)
        ]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return self.transform(img.float() / 255)

    def filter(self, imgs: List[Tensor]) -> List[Tensor]:
        """
        过滤掉尺寸小于crop_size的图像
        """
        return [
            img for img in imgs if (img.shape[1] >= self.crop_size[0]
                                    and img.shape[2] >= self.crop_size[1])
        ]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        1. 裁剪feature & label
        2. 将label的像素值转换成标签格式
        """
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        return (feature, voc_label_indices(label, self.colormap2label))

    def __len__(self) -> int:
        return len(self.features)


def load_data_voc(
    batch_size: int = 64,
    crop_size: Tuple[int, int] = (320, 480)
) -> Tuple[DataLoader, DataLoader]:
    """
    加载VOC语义分割数据集

    >>> train_iter, _ = load_data_voc(batch_size=64, crop_size=(320, 480))
    >>> for X, y in train_iter:
    >>>     assert X.shape == (64, 3, 320, 480)
    >>>     assert y.shape == (64, 320, 480)
    >>>     break
        read 1114 examples
        read 1078 examples

    返回: train_iter, test_iter
    """
    voc_dir = download_extract()
    voc_dir = os.path.join(voc_dir, 'VOC2012')
    train_iter = DataLoader(VOCSegDataset(True, crop_size, voc_dir),
                            batch_size,
                            shuffle=True,
                            drop_last=True)
    test_iter = DataLoader(VOCSegDataset(False, crop_size, voc_dir),
                           batch_size,
                           drop_last=True)
    return train_iter, test_iter


def bilinear_kernel(in_channels: int, out_channels: int,
                    kernel_size: int) -> Tensor:
    """
    双线性插值
    """
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels, kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight


def fcn_net(num_classes: int = 21) -> nn.Module:
    """
    输入和输出的height/width是不变的, 变的是channels
    [输出的channels=num_classes]

    >>> net = fcn_net(num_classes=21)
    >>> x = torch.randn((2, 3, 320, 480))
    >>> assert net(x).shape == (2, 21, 320, 480)
    """
    pretrained_net = torchvision.models.resnet18(pretrained=True)
    # 输入: [batch_size, 3, 320, 480]
    # output [batch_size, 512, 320/32=10, 480/32=15]
    net = nn.Sequential(*list(pretrained_net.children())[:-2])
    # output [batch_size, 21, 10, 15]
    net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
    # output [batch_size, 21, 320, 480]
    net.add_module(
        'transpose_conv',
        nn.ConvTranspose2d(num_classes,
                           num_classes,
                           kernel_size=64,
                           padding=16,
                           stride=32))

    # 初始化新添加的层
    nn.init.xavier_uniform_(net.final_conv.weight)
    # 使用双线性插值初始化转置卷积的权重
    W = bilinear_kernel(num_classes, num_classes, 64)
    net.transpose_conv.weight.data.copy_(W)
    return net


def try_gpu(i: int = 0) -> torch.device:
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def accuracy(y_hat: Tensor, y: Tensor) -> Tensor:
    """
    计算预测正确的数量

    参数:
    y_hat [batch_size, num_classes, height, width]
    y [batch_size, height, width]
    """
    _, predicted = torch.max(y_hat, 1)
    cmp = predicted.type(y.dtype) == y
    return cmp.type(y.dtype).sum()


def loss(inputs: Tensor, targets: Tensor) -> Tensor:
    """
    参数:
    inputs: [batch_size, num_classes, height, width]
    targets: [batch_size, height, width]

    返回:
    loss: [batch_size, ]
    """
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)


def train_gpu(net: nn.Module,
              train_iter: DataLoader,
              test_iter: DataLoader,
              num_epochs: int = 10,
              loss: Union[nn.Module, Callable] = None,
              optimizer: Optimizer = None,
              device: torch.device = None) -> List[List[Tuple[int, float]]]:
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
    for epoch in range(num_epochs):
        # 训练
        metric_train = [0.0] * 4  # 统计: 训练集损失之和, 训练集准确数量之和, 训练集样本数量之和, 特点数
        net.train()
        train_iter_tqdm = tqdm(train_iter, file=sys.stdout)
        for i, (X, y) in enumerate(train_iter_tqdm):
            t_start = time.time()
            optimizer.zero_grad()
            # X.shape [batch_size, 3, height, width]
            # y.shape [batch_size, height, width]
            X, y = X.to(device), y.to(device)
            # y_hat.shape [batch_size, num_classes, height, width]
            y_hat = net(X)
            # l.shape [batch_size, ]
            l = loss(y_hat, y)
            l.sum().backward()
            optimizer.step()
            with torch.no_grad():
                metric_train[0] += float(l.sum())
                metric_train[1] += float(accuracy(y_hat, y))
                metric_train[2] += float(X.shape[0])
                metric_train[3] += float(y.numel())
            times.append(time.time() - t_start)
            train_loss = metric_train[0] / metric_train[2]
            train_acc = metric_train[1] / metric_train[3]
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
                metric_test[1] += float(y.numel())
            test_acc = metric_test[0] / metric_test[1]
            history[2].append((epoch + 1, test_acc))
            print(f'epoch {epoch}, step {i+1}, train loss {train_loss:.3f}, '
                  f'train acc {train_acc:.3f}, test acc {test_acc:.3f}')

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


def predict(img: Tensor, net: nn.Module, test_iter: DataLoader,
            device: torch.device) -> Tensor:
    """
    对图片img进行前向传播, 返回预测的类别

    参数:
    img: [3, 320, 480]
    
    返回:
    pred: [320, 480]
    """
    # 将输入图像在各个通道做标准化, 保持和训练时候的操作一致
    # X.shape [1, 3, 320, 480]
    X = test_iter.dataset.normalize_image(img).unsqueeze(0)
    # pred.shape [1, 21, 320, 480]
    #         -> [1, 320, 480]
    pred = net(X.to(device)).argmax(dim=1)
    # pred.shape [320, 480]
    return pred.reshape(pred.shape[1], pred.shape[2])


def label2image(pred: Tensor, device: torch.device) -> Tensor:
    """
    将预测类别映射回它们在数据集中的标注颜色

    参数:
    pred: [320, 480]

    返回:
    colormap: [320, 480, 3]
    """
    # colormap.shape [num_classes=21, 3]
    colormap = torch.tensor(VOC_COLORMAP, device=device)
    X = pred.long()
    return colormap[X, :]


def test_predict(net: nn.Module, test_iter: DataLoader,
                 device: torch.device) -> None:
    voc_dir = download_extract()
    voc_dir = os.path.join(voc_dir, 'VOC2012')
    # features: List of [3, raw_height, raw_width]
    # labels: List of [3, raw_height, raw_width]
    test_images, test_labels = read_voc_images(voc_dir, False)
    n, imgs = 4, []
    for i in range(n):
        crop_rect = (0, 0, 320, 480)
        # X.shape [3, 320, 480]
        X = torchvision.transforms.functional.crop(test_images[i], *crop_rect)
        # pred.shape [320, 480, 3]
        pred = label2image(predict(X, net, test_iter, device), device)
        imgs += [
            # X.shape [320, 480, 3]
            X.permute(1, 2, 0),
            pred.cpu(),
            # [320, 480, 3]
            torchvision.transforms.functional.crop(test_labels[i],
                                                   *crop_rect).permute(
                                                       1, 2, 0)
        ]
    # 第一行是原始图片; 第二行是预测结果; 第三行是真实标签
    show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2)
    plt.show()


def run(device: torch.device) -> Tuple[nn.Module, DataLoader, DataLoader]:
    train_iter, test_iter = load_data_voc(batch_size=32, crop_size=(320, 480))
    net = fcn_net()
    kwargs = {
        'num_epochs': 5,
        'loss': loss,
        'optimizer': torch.optim.SGD(net.parameters(),
                                     lr=0.001,
                                     weight_decay=1e-3),
        'device': device
    }
    history = train_gpu(net, train_iter, test_iter, **kwargs)
    plot_history(history)
    return net, train_iter, test_iter


if __name__ == '__main__':
    device = try_gpu()
    net, train_iter, test_iter = run(device)
    test_predict(net, test_iter, device)
# training on cuda:0
# epoch 0, step 34, train loss 1.154, train acc 0.747: 100%|██████████| 34/34 [00:15<00:00,  2.17it/s]
# epoch 0, step 34, train loss 1.154, train acc 0.747, test acc 0.813
# epoch 1, step 34, train loss 0.581, train acc 0.836: 100%|██████████| 34/34 [00:15<00:00,  2.23it/s]
# epoch 1, step 34, train loss 0.581, train acc 0.836, test acc 0.833
# epoch 2, step 34, train loss 0.463, train acc 0.860: 100%|██████████| 34/34 [00:15<00:00,  2.21it/s]
# epoch 2, step 34, train loss 0.463, train acc 0.860, test acc 0.839
# epoch 3, step 34, train loss 0.409, train acc 0.872: 100%|██████████| 34/34 [00:15<00:00,  2.21it/s]
# epoch 3, step 34, train loss 0.409, train acc 0.872, test acc 0.842
# epoch 4, step 34, train loss 0.370, train acc 0.882: 100%|██████████| 34/34 [00:15<00:00,  2.20it/s]
# epoch 4, step 34, train loss 0.370, train acc 0.882, test acc 0.849
# train loss 0.370, train acc 0.882, test acc 0.849
# 83.6 examples/sec on cuda:0