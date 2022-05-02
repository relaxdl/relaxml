import os
from typing import List, Tuple, Union
import requests
import hashlib
import zipfile
import tarfile
import numpy as np
import torch
from torch import Tensor
import torchvision
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
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


if __name__ == '__main__':
    train_iter, _ = load_data_voc(batch_size=64, crop_size=(320, 480))
    for X, y in train_iter:
        assert X.shape == (64, 3, 320, 480)
        assert y.shape == (64, 320, 480)
        break