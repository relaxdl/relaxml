from typing import Tuple, List, Union
import os
import requests
import hashlib
import zipfile
import tarfile
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import torchvision
import matplotlib.pyplot as plt

_DATA_HUB = dict()
_DATA_HUB['cat1'] = (
    'f9c5b905d908b97eeeb64ff34a46fa8b143f88f8',
    'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/img/cat1.jpg')
_DATA_HUB['cat2'] = (
    'b712adcb9ca6af53081bd96426b719573f40053e',
    'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/img/cat2.jpg')
_DATA_HUB['cat3'] = (
    '80249a6aa841706d861f3f7efad582f6828ca3d0',
    'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/img/cat3.jpg')
_DATA_HUB['catdog'] = (
    '60b7d540db03eef6b9834329bccc4417ef349bf6',
    'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/img/catdog.jpg')
_DATA_HUB['elephant'] = (
    '9225c1b5be359a4fe7617d5f9cc6e6a28155b624',
    'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/img/elephant.jpg')
_DATA_HUB['banana'] = (
    'cdcc1e668faf05c173389b6a73d33ed8d44cf4b4',
    'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/img/banana.jpg')
_DATA_HUB['banana_detection'] = (
    '068e13f04d30c8b96d30e490394ce9ab7cbdf2d5',
    'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/banana_detection.zip')


def download(name: str, cache_dir: str = '../data') -> str:
    """
    下载数据
    """
    sha1_hash, url = _DATA_HUB[name]
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
    # e.g. ../data/file.zip
    return fname


def download_extract(name: str, cache_dir: str = '../data') -> str:
    """
    下载数据 & 解压
    """
    # 下载数据集
    fname = download(name, cache_dir)

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
    # e.g. ../data/file
    return data_dir


"""
香蕉数据集介绍:

1. banana_detection.zip解压之后的目录结构:
../data/banana_detection/
    bananas_train/
        /images/
            0.png
            1.png
            ...
            999.png
        /label.csv
    bananas_val/
        /images/
            0.png
            1.png
            ...
            99.png
        /label.csv
2. 训练集一共1000张图片, 尺寸为3x256x256; 验证集一共100张图片, 尺寸为3x256x256
3. label.csv格式:
img_name,label,xmin,ymin,xmax,ymax
0.png,0,104,20,143,58
1.png,0,68,175,118,223

labe从0开始, 香蕉数据集只有一类数据, 所以标签都为0 
"""


def read_data_bananas(is_train: bool = True) -> Tuple[Tensor, Tensor]:
    """
    读取香蕉检测数据集中的图像和标签

    参数:
    is_train: bool

    返回: (images, labels)
    images: list of Tensor, 每个Tensor的形状是: [C,H,W] = [3,256,256]
        数值转换为0-1.0之间的float, 方便神经网络处理
    targets: [num_examples, num_gt_boxes, 5]
        训练集: num_examples=1000; 验证集: num_examples=100
        num_gt_boxes=1, 这个数据集中每张图片有一个目标需要检测
        [label,xmin,ymin,xmax,ymax], 坐标都转换成了范围为0-1.相对坐标
    """
    data_dir = download_extract(name='banana_detection')
    csv_fname = os.path.join(data_dir,
                             'bananas_train' if is_train else 'bananas_val',
                             'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    # 使用read_image一张图片一张图片的读取, 这个方法可以把JPEG/PNG image转换成CHW格
    # 式的3-D RGB Tensor, 每个像素是取值在0-255之间的uint8; 之后再除以255, 将数值转换为
    # 0-1.0之间的float, 方便神经网络处理
    for img_name, target in csv_data.iterrows():
        images.append(
            torchvision.io.read_image(
                os.path.join(data_dir, 'bananas_train' if is_train else
                             'bananas_val', 'images', f'{img_name}')) / 255.)
        # [label,xmin,ymin,xmax,ymax]
        targets.append(list(target))

    # 图片尺寸的大小是256x256, 最终除以256表示坐标在图片的相对位置
    #
    # 如果is_train=True
    # image是一个长度为1000的list, 每个元素是一个Tensor, image.shape [3, 256, 256]
    # targets.shape [1000, 1, 5]
    #
    # 注意:
    # targets插入一个维度的原因是有时候一张图片会有多个ground_truth, 我们这个例子做了简化,
    # 每个图片只有一个目标需要检测, targets最后除以256是将ground_truth保存成0-1范围内的
    # 相对坐标
    return images, torch.tensor(targets).unsqueeze(1) / 256.


class BananasDataset(Dataset):
    """
    用于加载香蕉检测数据集的自定义数据集
    """

    def __init__(self, is_train: bool) -> None:
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) +
              (f' training examples' if is_train else f' validation examples'))

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return (self.features[idx], self.labels[idx])

    def __len__(self) -> None:
        return len(self.features)


def load_data_bananas(batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """
    加载香蕉检测数据集(是一个Mini的目标检测数据集)

    1. 图片尺寸为3x256x256, 像素值被处理成0-1.之间的浮点数
    2. 训练集一共1000张图片, 验证集一共100张图片
    3. 需要检测的目标只有1类, 就是: 香蕉(label=0)
    4. 每张图片只有1个目标需要检测
    5. 返回的数据格式是: [batch_size, max_gt_boxes, 5]
       max_gt_boxes任何图像中边界框可能出现的最大数量, 在我们的数据集中为1
    6. 标签的格式: [label, xmin, ymin, xmax, ymax]
       <1> 标签从0开始, 当前只有一类, 0 - 香蕉
       <2> 里面的4个坐标都被处理成了0-1.之间的相对坐标, 乘以图片原始的宽和高之后, 
           就可以得到真实的像素坐标了

    # Example1
    >>> train_iter, val_iter = load_data_bananas(batch_size=32)
    >>> for x, y in train_iter:
    >>>     assert x.shape == (32, 3, 256, 256)
    >>>     assert y.shape == (32, 1, 5)
    >>>     print(y[:5]) # 打印前5张张片对应的labels
            tensor([[[0.0000, 0.2812, 0.2539, 0.4805, 0.5234]],
                    [[0.0000, 0.2930, 0.4883, 0.5195, 0.7422]],
                    [[0.0000, 0.4492, 0.5234, 0.6641, 0.7344]],
                    [[0.0000, 0.0977, 0.2031, 0.3672, 0.3945]],
                    [[0.0000, 0.0469, 0.0742, 0.2461, 0.2695]]])
    >>>     break

    # Example2
    >>> train_iter, val_iter = load_data_bananas(batch_size=32)
    >>> edge_size = 256  # 图片的H/W
    >>> for x, y in train_iter:
    >>>     imgs = x[:10].permute(0, 2, 3, 1) # [batch_size, C, H, W] -> [batch_size, H, W, C]
    >>>     labels = y[:10] # [batch_size, max_gt_boxes=1, 5]
    >>>     break
    # 绘图, 显示imgs & labels
    >>> axes = show_images(imgs, 2, 5, scale=2)
    >>> for ax, label in zip(axes, labels):
    >>>     show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
    >>> plt.show()
    """
    train_iter = DataLoader(BananasDataset(is_train=True),
                            batch_size,
                            shuffle=True)
    val_iter = DataLoader(BananasDataset(is_train=False), batch_size)
    return train_iter, val_iter


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


def show_bboxes(axes: plt.Axes,
                bboxes: Tensor,
                labels: List[str] = None,
                colors: List[str] = None) -> None:
    """
    在图像(axes)上绘制多个bboxes

    # 下载显示图片
    >>> img = plt.imread(download('catdog'))  # img.shape [H, W, 3]
    >>> fig = plt.imshow(img)
    # 显示bboxes
    >>> h, w = img.shape[:2]  # h=561, w=728
    >>> bbox_scale = torch.tensor((w, h, w, h))
    # 5个锚框的相对坐标: [xmin, ymin, xmax, ymax]
    # 乘以`bbox_scale`之后就变为像素值了
    >>> bboxes = torch.Tensor([[0.0620, 0.0804, 0.6399, 0.8304],
                               [0.1583, 0.2054, 0.5436, 0.7054],
                               [0.2546, 0.3304, 0.4473, 0.5804],
                               [-0.0577, 0.1903, 0.7596, 0.7206],
                               [0.1466, -0.0749, 0.5553, 0.9858]])
    # sizes=[0.75, 0.5, 0.25]
    # ratios=[1, 2, 0.5]
    # 会得到下面的标签(包含sizes[0]和ratios[0]的所有组合), 所以:
    # labels = (0.75, 1), (0.5, 1), (0.25, 1), (0.75, 2), (0.75, 0.5)
    >>> labels = [
            's=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2',
            's=0.75, r=0.5'
        ]
    >>> show_bboxes(fig.axes, bboxes * bbox_scale, labels)
    >>> plt.show()

    参数:
    axes: matplotlib axes
    bboxes: [num_bboxes, 4]
        [xmin, ymin, xmax, ymax], 是像素坐标
    labels: 标签列表
    colors: 颜色列表
    """

    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (tuple, list)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        # 画rect
        color = colors[i % len(colors)]  # 当前边界框的颜色
        rect = bbox_to_rect(bbox.detach().numpy(), color)
        axes.add_patch(rect)

        # 在rect的左上角画label
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'  # 字体默认白色
            axes.text(rect.xy[0],
                      rect.xy[1],
                      labels[i],
                      va='center',
                      ha='center',
                      fontsize=9,
                      color=text_color,
                      bbox=dict(facecolor=color, lw=0))


def bbox_to_rect(bbox: Tuple[float, float, float, float],
                 color: str) -> plt.Rectangle:
    """
    将bbox转换成matplotlib rect, 可以直接add到axes上显示

    >>> img = plt.imread(download('catdog'))
    >>> fig = plt.imshow(img)
    >>> dog_bbox = [60.0, 45.0, 378.0, 516.0]
    >>> cat_bbox = [400.0, 112.0, 655.0, 493.0]
    >>> fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
    >>> fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'))
    >>> plt.show()

    参数:
    bbox: [xmin, ymin, xmax, ymax] - 里面的元素是像素值
        e.g. [60.0, 45.0, 378.0, 516.0]

    返回:
    matplotlib rect
    """
    return plt.Rectangle(xy=(bbox[0], bbox[1]),
                         width=bbox[2] - bbox[0],
                         height=bbox[3] - bbox[1],
                         fill=False,
                         edgecolor=color,
                         linewidth=2)


if __name__ == '__main__':
    train_iter, val_iter = load_data_bananas(batch_size=32)
    edge_size = 256  # 图片的H/W
    for x, y in train_iter:
        # [batch_size, C, H, W] -> [batch_size, H, W, C]
        imgs = x[:10].permute(0, 2, 3, 1)
        # labels.shape [batch_size, num_gt_boxes=1, 5]
        labels = y[:10]
        break
    # 绘图, 显示imgs & labels
    axes = show_images(imgs, 2, 5, scale=2)
    for ax, label in zip(axes, labels):
        show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
    plt.show()