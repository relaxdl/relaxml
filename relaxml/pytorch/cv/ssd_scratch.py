from typing import Any, List, Tuple, Union
import os
import requests
import time
import hashlib
import sys
import zipfile
import tarfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.optim.optimizer import Optimizer
from torch.nn import functional as F
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
"""
SSD从零实现

实现说明:
https://tech.foxrelax.com/cv/ssd_scratch/
"""

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
2. 训练集一共1000张图片, shape为3x256x256; 验证集一共100张图片, shape为3x256x256
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

    1. 图片shape为3x256x256, 像素值被处理成0-1.之间的浮点数
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


def set_figsize(figsize: Tuple[int, int] = (6, 4)) -> None:
    plt.rcParams['figure.figsize'] = figsize


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


def box_corner_to_center(boxes: Tensor) -> Tensor:
    """
    [xmin, ymin, xmax, ymax] -> [xcenter, ycenter, width, height]
    
    参数:
    boxes: [num_boxes, 4]
    
    返回: 
    boxes: [num_boxes, 4]
    """
    # x1.shape [num_boxes, ]
    # y1.shape [num_boxes, ]
    # x2.shape [num_boxes, ]
    # y2.shape [num_boxes, ]
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    # boxes.shape [num_boxes, 4]
    boxes = torch.stack((cx, cy, w, h), axis=-1)
    return boxes


def box_center_to_corner(boxes: Tensor) -> Tensor:
    """
    [xcenter, ycenter, width, height] -> [xmin, ymin, xmax, ymax]

    >>> dog_bbox = [60.0, 45.0, 378.0, 516.0]
    >>> cat_bbox = [400.0, 112.0, 655.0, 493.0]
    >>> boxes = torch.tensor((dog_bbox, cat_bbox))
    >>> box_center_to_corner(box_corner_to_center(boxes)) == boxes

    参数:
    boxes: [num_boxes, 4]
    
    返回: 
    boxes: [num_boxes, 4]
    """
    # cx.shape [num_boxes, ]
    # cy.shape [num_boxes, ]
    # w.shape [num_boxes, ]
    # h.shape [num_boxes, ]
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    # boxes.shape [num_boxes, 4]
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes


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


def multibox_prior(data: Tensor, sizes: List[float],
                   ratios: List[float]) -> Tensor:
    """
    输入图像(我们只关心图像的height和width)、尺寸(缩放比)列表和宽高比列表, 生成anchor box

    1. data是一个批量的图片, 具有相同的shape, 我们只会用到其宽度和高度, 函数会生成以data
    每个像素为中心具有不同形状的anchor box. 传递不同的sizes和ratios, 会生成不同数量的
    anchor box
    2. 每个像素生成的anchor box数量是: num_sizes + num_ratios - 1
    3. 要注意的一点是返回的数据会和data在同样的device上, 如果data是GPU上的数据, 返回的
    anchor box同样在GPU上
    3. 返回的anchor box格式: [xmin, ymin, xmax, ymax], 坐标是相对坐标, 需要乘以H和W才能
    还原出像素坐标: e.g. [0.1583, 0.2054, 0.5436, 0.7054]

    >>> batch_size, height, width, channels = 32, 561, 728, 3
    >>> X = torch.randn((batch_size, channels, height, width))
    >>> boxes = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
    # 锚框数量=height*width*(num_sizes+num_ratios-1)=561*728*(3+3-1)
    >>> assert boxes.shape == (1, height * width * (3 + 3 - 1), 4) # [1, 2042040, 4]

    # 我们将返回的Y的形状reshape成: [height, width, boxes_per_pixel, 4]
    # 这样就可以获取以任意一像素为中心的锚框了
    >>> boxes = boxes.reshape(height, width, 5, 4)

    # 获取以(255, 255)这个像素点为中心的第1个锚框
    >>> boxes[255, 255, 0, :]
        tensor([0.0620, 0.0804, 0.6399, 0.8304])

    # 获取以(255, 255)这个像素点为中心的第2个锚框
    >>> boxes[255, 255, 1, :]
        tensor([0.1583, 0.2054, 0.5436, 0.7054])

    # 获取以(255, 255)这个像素点为中心的所有锚框, 一共3x3-1=5个
    >>> boxes[255, 255, :, :]
        tensor([[ 0.0620,  0.0804,  0.6399,  0.8304],
                [ 0.1583,  0.2054,  0.5436,  0.7054],
                [ 0.2546,  0.3304,  0.4473,  0.5804],
                [-0.0577,  0.1903,  0.7596,  0.7206],
                [ 0.1466, -0.0749,  0.5553,  0.9858]])

    参数:
    data的: [batch_size, channels, height, width] height/width 的单位是像素
    sizes: [num_sizes, ] e.g. [0.75, 0.5, 0.25] 尺寸(缩放比)列表
    ratios: [num_ratios, ] e.g. [1, 2, 0.5] 宽高比列表

    返回:
    outputs: [1, height*width*boxes_per_pixel, 4]
        a. boxes_per_pixel表示每个像素anchor box数量
        b. [xmin, ymin, xmax, ymax], 是相对坐标, e.g. [0.1583, 0.2054, 0.5436, 0.7054]
    """
    # 图片的高度H和宽度W(像素)
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)  # 每个像素对应的anchor box数量
    # size_tensor.shape [num_sizes, ]
    size_tensor = torch.tensor(sizes, device=device)
    # ratio_tensor.shape [num_ratios, ]
    ratio_tensor = torch.tensor(ratios, device=device)

    # 为了将锚点移动到像素的中心, 需要设置偏移量
    # 因为一个像素的的高为1且宽为1, 我们选择偏移我们的中心0.5
    offset_h, offset_w = 0.5, 0.5
    # 像素坐标乘以缩放比例(steps_w, steps_h), 就可以得到相对坐标
    steps_h = 1.0 / in_height
    steps_w = 1.0 / in_width

    # 生成锚框的所有中心点, 相对坐标, 取值范围:0-1
    # center_h.shape [height, ]
    # center_w.shape [width, ]
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    # shift_x.shape [height, width]
    # shift_y.shape [height, width]
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    # shift_x.shape [height*width, ]
    # shift_y.shape [height*width, ]
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # 生成`boxes_per_pixel`个anchor box的高和宽(相对宽度和高度), 之后用于创建锚框的四角坐标
    # w.shape [boxes_per_pixel, ]
    # h.shape [boxes_per_pixel, ]
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                     size_tensor[0] * torch.sqrt(ratio_tensor[1:])))\
                     * in_height / in_width  # Handle rectangular inputs
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   size_tensor[0] / torch.sqrt(ratio_tensor[1:])))

    # 每个像素点都对应`boxes_per_pixel`个锚框, 所以整张图片就对应了`height*width*boxes_per_pixel`个锚框
    # anchor_manipulations.shape [height*width*boxes_per_pixel, 4]
    anchor_manipulations = torch.stack(
        (-w, -h, w, h)).T.repeat(in_height * in_width, 1) / 2  # 除以2来获得半高和半宽

    # 补充:
    # 这里要注意repeat()和repeat_interleave()在行为上的区别, 也就是在元素排列方式上的不同
    # 1. 将一个shape为[boxes_per_pixel, 4]的tensor通过repeat(H * W, 1)
    #    变为[height*width*boxes_per_pixel, 4]
    # 2. 将一个shape为[height*width, 4]的tensor通过repeat_interleave(boxes_per_pixel, dim=0)
    #    变为[height*width*boxes_per_pixel, 4]
    # 目的是为了让其对应逻辑元素对应, 之后可以相加

    # shift_x.shape [height*width, ]
    # shift_y.shape [height*width, ]
    # out_grid.shape [height*width, 4] -> [height*width*boxes_per_pixel, 4]
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                           dim=1).repeat_interleave(boxes_per_pixel, dim=0)

    # output.shape [height*width*boxes_per_pixel, 4]
    output = out_grid + anchor_manipulations

    # batch_size=1, 生成的anchor box和批量大小无关
    # output.shape [1, height*width*boxes_per_pixel, 4]
    return output.unsqueeze(0)


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


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    `边界框列表boxes1`和`边界框列表boxes2`中边界框成对的交并比-IOU

    >>> ground_truth = torch.tensor([[0.1, 0.08, 0.52, 0.92],
                                     [0.55, 0.2, 0.9, 0.88]])
    >>> anchors = torch.tensor([[0, 0.1, 0.2, 0.3], 
                                [0.15, 0.2, 0.4, 0.4],
                                [0.63, 0.05, 0.88, 0.98], 
                                [0.66, 0.45, 0.8, 0.8],
                                [0.57, 0.3, 0.92, 0.9]])
    >>> jaccard = box_iou(anchors, ground_truth)
    >>> assert jaccard.shape == (5, 2)

    参数:
    boxes1: [num_boxes1, 4]
        [xmin, ymin, xmax, ymax]
    boxes2: [num_boxes2, 4]
        [xmin, ymin, xmax, ymax]

    返回:
    output: [num_boxes1, num_boxes2]
    """
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # boxes1.shape [num_boxes1, 4]
    # boxes2.shape [num_boxes2, 4]
    # areas1.shape [num_boxes1,]
    # areas2.shape [num_boxes2,]
    areas1 = box_area(boxes1)  # boxes1中每个框的面积
    areas2 = box_area(boxes2)  # boxes2中每个框的面积

    # 最后一个维度表示左上角的坐标: [xmin, ymin]
    # inter_upperlefts.shape [num_boxes1, num_boxes2, 2]
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    # 最后一个维度表示右下角的坐标:[xmax, ymax]
    # inter_lowerrights.shape [num_boxes1, num_boxes2, 2]
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    # 最后一个维度表示并集(inner)的: [宽度, 高度]
    # inters.shape [num_boxes1, num_boxes2, 2]
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # inter_areas.shape [num_boxes1, num_boxes2]
    inter_areas = inters[:, :, 0] * inters[:, :, 1]  # 并集的面积
    # union_areas.shape [num_boxes1, num_boxes2]
    union_areas = areas1[:, None] + areas2 - inter_areas  # 交集的面积
    return inter_areas / union_areas


def assign_anchor_to_bbox(ground_truth: Tensor,
                          anchors: Tensor,
                          device: torch.device,
                          iou_threshold: float = 0.5) -> Tensor:
    """
    将真实边界框(ground_truth)分配给anchors

    分配规则:
    1. 一个ground_truth可能会被关联到多个anchor
    2. 每个ground_truth至少会被关联到一个anchor
    3. 有的anchor可能会没有关联ground_truth(索引为-1)

    下面的例子有2个ground_truth, 5个anchors:
    >>> ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                                     [1, 0.55, 0.2, 0.9, 0.88]])
    >>> anchors = torch.tensor([[0, 0.1, 0.2, 0.3], 
                                [0.15, 0.2, 0.4, 0.4],
                                [0.63, 0.05, 0.88, 0.98], 
                                [0.66, 0.45, 0.8, 0.8],
                                [0.57, 0.3, 0.92, 0.9]])
    >>> anchors_bbox_map = assign_anchor_to_bbox(ground_truth[:, 1:], anchors, None)
    >>> anchors_bbox_map
        tensor([-1,  0,  1, -1,  1])
    可以看到每个anchor分配的ground_truth的索引. 0这个ground_truth被分配给了1个anchor,
    1这个ground_truth被分配给了2个anchor. -1表示anchor没有分配到ground_truth

    参数:
    ground_truth: [num_gt_boxes, 4]
        [xmin, ymin, xmax, ymax]相对坐标
    anchors的形状: [num_anchors, 4]
        [xmin, ymin, xmax, ymax]相对坐标
    device: 返回的数据会放在device上
    iou_threshold: 低于iou_threshold的锚框不会关联真实的边界框

    返回:
    output: [num_anchors, ]
        每个anchor对应一个`ground_truth的索引`, 没有关联到ground_truth的anchor对应的索引是-1
    """

    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # 计算锚框和真实边界框的IoU
    # 位于第i行和第j列的元素x_ij是锚框i和真实边界框j的IoU
    # jaccard.shape [num_anchors, num_gt_boxes]
    jaccard = box_iou(anchors, ground_truth)

    # 每个锚框默认的真实边界框索引为: -1
    # anchors_bbox_map.shape [num_anchors, ]
    anchors_bbox_map = torch.full((num_anchors, ),
                                  -1,
                                  dtype=torch.long,
                                  device=device)

    # 根据阈值, 决定是否分配真实边界框(一个ground_truth可能会被关联到多个anchor)
    # max_ious.shape [num_anchors, ] - 每个anchor对应的最大的IoU
    # indices.shape [num_anchors, ] - 每个anchor最大的IoU对应的ground_truth索引
    max_ious, indices = torch.max(jaccard, dim=1)
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    box_j = indices[max_ious >= iou_threshold]
    # 到目前为止每一行有满足max_ious >= iou_threshold条件的anchor, 都会被分配
    # 一个ground_truth索引
    anchors_bbox_map[anc_i] = box_j

    # 下面这部分逻辑处理的是这种情况(A表示anchor, B表示ground_truth):
    # |    | B1 |  B2  |  B3  | B4 |
    # | A1 | .. | 0.85 | 0.80 | .. |
    # | A2 | .. | 0.98 | ..   | .. |
    # 上面的算法会把B2同时分配给A2和A1, 而B3并没有关联任何锚框, 执行下面的逻辑,
    # 会把B3分配给A1, 来覆盖之前的分配
    #
    # 下面的逻辑执行完后, 会保证: 每个ground_truth至少会被关联到一个anchor
    col_discard = torch.full((num_anchors, ), -1)  # 用来覆盖(丢弃)某一列的数据
    row_discard = torch.full((num_gt_boxes, ), -1)  # 用来覆盖(丢弃)某一行的数据
    for _ in range(num_gt_boxes):
        # max_idx是一个标量, 通过max_idx可以计算出最大值的行索引和列索引
        max_idx = torch.argmax(jaccard)
        box_idx = (max_idx % num_gt_boxes).long()  # 最大值的列索引
        anc_idx = (max_idx / num_gt_boxes).long()  # 最大值的行索引
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard

    # anchors_bbox_map.shape [num_anchors, ]
    return anchors_bbox_map


def offset_boxes(anchors: Tensor,
                 assigned_bb: Tensor,
                 eps: float = 1e-6) -> Tensor:
    """
    锚框 & 真实边界框 -> 偏移量

    计算锚框(anchors)相对于其关联的真实边界框(assigned_bb)的偏移量(offset), 
    返回偏移量

    >>> assigned_bb = torch.tensor([[0.1, 0.08, 0.52, 0.92],
                                    [0.55, 0.2, 0.9, 0.88]])
    >>> anchors = torch.tensor([[0.66, 0.45, 0.8, 0.8], 
                                [0.57, 0.3, 0.92, 0.9]])
    # anchors & 关联的真实边界框 -> offset
    >>> offset = offset_boxes(anchors, assigned_bb)
    >>> offset
        tensor([[-3.0000e+01, -3.5714e+00,  5.4931e+00,  4.3773e+00],
                [-5.7143e-01, -1.0000e+00,  4.1723e-06,  6.2582e-01]])
    # anchors & offset -> 真实边界框
    >>> predicted_bb = offset_inverse(anchors, offset)
    >>> predicted_bb
        tensor([[0.1000, 0.0800, 0.5200, 0.9200],
                [0.5500, 0.2000, 0.9000, 0.8800]])

    参数:
    anchors: [num_anchors, 4] 
        [xmin, ymin, xmax, ymax], 相对坐标
    assigned_bb: [num_anchors, 4] anchors关联的ground_truth
        [xmin, ymin, xmax, ymax], 相对坐标
    返回:
    offset: [num_anchors, 4]
    """
    # 计算的时候使用的是中心坐标

    # c_anc.shape [num_anchors, 4]
    # c_assigned_bb.shape [num_anchors, 4]
    c_anc = box_corner_to_center(anchors)
    c_assigned_bb = box_corner_to_center(assigned_bb)
    # offset_xy.shape [num_anchors, 2]
    # offset_wh.shape [num_anchors, 2]
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    # offset.shape [num_anchors, 4]
    offset = torch.cat([offset_xy, offset_wh], axis=1)
    return offset


def offset_inverse(anchors: Tensor, offset_preds: Tensor) -> Tensor:
    """
    锚框 & 预测偏移量 -> 真实边界框

    锚框(anchors)和预测的偏移量(offset_preds)来计算真实边界框(pred_bbox), 返回真实边界框

    参数:
    anchors: [num_anchors, 4]
        [xmin, ymin, xmax, ymax], 相对坐标
    offset_preds: [num_anchors, 4]

    返回:
    pred_bbox: [num_anchors, 4]
        [xmin, ymin, xmax, ymax], 相对坐标
    """
    # 计算的时候使用的是中心坐标

    # anc.shape [num_anchors, 4]
    anc = box_corner_to_center(anchors)

    # pred_bbox_xy.shape [num_anchors, 2]
    # pred_bbox_wh.shape [num_anchors, 2]
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]

    # pred_bbox.shape [num_anchors, 4]
    pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), axis=1)
    # pred_bbox.shape [num_anchors, 4]
    pred_bbox = box_center_to_corner(pred_bbox)
    return pred_bbox


def nms(boxes: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
    """
    使用最大抑制, 去掉多于的边界框, 返回最终保留下来的边界框的`索引`

    参数:
    boxes: [num_boxes, 4] 预测出来的真实边界框
    scores: [num_boxes, ] 每个真实边界框对应的预测概率最大类的概率值
    iou_threshold: IoU大于iou_threshold的会被抑制

    返回:
    output: [num_keep, ]
    返回的索引是按照预测概率从大到小排序过的, 其中num_keep <= num_boxes
    """
    # B.shape [num_boxes, ] 概率从大到小的索引
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []  # 保留预测边界框的索引, 最终会返回
    while B.numel() > 0:
        i = B[0]  # 取出当前概率最大索引
        keep.append(i)
        if B.numel() == 1: break
        # 参数1的形状: [1, 4]
        # 参数2的形状: [当前B中的元素个数-1, 4]
        # iou的形状:  [1, 当前B中的元素个数-1] -> [当前B中的元素个数-1, ]
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        # inds的形状: [当前B中满足条件的元素个数, ]
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        # inds的形状: [当前B中满足条件的元素个数, ]
        # 这里索引+1的原因是我们要跳过当前这一轮keep住的`i`
        B = B[inds + 1]
    # 返回: [num_keep, ]
    return torch.tensor(keep, device=boxes.device)


def multibox_target(anchors: Tensor,
                    labels: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """
    使用真实边界框(ground_truth)标记锚框(anchors)的`类别`和`offset`, 之后就可以训练了

    如果一个anchor没有被分配ground_truth, 我们只需将锚框的类别标记为"背景"(background).
    背景类别的anchor通常被称为"负类", 其余的被称为"正类". 我们使用真实边界框(labels参数)
    来标记anchor的类别和偏移量. 要注意的是: 下面函数会将背景类别的索引设置为0, 然后将类别的
    整数索引+1. 也就是标签中的真实类别是用0开始的, 但是经过这个函数处理后, 真实类别是从1开始的,
    因为0被分配给了背景

    遍历每张图片(label)
    1. 将真实边界框(ground_truth)分配给anchors [num_anchors, ]
       每个anchor对应一个`ground_truth的索引`, 没有关联ground_truth的anchor对应的索引是-1
    2. 根据#1的结果, 计算`bbox_mask` [num_anchors, 4]
       如果anchor被关联到了ground_truth, mask=1, 否则mask=0
    3. 根据#1的结果, 获取anchors对应的类的标签`class_labels`: [num_anchors, ]
       背景类为0, 其它类别从1开始. 要注意的是在这里我们默认把label的类别标签做了+1处理
    4. 根据#1的结果, 获取anchors对应的ground_truth坐标`assigned_bb`: [num_anchors, 4]
    5. 根据anchors和#4的结果, 也就是其关联的ground_truth坐标计算`bbox_offset`: [num_anchors, 4]
       如果对应的anchor没有关联到ground_truth, 则offset为0
    6. 将数据拉平后返回:
       <1> bbox_offset [num_anchors, 4] -> [num_anchors*4, ]
       <2> bbox_mask [num_anchors, 4] -> [num_anchors*4, ]
       <3> class_labels [num_anchors, ]

    例子:
    # 1张图片中有2个ground_truth
    # 标签中的类别是: 0-狗;1-猫
    # 处理之后类别会变为: 0-背景;1-狗;2-猫
    >>> batch_size, num_anchors, num_gt_boxes = 1, 5, 2
    >>> ground_truth = torch.tensor([
           [0, 0.1, 0.08, 0.52, 0.92],  # 0 - 狗
           [1, 0.55, 0.2, 0.9, 0.88]  # 1 - 猫
        ])
    # 有5个anchors box
    >>> anchors = torch.tensor([[0, 0.1, 0.2, 0.3], 
                                [0.15, 0.2, 0.4, 0.4],
                                [0.63, 0.05, 0.88, 0.98], 
                                [0.66, 0.45, 0.8, 0.8],
                                [0.57, 0.3, 0.92, 0.9]])
    >>> anchors = anchors.unsqueeze(dim=0)
    >>> anchors.shape == (1, num_anchors, 4)
    >>> ground_truth = ground_truth.unsqueeze(dim=0)
    >>> ground_truth.shape == (batch_size, num_gt_boxes, 5)
    >>> bbox_offset, bbox_mask, class_labels = multibox_target(
            anchors, ground_truth)
    >>> assert bbox_offset.shape == (batch_size, num_anchors * 4)
    >>> bbox_offset
        tensor([[-0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00,  1.4000e+00,
                 1.0000e+01,  2.5940e+00,  7.1754e+00, -1.2000e+00,  2.6882e-01,
                 1.6824e+00, -1.5655e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00,
                 -0.0000e+00, -5.7143e-01, -1.0000e+00,  4.1723e-06,  6.2582e-01]])
    >>> assert bbox_mask.shape == (batch_size, num_anchors * 4)
    >>> bbox_mask
        tensor([[0., 0., 0., 0., 
                 1., 1., 1., 1., 
                 1., 1., 1., 1., 
                 0., 0., 0., 0., 
                 1., 1., 1., 1.]])
    >>> assert class_labels.shape == (batch_size, num_anchors)
    >>> class_labels
        tensor([[0, 1, 2, 0, 2]])
        可以看到有1个anchor被分配到了1也就是狗对应的ground_truth; 有2个anchor被分配到了
        2也就是猫对应的ground_truth; 有2个anchor被分配到了背景0

    参数:
    anchors: [1, num_anchors, 4]
        [xmin, ymin, xmax, ymax] 相对坐标
        因为每张图片的anchors都是一样的, 所以这里的batch_size=1
    labels: [batch_size, num_gt_boxes, 5]  ground_truth(标签)
        [label, xmin, ymin, xmax, ymax] 相对坐标
        e.g. [0.0000, 0.5977, 0.5039, 0.8711, 0.7031]

    返回: (bbox_offset, bbox_mask, class_labels)
    bbox_offset: [batch_size, num_anchors*4] - 标记的偏移
        如果对应的anchor没有关联到ground_truth, 则offset为0
    bbox_mask: [batch_size, num_anchors*4] - 标记偏移的mask
        如果对应的anchor没有关联到ground_truth, 则mask为0; 否则为1
    class_labels: [batch_size, num_anchors] - 标记的类别
        背景类为0, 其它类从1开始
    """
    # 处理之后:
    # anchors.shape [1, num_anchors, 4]
    #            -> [num_anchors, 4]
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]

    # 遍历labels中的每张图片
    for i in range(batch_size):
        # label.shape [num_gt_boxes, 5]
        label = labels[i, :, :]

        # 将真实边界框(ground_truth)分配给anchors
        # 每个anchor对应一个`ground_truth的索引`, 没有关联ground_truth的anchor对应的索引是-1
        # anchors_bbox_map.shape [num_anchors, ]
        anchors_bbox_map = assign_anchor_to_bbox(label[:, 1:], anchors, device)

        # 如果anchor被关联到了ground_truth, mask=1, 否则mask=0
        # bbox_mask.shape [num_anchors, 4]
        # e.g.
        # tensor([[0., 0., 0., 0.],
        #         [1., 1., 1., 1.],
        #         [1., 1., 1., 1.],
        #         [0., 0., 0., 0.],
        #         [1., 1., 1., 1.]])
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(
            1, 4)
        # 将`类标签-class_labels`和`分配的边界框坐标-assigned_bb`初始化为零
        # class_labels.shape [num_anchors, ]
        class_labels = torch.zeros(num_anchors,
                                   dtype=torch.long,
                                   device=device)
        # assigned_bb.shape [num_anchors, 4]
        assigned_bb = torch.zeros((num_anchors, 4),
                                  dtype=torch.float32,
                                  device=device)
        # 使用ground_truth来标记anchor的类别
        # 如果一个anchor没有被分配, 我们标记其为背景(值为0)
        # indices_true.shape [正类锚框的数量, 1]
        # bb_idx.shape [正类锚框的数量, 1]
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        # class_labels.shape [num_anchors, ]
        # assigned_bb.shape [num_anchors, 4]
        # 注意: 数据集中的类是从0开始的, 我们这里+1, 让类从1开始(0为背景类)
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]

        # 偏移量转换
        # offset.shape [num_anchors, 4]
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))  # [num_anchors*4, ]
        batch_mask.append(bbox_mask.reshape(-1))  # [num_echors*4, ]
        batch_class_labels.append(class_labels)  # [num_anchors, ]

    bbox_offset = torch.stack(batch_offset)  # [batch_size, num_anchors*4]
    bbox_mask = torch.stack(batch_mask)  # [batch_size, num_anchors*4]
    class_labels = torch.stack(batch_class_labels)  # [batch_size, num_anchors]
    return (bbox_offset, bbox_mask, class_labels)


def multibox_detection(cls_probs: Tensor,
                       offset_preds: Tensor,
                       anchors: Tensor,
                       nms_threshold: float = 0.5,
                       pos_threshold: float = 0.009999999) -> Tensor:
    """
    使用预测的类别概率和offset来计算真实边界框, 并应用NMS, 返回最终保留下来的真实边界框.
    返回的数据是按照预测概率从大到小排序的

    遍历每张图片的预测结果:
    1. 找出预测概率最大的类`class_id`: [num_anchors, ] 
    2. 根据anchor和预测的偏移`offset_pred`计算出真实边界框`predicted_bb`: [num_anchors, 4]
    3. 在真实边界框`predicted_bb`上应用NMS, 去掉冗余的真实边界框
    4. 去掉预测概率小于`pos_threshold`的真实边界框`predicted_bb`, 会设置为背景

    注意:
    a. `multibox_target()`生成训练数据的时候对类别做了+1处理, 背景=0, 真实类别从1开始
    b. `multibox_detection()`对类别做了-1处理, 背景=-1, 真实类别从0开始, 这和数据集中
       的label是一致的

    例子:
    a. 假设预测的偏移量都是0, 这意味着预测的边界框即是anchor
    b. 一共4个anchor; 2个类别(0-狗, 1-猫)
    >>> batch_size, num_anchors, num_classes = 1, 4, 3
    >>> anchors = torch.tensor([[0.1, 0.08, 0.52, 0.92], 
                                [0.08, 0.2, 0.56, 0.95],
                                [0.15, 0.3, 0.62, 0.91], 
                                [0.55, 0.2, 0.9, 0.88]])
    >>> offset_preds = torch.tensor([0] * anchors.numel())
    >>> cls_probs = torch.tensor([
            [0] * 4,               # 背景的预测概率
            [0.9, 0.8, 0.7, 0.1],  # 狗的预测概率
            [0.1, 0.2, 0.3, 0.9]]) # 猫的预测概率
    >>> cls_probs = cls_probs.unsqueeze(dim=0)
    >>> assert cls_probs.shape == (batch_size, num_classes, num_anchors)
    >>> offset_preds = offset_preds.unsqueeze(dim=0)
    >>> assert offset_preds.shape == (batch_size, num_anchors * 4)
    >>> anchors = anchors.unsqueeze(dim=0)
    >>> assert anchors.shape == (1, num_anchors, 4)
    >>> output = multibox_detection(cls_probs,
                                    offset_preds,
                                    anchors,
                                    nms_threshold=0.5)
    >>> assert output.shape == (batch_size, num_anchors, 6)
    >>> output
        tensor([[[ 0.0000,  0.9000,  0.1000,  0.0800,  0.5200,  0.9200],
                 [ 1.0000,  0.9000,  0.5500,  0.2000,  0.9000,  0.8800],
                 [-1.0000,  0.8000,  0.0800,  0.2000,  0.5600,  0.9500],
                 [-1.0000,  0.7000,  0.1500,  0.3000,  0.6200,  0.9100]]])
    a. 第1个元素: 真实的类从0开始(0代表狗, 1代表猫), 值-1表示背景或在非极大值抑制中被移除了,
       在上面的例子中第一个anchor被预测为狗; 第二个anchor被预测为猫, 剩下两个被预测为背景
    b. 第2个元素: 预测的边界框的置信度(概率)
    c. 后面4个元素: 预测的真实边界框[xmin, ymin, xmax, ymax]相对坐标, 由于这个例子中
       offset设置为0, 所以预测的真是边界框坐标就是anchor的坐标

    参数:
    cls_probs [batch_size, num_classes, num_anchors] 预测的每个类别对应的概率
    offset_preds [batch_size, num_anchors*4] 预测的偏移
    anchors: [1, num_anchors, 4]
        [xmin, ymin, xmax, ymax] 相对坐标
        因为每张图片的anchors都是一样的, 所以这里的batch_size=1
    nms_threshold: 两个真实边界框IoU超过这个阈值, 会被非极大抑制去掉
    pos_threshold: 非背景预测的阈值, 预测概率小于这个阈值的会被认为是背景

    返回:
    output: [batch_size, num_anchors, 6]
        6个数字中, 第1个是类别, 真实的类从0开始, 背景对应-1
        第2个是对应的概率
        后面4个是预测的真实边界框[xmin, ymin, xmax, ymax]相对坐标
    """
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    # anchors.shape [num_anchors, 4]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    # 遍历每个预测结果
    for i in range(batch_size):
        # cls_prob.shape [num_classes, num_anchors]
        # offset_pred.shape [num_anchors, 4]
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        # cls_prob[0]表示的是背景的概率, 我们不考虑背景
        # conf.shape  [num_anchors, ], 每个anchor box对应的最大类的概率值
        # class_id.shape  [num_anchors, ], 每个anchor box对应的预测概率最大的类
        conf, class_id = torch.max(cls_prob[1:], 0)
        # predicted_bb.shape [num_anchors, 4]
        predicted_bb = offset_inverse(anchors, offset_pred)
        # keep.shape [num_keep, ]
        # 返回的索引是按照预测概率从大到小排序过的
        keep = nms(predicted_bb, conf, nms_threshold)

        # 找到所有的non_keep索引, 并将类设置为背景
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        # combined.shape [num_anchors + num_keep, ]
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        # non_keep.shape [num_anchors - num_keep, ]
        non_keep = uniques[counts == 1]

        # 按照预测概率排序过的anchors box的索引
        # all_id_sorted.shape [num_anchors, ]
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1  # 被NMS排除的预测(设置为背景)

        # 对class_id, conf, predicted_bb按照all_id_sorted排序
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # `pos_threshold`是一个用于非背景预测的阈值, 概率小于这个阈值的会被认为是背景
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1  # 预测出来的类别由于之前是+1的, 现在-1恢复正常标签
        conf[below_min_idx] = 1 - conf[below_min_idx]
        # pred_info.shape [num_anchors, 6]
        pred_info = torch.cat(
            (class_id.unsqueeze(1), conf.unsqueeze(1), predicted_bb), dim=1)
        out.append(pred_info)
    # [batch_size, num_anchors, 6]
    # 6个数字中, 第1个是类别, 真实的类从0开始, 背景对应-1
    # 第2个是对应的概率
    # 后面4个是预测的边框 [xmin, ymin, xmax, ymax]相对坐标
    return torch.stack(out)


def display_anchors(img: np.ndarray, fmap_w: int, fmap_h: int,
                    s: List[float]) -> None:
    """
    演示多尺度目标检测

    通常来说feature map宽度和高度(fmap_w, fmap_h):
    1. 越大, 缩放比会越小, 用来检测小目标, e.g. (fmap_w,fmap_h)=(4,4), s=[0.15]
    2. 越小, 缩放比会越大, 用来检测大目标, e.g. (fmap_w,fmap_h)=(2,2), s=[0.4]

    # 下面测试不同的尺寸(缩放比)
    >>> img = plt.imread(download('catdog'))
    >>> display_anchors(img, fmap_w=4, fmap_h=4, s=[0.15])
    >>> plt.show()
    >>> display_anchors(img, fmap_w=2, fmap_h=2, s=[0.4])
    >>> plt.show()
    >>> display_anchors(img, fmap_w=1, fmap_h=1, s=[0.8])
    >>> plt.show()

    参数:
    img: 原始图像
    fmap_h: feature map的高度
    fmap_w: feature map的宽度
    s: 尺寸(缩放比) e.g. s=[0.15]
    """
    set_figsize()
    h, w = img.shape[:2]  # (H,W,C)或者(H,W)
    # fmap.shape [batch_size, channels, fmap_h, fmap_w]
    fmap = torch.zeros((1, 10, fmap_h, fmap_w))
    anchors = multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = torch.tensor((w, h, w, h))
    show_bboxes(plt.imshow(img).axes, anchors[0] * bbox_scale)


def cls_predictor(num_inputs: int, num_anchors_per_pixel: int,
                  num_classes: int) -> nn.Module:
    """
    `类别`预测层. 输入feature map, 输出feature map每个位置对应的anchors对应
    的每个类别的概率

    输入: [batch_size, channels, height, width]
    输出: [batch_size, num_anchors_per_pixel * (num_classes + 1), height, width]

    构建两个不同比例的feature map Y1和Y2, Y2的高度和宽度是Y1的一半
    假设: Y1和Y2的每个像素分别生成了5个和3个anchors. 进一步假设目标类别的
    数量为10, 对于feature map Y1和Y2, 类别预测输出中的channels分别为
    Y1: num_anchors_per_pixel * (num_classes + 1) = 5 x (10 + 1) = 55
    Y2: num_anchors_per_pixel * (num_classes + 1) = 3 x (10 + 1) = 33

    >>> batch_size = 2
    # [batch_size, channels, height, width]
    >>> y1 = torch.zeros((batch_size, 8, 20, 20))
    >>> y2 = torch.zeros((batch_size, 16, 10, 10))
    >>> cls_pred1 = cls_predictor(8, 5, 10)
    >>> cls_pred2 = cls_predictor(16, 3, 10)
    >>> assert cls_pred1(y1).shape == (batch_size, 55, 20, 20)
    >>> assert cls_pred2(y2).shape == (batch_size, 33, 10, 10)

    参数:
    num_inputs: 输入通道数
    num_anchors_per_pixel: 每个像素对应的anchor数量
    num_classes: 真实类别数(不包括背景)
    """
    return nn.Conv2d(
        num_inputs,
        num_anchors_per_pixel * (num_classes + 1),  # 其中0类是背景
        kernel_size=3,
        padding=1)


def bbox_predictor(num_inputs: int, num_anchors_per_pixel: int) -> nn.Module:
    """
    `边界框`预测层, 输入feature map, 输出feature map每个位置对应的anchors对应
    的边界框偏移量(offset)

    输入: [batch_size, channels, height, width]
    输出: [batch_size, num_anchors_per_pixel * 4, height, width]

    构建两个不同比例的feature map Y1和Y2, Y2的高度和宽度是Y1的一半
    假设: Y1和Y2的每个像素分别生成了5个和3个anchors. 对于feature map
    Y1和Y2, 边界框预测输出中的channels分别为
    Y1: num_anchors_per_pixel * 4 = 5 x 4 = 20
    Y2: num_anchors_per_pixel * 4 = 3 x 4 = 12

    >>> batch_size = 2
    # [batch_size, channels, height, width]
    >>> y1 = torch.zeros((batch_size, 8, 20, 20))
    >>> y2 = torch.zeros((batch_size, 16, 10, 10))
    >>> bbox_pred1 = bbox_predictor(8, 5)
    >>> bbox_pred2 = bbox_predictor(16, 3)
    >>> assert bbox_pred1(y1).shape == (batch_size, 20, 20, 20)
    >>> assert bbox_pred2(y2).shape == (batch_size, 12, 10, 10)

    参数:
    num_inputs: 输入通道数
    num_anchors_per_pixel: 每个像素对应的anchor数量
    """
    return nn.Conv2d(num_inputs,
                     num_anchors_per_pixel * 4,
                     kernel_size=3,
                     padding=1)


def flatten_pred(pred: Tensor) -> Tensor:
    """
    参数
    pred: [batch_size, channels, height, width] - 一层feature map的预测

    返回:
    output:[batch_size, height*width*channels]
    """
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)


def concat_preds(preds: List[Tensor]) -> Tensor:
    """
    将多层feature map的预测拉平

    参数: 
    preds是一个list, 里面的元素是: [batch_size, channelsN, heightN, widthN] - 多层feature map的预测

    返回: 
    output: [batch_size, height1*width1*channels1 + height2*width2*channels2 + ...)
    """
    return torch.cat([flatten_pred(p) for p in preds], dim=1)


def down_sample_blk(in_channels: int, out_channels: int) -> nn.Module:
    """
    变换通道数, 高宽减半

    输入: [batch_size, in_channels, height, width]
    输出: [batch_size, out_channels, height/2, width/2]

    >>> batch_size, in_channels, out_channels = 2, 10, 20
    # [batch_size, channels, height, width]
    >>> x = torch.zeros(batch_size, in_channels, 64, 64)
    >>> blk = down_sample_blk(in_channels, out_channels)
    >>> assert blk(x).shape == (batch_size, out_channels, 32, 32)

    参数:
    in_channels: 输入通道数
    out_channels: 输出通道数
    """
    blk = []
    for _ in range(2):
        blk.append(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))  # 高/宽减半
    return nn.Sequential(*blk)


def base_net() -> nn.Module:
    """
    通道从3变为64, 尺寸缩小为原来的(2x2x2=8)分之一

    输入: [batch_size, 3, height, width]
    输出: [batch_size, 64, height/8, width/8]

    e.g. 
    输入: [batch_size, 3, 256, 256]
    输出: [batch_size, 3, 32, 32]

    >>> batch_size = 2
    # [batch_size, channels, height, width]
    >>> x = torch.zeros(batch_size, 3, 256, 256)
    >>> blk = base_net()
    >>> assert blk(x).shape == (batch_size, 64, 32, 32)

    >>> def forward(x, block):
    >>> return block(x)
    >>> Y = forward(torch.zeros((2, 3, 256, 256)), base_net())
    >>> assert Y.shape == (2, 64, 32, 32)
    """
    blk = []
    # [in_channels, out_channels]
    # blk1 - [3, 16]
    # blk2 - [16, 32]
    # blk3 - [32, 64]
    num_filters = [3, 16, 32, 64]  # 通道数
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i + 1]))
    return nn.Sequential(*blk)


def get_blk(i: int) -> nn.Module:
    """
    一个5个block

    输入的尺寸为: [3, 256, 256], 则每个block输出的feature map:
    0 - [3, 256, 256] -> [64, 32, 32]
    1 - [64, 32, 32]  -> [128, 16, 16]
    2 - [128, 16, 16] -> [128, 8, 8]
    3 - [128, 8, 8]   -> [128, 4, 4]
    4 - [128, 4, 4]   -> [128, 1, 1]

    参数:
    i: 0,1,2,3,4
    """
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1, 1))
    else:
        # i == 2, 3
        blk = down_sample_blk(128, 128)
    return blk


def blk_forward(
        X: Tensor, blk: nn.Module, size: List[float], ratio: List[float],
        cls_predictor: nn.Module,
        bbox_predictor: nn.Module) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    逻辑:
    1. 计算`X`的feature map `Y`
    2. 在当前尺度下, 根据feature map `Y`生成anchors
    3. 预测这些anchors的类别和偏移量(基于feature map `Y`)

    参数:
    X: [batch_size, channels, height, width]
    blk: Block
    size: 尺寸(缩放比)列表, e.g. [0.2, 0.272]
    ratio: 宽高比列表, e.g. [1, 2, 0.5]
    cls_predictor: `类别`预测层
    bbox_predictor: `边界框`预测层

    输出: Y, anchors, cls_preds, bbox_preds
    Y: [batch_size, f_channels, f_height, f_width]    - 生成的feature map
    anchors: [1, f_height*f_width*boxes_per_pixel, 4] - 生成的所有anchors
    cls_preds: [batch_size, out_channels, f_height, f_width] - 类别预测
        out_channels = num_anchors_per_pixel * (num_classes + 1)
    bbox_preds: [batch_size, out_channels, f_height, f_width] - 偏移量预测
        out_channels = num_anchors_per_pixel * 4
    """
    # 输出feature map
    # Y.shape [batch_size, f_channels, f_height, f_width]
    Y = blk(X)

    # 在当前尺度下根据feature map Y生成anchors
    # anchors.shape [1, f_height*f_width*boxes_per_pixel, 4]
    anchors = multibox_prior(Y, sizes=size, ratios=ratio)

    # 基于feature map Y, 预测这些anchors对应每个类别的概率
    # 预测的时候并没有去看anchors的信息, 因为cls_predictor知道num_anchors_per_pixel和
    # num_classes, 直接预测就可以
    #
    # out_channels = num_anchors_per_pixel * (num_classes + 1)
    # cls_preds.shape [batch_size, out_channels, f_height, f_width]
    cls_preds = cls_predictor(Y)

    # 基于特征图Y, 预测这些anchors的偏移量
    # 预测的时候并没有去看anchors的信息, 因为bbox_predictor知道num_anchors_per_pixel,
    # 直接预测就可以
    #
    # out_channels = num_anchors_per_pixel * 4
    # bbox_preds.shape [batch_size, out_channels, f_height, f_width]
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)


class TinySSD(nn.Module):
    """
    # 下面是5层对应的feature map:
    0 - [3, 256, 256] -> [64, 32, 32]
    1 - [64, 32, 32]  -> [128, 16, 16]
    2 - [128, 16, 16] -> [128, 8, 8]
    3 - [128, 8, 8]   -> [128, 4, 4]
    4 - [128, 4, 4]   -> [128, 1, 1]
    根据TinySSD默认配置, 每个像素会生成4个anchors, 所以5层feature map
    一共会生成的anchors数量为:
    32x32x4 + 16x16x4 + 8x8x4 + 4x4x4 + 1x1x4=5444

    >>> net = TinySSD(num_classes=1)
    >>> X = torch.zeros((32, 3, 256, 256))
    >>> anchors, cls_preds, bbox_preds = net(X)
    >>> assert anchors.shape == (1, 5444, 4)
    >>> assert cls_preds.shape == (32, 5444, 2)
    >>> assert bbox_preds.shape == (32, 5444*4)
    """

    def __init__(self, num_classes: int, **kwargs: Any) -> None:
        super(TinySSD, self).__init__(**kwargs)
        # 尺寸(缩放比)
        # 一共5层feature map, 每一层的缩放比是不同的:
        # a. 浅层的feature map缩放比比较小, 用来检测小目标
        # b. 深层的feature map缩放比比较大, 用来检测大目标
        self.sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
                      [0.88, 0.961]]
        # 宽高比(5层)
        self.ratios = [[1, 2, 0.5]] * 5
        # 每个像素生成的anchor数量: # 2+3-1=4
        self.num_anchors_per_pixel = len(self.sizes[0]) + len(
            self.ratios[0]) - 1
        self.num_classes = num_classes
        # 每层feature map对应的channels
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            # self.blk_i=get_blk(i)
            # self.cls_i=cls_predictor(num_inputs, ...)
            # self.bbox_i=bbox_predictor(num_inputs, ...)
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(
                self, f'cls_{i}',
                cls_predictor(idx_to_in_channels[i],
                              self.num_anchors_per_pixel, num_classes))
            setattr(
                self, f'bbox_{i}',
                bbox_predictor(idx_to_in_channels[i],
                               self.num_anchors_per_pixel))

    def forward(self, X: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        参数:
        X: [batch_size, channel, height, width]

        输出:
        anchors: [1, 5层的所有的锚框数量, 4]
        cls_preds: [batch_size, 5层的所有的锚框数量, num_classes + 1]
        bbox_preds: [batch_size, 5层的所有的锚框数量*4]
        """
        # 保存5层feature map的输出
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # X.shape [batch_size, channel, height, width]                        - 生成的feature map
            # anchors[i].shape [1, f_height*f_width*boxes_per_pixel, 4]           - 生成的所有anchors
            # cls_preds[i].shape [batch_size, out_channels, f_height, f_width]    - 类别预测
            #     out_channels = num_anchors_per_pixel * (num_classes + 1)
            # bbox_preds[i].shape - [batch_size, out_channels, f_height, f_width] - 偏移量预测
            #     out_channels = num_anchors_per_pixel * 4
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), self.sizes[i], self.ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        # anchors.shape [1, 5层的所有的anchors数量, 4]
        anchors = torch.cat(anchors, dim=1)
        # cls_preds.shape [batch_size, 5层的所有的锚框数量*(num_classes + 1)]
        cls_preds = concat_preds(cls_preds)
        # cls_preds.shape [batch_size, 5层的所有的锚框数量, num_classes + 1]
        cls_preds = cls_preds.reshape(cls_preds.shape[0], -1,
                                      self.num_classes + 1)
        # bbox_preds.shape [batch_size, 5层的所有的锚框数量*4]
        bbox_preds = concat_preds(bbox_preds)

        # 最终输出的形状:
        # anchors.shape [1, 5层的所有的锚框数量, 4]
        # cls_preds.shape [batch_size, 5层的所有的锚框数量, num_classes + 1]
        # bbox_preds.shape [batch_size, 5层的所有的锚框数量*4]
        return anchors, cls_preds, bbox_preds


def calc_loss(cls_preds: Tensor, cls_labels: Tensor, bbox_preds: Tensor,
              bbox_labels: Tensor, bbox_masks: Tensor, cls_loss: nn.Module,
              bbox_loss: nn.Module) -> Tensor:
    """
    参数:
    cls_preds: [batch_size, num_anchors, num_classes + 1] - 预测的类别
    cls_labels: class_labels: [batch_size, num_anchors] - 标记的类别标签
    bbox_preds: [batch_size, num_anchors*4] - 预测的偏移
    bbox_labels: [batch_size, num_anchors*4] - 标记的偏移
    bbox_masks: [batch_size, num_anchors*4] - 标记偏移的mask
    cls_loss: cls loss function
    bbox_loss: bbox loss function

    返回:
    loss: [batch_size, ] 在批量这个维度上求均值loss
    """
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    # cls_preds.shape [batch_size*num_anchors, num_classes + 1]
    # cls_labels.shape [batch_size*num_anchors, ]
    # cls.shape [batch_size, ]
    cls = cls_loss(cls_preds.reshape(-1, num_classes),
                   cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    # bbox.shape [batch_size, ]
    bbox = bbox_loss(bbox_preds * bbox_masks,
                     bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox


def cls_eval(cls_preds: Tensor, cls_labels: Tensor) -> float:
    """
    分类准确的anchors数量

    参数:
    cls_preds: [batch_size, num_anchors, num_classes + 1] - 预测的类别
    cls_labels: [batch_size, num_anchors] - 标记的类别
    """
    # 由于类别预测结果放在最后一维, argmax需要指定最后一维。
    return float(
        (cls_preds.argmax(dim=-1).type(cls_labels.dtype) == cls_labels).sum())


def bbox_eval(bbox_preds: Tensor, bbox_labels: Tensor,
              bbox_masks: Tensor) -> float:
    """
    绝对误差的和

    参数:
    bbox_preds: [batch_size, num_anchors*4]
    bbox_labels: [batch_size, num_anchors*4] - 标记的偏移
    bbox_masks: [batch_size, num_anchors*4] - 标记偏移的mask
    """
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())


def try_gpu(i: int = 0) -> torch.device:
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def train_gpu(net: nn.Module,
              train_iter: DataLoader,
              num_epochs: int = 10,
              cls_loss: nn.Module = None,
              bbox_loss: nn.Module = None,
              optimizer: Optimizer = None,
              device: torch.device = None):
    """
    用GPU训练模型
    """
    if device is None:
        device = try_gpu()

    print('training on', device)
    net.to(device)

    history = [[], []]  # 记录: class error, bbox mae, 方便后续绘图
    num_batches = len(train_iter)
    for epoch in range(num_epochs):
        # 训练
        # 统计: 训练精确度(分类准确的anchors数量和), 训练精确度中的示例数, 绝对误差的和, 绝对误差的和中的示例数
        metric_train = [0.0] * 4
        net.train()
        train_iter_tqdm = tqdm(train_iter, file=sys.stdout)
        for i, (X, y) in enumerate(train_iter_tqdm):
            t_start = time.time()
            optimizer.zero_grad()
            # X.shape [batch_size, channels=3, 256, 256]
            # y.shape [batch_size, num_gt_boxes=1, 5]
            X, y = X.to(device), y.to(device)
            # 生成多尺度的anchors, 为每个anchor预测类别和偏移量
            # anchors.shape [1, num_anchors, 4]
            # cls_preds.shape [batch_size, num_anchors, num_classes + 1]
            # bbox_preds.shape [batch_size, num_anchors*4]
            # 注意: num_anchors=5层的所有的锚框数量
            anchors, cls_preds, bbox_preds = net(X)
            # 为每个锚框标注类别和偏移量
            # bbox_labels: [batch_size, num_anchors*4] - 标记的偏移
            # bbox_masks: [batch_size, num_anchors*4] - 标记偏移的mask
            # cls_labels: [batch_size, num_anchors] - 标记的类别
            bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, y)
            # 根据类别和偏移量的预测和标注值计算损失函数
            # l.shape [batch_size, ] 在批量这个维度上求均值loss
            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                          bbox_masks, cls_loss, bbox_loss)
            l.mean().backward()
            optimizer.step()

            with torch.no_grad():
                metric_train[0] += cls_eval(cls_preds, cls_labels)
                metric_train[1] += float(cls_labels.numel())
                metric_train[2] += bbox_eval(bbox_preds, bbox_labels,
                                             bbox_masks)
                metric_train[3] += float(bbox_labels.numel())
                cls_err = 1 - metric_train[0] / metric_train[1]
                bbox_mae = metric_train[2] / metric_train[3]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                history[0].append((epoch + (i + 1) / num_batches, cls_err))
                history[1].append((epoch + (i + 1) / num_batches, bbox_mae))
                train_iter_tqdm.desc = f'epoch {epoch}, class err {cls_err:.3f}, bbox mae {bbox_mae:.3f}'
    print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
    print(
        f'{len(train_iter.dataset) / (time.time() - t_start):.1f} examples/sec on '
        f'{str(device)}')
    return history


def train(batch_size: int, num_epochs: int, lr: float, device: torch.device):
    train_iter, _ = load_data_bananas(batch_size=batch_size)
    net = TinySSD(num_classes=1)
    cls_loss = nn.CrossEntropyLoss(reduction='none')
    bbox_loss = nn.L1Loss(reduction='none')
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=5e-4)
    history = train_gpu(net, train_iter, num_epochs, cls_loss, bbox_loss,
                        optimizer, device)

    # plot
    plt.figure(figsize=(6, 4))
    plt.plot(*zip(*history[0]), '-', label='class err')
    plt.plot(*zip(*history[1]), 'm--', label=' bbox mae')
    plt.xlabel('epoch')
    # 从epoch=1开始显示, 0-1这个范围的数据丢弃不展示,
    # 因为只有训练完成1个epochs之后, 才会有第一条test acc记录
    plt.xlim((1, num_epochs))
    plt.grid()
    plt.legend()
    plt.show()
    return net


def predict(net: nn.Module, X: Tensor, device: torch.device) -> Tensor:
    """
    预测一张图片

    参数:
    X: [1, channels, height, width]

    返回：
    output: [keep_anchors, 6] - 预测出来的非背景类
    """
    net.eval()
    # 生成多尺度的anchors, 为每个anchor预测类别和偏移量
    # anchors.shape [1, num_anchors, 4]
    # cls_preds.shape [1, num_anchors, num_classes + 1]
    # bbox_preds.shape [1, num_anchors*4]
    # 注意: num_anchors=5层的所有的锚框数量
    anchors, cls_preds, bbox_preds = net(X.to(device))
    # cls_probs.shape [1, num_classes + 1, num_anchors]
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    # output: [1, num_anchors, 6]
    # 6个数字中, 第1个是类别, 真实的类从0开始, 背景对应-1
    # 第2个是对应的概率
    # 后面4个是预测的真实边界框[xmin, ymin, xmax, ymax]相对坐标
    output = multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]


def display(img: Tensor, output: Tensor, threshold: float) -> None:
    """
    参数:
    img: [3, 256, 256]
    output: [keep_anchors, 6] - 预测出来的非背景类
    threshold: 预测的概率小于这个值不会显示
    """
    set_figsize((5, 5))
    fig = plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')
    plt.show()


def test_predict(net: nn.Module, device: torch.device) -> None:
    X = torchvision.io.read_image(
        download('banana')).unsqueeze(0).float() / 255.
    assert X.shape == (1, 3, 256, 256)
    # img.shape [256, 256, 3]
    img = X.squeeze(0).permute(1, 2, 0)
    output = predict(net, X, device)
    display(img, output.cpu(), threshold=0.9)


if __name__ == '__main__':
    device = try_gpu()
    kwargs = {
        'batch_size': 32,
        'num_epochs': 20,
        'lr': 0.2,
        'device': device,
    }
    net = train(**kwargs)
    # class err 3.40e-03, bbox mae 3.24e-03
    # 18916.9 examples/sec on cuda:0
    test_predict(net, device)