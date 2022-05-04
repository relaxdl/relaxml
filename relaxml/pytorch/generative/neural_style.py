from typing import Any, List, Tuple
import os
import requests
import hashlib
from PIL import Image
import torch
import torch.nn as nn
from torch import Tensor
import torchvision
from torch.optim.optimizer import Optimizer
from matplotlib import pyplot as plt
from tqdm import tqdm
"""
风格迁移

说明:
https://tech.foxrelax.com/generative/neural_style/
"""

_DATA_HUB = dict()
_DATA_HUB['content'] = (
    'b0193ffd8ecea1631ac1092130017b884faafa30',
    'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/img/rainier.jpg')
_DATA_HUB['style'] = (
    'bbe11f01403b9726bbc625589550e574fc45980a',
    'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/img/autumn-oak.jpg')


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
    # e.g. ../data/rainier.png
    return fname


rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std = torch.tensor([0.229, 0.224, 0.225])
style_layers, content_layers = [0, 5, 10, 19, 28], [25]


def preprocess(
    img: Image, image_shape: Tuple[int, int] = (300, 450)) -> Tensor:
    """
    PIL Image -> Tensor

    内容图片和风格图片可以有自己的尺寸, 它们在放入网络抽取特征之前, 都会先被
    resize成需要生成的合成图片的尺寸, 这行提取的内容特征和风格特征就可以直接
    计算loss了

    参数: 
    img: PIL Image
    image_shape: 需要生成的合成图片的尺寸

    返回:
    output: [batch_size=1, channels, height, width]
    """
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)
    ])
    return transforms(img).unsqueeze(0)


def postprocess(img: Tensor) -> Image:
    """
    Tensor -> PIL Image

    >>> content_img = Image.open(download('content'))
    >>> img = preprocess(content_img)
    >>> assert img.shape == (1, 3, 300, 450)

    参数:
    img: [batch_size=1, channels=3, height=300, width=450]

    输出:
    output: PIL Image
    """
    # img.shape [channels, height, width]
    img = img[0].to(rgb_std.device)
    # 从RGB Normalize还原回原来的图片
    # img.shape [height, width, channels]
    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
    return torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))


def extract_features(
        net: nn.Module, X: Tensor, content_layers: List[int],
        style_layers: List[int]) -> Tuple[List[Tensor], List[Tensor]]:
    """
    提取`风格输出`和`内容输出`

    >>> net = vgg19(content_layers, style_layers)
    >>> x = torch.randn((1, 3, 300, 450))
    >>> contents, styles = extract_features(net, x, content_layers, style_layers)
    >>> for c, l in zip(contents, content_layers):
    >>>     print(f'content_layer#{l}, shape {c.shape}')
        content_layer#25, shape torch.Size([1, 512, 37, 56])
    >>> for s, l in zip(styles, style_layers):
    >>>     print(f'style_layer#{l}, shape {s.shape}')
        style_layer#0, shape torch.Size([1, 64, 300, 450])
        style_layer#5, shape torch.Size([1, 128, 150, 225])
        style_layer#10, shape torch.Size([1, 256, 75, 112])
        style_layer#19, shape torch.Size([1, 512, 37, 56])
        style_layer#28, shape torch.Size([1, 512, 18, 28])

    参数:
    net: VGG19
    X: [batch_size=1, channels=3, height=300, width=450] 原始图片
    content_layers: e.g. [25]
    style_layers: e.g. [0, 5, 10, 19, 28]

    返回: contents, styles
    contents: List of [batch_size=1, C, H, W] - 内容feature map 
        每一层的C/H/W不同
    styles: List of [batch_size=1, C, H, W] - 风格feature map 
        每一层的C/H/W不同
    """
    contents = []
    styles = []
    for i in range(len(net)):
        # 逐层计算
        X = net[i](X)
        # 保留内容层的输出
        if i in style_layers:
            styles.append(X)
        # 保留风格层的输出
        if i in content_layers:
            contents.append(X)
    return contents, styles


def get_contents(net: nn.Module, content_img: Image, image_shape: Tuple[int,
                                                                        int],
                 content_layers: List[int], style_layers: List[int],
                 device: torch.device) -> Tuple[Tensor, Tensor]:
    """
    获取`输入的原始内容图片`和`内容的feature map`

    参数:
    net: VGG19
    content_img: PIL Image
    image_shape: 需要生成的合成图片的尺寸
    content_layers: e.g. [25]
    style_layers: e.g. [0, 5, 10, 19, 28]

    返回: content_X, contents_Y
    content_X: [batch_size=1, channels=3, height=300, width=450]  输入的原始内容图片
    contents_Y: List of [batch_size=1, C, H, W] - 每一层的C/H/W不同 内容的feature map
    """
    # 预处理内容图片
    content_X = preprocess(content_img, image_shape).to(device)
    # 从内容图片抽取特征
    contents_Y, _ = extract_features(net, content_X, content_layers,
                                     style_layers)
    return content_X, contents_Y


def get_styles(net: nn.Module, style_img: Image, image_shape: Tuple[int, int],
               content_layers: List[int], style_layers: List[int],
               device: torch.device) -> Tuple[Tensor, Tensor]:
    """
    获取`输入的原始风格图片`和`风格的feature map`

    参数:
    net: VGG19
    style_img: PIL Image
    image_shape: 需要生成的合成图片的尺寸
    content_layers: e.g. [25]
    style_layers: e.g. [0, 5, 10, 19, 28]

    返回: content_X, contents_Y
    content_X: [batch_size=1, channels=3, height=300, width=450]  输入的原始风格图片
    contents_Y: List of [batch_size=1, C, H, W] - 每一层的C/H/W不同 风格的feature map
    """
    # 预处理风格图片
    style_X = preprocess(style_img, image_shape).to(device)
    # 从风格图片抽取风格
    _, styles_Y = extract_features(net, style_X, content_layers, style_layers)
    return style_X, styles_Y


def gram(X: Tensor) -> Tensor:
    """
    gram matrix

    >>> x = torch.randn((1, 32, 200, 300))
    >>> assert gram(x).shape == (32, 32)

    参数:
    X: [batch_size=1, C, H, W]

    返回:
    output: [C, C]
    """
    # n = H*W
    num_channels, n = X.shape[1], X.numel() // X.shape[1]
    # X.shape [C, H*W]
    X = X.reshape((num_channels, n))
    # output.shape [C, H*W]@[H*W, C] = [C, C]
    return torch.matmul(X, X.T) / (num_channels * n)


def content_loss(Y_hat: Tensor, Y: Tensor) -> Tensor:
    """
    参数:
    Y_hat: List of [batch_size=1, C, H, W] - 每一层的C/H/W不同 内容的feature map
    Y: List of [batch_size=1, C, H, W] - 每一层的C/H/W不同 内容的feature map

    返回:
    l: 标量
    """
    return torch.square(Y_hat - Y.detach()).mean()  # 注意: Y要detach


def style_loss(Y_hat: Tensor, gram_Y: Tensor) -> Tensor:
    """
    参数:
    Y_hat: List of [batch_size=1, C, H, W] - 每一层的C/H/W不同 风格的feature map
    gram_Y: List of [C, C] 每一层的C不同 gram matrix

    返回:
    l: 标量
    """
    return torch.square(gram(Y_hat) -
                        gram_Y.detach()).mean()  # 注意: gram_Y要detach


def tv_loss(Y_hat: Tensor) -> Tensor:
    """
    total variation denoising, 尽可能使邻近的像素值相似

    参数:
    Y_hat: [batch_size=1, channels=3, height=300, width=450] 需要训练的初始化图像'模型参数'

    返回:
    l: 标量
    """
    return 0.5 * (torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                  torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())


content_weight, style_weight, tv_weight = 1, 1e3, 10


def compute_loss(
    X: Tensor, contents_Y_hat: List[Tensor], styles_Y_hat: List[Tensor],
    contents_Y: List[Tensor], styles_Y_gram: List[Tensor]
) -> Tuple[List[Tensor], List[Tensor], Tensor, Tensor]:
    """
    计算下面三个损失

    1. 内容损失: 使合成图像与内容图像在内容特征上接近
    2. 风格损失: 使合成图像与风格图像在风格特征上接近
    3. TV损失: 则有助于减少合成图像中的噪点

    参数:
    X: [batch_size=1, channels=3, height=300, width=450] 需要训练的初始化图像'模型参数'
    contents_Y_hat:  List of [batch_size=1, C, H, W] - 每一层的C/H/W不同 内容的feature map
    styles_Y_hat: List of [batch_size=1, C, H, W] - 每一层的C/H/W不同 风格的feature map
    contents_Y: List of [batch_size=1, C, H, W] - 每一层的C/H/W不同 内容的feature map
    styles_Y_gram: List of [C, C] 每一层的C不同 gram matrix

    返回: contents_l, styles_l, tv_l, l
    contents_l: List of Tensor 内容损失
    styles_l: List of Tensor 风格损失
    tv_l: Tensor TV损失
    l: 损失加权和
    """
    # 分别计算内容损失、风格损失和全变分损失
    contents_l = [
        content_loss(Y_hat, Y) * content_weight
        for Y_hat, Y in zip(contents_Y_hat, contents_Y)
    ]
    styles_l = [
        style_loss(Y_hat, Y) * style_weight
        for Y_hat, Y in zip(styles_Y_hat, styles_Y_gram)
    ]
    tv_l = tv_loss(X) * tv_weight
    # 对所有损失求和
    l = sum(10 * styles_l + contents_l + [tv_l])
    return contents_l, styles_l, tv_l, l


def vgg19(content_layers: List[int], style_layers: List[int]) -> nn.Module:
    """
    返回预训练的VGG19用来提取特征

    参数:
    content_layers: e.g. [25]
    style_layers: e.g. [0, 5, 10, 19, 28]
    """
    pretrained_net = torchvision.models.vgg19(pretrained=True)
    net = nn.Sequential(*[
        pretrained_net.features[i]
        for i in range(max(content_layers + style_layers) + 1)
    ])
    return net


class SynthesizedImage(nn.Module):
    """
    将合成的图像视为'模型参数', 模型的前向传播只需返回'模型参数'即可
    """

    def __init__(self, img_shape: Tuple[int, int, int, int],
                 **kwargs: Any) -> None:
        """
        参数:
        img_shape: [batch_size=1, channels=3, height=300, width=450]
        """
        super(SynthesizedImage, self).__init__(**kwargs)
        # 可以训练的权重就是最终生成的图像
        self.weight = nn.Parameter(torch.rand(*img_shape))

    def forward(self) -> Tensor:
        """
        返回:
        weight: [batch_size=1, channels=3, height=300, width=450]
        """
        return self.weight


def get_inits(X: Tensor, device: torch.device, lr: float,
              styles_Y) -> Tuple[Tensor, List[Tensor], Optimizer]:
    """
    获取:
    1. 训练的初始化图像'模型参数'
    2. 风格的gram matrix
    3. 优化器(会更新`图像`)

    参数:
    X: [batch_size=1, channels=3, height=300, width=450]  输入的原始内容图片
    device: 设备
    lr: 学习率
    style_Y:  List of [batch_size=1, C, H, W] - 每一层的C/H/W不同 风格的feature map

    返回: X, styles_Y_gram, trainer
    X: [batch_size=1, channels=3, height=300, width=450] 需要训练的初始化图像'模型参数'
    styles_Y_gram: List of [C, C] 每一层的C不同 gram matrix 
    trainer: 优化器
    """
    # gen_img.shape [batch_size=1, channels=3, height=300, width=450]
    gen_img = SynthesizedImage(X.shape).to(device)
    gen_img.weight.data.copy_(X.data)  # 用原始图像的内容初始化gen_img, 可以提高训练速度
    trainer = torch.optim.Adam(gen_img.parameters(), lr=lr)  # 更新`图像`
    # styles_Y_gram List of [C, C] 每一层的C不同
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, trainer


def try_gpu(i: int = 0) -> torch.device:
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def train(net: nn.Module, X: Tensor, contents_Y: List[Tensor],
          styles_Y: List[Tensor], device: torch.device, lr: float,
          num_epochs: int, lr_decay_epoch: int) -> Tensor:
    """
    参数:
    net: VGG19
    X: [batch_size=1, channels=3, height=300, width=450]  输入的原始内容图片
    contents_Y: List of [batch_size=1, C, H, W] - 每一层的C/H/W不同 内容的feature map
    styles_Y: List of [batch_size=1, C, H, W] - 每一层的C/H/W不同 风格的feature map
    device: 设备
    lr: 学习率
    num_epochs: 训练的epochs
    lr_decay_epoch: 将多多少步lr做一次decay

    返回:
    X [batch_size=1, channels=3, height=300, width=450]
    """
    # X: [batch_size=1, channels=3, height=300, width=450] 需要训练的初始化图像'模型参数'
    # styles_Y_gram: List of [C, C] 每一层的C不同 gram matrix
    X, styles_Y_gram, trainer = get_inits(X, device, lr, styles_Y)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_decay_epoch, 0.8)
    history = [[], [], []]  # 记录: content loss, style loss, TV loss, 方便后续绘图
    tqdm_iter = tqdm(range(num_epochs))
    for epoch in tqdm_iter:
        trainer.zero_grad()
        # contents_Y_hat.shape List of [batch_size=1, C, H, W] - 每一层的C/H/W不同 内容的feature map
        # styles_Y_hat.shape List of [batch_size=1, C, H, W] - 每一层的C/H/W不同 风格的feature map
        contents_Y_hat, styles_Y_hat = extract_features(
            net, X, content_layers, style_layers)
        # contents_l: List of 标量 内容损失
        # styles_l: List of 标量 风格损失
        # tv_l: 标量 TV损失
        # l: 标量 损失加权和
        contents_l, styles_l, tv_l, l = compute_loss(X, contents_Y_hat,
                                                     styles_Y_hat, contents_Y,
                                                     styles_Y_gram)
        l.backward()
        trainer.step()
        scheduler.step()
        if (epoch + 1) % 5 == 0:
            history[0].append((epoch + 1, float(sum(contents_l))))
            history[1].append((epoch + 1, float(sum(styles_l))))
            history[2].append((epoch + 1, float(tv_l)))
            tqdm_iter.desc = f'epoch: {epoch} | content loss: {float(sum(contents_l)):.3f}, ' \
                             f'style loss: {float(sum(styles_l)):.3f}, TV loss: {float(tv_l):.3f}'

    plt.figure(figsize=(6, 4))
    # content loss, style loss, TV loss, 方便后续绘图
    plt.plot(*zip(*history[0]), '-', label='content loss')
    plt.plot(*zip(*history[1]), 'm--', label='style acc')
    plt.plot(*zip(*history[2]), 'g-.', label='TV acc')
    plt.xlabel('epoch')
    # 从epoch=1开始显示, 0-1这个范围的数据丢弃不展示,
    # 因为只有训练完成1个epochs之后, 才会有第一条test acc记录
    plt.xlim((1, num_epochs))
    plt.grid()
    plt.legend()
    plt.show()
    return X


def run(num_epochs: int = 500) -> None:
    device, image_shape = try_gpu(), (300, 450)
    content_img = Image.open(download('content'))
    style_img = Image.open(download('style'))
    net = vgg19(content_layers, style_layers)
    net = net.to(device)
    # content_X.shape [batch_size=1, channels=3, height=300, width=450]  输入的原始内容图片
    # contents_Y List of [batch_size=1, C, H, W] - 每一层的C/H/W不同 内容的feature map
    content_X, contents_Y = get_contents(net, content_img, image_shape,
                                         content_layers, style_layers, device)
    # styles_Y List of [batch_size=1, C, H, W] - 每一层的C/H/W不同 风格的feature map
    _, styles_Y = get_styles(net, style_img, image_shape, content_layers,
                             style_layers, device)
    # output.shape [batch_size=1, channels=3, height=300, width=450]
    output = train(net,
                   content_X,
                   contents_Y,
                   styles_Y,
                   device,
                   lr=0.3,
                   num_epochs=num_epochs,
                   lr_decay_epoch=50)
    img = postprocess(output)
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    run(num_epochs=500)
    # epoch: 499 | content loss: 0.778, style loss: 0.039, TV loss: 1.169
