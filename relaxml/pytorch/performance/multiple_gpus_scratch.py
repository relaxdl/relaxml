from typing import Tuple, Union, List
import time
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
"""
多GPU训练模型从零实现

实现说明:
https://tech.foxrelax.com/performance/multiple_gpus_scratch/
"""


def load_data_fashion_mnist(
        batch_size: int,
        resize: Union[int, Tuple[int, int]] = None,
        root: str = '../data') -> Tuple[DataLoader, DataLoader]:
    """
    下载Fashion-MNIST数据集, 然后将其加载到内存中

    1. 60000张训练图像和对应Label
    2. 10000张测试图像和对应Label
    3. 10个类别
    4. 每张图像28x28x1的分辨率

    >>> train_iter, test_iter = load_data_fashion_mnist(batch_size=256)
    >>> for x, y in train_iter:
    >>>     assert x.shape == (256, 1, 28, 28)
    >>>     assert y.shape == (256, )
    >>>     break
    """
    # 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式
    # 并除以255使得所有像素的数值均在0到1之间
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=root,
                                                    train=True,
                                                    transform=trans,
                                                    download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root=root,
                                                   train=False,
                                                   transform=trans,
                                                   download=True)
    return (DataLoader(mnist_train, batch_size, shuffle=True),
            DataLoader(mnist_test, batch_size, shuffle=False))


def try_gpu(i: int = 0) -> torch.device:
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus() -> List[torch.device]:
    devices = [
        torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())
    ]
    return devices if devices else [torch.device('cpu')]


def get_params(params: List[Tensor], device: torch.device) -> List[Tensor]:
    """
    将params拷贝到device上, 并附加梯度
    """
    new_params = [p.to(device) for p in params]
    for p in new_params:
        p.requires_grad_()  # 附加梯度
    return new_params


def allreduce(data: List[Tensor]) -> List[Tensor]:
    """
    跨多个设备对data求和(累加), len(data)就是设备数
    1. 将数据复制到一个GPU上进行累加
    2. 将结果广播(复制)到所有GPU上

    >>> data = [torch.ones((1, 2), device=try_gpu(i)) * (i + 1) for i in range(2)]
    >>> print('allreduce之前：\n', data[0], '\n', data[1])
        allreduce之前：
        tensor([[1., 1.]], device='cuda:0') 
        tensor([[2., 2.]], device='cuda:1')
    >>> allreduce(data)
    >>> print('allreduce之后：\n', data[0], '\n', data[1])
        allreduce之后：
        tensor([[3., 3.]], device='cuda:0') 
        tensor([[3., 3.]], device='cuda:1')
    """
    for i in range(1, len(data)):
        # 将数据复制到一个GPU上进行累加
        data[0][:] += data[i].to(data[0].device)
    for i in range(1, len(data)):
        # 将结果广播(复制)到所有GPU上
        data[i][:] = data[0].to(data[i].device)


def split_batch(
        X: Tensor, y: Tensor,
        devices: List[torch.device]) -> Tuple[Tuple[Tensor], Tuple[Tensor]]:
    """
    将X和y拆分到多个设备上

    >>> X = torch.randn((256, 1, 28, 28))
    >>> y = torch.randn((256, 1))
    >>> devices = [torch.device('cuda:0'), torch.device('cuda:1')]
    >>> X_shards, y_shards = split_batch(X, y, devices)
    >>> assert X_shards[0].shape == (128, 1, 28, 28)
    >>> assert X_shards[1].shape == (128, 1, 28, 28)
    >>> assert y_shards[0].shape == (128, 1)
    >>> assert y_shards[1].shape == (128, 1)
    """
    assert X.shape[0] == y.shape[0]
    return (nn.parallel.scatter(X, devices), nn.parallel.scatter(y, devices))


def lenet_params() -> List[Tensor]:
    """
    返回lenet params
    """
    # 初始化模型参数
    scale = 0.01
    W1 = torch.randn(size=(20, 1, 3, 3)) * scale
    b1 = torch.zeros(20)
    W2 = torch.randn(size=(50, 20, 5, 5)) * scale
    b2 = torch.zeros(50)
    W3 = torch.randn(size=(800, 128)) * scale
    b3 = torch.zeros(128)
    W4 = torch.randn(size=(128, 10)) * scale
    b4 = torch.zeros(10)
    params = [W1, b1, W2, b2, W3, b3, W4, b4]
    return params


def lenet(X: Tensor, params: List[Tensor]) -> Tensor:
    """
    LeNet

    要注意: X和params要在同样的设备上, 比如X在GPU0上, params同样需要在GPU0上

    >>> x = torch.randn((256, 1, 28, 28))
    >>> params = lenet_params()
    >>> assert lenet(x, params).shape == (256, 10)  # [batch_size, num_classes]
    """
    h1_conv = F.conv2d(input=X, weight=params[0], bias=params[1])
    h1_activation = F.relu(h1_conv)
    h1 = F.avg_pool2d(input=h1_activation, kernel_size=(2, 2), stride=(2, 2))
    h2_conv = F.conv2d(input=h1, weight=params[2], bias=params[3])
    h2_activation = F.relu(h2_conv)
    h2 = F.avg_pool2d(input=h2_activation, kernel_size=(2, 2), stride=(2, 2))
    h2 = h2.reshape(h2.shape[0], -1)
    h3_linear = torch.mm(h2, params[4]) + params[5]
    h3 = F.relu(h3_linear)
    y_hat = torch.mm(h3, params[6]) + params[7]
    return y_hat


def sgd(params: Tensor, lr: float, batch_size: int) -> None:
    """
    小批量随机梯度下降
    """
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


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


def train_batch(X: Tensor, y: Tensor, device_params: List[List[Tensor]],
                loss: nn.Module, devices: List[torch.device],
                lr: float) -> None:
    """
    在多个GPU上做一次小批量梯度下降, 更新参数`device_params`. 更新之后每个GPU上的参数
    仍然是一样的

    假设有num_gpus个GPU
    1. 将一个批量的数据X,y分成num_gpus份, 每一份的批量长度变成: batch_size/num_gpus
    2. 在每个GPU上分别计算loss, 会得到num_gpus个loss
    3. 在每个GPU上分别进行反向传播, 得到每个参数的梯度
    4. 遍历每个参数, 将每个参数在每个GPU上的梯度相加, 然后将结果广播(拷贝)到所有GPU上,
       使`所有GPU上每个参数有相同的梯度`, 这样可以保证每个GPU上的模型在执行一次梯度下降之后
       的参数仍然是一致的
    5. 在每个GPU上分别执行一次梯度下降, 更新模型参数
    
    参数:
    X: [batch_size, 1, 28, 28]
    y: [batch_size, ]
    device_params: 每个GPU上的参数列表
    loss: 损失函数
    devices: list GPU device
    lr: 学习率
    """
    # X_shards[i].shape = [batch_size/num_gpus, 1, 28, 28], i=[0, ..., num_gpus-1]
    # y_shards[i].shape = [batch_size/num_gpus, ], i=[0, ..., num_gpus-1]
    X_shards, y_shards = split_batch(X, y, devices)
    # 在每个GPU上分别计算损失
    # ls[0]-GPU0
    # ls[1]-GPU1
    # ...
    ls = [
        # X_shard-GPU0, device_W-GPU0, y_shard-GPU0
        # X_shard-GPU1, device_W-GPU1, y_shard-GPU1
        # ...
        loss(lenet(X_shard, device_W),
             y_shard).sum() for X_shard, y_shard, device_W in zip(
                 X_shards, y_shards, device_params)
    ]
    # 在每个GPU上分别进行反向传播, 得到每个参数的梯度
    for l in ls:
        l.backward()
    with torch.no_grad():
        # 遍历每个参数
        for i in range(len(device_params[0])):
            # 将每个参数在每个GPU上的梯度相加, 然后将结果广播(拷贝)到所有GPU上,
            # 使`所有GPU上每个参数有相同的梯度`, 这样可以保证每个GPU上的模型在执行一次
            # 梯度下降之后的参数仍然是一致的
            allreduce([device_params[c][i].grad for c in range(len(devices))])
    # 在每个GPU上分别执行一次梯度下降, 更新模型参数
    for param in device_params:
        # 注意: 在这里, 我们使用batch_size, 因为梯度已经做了`聚合`
        sgd(param, lr, X.shape[0])


def train(num_epochs: int = 10,
          num_gpus: int = 1,
          batch_size: int = 256,
          lr: float = 0.2) -> None:
    loss = nn.CrossEntropyLoss(reduction='none')
    params = lenet_params()
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    devices = [try_gpu(i) for i in range(num_gpus)]
    # 将模型参数复制到`num_gpus`个GPU
    # List[List[Tensor]]
    device_params = [get_params(params, d) for d in devices]
    times = []
    history = [[]]  # 记录: 训练集损失, 方便后续绘图
    # animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    # timer = d2l.Timer()
    for epoch in range(num_epochs):
        t_start = time.time()
        for X, y in train_iter:
            # 为单个小批量执行多GPU训练
            # X.shape [batch_size, 1, 28, 28]
            # y.shape [batch_size, ]
            train_batch(X, y, device_params, loss, devices, lr)
            torch.cuda.synchronize()  # 等待每个GPU上完成梯度下降
        times.append(time.time() - t_start)
        # 在GPU0上评估模型
        metric_test = [0.0] * 2  # 测试准确数量之和, 测试样本数量之和
        with torch.no_grad():
            for X, y in test_iter:
                X = X.to(devices[0])
                y = y.to(devices[0])
                metric_test[0] += float(accuracy(lenet(X, device_params[0]),
                                                 y))
                metric_test[1] += float(y.numel())
            test_acc = metric_test[0] / metric_test[1]
            history[0].append((epoch + 1, test_acc))
        print(f'epoch {epoch}, test acc {test_acc:.3f}')

    print(
        f'test acc {history[0][-1][1]:.2f}，{sum(times)/num_epochs:.1f} epoch/sec on {str(devices)}'
    )

    plt.figure(figsize=(6, 4))
    plt.plot(*zip(*history[0]), '-', label='test acc')
    plt.xlabel('epoch')
    # 从epoch=1开始显示, 0-1这个范围的数据丢弃不展示,
    # 因为只有训练完成1个epochs之后, 才会有第一条test acc记录
    plt.xlim((1, num_epochs))
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # train(num_epochs=10, num_gpus=1, batch_size=256, lr=0.2)
    # epoch 0, test acc 0.100
    # epoch 1, test acc 0.629
    # epoch 2, test acc 0.709
    # epoch 3, test acc 0.779
    # epoch 4, test acc 0.757
    # epoch 5, test acc 0.790
    # epoch 6, test acc 0.827
    # epoch 7, test acc 0.826
    # epoch 8, test acc 0.827
    # epoch 9, test acc 0.824
    # test acc 0.82，7.5 epoch/sec on [device(type='cuda', index=0)]

    train(num_epochs=10, num_gpus=2, batch_size=256, lr=0.2)
    # epoch 0, test acc 0.100
    # epoch 1, test acc 0.686
    # epoch 2, test acc 0.749
    # epoch 3, test acc 0.751
    # epoch 4, test acc 0.778
    # epoch 5, test acc 0.813
    # epoch 6, test acc 0.792
    # epoch 7, test acc 0.825
    # epoch 8, test acc 0.823
    # epoch 9, test acc 0.826
    # test acc 0.83，8.5 epoch/sec on [device(type='cuda', index=0), device(type='cuda', index=1)]
