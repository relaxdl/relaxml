from typing import Callable, Tuple, Union, List
import math
import sys
import time
import torch
from torch import Tensor
from torch import nn
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from tqdm import tqdm
"""
学习率调度器

说明:
https://tech.foxrelax.com/optimization/lr_scheduler/
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


def net_fn() -> nn.Module:
    """
    改进版的LeNet
    """
    net = nn.Sequential(nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2), nn.Flatten(),
                        nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
                        nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, 10))

    return net


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


def train_gpu(
    net: nn.Module,
    train_iter: DataLoader,
    test_iter: DataLoader,
    num_epochs: int = 10,
    loss: nn.Module = None,
    optimizer: Optimizer = None,
    device: torch.device = None,
    scheduler: Union[nn.Module,
                     Callable] = None) -> List[List[Tuple[int, float]]]:
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
            if test_acc > best_test_acc:
                best_test_acc = test_acc

        # 每一次epoch结束之后, 执行一次scheduler来更新`学习率`
        if scheduler:
            if scheduler.__module__ == lr_scheduler.__name__:
                # Using PyTorch In-Built scheduler
                scheduler.step()
            else:
                # Using custom defined scheduler
                for param_group in optimizer.param_groups:
                    param_group['lr'] = scheduler(epoch)

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


def demo1():
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)

    lr, num_epochs = 0.3, 30
    net = net_fn()
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    history = train_gpu(net=net,
                        train_iter=train_iter,
                        test_iter=test_iter,
                        num_epochs=num_epochs,
                        loss=nn.CrossEntropyLoss(reduction='mean'),
                        optimizer=trainer)
    plot_history(history)


class SquareRootScheduler:

    def __init__(self, lr=0.1):
        self.lr = lr

    def __call__(self, num_update: int) -> float:
        return self.lr * pow(num_update + 1.0, -0.5)


def demo2():
    """
    调度器
    """
    # train
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)
    scheduler = SquareRootScheduler(lr=0.1)
    lr, num_epochs = 0.3, 30
    net = net_fn()
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    history = train_gpu(net=net,
                        train_iter=train_iter,
                        test_iter=test_iter,
                        num_epochs=num_epochs,
                        loss=nn.CrossEntropyLoss(reduction='mean'),
                        optimizer=trainer,
                        scheduler=scheduler)
    plot_history(history)

    # plot scheduler
    plt.plot(torch.arange(num_epochs),
             [scheduler(t) for t in range(num_epochs)])
    plt.grid()
    plt.show()


class FactorScheduler:

    def __init__(self, factor=1, stop_factor_lr=1e-7, base_lr=0.1):
        # `factor`: 就是衰减因子, 如果factor=1则不衰减
        # `base_lr`: 表示初始学习率
        # `stop_factor_lr`: 表示小于这个学习率则不再衰减
        self.factor = factor
        self.stop_factor_lr = stop_factor_lr
        self.base_lr = base_lr

    def __call__(self, num_update: int) -> float:
        self.base_lr = max(self.stop_factor_lr, self.base_lr * self.factor)
        return self.base_lr


def demo3():
    """
    因子调度器
    """
    # plot scheduler
    num_epochs = 30
    # 初始学习率为2.0, 衰减因子是0.9
    scheduler = FactorScheduler(factor=0.9, stop_factor_lr=1e-2, base_lr=2.0)
    plt.plot(torch.arange(num_epochs),
             [scheduler(t) for t in range(num_epochs)])
    plt.grid()
    plt.show()


def demo4():
    """
    多因子调度器
    """
    # train
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)
    lr, num_epochs = 0.5, 30
    net = net_fn()
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    scheduler = lr_scheduler.MultiStepLR(trainer,
                                         milestones=[15, 30],
                                         gamma=0.5)
    history = train_gpu(net=net,
                        train_iter=train_iter,
                        test_iter=test_iter,
                        num_epochs=num_epochs,
                        loss=nn.CrossEntropyLoss(reduction='mean'),
                        optimizer=trainer,
                        scheduler=scheduler)
    plot_history(history)

    # plot scheduler
    num_epochs = 100
    scheduler = lr_scheduler.MultiStepLR(trainer,
                                         milestones=[15, 30],
                                         gamma=0.5)

    def get_lr(trainer, scheduler):
        lr = scheduler.get_last_lr()[0]  # 返回的lr是一个list, 我们取第一个group的lr
        trainer.step()
        scheduler.step()
        return lr

    plt.plot(torch.arange(num_epochs),
             [get_lr(trainer, scheduler) for t in range(num_epochs)])
    plt.grid()
    plt.show()


class CosineScheduler:

    def __init__(self,
                 max_update,
                 base_lr=0.01,
                 final_lr=0,
                 warmup_steps=0,
                 warmup_begin_lr=0):
        # `max_update`: 最大的更新步数, T
        # `base_lr`: 表示初始学习率, mu_0
        # `final_lr`: 表示最终学习率, mu_T
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        # 预热的逻辑:
        # 经过warmup_steps将学习率从warmup_begin_lr逐步增大为base_lr_orig
        increase = (self.base_lr_orig - self.warmup_begin_lr) \
                       * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase  # 每次增加increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (
                self.base_lr_orig - self.final_lr) * (1 + math.cos(
                    math.pi *
                    (epoch - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr


def demo5():
    """
    余弦调度器
    """
    # train
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)
    # 初始学习率为0.3, 最终学习率为0.01, 最大更新步骤为20
    scheduler = CosineScheduler(max_update=20, base_lr=0.3, final_lr=0.01)
    lr, num_epochs = 0.3, 30
    net = net_fn()
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    history = train_gpu(net=net,
                        train_iter=train_iter,
                        test_iter=test_iter,
                        num_epochs=num_epochs,
                        loss=nn.CrossEntropyLoss(reduction='mean'),
                        optimizer=trainer,
                        scheduler=scheduler)
    plot_history(history)

    # plot scheduler
    num_epochs = 30
    # 初始学习率为0.3, 最终学习率为0.01, 最大更新步骤为20
    scheduler = CosineScheduler(max_update=20, base_lr=0.3, final_lr=0.01)
    plt.plot(torch.arange(num_epochs),
             [scheduler(t) for t in range(num_epochs)])
    plt.grid()
    plt.show()


def demo6():
    """
    预热
    """
    # train
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)
    # 初始学习率为0.3, 最终学习率为0.01, 最大更新步骤为20
    scheduler = CosineScheduler(20, warmup_steps=5, base_lr=0.3, final_lr=0.01)
    lr, num_epochs = 0.3, 30
    net = net_fn()
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    history = train_gpu(net=net,
                        train_iter=train_iter,
                        test_iter=test_iter,
                        num_epochs=num_epochs,
                        loss=nn.CrossEntropyLoss(reduction='mean'),
                        optimizer=trainer,
                        scheduler=scheduler)
    plot_history(history)

    # plot scheduler
    num_epochs = 30
    scheduler = CosineScheduler(20, warmup_steps=5, base_lr=0.3, final_lr=0.01)
    plt.plot(torch.arange(num_epochs),
             [scheduler(t) for t in range(num_epochs)])
    plt.grid()
    plt.show()


if __name__ == '__main__':
    demo6()