from typing import Tuple, Union, List
import sys
import time
import torch
from torch import Tensor
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm
"""
softmax回归从零实现

说明:
https://tech.foxrelax.com/basic/softmax_regression_scratch/
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


def softmax(X: Tensor) -> Tensor:
    """
    >>> X = torch.normal(0, 1, (2, 5))
    >>> X_prob = softmax(X)
    >>> print(X_prob)
        tensor([[0.1793, 0.0716, 0.0739, 0.0880, 0.5872],
                [0.0373, 0.3591, 0.0533, 0.3523, 0.1980]])
    >>> print(X_prob.sum(1))
        tensor([1.0000, 1.0000])
    """
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制


def gen_params(num_inputs: int = 784,
               num_outputs: int = 10) -> Tuple[Tensor, Tensor]:
    """
    生成模型参数: [w, b]
    """
    W = torch.normal(0,
                     0.01,
                     size=(num_inputs, num_outputs),
                     requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)
    return W, b


def net(X: Tensor, W: Tensor, b: Tensor) -> Tensor:
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


def cross_entropy(y_hat: Tensor, y: Tensor, reduction: str = 'mean') -> Tensor:
    """
    >>> y = torch.tensor([0, 2])
    >>> y_hat = torch.tensor([[0.1, 0.3, 0.6], 
                              [0.3, 0.2, 0.5]])
    >>> l = cross_entropy(y_hat, y, reduction='mean')
        tensor(1.4979)
    >>> l = cross_entropy(y_hat, y, reduction='sum')
        tensor(2.9957)
    >>> l = cross_entropy(y_hat, y, reduction='none')
        tensor([2.3026, 0.6931])
    """
    if reduction == 'mean':
        return -torch.log(y_hat[range(len(y_hat)), y]).mean()
    elif reduction == 'none':
        return -torch.log(y_hat[range(len(y_hat)), y])
    elif reduction == 'sum':
        return -torch.log(y_hat[range(len(y_hat)), y]).sum()


def sgd(params: Union[Tuple, List], lr: float, batch_size: int) -> None:
    """
    小批量随机梯度下降
    """
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad
            param.grad.zero_()


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


def train(batch_size: int = 256,
          num_epochs: int = 10,
          lr: float = 0.1) -> List[List[Tuple[int, float]]]:
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    W, b = gen_params()
    loss = cross_entropy
    times = []
    history = [[], [], []]  # 记录: 训练集损失, 训练集准确率, 测试集准确率, 方便后续绘图
    num_batches = len(train_iter)
    best_test_acc = 0.0
    for epoch in range(num_epochs):
        # 训练
        metric_train = [0.0] * 3  # 统计: 训练集损失之和, 训练集准确数量之和, 训练集样本数量之和
        train_iter_tqdm = tqdm(train_iter, file=sys.stdout)
        for i, (X, y) in enumerate(train_iter_tqdm):
            t_start = time.time()
            y_hat = net(X, W, b)
            l = loss(y_hat, y)
            l.backward()
            sgd([W, b], lr, batch_size)
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
        with torch.no_grad():
            for X, y in test_iter:
                metric_test[0] += float(accuracy(net(X, W, b), y))
                metric_test[1] += float(X.shape[0])
            test_acc = metric_test[0] / metric_test[1]
            history[2].append((epoch + 1, test_acc))
            print(f'epoch {epoch}, step {i+1}, train loss {train_loss:.3f}, '
                  f'train acc {train_acc:.3f}, test acc {test_acc:.3f}')
            if test_acc > best_test_acc:
                best_test_acc = test_acc

    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric_train[2] * num_epochs / sum(times):.1f} examples/sec')
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
    history = train(num_epochs=10, lr=0.1)
    plot_history(history)


if __name__ == '__main__':
    run()
# epoch 0, step 235, train loss 0.786, train acc 0.748: 100%|█████████████████████████████████████| 235/235 [00:01<00:00, 156.26it/s]
# epoch 0, step 235, train loss 0.786, train acc 0.748, test acc 0.778
# epoch 1, step 235, train loss 0.572, train acc 0.813: 100%|█████████████████████████████████████| 235/235 [00:01<00:00, 166.79it/s]
# epoch 1, step 235, train loss 0.572, train acc 0.813, test acc 0.809
# epoch 2, step 235, train loss 0.524, train acc 0.827: 100%|█████████████████████████████████████| 235/235 [00:01<00:00, 169.41it/s]
# epoch 2, step 235, train loss 0.524, train acc 0.827, test acc 0.814
# epoch 3, step 235, train loss 0.501, train acc 0.832: 100%|█████████████████████████████████████| 235/235 [00:01<00:00, 167.22it/s]
# epoch 3, step 235, train loss 0.501, train acc 0.832, test acc 0.805
# epoch 4, step 235, train loss 0.486, train acc 0.837: 100%|█████████████████████████████████████| 235/235 [00:01<00:00, 165.61it/s]
# epoch 4, step 235, train loss 0.486, train acc 0.837, test acc 0.808
# epoch 5, step 235, train loss 0.473, train acc 0.841: 100%|█████████████████████████████████████| 235/235 [00:01<00:00, 165.58it/s]
# epoch 5, step 235, train loss 0.473, train acc 0.841, test acc 0.827
# epoch 6, step 235, train loss 0.465, train acc 0.843: 100%|█████████████████████████████████████| 235/235 [00:01<00:00, 168.86it/s]
# epoch 6, step 235, train loss 0.465, train acc 0.843, test acc 0.817
# epoch 7, step 235, train loss 0.458, train acc 0.845: 100%|█████████████████████████████████████| 235/235 [00:01<00:00, 167.03it/s]
# epoch 7, step 235, train loss 0.458, train acc 0.845, test acc 0.833
# epoch 8, step 235, train loss 0.452, train acc 0.847: 100%|█████████████████████████████████████| 235/235 [00:01<00:00, 163.80it/s]
# epoch 8, step 235, train loss 0.452, train acc 0.847, test acc 0.832
# epoch 9, step 235, train loss 0.447, train acc 0.848: 100%|█████████████████████████████████████| 235/235 [00:01<00:00, 164.01it/s]
# epoch 9, step 235, train loss 0.447, train acc 0.848, test acc 0.806
# train loss 0.447, train acc 0.848, test acc 0.806
# 925525.1 examples/sec