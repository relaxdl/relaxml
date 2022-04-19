from typing import Tuple, Union
import time
import torch
from torch import Tensor
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
"""
计算图

说明:
https://tech.foxrelax.com/basic/graph/
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


def relu(X: Tensor) -> Tensor:
    a = torch.zeros_like(X)
    return torch.max(X, a)


def relu_derivative(X: Tensor) -> Tensor:
    return torch.ones_like(X) * (X >= 0)


def softmax(X: Tensor) -> Tensor:
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制


def cross_entropy(y_hat: Tensor, y: Tensor) -> Tensor:
    return -torch.log(y_hat[range(len(y_hat)), y])


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


def train():
    # 定义网络模型
    num_inputs, num_outpus, num_hiddens = 784, 10, 256
    W1 = torch.randn(num_inputs, num_hiddens) * 0.01
    b1 = torch.zeros(num_hiddens)
    W2 = torch.randn(num_hiddens, num_outpus) * 0.01
    b2 = torch.zeros(num_outpus)

    def net(X: Tensor) -> Tensor:
        X = X.reshape((-1, num_inputs))
        O1 = X @ W1 + b1
        H1 = relu(O1)

        O2 = H1 @ W2 + b2
        y_hat = softmax(O2)
        return y_hat

    # 加载数据集
    batch_size = 128
    train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)

    # 训练
    lr, num_epochs = 0.1, 10
    history = [[], [], []]  # 记录: 训练集损失, 训练集准确率, 测试集准确率, 方便后续绘图
    times, num_batches = [], len(train_iter)
    for epoch in range(num_epochs):
        metric_train = [0.0] * 3  # 统计: 训练集损失之和, 训练集准确数量之和, 训练集样本数量之和
        for i, (X, y) in enumerate(train_iter):
            t_start = time.time()
            # `前向传播`
            # 没有直接调用nex(X)的原因是`后向传播`的过程中会用到前向传播产生的中间变量,
            # 所以将nex(X)展开在这里调用
            X = X.reshape((-1, num_inputs))  # [b, 784]
            O1 = X @ W1 + b1  # [b, 256]
            H1 = relu(O1)  # [b, 256]

            O2 = H1 @ W2 + b2  # [b, 10]
            y_hat = softmax(O2)  # [b, 10]
            l = cross_entropy(y_hat, y)

            # `后向传播`
            y_onehot = F.one_hot(y)  # [b, 10]
            O2_derivative = softmax(O2) - y_onehot  # [b, 10]

            W2_derivative = H1.unsqueeze(2) @ O2_derivative.unsqueeze(
                1)  # [b, 256, 1] @ [b, 1, 10] = [b, 256, 10]
            W2_derivative = W2_derivative.mean(0)  # [256, 10]
            b2_derivative = O2_derivative  # [b, 10]
            b2_derivative = b2_derivative.mean(0)  # [10, ]

            H1_derivative = O2_derivative @ W2.T  # [b, 10] @ [10, 256] = [b, 256]
            O1_derivative = H1_derivative * relu_derivative(
                O1)  # [b, 256] * [b, 256] = [b, 256]

            W1_derivative = X.unsqueeze(2) @ O1_derivative.unsqueeze(
                1)  # [b, 784, 1] @ [b, 1, 256] = [b, 784, 256]
            W1_derivative = W1_derivative.mean(0)  # [784, 256]
            b1_derivative = O1_derivative  # [b, 256]
            b1_derivative = b1_derivative.mean(0)  # [10, ]

            # 更新参数
            W1 -= lr * W1_derivative
            b1 -= lr * b1_derivative
            W2 -= lr * W2_derivative
            b2 -= lr * b2_derivative

            metric_train[0] += float(l.sum())
            metric_train[1] += float(accuracy(y_hat, y))
            metric_train[2] += float(X.shape[0])
            times.append(time.time() - t_start)
            train_loss = metric_train[0] / metric_train[2]
            train_acc = metric_train[1] / metric_train[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                history[0].append((epoch + (i + 1) / num_batches, train_loss))
                history[1].append((epoch + (i + 1) / num_batches, train_acc))

        # 评估
        metric_test = [0.0] * 2  # 测试准确数量之和, 测试样本数量之和
        with torch.no_grad():
            for X, y in test_iter:
                metric_test[0] += float(accuracy(net(X), y))
                metric_test[1] += float(X.shape[0])
            test_acc = metric_test[0] / metric_test[1]
            history[2].append((epoch + 1, test_acc))
            print(f'epoch {epoch}, step {i+1}, train loss {train_loss:.3f}, '
                  f'train acc {train_acc:.3f}, test acc {test_acc:.3f}')

    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric_train[2] * num_epochs / sum(times):.1f} examples/sec')

    plt.figure(figsize=(6, 4))
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


if __name__ == '__main__':
    train()
# epoch 0, step 469, train loss 0.834, train acc 0.710, test acc 0.773
# epoch 1, step 469, train loss 0.511, train acc 0.820, test acc 0.799
# epoch 2, step 469, train loss 0.454, train acc 0.839, test acc 0.828
# epoch 3, step 469, train loss 0.421, train acc 0.850, test acc 0.837
# epoch 4, step 469, train loss 0.398, train acc 0.859, test acc 0.809
# epoch 5, step 469, train loss 0.379, train acc 0.865, test acc 0.858
# epoch 6, step 469, train loss 0.363, train acc 0.869, test acc 0.854
# epoch 7, step 469, train loss 0.352, train acc 0.874, test acc 0.861
# epoch 8, step 469, train loss 0.341, train acc 0.877, test acc 0.856
# epoch 9, step 469, train loss 0.332, train acc 0.881, test acc 0.863
# train loss 0.332, train acc 0.881, test acc 0.863
# 6386.3 examples/sec