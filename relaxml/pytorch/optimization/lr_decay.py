from typing import List, Tuple, Callable
import math
import torch
from torch import Tensor
import matplotlib.pyplot as plt
"""
学习率衰减

说明:
https://tech.foxrelax.com/optimization/lr_decay/
"""


def train_2d(trainer: Callable,
             steps: int = 20,
             f_grad: Callable = None) -> List[Tuple[Tensor, Tensor]]:
    """
    用定制的训练机优化2D目标函数
    """
    x1, x2, s1, s2 = -5, -2, 0, 0  # 初始值
    results = [(x1, x2)]
    for i in range(steps):
        if f_grad:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2, f_grad)
        else:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print(f'epoch {i + 1}, x1: {float(x1):f}, x2: {float(x2):f}')
    return results


def show_trace_2d(results: List[Tuple[Tensor, Tensor]], f: Callable) -> None:
    """
    显示优化过程中2D变量的轨迹
    """
    plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = torch.meshgrid(torch.arange(-5.5, 1.0, 0.1),
                            torch.arange(-3.0, 1.0, 0.1))
    plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


def f(x1: Tensor, x2: Tensor) -> Tensor:
    """
    目标函数
    """
    return x1**2 + 2 * x2**2


def f_grad(x1: Tensor, x2: Tensor) -> Tuple[Tensor, Tensor]:
    """
    目标函数的梯度
    """
    return (2 * x1, 4 * x2)


t = 1


def demo1():
    """
    指数衰减
    """

    def exponential_lr() -> float:
        global t
        t += 1
        return math.exp(-0.1 * t)

    eta = 0.1
    lr = exponential_lr

    def sgd(x1: Tensor, x2: Tensor, s1: Tensor, s2: Tensor,
            f_grad: Callable) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        优化函数, 用来更新参数(做一次梯度下降)
        """
        g1, g2 = f_grad(x1, x2)
        # 添加平均值0和方差1的随机噪声来模拟`随机梯度下降`
        g1 += torch.normal(0.0, 1, (1, ))
        g2 += torch.normal(0.0, 1, (1, ))
        eta_t = eta * lr()
        return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)

    show_trace_2d(train_2d(sgd, steps=50, f_grad=f_grad), f)


def demo2():
    """
    多项式衰减
    """

    def polynomial_lr():
        global t
        t += 1
        return (1 + 0.1 * t)**(-0.5)

    eta = 0.1
    lr = polynomial_lr

    def sgd(x1: Tensor, x2: Tensor, s1: Tensor, s2: Tensor,
            f_grad: Callable) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        优化函数, 用来更新参数(做一次梯度下降)
        """
        g1, g2 = f_grad(x1, x2)
        # 添加平均值0和方差1的随机噪声来模拟`随机梯度下降`
        g1 += torch.normal(0.0, 1, (1, ))
        g2 += torch.normal(0.0, 1, (1, ))
        eta_t = eta * lr()
        return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)

    show_trace_2d(train_2d(sgd, steps=50, f_grad=f_grad), f)


if __name__ == '__main__':
    demo2()
