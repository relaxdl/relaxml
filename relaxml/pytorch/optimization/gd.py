from typing import Any, Callable, List, Tuple
import torch
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
"""
梯度下降

说明:
https://tech.foxrelax.com/optimization/gd/
"""


def plots(
    xs: List[Tensor],
    ys: List[Tensor],
    xlabel: str = None,
    ylabel: str = None,
    fmts: Tuple[str] = ('-', 'm--', 'g-.', 'r:'),
    figsize=(6, 4)
) -> None:
    plt.figure(figsize=figsize)
    for x, y, fmt in zip(xs, ys, fmts):
        plt.plot(x, y, fmt)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()


def gd(eta: float, f_grad: Callable) -> List[Tensor]:
    """
    执行10次梯度下降, 返回梯度下降过程中x的所有取值轨迹

    eta: 学习率
    f_grad: 梯度函数
    """
    x = 10.0  # x的初始值, 梯度下降的开始位置
    results = [x]  # 保存x的变化
    for i in range(10):
        x -= eta * f_grad(x)
        results.append(x)
    print(f'epoch 10, x: {x:f}')
    return results


def show_trace(results: List[Tensor], f: Callable) -> None:
    """
    显示x的轨迹
    """
    n = max(abs(min(results)), abs(max(results)))
    f_line = torch.arange(-n, n, 0.01)
    plots([f_line, results], [[f(x) for x in f_line], [f(x) for x in results]],
          'x',
          'f(x)',
          fmts=['-', '-o'])
    plt.show()


def f(x: Tensor) -> Tensor:
    """
    目标函数
    """
    return x**2


def f_grad(x: Tensor) -> Tensor:
    """
    目标函数的梯度(导数)
    """
    return 2 * x


def f_2d(x1: Tensor, x2: Tensor) -> Tensor:
    """
    目标函数
    """
    return x1**2 + 2 * x2**2


def f_2d_grad(x1: Tensor, x2: Tensor) -> Tuple[Tensor, Tensor]:
    """
    目标函数的梯度
    """
    return (2 * x1, 4 * x2)


def gd_2d(x1: Tensor, x2: Tensor, s1: Tensor, s2: Tensor,
          f_grad: Callable) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    优化函数, 用来更新参数(做一次梯度下降)
    """
    eta = 0.1
    g1, g2 = f_grad(x1, x2)
    return (x1 - eta * g1, x2 - eta * g2, 0, 0)


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


def demo1():
    """
    一维梯度下降: 学习率=0.2
    """
    show_trace(gd(0.2, f_grad), f)


def demo2():
    """
    一维梯度下降: 学习率=0.05
    """
    show_trace(gd(0.05, f_grad), f)


def demo3():
    """
    一维梯度下降: 学习率=1.1
    """
    show_trace(gd(1.1, f_grad), f)


def demo4():
    """
    局部最小值
    """
    c = torch.tensor(0.15 * np.pi)

    def _f(x):
        # 目标函数
        return x * torch.cos(c * x)

    def _f_grad(x):
        # 目标函数的梯度(导数)
        return torch.cos(c * x) - c * x * torch.sin(c * x)

    show_trace(gd(2, _f_grad), _f)


def demo5():
    """
    多元梯度下降
    """
    show_trace_2d(train_2d(trainer=gd_2d, f_grad=f_2d_grad), f_2d)


if __name__ == '__main__':
    demo5()
