from typing import List, Tuple
from matplotlib.text import Annotation
import numpy as np
import torch
from torch import Tensor
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


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


def annotate(text: str, xy: Tuple[int, int], xytext: Tuple[int,
                                                           int]) -> Annotation:
    plt.gca().annotate(text, xy, xytext, arrowprops=dict(arrowstyle='->'))


def f(x: Tensor) -> Tensor:
    return x * torch.cos(np.pi * x)


def g(x: Tensor) -> Tensor:
    return f(x) + 0.2 * torch.cos(5 * np.pi * x)


def demo1():
    """
    优化的目标
    """
    x = torch.arange(0.5, 1.5, 0.01)
    plots([x, x], [f(x), g(x)], 'x', 'risk')
    annotate('min of\nempirical risk', (1.0, -1.2), (0.5, -1.1))
    annotate('min of risk', (1.1, -1.05), (0.95, -0.5))
    plt.show()


def demo2():
    """
    局部最小值
    """
    x = torch.arange(-1.0, 2.0, 0.01)
    plots([x], [f(x)], 'x', 'f(x)')
    annotate('local minimum', (-0.3, -0.25), (-0.77, -1.0))
    annotate('global minimum', (1.1, -0.95), (0.6, 0.8))
    plt.show()


def demo3():
    """
    鞍点1
    """
    x = torch.arange(-2.0, 2.0, 0.01)
    plots([x], [x * x * x], 'x', 'f(x)')
    annotate('saddle point', (0, -0.2), (-0.52, -5.0))
    plt.show()


def demo4():
    """
    鞍点2
    """
    x, y = torch.meshgrid(torch.linspace(-1.0, 1.0, 101),
                          torch.linspace(-1.0, 1.0, 101))
    z = x**2 - y**2
    ax = plt.figure().add_subplot(111, projection='3d')
    ax.plot_wireframe(x, y, z, **{'rstride': 10, 'cstride': 10})
    ax.plot([0], [0], [0], 'rx')
    ticks = [-1, 0, 1]
    plt.xticks(ticks)
    plt.yticks(ticks)
    ax.set_zticks(ticks)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def demo5():
    """
    梯度消失
    """
    x = torch.arange(-2.0, 5, 0.01)
    plots([x], [torch.tanh(x)], 'x', 'f(x)')
    annotate('vanishing gradient', (4, 1), (2, 0.0))
    plt.show()


if __name__ == '__main__':
    demo4()