import torch
from typing import Callable, List, Tuple
from torch import Tensor
import matplotlib.pyplot as plt


def plot_activation(activation: Callable,
                    name: str,
                    start: int,
                    end: int,
                    steps: int = 100) -> None:
    """
    画激活函数及其对应的梯度
    1. 激活函数的图形是蓝色
    2. 梯度的图像是绿色
    """
    # 计算outputs & grads
    inputs = torch.linspace(start, end, steps)
    outputs = activation(inputs)
    grads = []
    for x in inputs:
        x.requires_grad = True
        y = activation(x)
        y.backward()
        grads.append(x.grad)

    # plot
    _, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(inputs.detach(), outputs.detach(), '-', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid()
    ax1.set_title(name)

    ax2 = ax1.twinx()  # 创建共用x轴的第二个y轴
    ax2.plot(inputs.detach(), grads, '--', color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')
    ax2.grid()
    ax2.set_title(name)

    plt.show()


def plot_activations(activations: List[Tuple[Callable, str]],
                     start: int,
                     end: int,
                     steps: int = 100) -> None:
    """
    对比多个激活函数的图像
    """
    plt.figure(figsize=(6, 4))
    # 计算outputs
    inputs = torch.linspace(start, end, steps)
    for activation, name in activations:
        outpus = activation(inputs)
        plt.plot(inputs, outpus, label=name)
    plt.grid()
    plt.legend()
    plt.show()


def sigmoid(x: Tensor) -> Tensor:
    y = 1 / (1 + torch.exp(-x))
    return y


def h_sigmoid(x: Tensor) -> Tensor:
    y = torch.maximum(torch.tensor(0.0),
                      torch.minimum((x + 1) / 2, torch.tensor(1.0)))
    return y


def tanh(x: Tensor) -> Tensor:
    y = (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))
    return y


def relu(x: Tensor) -> Tensor:
    y = torch.maximum(x, torch.tensor(0.0))
    return y


def relu6(x: Tensor) -> Tensor:
    y = torch.minimum(torch.maximum(x, torch.tensor(0.0)), torch.tensor(6.0))
    return y


def leaky_relu(x: Tensor, negative_slope: float = 0.1) -> Tensor:
    y = torch.maximum(x, torch.tensor(0.0)) + negative_slope * torch.minimum(
        x, torch.tensor(0.0))
    return y


def swish(x: Tensor, beta: float = 1.0) -> Tensor:
    y = x * sigmoid(beta * x)
    return y


def h_swish(x: Tensor, beta: float = 1.0) -> Tensor:
    y = x * h_sigmoid(beta * x)
    return y


if __name__ == '__main__':
    plot_activation(h_swish, 'h_swish beta=1.0', -6, 6)
    plot_activations([(swish, 'swish'), (h_swish, 'h_swish')], -6, 6)
