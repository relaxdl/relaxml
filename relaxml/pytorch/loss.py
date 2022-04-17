from typing import Callable
import torch
from torch import Tensor
import torch.nn as nn
import matplotlib.pyplot as plt


def plot_regression_loss(loss: Callable,
                         name: str,
                         start: int,
                         end: int,
                         steps: int = 100) -> None:
    """
    画Regression Loss函数及其对应的梯度
    1. 蓝色实线是当真实值y=0时, 随着预测值y*的变化, 损失函数的曲线
    2. 橙色虚线是导数(梯度)
    """
    inputs = torch.linspace(start, end, steps)
    target = torch.tensor(0.0)
    outputs = []
    grads = []
    for x in inputs:
        x.requires_grad = True
        l = loss(x, target)
        l.backward()
        outputs.append(l.detach())
        grads.append(x.grad)

    # plot
    plt.figure(figsize=(6, 4))
    plt.plot(
        inputs.detach(),
        outputs,
        label='loss',
    )
    plt.plot(inputs.detach(), grads, '--', label='derivative')
    plt.grid(ls='--')
    plt.title(name)
    plt.legend()
    plt.show()


class MaskedBCELoss(nn.Module):
    """
    带遮蔽的BCELoss

    >>> input = torch.tensor([[0.1, 0.9, 0.1, 0.9], 
                              [0.2, 0.8, 0.2, 0.8]])
    >>> target = torch.tensor([[0., 1., 0., 1.], 
                               [0., 1., 0., 1.]])
    >>> mask = torch.tensor([[0, 0, 1, 1], 
                             [0, 1, 1, 1]])
    >>> loss = MaskedBCELoss()
    >>> output = loss(input, target, mask)
    >>> print(output)
        tensor(0.1100)
    """

    def forward(self,
                input: Tensor,
                target: Tensor,
                mask: Tensor = None,
                reduction: str = 'mean') -> Tensor:
        output = nn.functional.binary_cross_entropy(input,
                                                    target,
                                                    mask,
                                                    reduction=reduction)
        return output


if __name__ == '__main__':
    plot_regression_loss(nn.HuberLoss(), 'HuberLoss', -5, 5)
