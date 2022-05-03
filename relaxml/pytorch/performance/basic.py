from typing import List
import time
import numpy as np
import torch
import torch.nn as nn
"""
基础

实现说明:
https://tech.foxrelax.com/performance/basic/
"""


class Benchmark:

    def __init__(self, description: str = 'Done') -> None:
        self.description = description

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        print(f'{self.description}: {time.time()-self.start:.4f} sec')


def cpu() -> torch.device:
    return torch.device('cpu')


def try_gpu(i: int = 0) -> torch.device:
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus() -> List[torch.device]:
    devices = [
        torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())
    ]
    return devices if devices else [torch.device('cpu')]


def copy_to_cpu(x: List[torch.Tensor],
                non_blocking: bool = False) -> List[torch.Tensor]:
    """
    将x拷贝到CPU上
    """
    return [y.to('cpu', non_blocking=non_blocking) for y in x]


def get_net() -> nn.Module:
    """
    生产网络的工厂模式
    """
    net = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 128),
                        nn.ReLU(), nn.Linear(128, 2))
    return net


def demo1():
    x = torch.randn(size=(1, 512))
    net = get_net()
    with Benchmark('无torchscript'):
        for i in range(100000):
            net(x)

    net = torch.jit.script(net)
    with Benchmark('有torchscript'):
        for i in range(100000):
            net(x)
    # 无torchscript: 2.5045 sec
    # 有torchscript: 1.9508 sec


def demo2():
    device = try_gpu()
    with Benchmark('numpy'):
        for _ in range(10):
            a = np.random.normal(size=(1000, 1000))
            b = np.dot(a, a)

    with Benchmark('torch'):
        for _ in range(10):
            a = torch.randn(size=(1000, 1000), device=device)
            b = torch.mm(a, a)

    with Benchmark('torch.synchronize'):
        for _ in range(10):
            a = torch.randn(size=(1000, 1000), device=device)
            b = torch.mm(a, a)
        torch.cuda.synchronize(device)
    # numpy: 0.7022 sec
    # torch: 0.0011 sec
    # torch.synchronize: 0.0156 sec


def demo3():
    """
    注意: 这个案例必须在有至少2个GPU的机器上运行
    """
    devices = try_all_gpus()

    def run(x):
        return [x.mm(x) for _ in range(100)]

    x_gpu1 = torch.rand(size=(4000, 4000), device=devices[0])
    x_gpu2 = torch.rand(size=(4000, 4000), device=devices[1])

    run(x_gpu1)
    run(x_gpu2)  # 预热设备
    torch.cuda.synchronize(devices[0])
    torch.cuda.synchronize(devices[1])

    with Benchmark('GPU1 time'):
        run(x_gpu1)
        torch.cuda.synchronize(devices[0])

    with Benchmark('GPU2 time'):
        run(x_gpu2)
        torch.cuda.synchronize(devices[1])

    with Benchmark('GPU1 & GPU2'):
        run(x_gpu1)
        run(x_gpu2)
        torch.cuda.synchronize()
    # GPU1 time: 0.9227 sec
    # GPU2 time: 0.9103 sec
    # GPU1 & GPU2: 0.9217 sec


def demo4():
    devices = try_all_gpus()

    def run(x):
        return [x.mm(x) for _ in range(50)]

    x_gpu1 = torch.rand(size=(4000, 4000), device=devices[0])

    with Benchmark('在GPU1上运行'):
        y = run(x_gpu1)
        torch.cuda.synchronize()

    with Benchmark('复制到CPU'):
        y_cpu = copy_to_cpu(y)
        torch.cuda.synchronize()

    with Benchmark('在GPU1上运行并复制到CPU'):
        y = run(x_gpu1)
        y_cpu = copy_to_cpu(y, True)
        torch.cuda.synchronize()
    # 在GPU1上运行: 0.4713 sec
    # 复制到CPU: 2.4710 sec
    # 在GPU1上运行并复制到CPU: 1.8784 sec


if __name__ == '__main__':
    demo3()