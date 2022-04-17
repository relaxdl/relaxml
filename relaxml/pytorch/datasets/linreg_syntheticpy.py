from typing import List, Tuple
import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader


def synthetic_data(w: Tensor,
                   b: Tensor,
                   num_examples: int = 1000) -> Tuple[Tensor, Tensor]:
    """
    生成: y=Xw+b+噪声

    >>> num_examples = 1000
    >>> true_w = torch.tensor([2, -3.4])
    >>> true_b = 4.2
    >>> features, labels = synthetic_data(true_w, true_b, num_examples)
    >>> assert features.shape == (num_examples, 2)
    >>> assert labels.shape == (num_examples, 1)
    """
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)

    # X.shape [num_examples, num_features]
    # y.shape [num_examples, 1]
    return X, y.reshape((-1, 1))


def load_linreg_synthetic(data_arrays: List[Tensor],
                          batch_size: int,
                          is_train: bool = True) -> DataLoader:
    """
    加载线性回归数据集

    >>> num_examples = 1000
    >>> true_w = torch.tensor([2, -3.4])
    >>> true_b = 4.2
    >>> features, labels = synthetic_data(true_w, true_b, num_examples)
    >>> batch_size = 10
    >>> for X, y in load_linreg_synthetic((features, labels), batch_size):
    >>>     assert X.shape == (batch_size, 2)
    >>>     assert y.shape == (batch_size, 1)
    >>>     break
    """
    dataset = TensorDataset(*data_arrays)
    return DataLoader(dataset, batch_size, shuffle=is_train)


if __name__ == '__main__':
    num_examples = 1000
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, num_examples)
    batch_size = 10
    for X, y in load_linreg_synthetic((features, labels), batch_size):
        assert X.shape == (batch_size, 2)
        assert y.shape == (batch_size, 1)
        break