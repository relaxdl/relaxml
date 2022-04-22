from typing import Any, List, Tuple
import math
import torch
from torch import Tensor
import torch.nn as nn
import matplotlib.pyplot as plt


def show_heatmaps(matrices: Tensor,
                  xlabel: str,
                  ylabel: str,
                  titles: List[str] = None,
                  figsize: Tuple[int, int] = (3.5, 3.5),
                  cmap: str = 'Reds') -> None:
    """
    可视化注意力(显示的注意力分数)

    `num_queries`个queries和`num_keys`个keys会产生`num_queries x num_keys`个分数

    >>> attention_weights = torch.eye(10).reshape((1, 1, 10, 10)).repeat((2, 3, 1, 1))
    >>> show_heatmaps(attention_weights,
                      xlabel='Keys',
                      ylabel='Queries',
                      titles=['Title1', 'Title2', 'Title3'])

    参数:
    matrices: [num_rows, num_cols, num_queries, num_keys] attention_weights
    xlabel: e.g. Keys
    ylabel: e.g. Queries
    titles: 每一列一个标题
    figsize: 尺寸
    cmap: e.g. Reds
    """
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = plt.subplots(num_rows,
                             num_cols,
                             figsize=figsize,
                             sharex=True,
                             sharey=True,
                             squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6)
    plt.show()


def sequence_mask(X: Tensor, valid_len: Tensor, value: int = 0) -> Tensor:
    """
    在序列中屏蔽不相关的项

    >>> X = torch.tensor([[1, 2, 3], 
                          [4, 5, 6]])
    >>> sequence_mask(X, torch.tensor([1, 2]))
        tensor([[1, 0, 0],
                [4, 5, 0]])

    参数:
    X: [batch_size, num_steps]
       [batch_size, num_steps, size]
    valid_len: [batch_size,]
               [batch_size, num_steps]

    返回:
    X: [batch_size, num_steps]
       [batch_size, num_steps, size]
    """
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


def masked_softmax(X: Tensor, valid_lens: Tensor) -> Tensor:
    """
    通过在最后一个轴上遮蔽元素来执行softmax操作

    >>> masked_softmax(torch.rand(2, 2, 4), 
                       torch.tensor([2, 3]))
        tensor([[[0.4773, 0.5227, 0.0000, 0.0000],
                 [0.4483, 0.5517, 0.0000, 0.0000]],

                [[0.4079, 0.2658, 0.3263, 0.0000],
                 [0.3101, 0.2718, 0.4182, 0.0000]]])

    >>> masked_softmax(torch.rand(2, 2, 4), 
                       torch.tensor([[1, 3], 
                                     [2, 4]]))
        tensor([[[1.0000, 0.0000, 0.0000, 0.0000],
                 [0.3612, 0.2872, 0.3516, 0.0000]],

                [[0.5724, 0.4276, 0.0000, 0.0000],
                 [0.3007, 0.2687, 0.1585, 0.2721]]])

    参数:
    X: [batch_size, num_steps, size]
    valid_len: [batch_size,]
               [batch_size, num_steps]

    输出:
    output: [batch_size, num_steps, size]
    """
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            # valid_lens.shape [batch_size, ]
            #               -> [batch_size*num_steps, ]
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            # valid_lens.shape [batch_size, num_steps]
            #               -> [batch_size*num_steps, ]
            valid_lens = valid_lens.reshape(-1)

        # 在最后的轴上, 被遮蔽的元素使用一个非常大的负值替换, 从而其softmax(指数)输出为0
        # 参数:
        # X.shape [batch_size, num_steps, size]
        #      -> [batch_size*num_steps, size]
        # valid_lens.shape [batch_size*num_steps, ]
        # 最终:
        # X.shape [batch_size*num_steps, size]
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)

        # X.shape [batch_size, num_steps, size]
        return nn.functional.softmax(X.reshape(shape), dim=-1)


class AdditiveAttention(nn.Module):
    """
    加性注意力(query和key的是不同长度的矢量)

    >>> batch_size, num_queries, num_kvs, query_size, key_size, value_size = 2, 1, 10, 20, 2, 4
    >>> queries = torch.normal(0, 1, (batch_size, num_queries, query_size))
    >>> keys = torch.ones((batch_size, num_kvs, key_size))
    # values的小批量, 两个值矩阵是相同的
    >>> values = torch.arange(40, dtype=torch.float32).reshape(
            1, num_kvs, value_size).repeat(batch_size, 1, 1)
    >>> valid_lens = torch.tensor([2, 6])

    >>> attention = AdditiveAttention(
            key_size=key_size,  # 2
            query_size=query_size,  # 20
            num_hiddens=8,
            dropout=0.1)
    >>> attention.eval()
    >>> output = attention(queries, keys, values, valid_lens)
    >>> assert output.shape == (batch_size, num_queries, value_size)
    >>> output
        tensor([[[ 2.0000,  3.0000,  4.0000,  5.0000]],
                [[10.0000, 11.0000, 12.0000, 13.0000]]], grad_fn=<BmmBackward0>)
    >>> show_heatmaps(attention.attention_weights.reshape(
            (1, 1, batch_size * num_queries, num_kvs)), xlabel='Keys', ylabel='Queries')
    """

    def __init__(self, key_size: int, query_size: int, num_hiddens: int,
                 dropout: float, **kwargs: Any) -> None:
        super(AdditiveAttention, self).__init__(**kwargs)
        """
        三个需要学习的参数: W_k, W_q, w_v
        """
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        """
        参数:
        queries: [batch_size, num_queries, query_size]
        keys: [batch_size, num_kvs, key_size]
        values: [batch_size, num_kvs, value_size]
        valid_lens: [batch_size, ]
                    [batch_size, num_queries]

        输出:
        output: [batch_size, num_queries, value_size]
        """
        # queries.shape [batch_size, num_queries, num_hidden]
        # keys.shape [batch_size, num_kvs, num_hidden]
        queries, keys = self.W_q(queries), self.W_k(keys)

        # 扩展维度:
        # queries.shape [batch_size, num_queries, 1, num_hidden]
        # keys.shape [batch_size, 1, num_kvs, num_hidden]
        # 使用广播方式进行求和
        # features.shape [batch_size, num_queries, num_kvs, num_hidden]
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # scores.shape [batch_size, num_queries, num_kvs, 1]
        #           -> [batch_size, num_queries, num_kvs]
        scores = self.w_v(features).squeeze(-1)
        # self.attention_weights.shape [batch_size, num_queries, num_kvs]
        self.attention_weights = masked_softmax(scores, valid_lens)
        # output.shape: [batch_size, num_queries, value_size]
        return torch.bmm(self.dropout(self.attention_weights), values)


class DotProductAttention(nn.Module):
    """
    缩放点积注意力(query和key的是相同长度的矢量)

    >>> batch_size, num_queries, num_kvs, query_size, key_size, value_size = 2, 1, 10, 2, 2, 4
    >>> assert query_size == key_size  # 这是缩放点积注意力的要求
    >>> queries = torch.normal(0, 1, (batch_size, num_queries, query_size))
    >>> keys = torch.ones((batch_size, num_kvs, key_size))
    # values的小批量, 两个值矩阵是相同的
    >>> values = torch.arange(40, dtype=torch.float32).reshape(
            1, num_kvs, value_size).repeat(batch_size, 1, 1)
    >>> valid_lens = torch.tensor([2, 6])

    >>> attention = AdditiveAttention(
            key_size=key_size,  # 2
            query_size=query_size,  # 20
            num_hiddens=8,
            dropout=0.1)
    >>> attention.eval()
    >>> output = attention(queries, keys, values, valid_lens)
    >>> assert output.shape == (batch_size, num_queries, value_size)
    >>> output
        tensor([[[ 2.0000,  3.0000,  4.0000,  5.0000]],
                [[10.0000, 11.0000, 12.0000, 13.0000]]], grad_fn=<BmmBackward0>)
    >>> show_heatmaps(attention.attention_weights.reshape(
            (1, 1, batch_size * num_queries, num_kvs)), xlabel='Keys', ylabel='Queries')
    """

    def __init__(self, dropout: float, **kwargs: Any) -> None:
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        """
        参数:
        queries: [batch_size, num_queries, d]
        keys: [batch_size, num_kvs, d]
        values: [batch_size, num_kvs, value_size]
        valid_lens: [batch_size, ]
                    [batch_size, num_queries]

        输出:
        output: [batch_size, num_queries, value_size]
        """
        # d =
        d = queries.shape[-1]
        # scores.shape [batch_size, num_queries, num_kvs]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        # self.attention_weights.shape [batch_size, num_queries, num_kvs]
        self.attention_weights = masked_softmax(scores, valid_lens)
        # output.shape: [batch_size, num_queries, value_size]
        return torch.bmm(self.dropout(self.attention_weights), values)


if __name__ == '__main__':
    pass