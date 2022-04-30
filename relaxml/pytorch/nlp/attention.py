from typing import Any, List, Tuple
import math
import torch
from torch import Tensor
import torch.nn as nn
import matplotlib.pyplot as plt
"""
注意力

实现说明:
https://tech.foxrelax.com/nlp/attention/
"""


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

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor,
                valid_lens: Tensor) -> Tensor:
        """
        参数:
        queries: [batch_size, num_queries, query_size]
        keys: [batch_size, num_kvs, key_size]
        values: [batch_size, num_kvs, value_size]
        valid_lens: [batch_size, ] 在计算注意力的时候需要看多少个k/v pairs
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
        # output.shape [batch_size, num_queries, value_size]
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

    def forward(self,
                queries: Tensor,
                keys: Tensor,
                values: Tensor,
                valid_lens: Tensor = None) -> Tensor:
        """
        参数:
        queries: [batch_size, num_queries, d]
        keys: [batch_size, num_kvs, d]
        values: [batch_size, num_kvs, value_size]
        valid_lens: [batch_size, ] 在计算注意力的时候需要看多少个k/v pairs
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
        # output.shape [batch_size, num_queries, value_size]
        return torch.bmm(self.dropout(self.attention_weights), values)


class MultiHeadAttention(nn.Module):
    """
    实现多头注意力

    >>> query_size, key_size, value_size, num_hiddens, num_heads = 5, 4, 3, 100, 5
    >>> attention = MultiHeadAttention(key_size, query_size, value_size,
                                       num_hiddens, num_heads, 0.5)
    >>> attention.eval()
    >>> batch_size, num_queries, num_kvs = 2, 4, 6
    >>> valid_lens = torch.tensor([3, 2])
    >>> Queries = torch.ones((batch_size, num_queries, query_size))
    >>> Keys = torch.ones((batch_size, num_kvs, key_size))
    >>> Values = torch.ones((batch_size, num_kvs, value_size))
    >>> output = attention(Queries, Keys, Values, valid_lens)
    >>> assert output.shape == (batch_size, num_queries, num_hiddens)
    """

    def __init__(self,
                 key_size: int,
                 query_size: int,
                 value_size: int,
                 num_hiddens: int,
                 num_heads: int,
                 dropout: float,
                 bias: bool = False,
                 **kwargs: Any) -> None:
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads

        # 缩放点积注意力作为每一个注意力头
        self.attention = DotProductAttention(dropout)

        # 注意:
        # 每个头应该有一个单独的W_q, W_k, W_v, 在这里我们为了实现多个头的`并行计算`,
        # 将num_heads个头的W_q, W_k, W_v合并到一起, 这样多个头可以`并行计算`, 效率更高
        #
        # 举例说明:
        # 如果有8个头, 我们每个头会有24个矩阵:
        # W_q_1, W_q_2, ....W_q_8, 形状为: [query_size, num_hiddens/8]
        # W_k_1, W_k_2, ....W_k_8, 形状为: [key_size, num_hiddens/8]
        # W_v_1, W_v_2, ....W_v_8, 形状为: [value_size, num_hiddens/8]
        #
        # 当前的并行版本将8个头的24个矩阵合并为3个矩阵:
        # W_q, 形状为: [query_size, num_hiddens]
        # W_k, 形状为: [key_size, num_hiddens]
        # W_v, 形状为: [value_size, num_hiddens]
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor,
                valid_lens: Tensor) -> Tensor:
        """
        当前的版本是效率比较高的实现方式, 我们没有单独计算每个头, 
        而是通过变换, 并行的计算多个头

        参数:
        queries: [batch_size, num_queries, query_size]
        keys: [batch_size, num_kvs, key_size]
        values: [batch_size, num_kvs, value_size]
        valid_lens: [batch_size, ] 在计算注意力的时候需要看多少个k/v pairs
                    [batch_size, num_queries]

        输出:
        output: [batch_size, num_queries, num_hiddens]
        """
        # queries.shape [batch_size, num_queries, num_hiddens]
        #            -> [batch_size*num_heads, num_queries, num_hiddens/num_heads]
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        # keys.shape [batch_size, num_kvs, num_hiddens]
        #            -> [batch_size*num_heads, num_kvs, num_hiddens/num_heads]
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        # values.shape [batch_size, num_kvs, num_hiddens]
        #            -> [batch_size*num_heads, num_kvs, num_hiddens/num_heads]
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # [batch_size, ] -> [batch_size*num_heads, ]
            # [batch_size, num_queries] -> [batch_size*num_heads, num_queries]
            valid_lens = torch.repeat_interleave(valid_lens,
                                                 repeats=self.num_heads,
                                                 dim=0)

        # 缩放点积注意力
        # output.shape [batch_size*num_heads, num_queries, num_hiddens/num_heads]
        output = self.attention(queries, keys, values, valid_lens)

        # output_concat.shape [batch_size, num_queries, num_hiddens]
        output_concat = transpose_output(output, self.num_heads)

        # output.shape [batch_size, num_queries, num_hiddens]
        return self.W_o(output_concat)


def transpose_qkv(X: Tensor, num_heads: int) -> Tensor:
    """
    参数:
    X: [batch_size, num_qkv, num_hiddens]
    num_heads: 头数

    输出:
    output: [batch_size*num_heads, num_qkv, num_hiddens/num_heads]
    """

    # X.shape [batch_size, num_qkv, num_heads, num_hiddens/num_heads]
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # X.shape [batch_size, num_heads, num_qkv, num_hiddens/num_heads]
    X = X.permute(0, 2, 1, 3)

    # output.shape [batch_size*num_heads, num_qkv, num_hiddens/num_heads]
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X: Tensor, num_heads: int) -> Tensor:
    """
    逆转`transpose_qkv`函数的操作

    参数:
    X: [batch_size*num_heads, num_qkv, num_hiddens/num_heads]
    num_heads: 头数

    输出:
    output: [batch_size, num_qkv, num_hiddens]
    """
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class PositionalEncoding(nn.Module):
    """
    位置编码
    
    >>> batch_size, num_hiddens, num_steps = 1, 32, 60
    >>> pos_encoding = PositionalEncoding(num_hiddens, 0)
    >>> pos_encoding.eval()
    >>> X = pos_encoding(torch.zeros((batch_size, num_steps, num_hiddens)))
    >>> assert X.shape == (batch_size, num_steps, num_hiddens)
    >>> P = pos_encoding.P[:, :X.shape[1], :]
    >>> assert P.shape == (1, num_steps, num_hiddens)
    >>> P = P[0, :, :].unsqueeze(0).unsqueeze(0)
    >>> assert P.shape == (1, 1, num_steps, num_hiddens)
    >>> show_heatmaps(P, xlabel='Column (encoding dimension)',
                      ylabel='Row (position)', figsize=(3.5, 4),
                      cmap='Blues')
    """

    def __init__(self,
                 num_hiddens: int,
                 dropout: float,
                 max_len: int = 1000) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的`P`
        # P.shape [1, max_len, num_hiddens]
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(
                10000,
                torch.arange(0, num_hiddens, 2, dtype=torch.float32) /
                num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X: Tensor) -> Tensor:
        """
        参数: 
        X: [batch_size, num_steps, num_hiddens]
        
        返回:
        output: [batch_size, num_steps, num_hiddens]
        """
        # P.shape [1, max_len, num_hiddens]
        # 在相加的时候, P在第一个维度可以通过广播来进行计算
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


if __name__ == '__main__':
    pass