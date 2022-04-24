from typing import Union, List, Dict, Tuple, Any
import collections
import os
import time
import math
import zipfile
import tarfile
import requests
import random
import hashlib
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm
"""
Bert从零实现

实现说明:
https://tech.foxrelax.com/nlp/bert_scratch/
"""


class PositionWiseFFN(nn.Module):
    """
    基于位置的前馈网络

    >>> ffn_num_input, ffn_num_hiddens, ffn_num_outputs = 6, 5, 4
    >>> batch_size, num_steps = 2, 10
    >>> ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, ffn_num_outputs)
    >>> X = torch.randn((batch_size, num_steps, ffn_num_input))
    >>> assert ffn(X).shape == (batch_size, num_steps, ffn_num_outputs)
    """

    def __init__(self, ffn_num_input: int, ffn_num_hiddens: int,
                 ffn_num_outputs: int, **kwargs: Any) -> None:
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X: Tensor) -> Tensor:
        """
        参数:
        X: [batch_size, num_steps, ffn_num_input]

        返回:
        output: [batch_size, num_steps, ffn_num_outputs]
        """
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(nn.Module):
    """
    Add & Norm

    >>> batch_size, num_steps, num_hiddens = 12, 10, 8
    >>> normalized_shape = (num_steps, num_hiddens)
    >>> addnorm = AddNorm(normalized_shape, 0.1)
    >>> X = torch.randn((batch_size, num_steps, num_hiddens))
    >>> Y = torch.randn((batch_size, num_steps, num_hiddens))
    >>> assert addnorm(X, Y).shape == (batch_size, num_steps, num_hiddens)
    """

    def __init__(self, normalized_shape: Tuple[int, int], dropout: float,
                 **kwargs: Any) -> None:
        """
        参数:
        normalized_shape: (num_steps, num_hiddens)
        dropout: dropout操作设置为0的元素的概率
        """
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        """
        将相同形状的X, Y按元素相加, 然后再做一次LayerNorm后输出,
        输出的形状和X, Y的形状是一样的

        参数:
        X: [batch_size, num_steps, num_hiddens]
        Y: [batch_size, num_steps, num_hiddens]

        返回:
        output: [batch_size, num_steps, num_hiddens]
        """
        assert X.shape == Y.shape
        return self.ln(self.dropout(Y) + X)


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
        # output.shape: [batch_size, num_queries, value_size]
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


class EncoderBlock(nn.Module):
    """
    >>> key_size, query_size, value_size, num_hiddens, num_steps = 24, 24, 24, 24, 100
    >>> batch_size, ffn_num_input, ffn_num_hiddens, num_heads = 2, 24, 48, 8
    >>> norm_shape = (num_steps, num_hiddens)
    >>> encoder_blk = EncoderBlock(key_size, query_size, value_size, num_hiddens,
                                   norm_shape, ffn_num_input, ffn_num_hiddens,
                                   num_heads, 0.1)
    >>> X = torch.randn((batch_size, num_steps, num_hiddens))
    >>> valid_lens = torch.tensor([3, 2])
    >>> assert encoder_blk(X, valid_lens).shape == (batch_size, num_steps,
                                                    num_hiddens)
    """

    def __init__(self,
                 key_size: int,
                 query_size: int,
                 value_size: int,
                 num_hiddens: int,
                 norm_shape: Tuple[int, int],
                 ffn_num_input: int,
                 ffn_num_hiddens: int,
                 num_heads: int,
                 dropout: float,
                 use_bias: bool = False,
                 **kwargs: Any) -> None:
        """
        参数:
        key_size: key的特征长度
        query_size: query的特征长度
        value_size: value的特征长度
        num_hiddens: 隐藏单元的特征长度(也是多头注意力最终输出的特征长度)
        norm_shape: (num_steps, num_hiddens)
        ffn_num_input: ffn输入的维度
        ffn_num_hiddens: ffn隐藏层的维度
        num_heads: 头数
        dropout: dropout操作设置为0的元素的概率
        use_bias: 是否开启bias
        """
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(key_size, query_size, value_size,
                                            num_hiddens, num_heads, dropout,
                                            use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X: Tensor, valid_lens: Tensor) -> Tensor:
        """
        参数:
        X: [batch_size, num_steps, num_hiddens]
        valid_lens的形状: [batch_size, ] 表示X对应的有效token个数

        返回:
        output的形状: [batch_size, num_steps, num_hiddens]
        """
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))


class BERTEncoder(nn.Module):
    """
    BERT编码器

    本质就是改进版本的Transformer Encoder

    >>> vocab_size, key_size, query_size, value_size, num_hiddens, num_steps = 10000, 24, 24, 24, 24, 100
    >>> batch_size, ffn_num_input, ffn_num_hiddens, num_heads, num_layers, max_len = 2, 24, 48, 8, 5, 1000
    >>> norm_shape = (num_steps, num_hiddens)
    >>> encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input,
                              ffn_num_hiddens, num_heads, num_layers, 0.1, max_len,
                              key_size, query_size, value_size)
    >>> tokens = torch.ones((batch_size, num_steps), dtype=torch.long)
    >>> segments = torch.ones((batch_size, num_steps), dtype=torch.long)
    >>> valid_lens = torch.tensor([3, 2])
    >>> assert encoder(tokens, segments,
                       valid_lens).shape == (batch_size, num_steps, num_hiddens)
    """

    def __init__(self,
                 vocab_size: int,
                 num_hiddens: int,
                 norm_shape: Tuple[int, int],
                 ffn_num_input: int,
                 ffn_num_hiddens: int,
                 num_heads: int,
                 num_layers: int,
                 dropout: float,
                 max_len: int = 1000,
                 key_size: int = 768,
                 query_size: int = 768,
                 value_size: int = 768,
                 **kwargs: Any) -> None:
        """
        参数:
        vocab_size: 字典大小
        num_hiddens: Transformer EncoderBlock隐藏单元的特征长度(也是多头注意力最终输出的特征长度)
        norm_shape: (num_steps, num_hiddens)
        ffn_num_input: ffn输入的维度
        ffn_num_hiddens: ffn隐藏层的维度
        num_heads: 头数
        num_layers: EncoderBlock的数量
        dropout: dropout操作设置为0的元素的概率
        max_len: Pos Embedding生成的向量的最大长度
        key_size: key的特征长度
        query_size: query的特征长度
        value_size: value的特征长度
        """
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                f"{i}",
                EncoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, True))

        # 在BERT中, 位置嵌入是可学习的, 因此我们创建一个足够长的位置嵌入参数
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, num_hiddens))

    def forward(self, tokens: Tensor, segments: Tensor,
                valid_lens: Tensor) -> Tensor:
        """
        参数:
        tokens: [batch_size, num_steps]
        segments: [batch_size, num_steps]
        valid_lens: [batch_size, ], 表示tokens对应的有效token个数

        返回:
        output: [batch_size, num_steps, num_hiddens]
        """

        # X.shape [batch_size, num_steps, num_hiddens]
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X


class MaskLM(nn.Module):
    """
    BERT的Masked Language Modeling

    >>> batch_size, num_steps = 2, 8
    >>> vocab_size, num_hiddens = 10000, 768
    >>> mlm = MaskLM(vocab_size, num_hiddens)
    >>> encoded_X = torch.randn((batch_size, num_steps, num_hiddens))
    >>> mlm_positions = torch.tensor([[1, 5, 2], [6, 1, 5]])
    >>> mlm_Y_hat = mlm(encoded_X, mlm_positions)
    >>> assert mlm_Y_hat.shape == (batch_size, 3, vocab_size)
    """

    def __init__(self,
                 vocab_size: int,
                 num_hiddens: int,
                 num_inputs: int = 768,
                 **kwargs: Any) -> None:
        """
        参数:
        vocab_size: 字典大小
        num_hiddens: 隐藏层大小
        num_inputs: 输入的维度
        """
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens), nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, vocab_size))

    def forward(self, X: Tensor, pred_positions: Tensor) -> Tensor:
        """
        做2两件事情:
        1. 把X中pred_positions对应的特征`masked_X`抽取出来
           masked_X.shape [batch_size, num_steps, num_hiddens]
        2. 将抽取出来的特征送入MLP, 处理成最终输出: [batch_size, num_pred, vocab_size]

        参数:
        X: [batch_size, num_steps, num_hiddens]
        pred_positions: [batch_size, num_pred]

        返回:
        output: [batch_size, num_pred, vocab_size]
        """
        num_pred_positions = pred_positions.shape[1]  # num_pred
        # pred_positions.shape [batch_size*num_pred, ]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        # batch_idx.shape [batch_size, ]
        batch_idx = torch.arange(0, batch_size)
        # 假设: batch_size=2，num_pred_positions=3
        # 则: batch_idx = tensor([0, 0, 0, 1, 1, 1])
        # batch_idx.shape [batch_size*num_pred, ]
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)

        # masked_X.shape [batch_size*num_pred, num_hiddens]
        masked_X = X[batch_idx, pred_positions]
        # masked_X.shape [batch_size, num_pred, num_hiddens]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        # mlm_Y_hat.shape [batch_size, num_pred, vocab_size]
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat


class NextSentencePred(nn.Module):
    """
    BERT的下一句预测任务
    
    >>> batch_size, num_steps, num_hiddens = 2, 8, 768
    >>> nsp = NextSentencePred(num_hiddens)
    >>> encoded_X = torch.randn((batch_size, num_steps, num_hiddens))
    # 0是'<cls>'标记的索引
    >>> nsp_Y_hat = nsp(encoded_X[:, 0, :])
    >>> assert nsp_Y_hat.shape == (batch_size, 2)
    """

    def __init__(self, num_hiddens: int, **kwargs: Any) -> None:
        """
        参数:
        num_hiddens: 输入的维度
        """
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Linear(num_hiddens, 2)

    def forward(self, X: Tensor) -> Tensor:
        """
        参数:
        X: [batch_size, num_hiddens]

        返回:
        output: [batch_size, 2]
        """
        return self.output(X)


class BERTModel(nn.Module):
    """
    BERT模型
    """

    def __init__(self,
                 vocab_size: int,
                 num_hiddens: int,
                 norm_shape: int,
                 ffn_num_input: int,
                 ffn_num_hiddens: int,
                 num_heads: int,
                 num_layers: int,
                 dropout: float,
                 max_len: int = 1000,
                 key_size: int = 768,
                 query_size: int = 768,
                 value_size: int = 768,
                 hid_in_features: int = 768,
                 mlm_in_features: int = 768,
                 nsp_in_features: int = 768) -> None:
        """
        参数:
        vocab_size: 字典大小
        num_hiddens: Transformer EncoderBlock隐藏单元的特征长度(也是多头注意力最终输出的特征长度)
        norm_shape: (num_steps, num_hiddens)
        ffn_num_input: ffn输入的维度
        ffn_num_hiddens: ffn隐藏层的维度
        num_heads: 头数
        num_layers: EncoderBlock的数量
        dropout: dropout操作设置为0的元素的概率
        max_len: Pos Embedding生成的向量的最大长度
        key_size: key的特征长度
        query_size: query的特征长度
        value_size: value的特征长度
        hid_in_features: hidden层输入的维度
        mlm_in_features: MLM输入的维度
        nsp_in_features: NSP输入的维度
        """
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size,
                                   num_hiddens,
                                   norm_shape,
                                   ffn_num_input,
                                   ffn_num_hiddens,
                                   num_heads,
                                   num_layers,
                                   dropout,
                                   max_len=max_len,
                                   key_size=key_size,
                                   query_size=query_size,
                                   value_size=value_size)
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens),
                                    nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(
            self,
            tokens: Tensor,
            segments: Tensor,
            valid_lens: Tensor = None,
            pred_positions: Tensor = None) -> Tuple[Tensor, Tensor, Tensor]:
        """
        逻辑:
        1. 将tokens送入BERTEncoder提取特征, 得到encoded_X: 
           [batch_size, num_steps, num_hiddens]
        2. 将提取出来的特征encoded_X和pred_positions送入MLM, 得到MLM的输出: 
           [batch_size, num_pred, vocab_size]
        3. 将提取出来的特征encoded_X[:, 0, :])形状为[batch_size, num_hiddens], 
           也就是每个句子的<cls>, 送入NSP, 得到NSP的输出: [batch_size, 2]

        参数:
        tokens: [batch_size, num_steps]
        segments: [batch_size, num_steps]
        valid_lens: [batch_size, ], 表示tokens对应的有效token个数
        pred_positions: [batch_size, num_pred]

        返回: (encoded_X, mlm_Y_hat, nsp_Y_hat)
        encoded_X: [batch_size, num_steps, num_hiddens]
        mlm_Y_hat: [batch_size, num_pred, vocab_size]
        nsp_Y_hat: [batch_size, 2]
        """

        # encoded_X.shape [batch_size, num_steps, num_hiddens]
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            # mlm_Y_hat.shape [batch_size, num_pred, vocab_size]
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None

        # 用于下一句预测的多层感知机分类器的隐藏层, 0是'<cls>'标记的索引
        # 将[batch_size, num_hiddens]的数据送入hidden -> [batch_size, num_hiddens]
        # 将[batch_size, num_hiddens]的数据送入nsp -> [batch_size, 2]
        # nsp_Y_hat.shape [batch_size, 2]
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat


def _get_batch_loss_bert(net: BERTModel, loss: nn.CrossEntropyLoss,
                         vocab_size: int, tokens_X: Tensor, segments_X: Tensor,
                         valid_lens_x: Tensor, pred_positions_X: Tensor,
                         mlm_weights_X: Tensor, mlm_Y: Tensor,
                         nsp_y: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """
    针对一个批量的数据:
    1. BertModel前向传播
    2. 计算MLM Loss
    3. 计算NSP Loss
    4. 计算最终Loss = MLM Loss + NPS Loss

    参数:
    net: BERTModel实例
    loss: nn.CrossEntropyLoss实例
    vocab_size: 字典大小
    tokens_X: [batch_size, num_steps]
    segments_X: [batch_size, num_steps]
    valid_lens_x: [batch_size, ]
    pred_positions_X: [batch_size, num_preds]
    mlm_weights_X: [batch_size, num_preds]
    mlm_Y: [batch_size, num_preds]
    nsp_y: [batch_size, ]

    返回: (mlm_l, nsp_l, l)
    mlm_l: MLM Loss 均值
    nsp_l: NSP Loss 均值
    l: mlm_l+nsp_l
    """
    # 前向传播
    # mlm_Y_hat.shape [batch_size, num_preds, vocab_size]
    # nsp_Y_hat.shape [batch_size, 2]
    _, mlm_Y_hat, nsp_Y_hat = net(tokens_X, segments_X,
                                  valid_lens_x.reshape(-1), pred_positions_X)
    # 计算遮蔽语言模型损失
    # 将mlm_Y_hat处理成 [batch_size*num_preds, vocab_size]
    # 将mlm_Y处理成 [batch_size*num_preds,]
    # 将mlm_weights_X处理成 [batch_size*num_preds,]
    # mlm_l处理成 [batch_size*num_preds,]
    loss.reduction = 'none'
    # mlm_l.shape [batch_size*num_preds,]
    mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size),
                 mlm_Y.reshape(-1)) * mlm_weights_X.reshape(-1, )
    # mlm_l标量(自己计算mean)
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)

    # 计算下一句子预测任务的损失
    loss.reduction = 'mean'
    # nsp_l标量
    nsp_l = loss(nsp_Y_hat, nsp_y)
    l = mlm_l + nsp_l  # MLM Loss和NSP Loss可以分别乘以weight再相加, 我们实现的版本直接相加了
    return mlm_l, nsp_l, l


def download(cache_dir: str = '../data') -> str:
    """
    下载数据
    """
    sha1_hash = '81d2333b501a1d8c32bfe96859e2490991fee293'
    url = 'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/wikitext_2.zip'
    fname = os.path.join(cache_dir, url.split('/ml/')[-1])
    fdir = os.path.dirname(fname)
    os.makedirs(fdir, exist_ok=True)
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname
    print(f'download {url} -> {fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    print(f'download {fname} success!')
    # e.g. ../data/wikitext_2.zip
    return fname


def download_extract(cache_dir: str = '../data') -> str:
    """
    下载数据 & 解压
    """
    # 下载数据集
    fname = download(cache_dir)

    # 解压
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, 'Only zip/tar files can be extracted.'
    fp.extractall(base_dir)
    # e.g. ../data/wikitext_2
    return data_dir


def tokenize(lines: List[str], token='word') -> List[List[str]]:
    """
    >>> lines = ['this moive is great', 'i like it']
    >>> tokenize(lines)
    [['this', 'moive', 'is', 'great'], 
     ['i', 'like', 'it']]
    """
    assert token in ('word', 'char'), 'Unknown token type: ' + token
    return [line.split() if token == 'word' else list(line) for line in lines]


def count_corpus(tokens: Union[List[str], List[List[str]]]) -> Dict[str, int]:
    """
    统计token的频率
    
    e.g.
    Counter({'the': 2261, 'i': 1267, 'and': 1245, 'of': 1155...})
    """
    # Flatten a 2D list if needed
    if tokens and isinstance(tokens[0], list):
        # 将词元列表展平成使用词元填充的一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


class Vocab:
    """
    文本词表
    
    token内部的排序方式:
    <unk>, reserved_tokens, 其它按照token出现的频率从高到低排序
    """

    def __init__(self,
                 tokens: Union[List[str], List[List[str]]] = None,
                 min_freq: int = 0,
                 reserved_tokens: List[str] = None) -> None:
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序(出现频率最高的排在最前面)
        counter = count_corpus(tokens)
        self.token_freqs = sorted(counter.items(),
                                  key=lambda x: x[1],
                                  reverse=True)
        # <unk>的索引为0
        self.unk, uniq_tokens = 0, ['<unk>'] + reserved_tokens
        uniq_tokens += [
            token for token, freq in self.token_freqs
            if freq >= min_freq and token not in uniq_tokens
        ]
        self.idx_to_token, self.token_to_idx = [], dict()
        for token in uniq_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self) -> int:
        return len(self.idx_to_token)

    def __getitem__(self, tokens: Union[str, List[str],
                                        Tuple[str]]) -> List[int]:
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices: Union[int, List[int],
                                       Tuple[int]]) -> List[str]:
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]


def _read_wiki(data_dir: str) -> List[List[str]]:
    """
    
    1. paragraphs是一个paragraph列表
    2. 每个paragraph包含多个sentence(每个sentence可以分成多个tokens, 
       这里返回的sentence还没有进行分词)

    >>> paragraphs = _read_wiki(download_extract('wikitext_2'))
    >>> len(paragraphs)
        15496
    # 返回第1个paragraph
    >>> paragraphs[0]
       ['when he died at the age of 78 , the daily telegraph , 
        guardian and times published his obituary , and the museum of 
        london added his pamphlets and placards to their collection', 
        'in 2006 his biography was included in the oxford dictionary of 
        national biography .']
    # 返回第1个paragraph的第1个sentence 
    >>> paragraphs[0][0]
        when he died at the age of 78 , the daily telegraph , 
        guardian and times published his obituary , and the museum of 
        london added his pamphlets and placards to their collection
    """
    file_name = os.path.join(data_dir, 'wiki.train.tokens')
    with open(file_name, 'r') as f:
        lines = f.readlines()
    # 大写字母转换为小写字母
    paragraphs = [
        line.strip().lower().split(' . ') for line in lines
        if len(line.split(' . ')) >= 2
    ]
    random.shuffle(paragraphs)  # 随机打乱段落的顺序
    return paragraphs


def get_tokens_and_segments(
        tokens_a: List[str],
        tokens_b: List[str] = None) -> Tuple[List[str], List[int]]:
    """
    将tokens_a和tokens_b拼接起来, 返回拼接后的tokens及其segments

    >>> tokens_a = ['this', 'movie', 'is', 'great']
    >>> tokens_b = ['i', 'like', 'it']
    >>> tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)
    >>> tokens
    ['<cls>', 'this', 'movie', 'is', 'great', '<sep>', 'i', 'like', 'it', '<sep>']
    >>> segments
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
    """
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0和1分别标记片段A和B
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments


def _get_next_sentence(
        sentence: List[str], next_sentence: List[str],
        paragraphs: List[List[List[str]]]
) -> Tuple[List[str], List[str], bool]:
    """
    生成NSP任务的训练样本

    50%的概率返回下一个句子, 50%的概率返回随机句子. 也就是生成正样本和
    负样本的数量是一致的

    输入: 
    sentence: e.g. ['this', 'movie', 'is', 'great']
    next_sentence: e.g. ['i', 'like', 'it']
    paragraphs: list of paragraph

    返回: (sentence, next_sentence, is_next)
    sentence: e.g. ['this', 'movie', 'is', 'great']
    next_sentence: e.g. ['i', 'like', 'it']
    is_next: True | False
    """
    if random.random() < 0.5:
        is_next = True
    else:
        # 先随机选择一个paragraph, 在从这个paragraph中随机选择一个sentence
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next


def _get_nsp_data_from_paragraph(
        paragraph: List[List[str]], paragraphs: List[List[List[str]]],
        vocab: Vocab, max_len: int) -> List[Tuple[List[str], List[int], bool]]:
    """
    处理一个paragraph, 返回训练NSP的训练样本

    参数:
    paragraph: 句子列表, 其中每个句子都是token列表
        e.g. [['this', 'movie', 'is', 'great'], 
              ['i', 'like', 'it']]
    paragraphs: list of paragraph
    vocab: 字典
    max_len: 预训练期间的BERT输入序列的最大长度(超过最大长度的tokens忽略掉)

    返回: list of (tokens, segments, is_next)
    tokens: e.g. ['<cls>', 'this', 'movie', 'is', 'great', '<sep>', 'i', 'like', 'it', '<sep>']
    segments: e.g. [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
    is_next: True | False
    """
    nsp_data_from_paragraph = []  # [(tokens, segments, is_next)]
    for i in range(len(paragraph) - 1):
        tokens_a, tokens_b, is_next = _get_next_sentence(
            paragraph[i], paragraph[i + 1], paragraphs)
        # 考虑1个'<cls>'和2个'<sep>', 超过最大长度max_len的tokens忽略掉
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph


def _replace_mlm_tokens(
        tokens: List[str], candidate_pred_positions: List[int],
        num_mlm_preds: int,
        vocab: Vocab) -> Tuple[List[str], List[Tuple[int, str]]]:
    """
    处理一句话(tokens), 返回`MLM的输入, 预测位置以及标签`

    参数:
    tokens: 表示BERT输入序列的token列表
        e.g. ['<cls>', 'this', 'movie', 'is', 'great', '<sep>', 'i', 'like', 'it', '<sep>']
    candidate_pred_positions: 后选预测位置的索引, 会在tokens中过滤掉<cls>, <sep>, 剩下的都算后选预测位置
        (特殊token <cls>, <sep>在MLM任务中不被预测)
        e.g. ['<cls>', 'this', 'movie', 'is', 'great', '<sep>', 'i', 'like', 'it', '<sep>']对应的
        后选预测位置是: [1,2,3,4,6,7,8]
    num_mlm_preds: 需要预测多少个token, 通常是len(tokens)的15%
    vocab: 字典

    返回: (mlm_input_tokens, pred_positions_and_labels)
    mlm_input_tokens: 处理后的tokens, 15%的tokens已经做了替换
        e.g. ['<cls>', 'this', 'movie', '<mask>', 'great', '<sep>', 'i', 'like', 'it', '<sep>']
    pred_positions_and_labels: list of (mlm_pred_position, token)
        mlm_pred_position: 需要预测的位置, e.g. 3
        token: 需要预测的标签, e.g. 'is'
    """
    # 为遮蔽语言模型的输入创建新的token副本，其中输入可能包含替换的'<mask>'或随机token
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []  # [(mlm_pred_position, token)]
    # 打乱后用于在遮蔽语言模型任务中获取15%的随机token进行预测
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            # 已经预测足够的tokens, 返回
            break
        masked_token = None
        # 80%的时间: 将token替换为<mask>
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            # 10%的时间: 保token不变
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            # 10%的时间: 用随机token替换该token
            else:
                masked_token = random.choice(vocab.idx_to_token)
        mlm_input_tokens[mlm_pred_position] = masked_token  # 替换成masked_token
        pred_positions_and_labels.append(
            (mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels


def _get_mlm_data_from_tokens(
        tokens: List[str],
        vocab: Vocab) -> Tuple[List[int], List[int], List[int]]:
    """
    处理一个tokens, 返回训练MLM的数据

    参数:
    tokens: e.g. ['<cls>', 'this', 'movie', 'is', 'great', '<sep>', 'i', 'like', 'it', '<sep>']
    vocab: 字典

    返回: (mlm_input_tokens_ids, pred_positions, mlm_pred_labels_ids)
    mlm_input_tokens_ids: 输入tokens的索引
        e.g. vocab[['<cls>', 'this', 'movie', '<mask>', 'great', '<sep>', 'i', 'like', 'it', '<sep>']]
    pred_positions: 需要预测的位置索引, e.g. [3, ...]
    mlm_pred_labels_ids: 预测的标签索引, e.g. vocab[['is', ...]]
    """
    candidate_pred_positions = []  # list of int
    for i, token in enumerate(tokens):
        # 在遮蔽语言模型任务中不会预测特殊token
        if token in ['<cls>', '<sep>']:
            continue
        candidate_pred_positions.append(i)
    # 遮蔽语言模型任务中预测15%的随机token
    num_mlm_preds = max(1, round(len(tokens) * 0.15))

    # mlm_input_tokens: 处理后的tokens, 15%的tokens已经做了替换
    #   e.g. ['<cls>', 'this', 'movie', '<mask>', 'great', '<sep>', 'i', 'like', 'it', '<sep>']
    # pred_positions_and_labels: list of (mlm_pred_position, token)
    #   mlm_pred_position: 需要预测的位置, e.g. 3
    #   token: 需要预测的标签, e.g. 'is'
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(
        tokens, candidate_pred_positions, num_mlm_preds, vocab)
    pred_positions_and_labels = sorted(pred_positions_and_labels,
                                       key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels
                      ]  # list of int, e.g. [3, ...]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels
                       ]  # list of token, e.g. ['is', ...]
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]


def _pad_bert_inputs(
    examples: List[Tuple[List[int], List[int], List[int], List[int],
                         bool]], max_len: int, vocab: Vocab
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    填充(pad)样本

    参数:
    examples: [(mlm_input_tokens_ids, pred_positions, mlm_pred_labels_ids, segments, is_next)]
        mlm_input_tokens_ids: e.g. vocab[['<cls>', 'this', 'movie', '<mask>', 'great', '<sep>', 'i', 'like', 'it', '<sep>']]
        pred_positions: e.g. [3, ...]
        mlm_pred_labels_ids: e.g. vocab[['is', ...]]
        segments: e.g. [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
        is_next: True | False
    max_len: 最大长度
    vocab: 字典

    返回: (all_token_ids, all_segments, valid_lens, all_pred_positions,
           all_mlm_weights, all_mlm_labels, nsp_labels)
    all_token_ids: [num_examples, max_len], 每个token_ids长度为max_len, 长度不足的用<pad>补足
    all_segments: [num_examples, max_len], 每个segments的长度为max_len, 长度不足的用0补足
    valid_lens: [num_examples, ], 每个token_ids的有效长度, 不包括<pad>
    all_pred_positions: [num_examples, max_num_mlm_preds], 每个pred_positions长度为max_num_mlm_preds, 长度不足的用0补足
    all_mlm_weights: [num_examples, max_num_mlm_preds], 有效的pred_positions对应的权重为1, 填充对应的权重为0
    all_mlm_labels: [num_examples, max_num_mlm_preds], 每个pred_label_ids长度为max_num_mlm_preds, 长度不足的用0补足
    nsp_labels: [num_examples, ]
    """
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens, = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,
         is_next) in examples:
        all_token_ids.append(
            torch.tensor(token_ids + [vocab['<pad>']] *
                         (max_len - len(token_ids)),
                         dtype=torch.long))
        all_segments.append(
            torch.tensor(segments + [0] * (max_len - len(segments)),
                         dtype=torch.long))
        # valid_lens不包括'<pad>'的计数
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
        all_pred_positions.append(
            torch.tensor(pred_positions + [0] *
                         (max_num_mlm_preds - len(pred_positions)),
                         dtype=torch.long))
        # 填充token的预测将通过乘以0权重在损失中过滤掉
        all_mlm_weights.append(
            torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] *
                         (max_num_mlm_preds - len(pred_positions)),
                         dtype=torch.float32))
        all_mlm_labels.append(
            torch.tensor(mlm_pred_label_ids + [0] *
                         (max_num_mlm_preds - len(mlm_pred_label_ids)),
                         dtype=torch.long))
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)


class _WikiTextDataset(Dataset):
    """
    数据处理流程:

    1. 分词, 将原始文本数据处理成`paragraphs`和`sentences`
       <1> paragraphs: List[List[List[str]]] paragraphs -> paragraph -> sentence
       <2> sentences: List[List[str]] sentences -> sentence [去掉了段落的概念]
    2. 使用sentences构建Vocab
    3. 遍历每一个paragraph, 获取下一句子预测任务(NSP)的训练样本(exmaples)
    4. 遍历步骤3的每一个样本(examples), 获取遮蔽语言模型任务(MLM)的训练数据, 拼接到一起
    5. 填充(pad)样本
    """

    def __init__(self, paragraphs: List[List[str]], max_len: int) -> None:
        """
        参数:
        paragraphs: 段落的列表, 每个元素是多个句子列表, e.g. ['this moive is great', 'i like it']
        max_len: 最大长度
        """
        # 1. 分词, 将原始文本数据处理成`paragraphs`和`sentences`
        # 处理前的paragraphs[i]表示句子的列表,
        # e.g. ['this moive is great', 'i like it']
        # 经过处理后的paragraphs[i]表示一个段落句子的token列表,
        # e.g. [['this', 'movie', 'is', 'great'], ['i', 'like', 'it']]
        # paragraphs List[List[List[str]]]
        paragraphs = [
            tokenize(paragraph, token='word') for paragraph in paragraphs
        ]
        # 经过处理后的sentences[i]表示一个句子的token列表, e.g. ['this', 'movie', 'is', 'great']
        # sentences List[List[str]]
        sentences = [
            sentence for paragraph in paragraphs for sentence in paragraph
        ]

        # 2. 使用sentences构建Vocab
        self.vocab = Vocab(
            sentences,
            min_freq=5,
            reserved_tokens=['<pad>', '<mask>', '<cls>', '<sep>'])

        examples = []  # 训练样本的集合
        # 3. 遍历每一个paragraph, 获取下一句子预测任务(NSP)的训练样本(exmaples)
        # 此时的examples: [(tokens, segments, is_next)]
        # tokens: e.g. ['<cls>', 'this', 'movie', 'is', 'great', '<sep>', 'i', 'like', 'it', '<sep>']
        # segments: e.g. [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
        # is_next: True | False
        for paragraph in paragraphs:
            examples.extend(
                _get_nsp_data_from_paragraph(paragraph, paragraphs, self.vocab,
                                             max_len))

        # 4. 遍历步骤3的每一个样本(examples), 获取遮蔽语言模型任务(MLM)的训练数据, 拼接到一起
        # 此时的examples: [(mlm_input_tokens_ids, pred_positions, mlm_pred_labels_ids, segments, is_next)]
        # mlm_input_tokens_ids: 输入tokens的索引 e.g. vocab[['<cls>', 'this', 'movie', '<mask>', 'great', '<sep>', 'i', 'like', 'it', '<sep>']]
        # pred_positions: 需要预测的位置索引, e.g. [3, ...]
        # mlm_pred_labels_ids: 预测的标签索引, e.g. vocab[['is', ...]]
        # segments: e.g. [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
        # is_next: True | False
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab) +
                     (segments, is_next))
                    for tokens, segments, is_next in examples]

        # 5. 填充(pad)样本
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights, self.all_mlm_labels,
         self.nsp_labels) = _pad_bert_inputs(examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)


def load_data_wiki(batch_size: int, max_len: int) -> Tuple[DataLoader, Vocab]:
    """
    加载WikiText_2数据集

    >>> batch_size, max_len = 512, 64
    >>> max_num_mlm_preds = round(max_len * 0.15)  # 根据公式计算
    >>> train_iter, vocab = load_data_wiki(batch_size, max_len)
    >>> for (tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X,
             mlm_Y, nsp_y) in train_iter:
    >>>     assert tokens_X.shape == (batch_size, max_len)
    >>>     assert segments_X.shape == (batch_size, max_len)
    >>>     assert valid_lens_x.shape == (batch_size, )
    >>>     assert pred_positions_X.shape == (batch_size, max_num_mlm_preds)
    >>>     assert mlm_weights_X.shape == (batch_size, max_num_mlm_preds)
    >>>     assert mlm_Y.shape == (batch_size, max_num_mlm_preds)
    >>>     assert nsp_y.shape == (batch_size, )
    >>>     break
    """
    paragraphs = _read_wiki(download_extract('wikitext_2'))
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = DataLoader(train_set, batch_size, shuffle=True)
    return train_iter, train_set.vocab


def try_gpu(i: int = 0) -> torch.device:
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def train_bert_gpu(train_iter: DataLoader, net: BERTModel,
                   loss: nn.CrossEntropyLoss, vocab_size: int,
                   device: torch.device, num_steps: int) -> None:
    net = net.to(device)
    trainer = torch.optim.Adam(net.parameters(), lr=0.01)
    step = 0
    times = []
    history = [[], []]  # 遮蔽语言模型损失, 下一句预测任务损失
    # 遮蔽语言模型损失的和, 下一句预测任务损失的和, 句子对的数量, 计数
    metric = [0.0] * 4
    num_steps_reached = False
    pbar = tqdm(total=num_steps)
    while step < num_steps and not num_steps_reached:
        for tokens_X, segments_X, valid_lens_x, pred_positions_X,\
            mlm_weights_X, mlm_Y, nsp_y in train_iter:
            tokens_X = tokens_X.to(device)
            segments_X = segments_X.to(device)
            valid_lens_x = valid_lens_x.to(device)
            pred_positions_X = pred_positions_X.to(device)
            mlm_weights_X = mlm_weights_X.to(device)
            mlm_Y, nsp_y = mlm_Y.to(device), nsp_y.to(device)
            trainer.zero_grad()
            t_start = time.time()
            mlm_l, nsp_l, l = _get_batch_loss_bert(net, loss, vocab_size,
                                                   tokens_X, segments_X,
                                                   valid_lens_x,
                                                   pred_positions_X,
                                                   mlm_weights_X, mlm_Y, nsp_y)
            l.backward()
            trainer.step()
            metric[0] += float(mlm_l)
            metric[1] += float(nsp_l)
            metric[2] += float(tokens_X.shape[0])
            metric[3] += 1
            times.append(time.time() - t_start)
            history[0].append((step + 1, metric[0] / metric[3]))
            history[1].append((step + 1, metric[1] / metric[3]))
            step += 1
            pbar.update(1)
            pbar.desc = f'step {step}, MLM loss {metric[0] / metric[3]:.3f}, NSP loss {metric[1] / metric[3]:.3f}'
            if step == num_steps:
                num_steps_reached = True
                break

    print(f'MLM loss {metric[0] / metric[3]:.3f}, '
          f'NSP loss {metric[1] / metric[3]:.3f}')
    print(f'{metric[2] / sum(times):.1f} sentence pairs/sec on '
          f'{str(device)}')

    plt.figure(figsize=(6, 4))
    # 训练集损失
    plt.plot(*zip(*history[0]), '-', label='MLM loss')
    plt.plot(*zip(*history[1]), 'm--', label='NSP loss')
    plt.xlabel('step')
    plt.xlim((1, num_steps))
    plt.grid()
    plt.legend()
    plt.show()


def train(batch_size: int, max_len: int, num_steps: int, num_hiddens: int,
          norm_shape: Tuple[int, int], ffn_num_input: int,
          ffn_num_hiddens: int, num_heads: int, num_layers: int,
          dropout: float, key_size: int, query_size: int, value_size: int,
          hid_in_features: int, mlm_in_features: int, nsp_in_features: int,
          device: torch.device) -> Tuple[BERTModel, Vocab]:
    train_iter, vocab = load_data_wiki(batch_size, max_len)
    net = BERTModel(len(vocab),
                    num_hiddens=num_hiddens,
                    norm_shape=norm_shape,
                    ffn_num_input=ffn_num_input,
                    ffn_num_hiddens=ffn_num_hiddens,
                    num_heads=num_heads,
                    num_layers=num_layers,
                    dropout=dropout,
                    key_size=key_size,
                    query_size=query_size,
                    value_size=value_size,
                    hid_in_features=hid_in_features,
                    mlm_in_features=mlm_in_features,
                    nsp_in_features=nsp_in_features)
    loss = nn.CrossEntropyLoss()
    train_bert_gpu(train_iter, net, loss, len(vocab), device, num_steps)
    return net, vocab


def get_bert_encoding(net: BERTModel,
                      tokens_a: List[str],
                      tokens_b: List[str] = None,
                      device: torch.device = None) -> Tensor:
    """
    返回tokens_a和tokens_b中所有token的BERT表示

    参数:
    net: BERTModel实例
    tokens_a: e.g. ['this', 'movie', 'is', 'great']
    tokens_b: e.g. ['i', 'like', 'it']

    返回:
    encoded_X: [1, num_steps, num_hiddens]
    """
    tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)
    # token_ids.shape [1, num_steps]
    token_ids = torch.tensor(vocab[tokens], device=device).unsqueeze(0)
    # segments.shape [1, num_steps]
    segments = torch.tensor(segments, device=device).unsqueeze(0)
    # valid_len.shape [1, ]
    valid_len = torch.tensor(len(tokens), device=device).unsqueeze(0)
    # encoded_X.shape [1, num_steps, num_hiddens]
    encoded_X, _, _ = net(token_ids, segments, valid_len)
    return encoded_X


if __name__ == '__main__':
    device = try_gpu()
    kwargs = {
        'batch_size': 512,
        'max_len': 64,
        'num_steps': 1000,
        'num_hiddens': 128,
        'norm_shape': [128],
        'ffn_num_input': 128,
        'ffn_num_hiddens': 256,
        'num_heads': 2,
        'num_layers': 2,
        'dropout': 0.2,
        'key_size': 128,
        'query_size': 128,
        'value_size': 128,
        'hid_in_features': 128,
        'mlm_in_features': 128,
        'nsp_in_features': 128,
        'device': device,
    }
    net, vocab = train(**kwargs)
    # MLM loss 6.943, NSP loss 0.701
    # 8333.0 sentence pairs/sec on cuda:0

    tokens_a = ['a', 'crane', 'is', 'flying']
    encoded_text = get_bert_encoding(net, tokens_a, device=device)
    # tokens: '<cls>','a','crane','is','flying','<sep>'
    encoded_text_cls = encoded_text[:, 0, :]
    encoded_text_crane = encoded_text[:, 2, :]
    print(encoded_text.shape, encoded_text_cls.shape,
          encoded_text_crane[0][:3])
    # torch.Size([1, 6, 128])
    # torch.Size([1, 128])
    # tensor([-0.2938,  0.0296, -0.0676], device='cuda:0', grad_fn=<SliceBackward0>)

    tokens_a, tokens_b = ['a', 'crane', 'driver',
                          'came'], ['he', 'just', 'left']
    encoded_pair = get_bert_encoding(net, tokens_a, tokens_b, device=device)
    # 词元：'<cls>','a','crane','driver','came','<sep>','he','just','left','<sep>'
    encoded_pair_cls = encoded_pair[:, 0, :]
    encoded_pair_crane = encoded_pair[:, 2, :]
    print(encoded_pair.shape, encoded_pair_cls.shape,
          encoded_pair_crane[0][:3])
    # torch.Size([1, 10, 128])
    # torch.Size([1, 128])
    # tensor([-0.1867,  0.1516, -0.5738], device='cuda:0', grad_fn=<SliceBackward0>)
