from typing import Union, List, Dict, Tuple, Any
import collections
import os
import time
import sys
import math
import requests
import hashlib
import zipfile
import tarfile
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt
from tqdm import tqdm
"""
Transformer从零实现

实现说明:
https://tech.foxrelax.com/nlp/transformer_scratch/
"""


class Encoder(nn.Module):
    """
    编码器-解码器结构的基本编码器接口
    """

    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X: Tensor, *args) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError


class Decoder(nn.Module):
    """
    编码器-解码器结构的基本编码器接口
    """

    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs: Tuple[Tensor, Tensor], *args) -> Tensor:
        raise NotImplementedError

    def forward(self, X: Tensor, state: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError


class EncoderDecoder(nn.Module):
    """
    编码器-解码器结构的基类
    """

    def __init__(self, encoder: Encoder, decoder: Decoder,
                 **kwargs: Any) -> None:
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X: Tensor, dec_X: Tensor,
                *args: Any) -> Tuple[Tensor, Tensor]:
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)


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


class TransformerEncoder(Encoder):
    """
    >>> vocab_size, key_size, query_size, value_size, num_hiddens, num_steps = 10000, 24, 24, 24, 24, 100
    >>> batch_size, ffn_num_input, ffn_num_hiddens, num_heads, num_layers = 2, 24, 48, 8, 5
    >>> norm_shape = (num_steps, num_hiddens)
    >>> encoder = TransformerEncoder(vocab_size, key_size, query_size, value_size,
                                     num_hiddens, norm_shape, ffn_num_input,
                                     ffn_num_hiddens, num_heads, num_layers, 0.1)
    >>> X = torch.ones((batch_size, num_steps), dtype=torch.long)
    >>> valid_lens = torch.tensor([3, 2])
    >>> assert encoder(X, valid_lens).shape == (batch_size, num_steps, num_hiddens)
    """

    def __init__(self,
                 vocab_size: int,
                 key_size: int,
                 query_size: int,
                 value_size: int,
                 num_hiddens: int,
                 norm_shape: Tuple[int, int],
                 ffn_num_input: int,
                 ffn_num_hiddens: int,
                 num_heads: int,
                 num_layers: int,
                 dropout: float,
                 use_bias: bool = False,
                 **kwargs: Any) -> None:
        """
        参数:
        vocab_size: 字典大小
        key_size: key的特征长度
        query_size: query的特征长度
        value_size: value的特征长度
        num_hiddens: 隐藏单元的特征长度(也是多头注意力最终输出的特征长度)
        norm_shape: (num_steps, num_hiddens)
        ffn_num_input: ffn输入的维度
        ffn_num_hiddens: ffn隐藏层的维度
        num_heads: 头数
        num_layers: EncoderBlock的数量
        dropout: dropout操作设置为0的元素的概率
        use_bias: 是否开启bias
        """
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                "block" + str(i),
                EncoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, use_bias))

    def forward(self, X: Tensor, valid_lens: Tensor, *args: Any) -> Tensor:
        """
        输入:
        X: [batch_size, num_steps]
        valid_lens的形状: [batch_size, ] 表示X对应的有效token个数

        返回:
        output: [batch_size, num_steps, num_hiddens]
        """
        # 因为位置编码值在-1和1之间, 因此嵌入值乘以嵌入维度的平方根进行缩放,
        # 然后再与位置编码相加
        #
        # X.shape [batch_size, num_steps, num_hiddens]
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))

        # 每一个block记录一个attention_weight
        # 每个的attention_weight的形状是根据送入attention时的queries, keys, values的形状
        # 计算出来的: [batch_size, num_queries, num_kvs]
        self.attention_weights = [None] * len(self.blks)

        # 遍历每个EncoderBlock
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            # 记录下attention_weights, 后续可以显示
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        return X


class DecoderBlock(nn.Module):
    """
    解码器中第i个块
    """

    def __init__(self, key_size: int, query_size: int, value_size: int,
                 num_hiddens: int, norm_shape: Tuple[int, int],
                 ffn_num_input: int, ffn_num_hiddens: int, num_heads: int,
                 dropout: float, i: int, **kwargs: Any) -> None:
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
        i: 第几个DecoderBlock
        """
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = MultiHeadAttention(key_size, query_size, value_size,
                                             num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(key_size, query_size, value_size,
                                             num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X: Tensor, state: List[Any]) -> Tuple[Tensor, List[Any]]:
        """
        参数:
        X的: [batch_size, num_steps, num_hiddens]
        state: [enc_outputs, enc_valid_lens, key_values]
          enc_outputs: [batch_size, num_steps, num_hiddens]
          enc_valid_lens: [batch_size, ]
          key_values: [None] * self.num_layers | [num_kvs, num_hiddens] * self.num_layers

        返回: (output, state)
        output: [batch_size, num_steps, num_hiddens]
        state: [enc_outputs, enc_valid_lens, key_values]
          enc_outputs: [batch_size, num_steps, num_hiddens]
          enc_valid_lens: [batch_size, ]
          key_values: [num_kvs, num_hiddens] * self.num_layers
        """
        enc_outputs, enc_valid_lens = state[0], state[1]

        # 1. 训练模式, 输出序列的所有token都在同一时间处理,
        #    因此state[2][self.i]初始化为None.
        # 2. 预测模式, 输出序列是通过token一个接着一个解码的, 因此state[2][self.i]包含
        #    着直到当前时间步第i个DecoderBlock解码的输出表示
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values

        if self.training:
            batch_size, num_steps, _ = X.shape
            # dec_valid_lens.shape [batch_size, num_steps]
            #
            # 例如: [batch_size=4, num_steps=6]
            # tensor([[1, 2, 3, 4, 5, 6],
            #         [1, 2, 3, 4, 5, 6],
            #         [1, 2, 3, 4, 5, 6],
            #         [1, 2, 3, 4, 5, 6]])
            dec_valid_lens = torch.arange(1, num_steps + 1,
                                          device=X.device).repeat(
                                              batch_size, 1)
        else:
            dec_valid_lens = None

        # 自注意力
        # X2.shape [batch_size, num_steps, num_hiddens]
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)

        # 编码器－解码器注意力
        # Y2.shape [batch_size, num_steps, num_hiddens]
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state


class TransformerDecoder(Decoder):
    """
    >>> vocab_size, key_size, query_size, value_size, num_hiddens, num_steps = 10000, 24, 24, 24, 24, 100
    >>> batch_size, ffn_num_input, ffn_num_hiddens, num_heads, num_layers = 2, 24, 48, 8, 5
    >>> norm_shape = (num_steps, num_hiddens)
    >>> encoder = TransformerEncoder(vocab_size, key_size, query_size, value_size,
                                     num_hiddens, norm_shape, ffn_num_input,
                                     ffn_num_hiddens, num_heads, num_layers, 0.1)
    >>> decoder = TransformerDecoder(vocab_size, key_size, query_size, value_size,
                                     num_hiddens, norm_shape, ffn_num_input,
                                     ffn_num_hiddens, num_heads, num_layers, 0.1)
    >>> enc_X = torch.ones((batch_size, num_steps), dtype=torch.long)
    >>> enc_valid_lens = torch.tensor([3, 2])
    >>> dec_X = torch.ones((batch_size, num_steps), dtype=torch.long)

    # 编码
    >>> enc_outputs = encoder(enc_X, enc_valid_lens)
    >>> assert enc_outputs.shape == (batch_size, num_steps, num_hiddens)

    # 初始化解码器state
    >>> dec_state = decoder.init_state(enc_outputs, enc_valid_lens)
    # 解码
    >>> dec_outputs = decoder(dec_X, dec_state)
    >>> assert dec_outputs[0].shape == (batch_size, num_steps, vocab_size)
    >>> assert dec_outputs[1][0].shape == (batch_size, num_steps, num_hiddens)
    >>> assert len(dec_outputs[1][2]) == num_layers
    """

    def __init__(self, vocab_size: int, key_size: int, query_size: int,
                 value_size: int, num_hiddens: int, norm_shape: Tuple[int,
                                                                      int],
                 ffn_num_input: int, ffn_num_hiddens: int, num_heads: int,
                 num_layers: int, dropout: float, **kwargs: Any) -> None:
        """
        vocab_size: 字典大小
        key_size: key的特征长度
        query_size: query的特征长度
        value_size: value的特征长度
        num_hiddens: 隐藏单元的特征长度(也是多头注意力最终输出的特征长度)
        norm_shape: (num_steps, num_hiddens)
        ffn_num_input: ffn输入的维度
        ffn_num_hiddens: ffn隐藏层的维度
        num_heads: 头数
        num_layers: DecoderBlock的数量
        dropout: dropout操作设置为0的元素的概率
        """
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                "block" + str(i),
                DecoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)  # 最终的输出层

    def init_state(self, enc_outputs: Tensor, enc_valid_lens: Tensor,
                   *args: Any) -> List[Any]:
        """
        参数:
        enc_outputs: [batch_size, num_steps, num_hiddens]
        enc_valid_lens: [batch_size, ]

        返回: [enc_outputs, enc_valid_lens, [None] * self.num_layers]
        enc_outputs: [batch_size, num_steps, num_hiddens]
        enc_valid_lens: [batch_size, ]
        key_values: [None] * self.num_layers
                    保存每一层的key_values, 因为在预测模式下, key_values每预测一步
                    会增加一个token进来
        """
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X: Tensor, state: List[Any]) -> Tuple[Tensor, List[Any]]:
        """
        参数:
        X: [batch_size, num_steps)
        state: [enc_outputs, enc_valid_lens, [None] * self.num_layers]
          enc_outputs: [batch_size, num_steps, num_hiddens]
          enc_valid_lens: [batch_size, ]
          key_values: [None] * self.num_layers

        返回: (output, state)
        output: [batch_size, num_steps, vocab_size]
        state: [enc_outputs, enc_valid_lens, key_values]
          enc_outputs: [batch_size, num_steps, num_hiddens]
          enc_valid_lens: [batch_size, ]
          key_values: [num_kvs, num_hiddens] * self.num_layers
        """
        # X.shape [batch_size, num_steps, num_hiddens]
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            # X的形状: (batch_size, num_steps, num_hiddens)
            X, state = blk(X, state)
            # 解码器自注意力权重
            self._attention_weights[0][
                i] = blk.attention1.attention.attention_weights
            # '编码器－解码器'自注意力权重
            self._attention_weights[1][
                i] = blk.attention2.attention.attention_weights

        # 经过一个线性层之后输出最终结果
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """
    带遮蔽的softmax交叉熵损失函数

    计算过程如下:
    假设pred的形状为: [2, 5, 10], label的形状为: [2, 5], 则reduction=none时, 计算出来
    的loss的形状为: [2, 5], 如下:
    tensor([[2.4712, 1.7931, 1.6518, 2.3004, 1.0466],
            [3.5565, 2.1062, 3.2549, 3.9885, 2.7302]])

    我们叠加如下的valid_len=tensor([5, 2]), 则会生成如下weights
    tensor([[1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0]])

    所以最终计算出来的loss为:
    tensor([[2.4712, 1.7931, 1.6518, 2.3004, 1.0466],
            [3.5565, 2.1062, 0, 0, 0]])
    最终得到的loss为: tensor([1.8526, 1.1325])
    (2.4712+1.7931+1.6518+2.3004+1.0466)/5 = 1.8526
    (3.5565+2.1062)/5 = 1.1325
    """

    def forward(self, pred, label, valid_len):
        """
        参数:
        pred: [batch_size, num_steps, vocab_size]
        label: [batch_size, num_steps]
        valid_len: [batch_size, ]

        输出:
        weighted_loss: [batch_size, ]
        """
        weights = torch.ones_like(label)
        # weights.shape [batch_size, num_steps]
        weights = sequence_mask(weights, valid_len)
        self.reduction = 'none'
        # unweighted_loss.shape [batch_size, num_steps]
        unweighted_loss = super(MaskedSoftmaxCELoss,
                                self).forward(pred.permute(0, 2, 1), label)
        # weighted_loss.shape [batch_size, num_steps]
        #                  -> [batch_size, ]
        weighted_loss = (weights * unweighted_loss).mean(dim=1)
        return weighted_loss


def try_gpu(i: int = 0) -> torch.device:
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def grad_clipping(net: Union[nn.Module, object], theta: float) -> None:
    """
    裁剪梯度

    训练RNN网络的常用技巧
    """
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad**2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def train_seq2seq_gpu(net: nn.Module,
                      data_iter: DataLoader,
                      lr: float,
                      num_epochs: int,
                      tgt_vocab: Any,
                      device: torch.device = None) -> None:
    """
    用GPU训练模型
    """
    if device is None:
        device = try_gpu()

    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)
        elif type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if 'weight' in param:
                    nn.init.xavier_normal_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    history = [[]]  # 记录: 训练集损失, 方便后续绘图
    for epoch in range(num_epochs):
        t_start = time.time()
        metric_train = [0.0] * 2  # 训练损失总和, 词元数量
        data_iter_tqdm = tqdm(data_iter, file=sys.stdout)
        for i, batch in enumerate(data_iter_tqdm):
            optimizer.zero_grad()
            # X.shape [batch_size, num_steps]
            # X_valid_len.shape [batch_size, ]
            # Y.shape [batch_size, num_steps]
            # Y_valid_len.shape [batch_size, ]
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            # bos.shape [batch_size, 1]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)
            # dec_input.shape [batch_size, num_steps]
            dec_input = torch.cat((bos, Y[:, :-1]), 1)
            # Y_hat.shape [batch_size, num_steps, vocab_size]
            Y_hat, _ = net(X, dec_input, X_valid_len)
            # l.shape [batch_size, ]
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()  # 损失函数的标量进行'反传'
            grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric_train[0] += float(l.sum())
                metric_train[1] += float(num_tokens)
            data_iter_tqdm.desc = f'epoch {epoch}, step {i+1}, train loss {metric_train[0] / metric_train[1]:.3f}'
        if (epoch + 1) % 10 == 0:
            history[0].append((epoch + 1, metric_train[0] / metric_train[1]))
    print(
        f'loss {metric_train[0] / metric_train[1]:.3f}, {metric_train[1] / (time.time() - t_start):.1f} '
        f'tokens/sec on {str(device)}')

    # plot 训练集损失
    plt.figure(figsize=(6, 4))
    # 训练集损失
    plt.plot(*zip(*history[0]), '-', label='train loss')
    plt.xlabel('epoch')
    # 从epoch=1开始显示, 0-1这个范围的数据丢弃不展示,
    # 因为只有训练完成1个epochs之后, 才会有第一条test acc记录
    plt.xlim((1, num_epochs))
    plt.grid()
    plt.legend()
    plt.show()


def predict_seq2seq(
        net: nn.Module,
        src_setence: List[str],
        src_vocab: Any,
        tgt_vocab: Any,
        num_steps: int,
        device: torch.device,
        save_attention_weights: bool = False) -> Tuple[str, List[Tensor]]:
    """
    序列到序列模型的预测
    """
    net.eval()
    src_tokens = src_vocab[src_setence.lower().split(' ')] + [
        src_vocab['<eos>']
    ]
    # enc_valid_len.shape [1, num_tokens]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # enc_X.shape [1, num_steps]
    enc_X = torch.unsqueeze(torch.tensor(src_tokens,
                                         dtype=torch.long,
                                         device=device),
                            dim=0)
    # enc_outputs[0].shape [num_steps, 1, num_hiddens]
    # enc_outputs[1].shape [num_layers, 1, num_hiddens]
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # dec_X.shape [1, 1]
    dec_X = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']],
                                         dtype=torch.long,
                                         device=device),
                            dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        # Y.shape [1, 1, vocab_size]
        # dec_state.shape [num_layers, 1, num_hiddens]
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 使用具有预测最高可能性的token, 作为解码器在下一时间步的输入
        # dex_X.shape [1, 1]
        dec_X = Y.argmax(dim=2)
        # pred.shape [1,]
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 保存注意力权重, 每一个step都会保存一个
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列结束词元被预测, 输出序列的生成就完成了
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq


def bleu(pred_seq: str, label_seq: str, k: int) -> float:
    """
    计算 BLEU
    """
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[''.join(label_tokens[i:i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[''.join(pred_tokens[i:i + n])] > 0:
                num_matches += 1
                label_subs[''.join(pred_tokens[i:i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score


def download(cache_dir: str = '../data') -> str:
    """
    下载数据
    """
    sha1_hash = '02eee9efbc64e076be914c6c163740dd5d448b36'
    url = 'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/fra_eng.zip'
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
    # e.g. ../data/fra_eng.zip
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
    # e.g. ../data/fra_eng
    return data_dir


def load_array(data_arrays: List[Tensor],
               batch_size: int,
               is_train: bool = True) -> DataLoader:
    """
    构造一个PyTorch数据迭代器
    """
    dataset = TensorDataset(*data_arrays)
    return DataLoader(dataset, batch_size, shuffle=is_train)


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

    >>> raw_text = read_data_nmt()
    >>> text = preprocess_nmt(raw_text)
    >>> source, target = tokenize_nmt(text)
    >>> src_vocab = Vocab(source,
                          min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    >>> tgt_vocab = Vocab(target,
                          min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    >>> len(src_vocab)
        10012
    >>> len(tgt_vocab)
        17851
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
        """
        tokens -> int list
        """
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices: Union[int, List[int],
                                       Tuple[int]]) -> List[str]:
        """
        int list -> tokens
        """
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]


def read_data_nmt() -> str:
    """
    载入'英语－法语'数据集
    
    >>> raw_text = read_data_nmt()
    >>> raw_text[:75]
        Go.     Va !
        Hi.     Salut !
        Run!    Cours !
        Run!    Courez !
        Who?    Qui ?
        Wow!    Ça alors !
    """
    data_dir = download_extract('fra_eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r', encoding='utf-8') as f:
        return f.read()


def preprocess_nmt(text: str) -> str:
    """
    预处理'英语－法语'数据集

    1. 使用空格替换不间断空格
    2. 使用小写字母替换大写字母
    3. 在单词和标点符号之间插入空格

    >>> raw_text = read_data_nmt()
    >>> text = preprocess_nmt(raw_text)
    >>> text[:80]
        go .    va !
        hi .    salut !
        run !   cours !
        run !   courez !
        who ?   qui ?
        wow !   ça alors !
    """

    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 使用空格替换不间断空格
    # 使用小写字母替换大写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [
        ' ' + char if i > 0 and no_space(char, text[i - 1]) else char
        for i, char in enumerate(text)
    ]
    return ''.join(out)


def tokenize_nmt(
        text: str,
        num_examples: int = None) -> Tuple[List[List[str]], List[List[str]]]:
    """
    词元化'英语－法语'数据数据集

    >>> raw_text = read_data_nmt()
    >>> text = preprocess_nmt(raw_text)
    >>> source, target = tokenize_nmt(text)
    >>> source[:6]
        [['go', '.'], ['hi', '.'], ['run', '!'], 
         ['run', '!'], ['who', '?'], ['wow', '!']]
    >>> target[:6]
        [['va', '!'], ['salut', '!'], ['cours', '!'], 
         ['courez', '!'], ['qui', '?'], ['ça', 'alors', '!']]

    参数:
    text: 文本
    num_examples: 返回的最大样本数

    返回: (source, target)
    source: 源语言
    target: 目标语言
    """
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target


def truncate_pad(line: List[int], num_steps: int,
                 padding_token: int) -> List[int]:
    """
    截断或填充文本序列, 返回序列的长度为`num_steps`
    
    填充
    >>> line = [47, 4]
    >>> padding_token = 1
    >>> padding_line = truncate_pad(line, 10, padding_token)
    >>> padding_line
        [47, 4, 1, 1, 1, 1, 1, 1, 1, 1]

    截断
    >>> line = [47, 4, 48, 90, 98, 19, 12, 34, 56, 654, 123, 198, 65]
    >>> padding_token = 1
    >>> padding_line = truncate_pad(line, 10, padding_token)
        [47, 4, 48, 90, 98, 19, 12, 34, 56, 654]
    """
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充


def build_array_nmt(lines: List[List[str]], vocab: Vocab,
                    num_steps: int) -> Tuple[Tensor, Tensor]:
    """
    将机器翻译的文本序列转换成小批量

    返回:
    array.shape [num_examples, num_steps]
    valid_len.shape [num_examples, ]
    """
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = torch.tensor(
        [truncate_pad(l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len


def load_data_nmt(batch_size: int,
                  num_steps: int,
                  num_examples: int = 1000) -> Tuple[DataLoader, Vocab, Vocab]:
    """
    返回翻译数据集的迭代器和词汇表

    >>> train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)
    >>> for X, X_valid_len, Y, Y_valid_len in train_iter:
    >>>     assert X.shape == (2, 8)
    >>>     assert X_valid_len.shape == (2, )
    >>>     assert Y.shape == (2, 8)
    >>>     assert X_valid_len.shape == (2, )
    >>>     print('X:', X)
    >>>     print('X的有效长度:', X_valid_len)
    >>>     print('Y:', Y)
    >>>     print('Y的有效长度:', Y_valid_len)
    >>>     break
        X: tensor([[ 13,   0,   4,   3,   1,   1,   1,   1],
                   [  7, 134,   4,   3,   1,   1,   1,   1]])
        X的有效长度: tensor([4, 4])
        Y: tensor([[10,  0,  4,  3,  1,  1,  1,  1],
                   [ 0, 39,  4,  3,  1,  1,  1,  1]])
        Y的有效长度: tensor([4, 4])

    返回: (data_iter, src_vocab, tgt_vocab)
    data_iter: DataLoader
    src_vocab: 源语言词汇表
    tgt_vocab: 目标语言词汇表
    """
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = Vocab(source,
                      min_freq=2,
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = Vocab(target,
                      min_freq=2,
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab


def mnt_seq2seq(src_vocab_size: int, tgt_vocab_size: int, key_size: int,
                query_size: int, value_size: int, num_hiddens: int,
                norm_shape: Tuple[int, int], ffn_num_input: int,
                ffn_num_hiddens: int, num_heads: int, num_layers: int,
                dropout: float) -> EncoderDecoder:
    encoder = TransformerEncoder(src_vocab_size, key_size, query_size,
                                 value_size, num_hiddens, norm_shape,
                                 ffn_num_input, ffn_num_hiddens, num_heads,
                                 num_layers, dropout)
    decoder = TransformerDecoder(tgt_vocab_size, key_size, query_size,
                                 value_size, num_hiddens, norm_shape,
                                 ffn_num_input, ffn_num_hiddens, num_heads,
                                 num_layers, dropout)
    return EncoderDecoder(encoder, decoder)


def train(key_size: int,
          query_size: int,
          value_size: int,
          num_hiddens: int,
          norm_shape: Tuple[int, int],
          ffn_num_input: int,
          ffn_num_hiddens: int,
          num_heads: int,
          num_layers: int,
          dropout: float,
          batch_size: int,
          num_steps: int,
          lr: float,
          num_epochs: int,
          device: torch.device = None) -> None:
    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)
    net = mnt_seq2seq(len(src_vocab), len(tgt_vocab), key_size, query_size,
                      value_size, num_hiddens, norm_shape, ffn_num_input,
                      ffn_num_hiddens, num_heads, num_layers, dropout)
    train_seq2seq_gpu(net, train_iter, lr, num_epochs, tgt_vocab, device)
    return net, src_vocab, tgt_vocab


def test(net: EncoderDecoder, src_vocab: Vocab, tgt_vocab: Vocab,
         num_steps: int, device: torch.device) -> None:
    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        translation, _ = predict_seq2seq(net, eng, src_vocab, tgt_vocab,
                                         num_steps, device, True)
        print(
            f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')


if __name__ == '__main__':
    device = try_gpu()
    kwargs = {
        'key_size': 32,
        'query_size': 32,
        'value_size': 32,
        'num_hiddens': 32,
        'norm_shape': [32],
        'ffn_num_input': 32,
        'ffn_num_hiddens': 64,
        'num_heads': 4,
        'num_layers': 2,
        'dropout': 0.1,
        'batch_size': 64,
        'num_steps': 20,
        'lr': 0.005,
        'num_epochs': 200,
        'device': device
    }
    net, src_vocab, tgt_vocab = train(**kwargs)
    kwargs_test = {'num_steps': 10, 'device': device}
    test(net, src_vocab, tgt_vocab, **kwargs_test)
# go . => va !, bleu 1.000
# i lost . => j'ai perdu ., bleu 1.000
# he's calm . => il est calme ., bleu 1.000
# i'm home . => je suis chez moi ., bleu 1.000