import datetime
from typing import Any, Dict, List, Tuple
import numpy as np
import sys
import time
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.functional import cross_entropy, softmax, relu
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
"""
Simple Transformer

实现说明:
https://tech.foxrelax.com/nlp/simple_transformer/
"""

MAX_LEN = 11
PAD_ID = 0
"""
训练集是这个范围内的时间: [1974年7月23日18时19分45秒, 2034年10月7日12时6分25秒]. 
我们选择这个时间范围的数据隐藏了另外一个规律: 就是如果中文年份大于74这个数字, 翻译出
来的英文年份前面应该补19; 如果中文年份小于23这个数字, 翻译出来的英文年份前面应该补20

我们希望神经网络可以自动识别出这个规律
"""


class DateData:
    """
    时间范围: (1974, 7, 23, 18, 19, 45) -> (2034, 10, 7, 12, 6, 25)
    中文的: "年-月-日", e.g. "98-02-26"
    英文的: "day/month/year", e.g. "26/Feb/1998"

    1. 中文样本的生成:
    04-07-18 -> ['0', '4', '-', '0', '7', '-', '1', '8']  分词
             -> [3, 7, 1, 3, 10, 1, 4, 11]                转化成token_id list

    2. 英文样本的生成(因为英文是目标对象, 所以我们增加了<BOS>, <EOS>):
    18/Jul/2004 -> ['1', '8', '/', 'Jul', '/', '2', '0', '0', '4']                   分词
                -> ['<BOS>', '1', '8', '/', 'Jul', '/', '2', '0', '0', '4', '<EOS>'] 添加开头和结尾
                -> [13, 4, 11, 2, 16, 2, 5, 3, 3, 7, 14]                             转化成token_id list

    >>> dataset = DateData(4000)
    >>> dataset.date_cn[:3]
        ['31-04-26', '04-07-18', '33-06-06']
    >>> dataset.date_en[:3]
        ['26/Apr/2031', '18/Jul/2004', '06/Jun/2033']
    >>> dataset.vocab
        {'Apr', 'Feb', 'Oct', 'Jun', 'Jul', 'May', 'Nov',
         'Mar', 'Aug',
         '<PAD>', '<BOS>', 'Jan', 'Dec', '9', '8', '5',
         '4', '7', '6', '1', '0', '3', '2', '-', '<EOS>',
         '/', 'Sep'}
    >>> dataset.x[0], dataset.idx2str(dataset.x[0])
        [6 4 1 3 7 1 5 9] 31-04-26
    >>> dataset.y[0], dataset.idx2str(dataset.y[0])
        [13  5  9  2 15  2  5  3  6  4 14] <BOS>26/Apr/2031<EOS>
    >>> dataset[0]
        (array([6, 4, 1, 3, 7, 1, 5, 9]), 
         array([13,  5,  9,  2, 15,  2,  5,  3,  6,  4, 14]), 
         10)

    """
    PAD_ID = 0

    def __init__(self, n: int):
        """
        参数:
        n: 生成的样本数量
        """
        np.random.seed(1)

        self.date_cn = []  # list of cn data, e.g. 34-10-07
        self.date_en = []  # list of en data, e.g. 07/Oct/203
        # 时间范围: (1974, 7, 23, 18, 19, 45) -> (2034, 10, 7, 12, 6, 25)
        # 中文: 74-07-23 -> 34-10-07
        # 英文: 23/Jul/1974 -> 07/Oct/203
        for timestamp in np.random.randint(143835585, 2043835585, n):
            date = datetime.datetime.fromtimestamp(timestamp)
            self.date_cn.append(date.strftime("%y-%m-%d"))
            self.date_en.append(date.strftime("%d/%b/%Y"))
        # vocab:
        # {'2', '1', '<EOS>', '/', 'Mar', 'Jun', 'Apr', 'Aug', '3', 'Jan', '5',
        #   '<BOS>', '8', '0', 'May',
        #   'Nov', 'Jul', 'Oct', 'Sep', '9', '6', '4', 'Dec', '7', '-', 'Feb'}
        #
        # 包含三个部分(str):
        # 1. 0,1,2,3,4,5,6,7,8,9
        # 2. <BOS>, <EOS>, <PAD>, -, /
        # 3. Jun, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec
        self.vocab = set([str(i) for i in range(0, 10)] +
                         ["-", "/", "<BOS>", "<EOS>"] +
                         [i.split("/")[1] for i in self.date_en])

        self.token_to_id = {
            v: i
            for i, v in enumerate(sorted(list(self.vocab)), start=1)
        }  # id从1开始, 0留给<PAD>
        self.token_to_id["<PAD>"] = DateData.PAD_ID
        self.vocab.add("<PAD>")
        self.id_to_token = {i: v for v, i in self.token_to_id.items()}

        self.x, self.y = [], []
        for cn, en in zip(self.date_cn, self.date_en):
            # 中文: 34-10-07
            # e.g.
            # 04-07-18 -> ['0', '4', '-', '0', '7', '-', '1', '8']
            #          -> [3, 7, 1, 3, 10, 1, 4, 11]
            self.x.append([self.token_to_id[v] for v in cn])
            # 英文: 07/Oct/203
            # e.g.
            # 18/Jul/2004 -> ['<BOS>', '1', '8', '/', 'Jul', '/', '2', '0', '0', '4', '<EOS>']
            #             -> [13, 4, 11, 2, 16, 2, 5, 3, 3, 7, 14]
            self.y.append([
                self.token_to_id["<BOS>"],
            ] + [self.token_to_id[v] for v in en[:3]] + [
                self.token_to_id[en[3:6]],
            ] + [self.token_to_id[v] for v in en[6:]] + [
                self.token_to_id["<EOS>"],
            ])
        self.x, self.y = np.array(self.x), np.array(self.y)

        self.start_token = self.token_to_id["<BOS>"]
        self.end_token = self.token_to_id["<EOS>"]

    def __len__(self) -> int:
        return len(self.x)

    @property
    def num_word(self) -> int:
        """
        词典长度
        """
        return len(self.vocab)

    def __getitem__(self, index: int) -> Tuple[np.array, np.array, int]:
        """
        采样:

        e.g.
        (array([6, 4, 1, 3, 7, 1, 5, 9]), 
         array([13,  5,  9,  2, 15,  2,  5,  3,  6,  4, 14]), 
         10)

        返回: (bx, by, decoder_len)
        """
        # 返回的decoder_len-1是为了去掉开头的<BOS>
        return self.x[index], self.y[index], len(self.y[index]) - 1

    def idx2str(self, idx: List[str]) -> List[int]:
        """
        将token_id list转换为token list

        >>> idx2str([ 4,  3,  1,  3, 10,  1,  5,  8])
        10-07-25
        >>> idx2str([13,  5,  8,  2, 20,  2,  5,  3,  4,  3, 14])
        <BOS>25/Jul/2010<EOS>
        """

        x = []
        for i in idx:
            x.append(self.id_to_token[i])
            if i == self.end_token:
                break
        return "".join(x)


def load_date(batch_size: int = 32,
              num_examples: int = 4000) -> Tuple[DataLoader, DateData]:
    """
    >>> batch_size = 32
    # 04-07-18, 拆分成token后长度为8
    # <BOS>18/Jul/2004<EOS>, 拆分成token后长度为11
    >>> num_steps_x, num_steps_y = 8, 11
    >>> data_iter, dataset = load_date(batch_size=batch_size)
    >>> for X, y, decoder_len in data_iter:
    >>>     print(X[0], dataset.idx2str(X[0].numpy()))
    >>>     print(y[0], dataset.idx2str(y[0].numpy()))
    >>>     print(decoder_len[0])  # 去掉了y开始的<BOS>
    >>>     assert X.shape == (batch_size, num_steps_x)
    >>>     assert y.shape == (batch_size, num_steps_y)
    >>>     assert decoder_len.shape == (batch_size, )
    >>>     break
        tensor([11,  5,  1,  4,  5,  1,  3,  7]) 82-12-04
        tensor([13,  3,  7,  2, 17,  2,  4, 12, 11,  5, 14]) <BOS>04/Dec/1982<EOS>
        tensor(10)
    """
    dataset = DateData(num_examples)
    data_iter = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_iter, dataset


def pad_zero(seqs: np.ndarray, max_len: int) -> np.ndarray:
    """
    >>> x = np.array([[2, 4], [3, 4]])
    >>> padded = pad_zero(x, 4)
    >>> assert padded.shape == (2, 4)
    >>> padded
        [[2 4 0 0]
         [3 4 0 0]]

    参数:
    seqs: [batch_size, steps]

    返回:
    padded: [batch_size, max_len]
    """
    padded = np.full((len(seqs), max_len), fill_value=PAD_ID, dtype=np.int32)
    for i, seq in enumerate(seqs):
        padded[i, :len(seq)] = seq
    return padded


class MultiHead(nn.Module):
    """
    实现多头注意力

    >>> batch_size, num_steps, num_hiddens = 2, 10, 32
    >>> mh = MultiHead(2, num_hiddens, 0.1)
    >>> q = torch.randn((batch_size, num_steps, num_hiddens))
    >>> k = torch.randn((batch_size, num_steps, num_hiddens))
    >>> v = torch.randn((batch_size, num_steps, num_hiddens))
    >>> mask = torch.randint(0, 2, (batch_size, 1, num_steps, num_steps))
    >>> assert mh(q, k, v, mask, None).shape == (batch_size, num_steps, num_hiddens)
    """

    def __init__(self, n_head: int, num_hiddens: int,
                 drop_rate: float) -> None:
        """
        参数:
        n_head: 头数
        num_hiddens: 隐藏单元的特征长度(也是多头注意力最终输出的特征长度)
        drop_rate: dropout rate
        """
        super().__init__()
        self.head_dim = num_hiddens // n_head
        self.n_head = n_head
        self.num_hiddens = num_hiddens
        self.wq = nn.Linear(num_hiddens, n_head * self.head_dim)
        self.wk = nn.Linear(num_hiddens, n_head * self.head_dim)
        self.wv = nn.Linear(num_hiddens, n_head * self.head_dim)

        self.o_dense = nn.Linear(num_hiddens, num_hiddens)
        self.o_drop = nn.Dropout(drop_rate)
        self.layer_norm = nn.LayerNorm(num_hiddens)
        # 保存最新的一次注意力权重, 调用一次forward(), 会更新一次
        self.attention = None  # [batch_size, n_head, num_steps, num_steps]

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor,
                training: Any) -> Tensor:
        """
        参数:
        q: [batch_size, num_steps, num_hiddens]
        k: [batch_size, num_steps, num_hiddens]
        v: [batch_size, num_steps, num_hiddens]
        mask: [batch_size, 1, num_steps, num_steps]
              最后两个维度的形状[num_steps, num_steps]其实就是attention的形状

        返回:
        o: [batch_size, num_steps, num_hiddens]
        """
        # residual connect
        # residual.shape [batch_size, num_steps, num_hiddens]
        residual = q

        # linear projection
        # key.shape [batch_size, num_steps, n_head * head_dim]
        # value.shape [batch_size, num_steps, n_head * head_dim]
        # query.shape [batch_size, num_steps, n_head * head_dim]
        key = self.wk(k)
        value = self.wv(v)
        query = self.wq(q)

        # split by head
        # query.shape [batch_size, n_head, num_steps, head_dim]
        # key.shape [batch_size, n_head, num_steps, head_dim]
        # value.shape [batch_size, n_head, num_steps, head_dim]
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)
        # context.shape [batch_size, num_steps, n_head * head_dim]
        #               [batch_size, num_steps, num_hiddens]
        context = self.scaled_dot_product_attention(query, key, value, mask)
        # o.shape [batch_size, num_steps, num_hiddens]
        o = self.o_dense(context)
        o = self.o_drop(o)

        # o.shape [batch_size, num_steps, num_hiddens]
        o = self.layer_norm(residual + o)
        return o

    def split_heads(self, x: Tensor) -> Tensor:
        """
        参数:
        x: [batch_size, num_steps, n_head * head_dim]

        返回:
        output: [batch_size, n_head, num_steps, head_dim]
        """
        # x.shape [batch_size, num_steps, n_head, head_dim]
        x = torch.reshape(x,
                          (x.shape[0], x.shape[1], self.n_head, self.head_dim))
        # output.shape [batch_size, n_head, num_steps, head_dim]
        return x.permute(0, 2, 1, 3)

    def scaled_dot_product_attention(self,
                                     q: Tensor,
                                     k: Tensor,
                                     v: Tensor,
                                     mask: Any = None) -> Tensor:
        """
        计算缩放点积注意力

        参数:
        q: [batch_size, n_head, num_steps, head_dim]
        k: [batch_size, n_head, num_steps, head_dim]
        v: [batch_size, n_head, num_steps, head_dim]
        mask: [batch_size, 1, num_steps, num_steps]

        返回:
        context: [batch_size, num_steps, n_head * head_dim]
        """
        # dk标量
        dk = torch.tensor(k.shape[-1]).type(torch.float)
        # q.shape [batch_size, n_head, num_steps, head_dim]
        # k被处理成: [batch_size, n_head, head_dim, num_steps]
        # score.shape [batch_size, n_head, num_steps, num_steps]
        score = torch.matmul(q, k.permute(0, 1, 3,
                                          2)) / (torch.sqrt(dk) + 1e-8)
        if mask is not None:
            # 将mask为True的位置赋予一个很大的负数, 这样计算softmax的时候
            # 这一项就接近于0
            score = score.masked_fill_(mask, -np.inf)
        # self.attention.shape [batch_size, n_head, num_steps, num_steps]
        self.attention = softmax(score, dim=-1)
        # context.shape [batch_size, n_head, num_steps, head_dim]
        context = torch.matmul(self.attention, v)
        # context.shape [batch_size, num_steps, n_head, head_dim]
        context = context.permute(0, 2, 1, 3)
        # context.shape [batch_size, num_steps, n_head * head_dim]
        context = context.reshape((context.shape[0], context.shape[1], -1))
        return context


class PositionWiseFFN(nn.Module):
    """
    基于位置的前馈网络

    >>> batch_size, num_steps, num_hiddens = 2, 10, 32
    >>> ffn = PositionWiseFFN(num_hiddens)
    >>> x = torch.randn((batch_size, num_steps, num_hiddens))
    >>> assert ffn(x).shape == (batch_size, num_steps, num_hiddens)
    """

    def __init__(self, num_hiddens: int, dropout: float = 0.0) -> None:
        """
        参数:
        num_hiddens: 隐藏单元的特征长度(也是多头注意力最终输出的特征长度)
        dropout: dropout rate
        """
        super().__init__()
        dff = num_hiddens * 4
        self.l = nn.Linear(num_hiddens, dff)
        self.o = nn.Linear(dff, num_hiddens)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(num_hiddens)

    def forward(self, x: Tensor) -> Tensor:
        """
        参数:
        x: [batch_size, num_steps, num_hiddens]

        输出:
        o: batch_size, num_steps, num_hiddens]
        """
        # o.shape [batch_size, num_steps, num_hiddens * 4]
        o = relu(self.l(x))
        # o.shape [batch_size, num_steps, num_hiddens]
        o = self.o(o)
        o = self.dropout(o)
        # o.shape [batch_size, num_steps, num_hiddens]
        o = self.layer_norm(x + o)
        return o


class EncoderLayer(nn.Module):
    """
    >>> batch_size, num_steps, num_hiddens = 2, 10, 32
    >>> batch_size, num_steps, num_hiddens = 2, 10, 32
    >>> encoder_layer = EncoderLayer(2, num_hiddens, 0.1)
    >>> xz = torch.randn((batch_size, num_steps, num_hiddens))
    >>> mask = torch.randint(0, 2, (batch_size, 1, num_steps, num_steps))
    >>> assert encoder_layer(xz, None, mask).shape == (batch_size, num_steps, num_hiddens)
    """

    def __init__(self, n_head: int, num_hiddens: int,
                 drop_rate: float) -> None:
        """
        参数:
        n_head: 头数
        num_hiddens: 隐藏单元的特征长度(也是多头注意力最终输出的特征长度)
        drop_rate: dropout rate
        """
        super().__init__()
        self.mh = MultiHead(n_head, num_hiddens, drop_rate)
        self.ffn = PositionWiseFFN(num_hiddens, drop_rate)

    def forward(self, xz: Tensor, training: Any, mask: Tensor) -> Tensor:
        """
        参数:
        xz: [batch_size, num_steps, num_hiddens]
        mask: [batch_size, 1, num_steps, num_steps]

        返回:
        o: [batch_size, num_steps, num_hiddens]
        """
        # context.shape [batch_size, num_steps, num_hiddens]
        context = self.mh(xz, xz, xz, mask, training)
        # o.shape [batch_size, num_steps, num_hiddens]
        o = self.ffn(context)
        return o


class Encoder(nn.Module):
    """
    Transformer Encoder
    >>> batch_size, num_steps, num_hiddens = 2, 10, 32
    >>> batch_size, num_steps, num_hiddens = 2, 10, 32
    >>> encoder = Encoder(2, num_hiddens, 0.1, 4)
    >>> xz = torch.randn((batch_size, num_steps, num_hiddens))
    >>> mask = torch.randint(0, 2, (batch_size, 1, num_steps, num_steps))
    >>> assert encoder(xz, None, mask).shape == (batch_size, num_steps, num_hiddens)
    """

    def __init__(self, n_head: int, num_hiddens: int, drop_rate: float,
                 n_layer: int) -> None:
        """
        参数:
        n_head: 头数
        num_hiddens: 隐藏单元的特征长度(也是多头注意力最终输出的特征长度)
        drop_rate: dropout rate
        n_layer: 层数(EncoderLayer的数量)
        """
        super().__init__()
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(n_head, num_hiddens, drop_rate)
            for _ in range(n_layer)
        ])

    def forward(self, xz: Tensor, training: Any, mask: Tensor) -> Tensor:
        """
        参数:
        xz: [batch_size, num_steps, num_hiddens]
        mask: [batch_size, 1, num_steps, num_steps]

        返回:
        xz: [batch_size, num_steps, num_hiddens]
        """
        for encoder in self.encoder_layers:
            # xz.shape [batch_size, num_steps, num_hiddens]
            xz = encoder(xz, training, mask)
        return xz


class DecoderLayer(nn.Module):
    """
    >>> batch_size, num_steps, num_hiddens = 2, 10, 32
    >>> batch_size, num_steps, num_hiddens = 2, 10, 32
    >>> decoder_layer = DecoderLayer(2, num_hiddens, 0.1)
    >>> yz = torch.randn((batch_size, num_steps, num_hiddens))
    >>> xz = torch.randn((batch_size, num_steps, num_hiddens))
    >>> yz_look_ahead_mask = torch.randint(0, 2,
                                           (batch_size, 1, num_steps, num_steps))
    >>> xz_pad_mask = torch.randint(0, 2, (batch_size, 1, num_steps, num_steps))
    >>> assert decoder_layer(yz, xz, None, yz_look_ahead_mask,
                             xz_pad_mask).shape == (batch_size, num_steps, num_hiddens)
    """

    def __init__(self, n_head: int, num_hiddens: int,
                 drop_rate: float) -> None:
        """
        n_head: 头数
        num_hiddens: 隐藏单元的特征长度(也是多头注意力最终输出的特征长度)
        drop_rate: dropout rate
        n_layer: 层数(EncoderLayer的数量)
        """
        super().__init__()
        # 2个MultiHead
        self.mh = nn.ModuleList(
            [MultiHead(n_head, num_hiddens, drop_rate) for _ in range(2)])
        self.ffn = PositionWiseFFN(num_hiddens, drop_rate)

    def forward(self, yz: Tensor, xz: Tensor, training: Any,
                yz_look_ahead_mask: Tensor, xz_pad_mask: Tensor) -> Tensor:
        """
        参数:
        yz: [batch_size, num_steps, num_hiddens]
        xz: [batch_size, num_steps, num_hiddens]
        yz_look_ahead_mask: [batch_size, 1, num_steps, num_steps]
        xz_pad_mask: [batch_size, 1, num_steps, num_steps]

        返回:
        dec_output: [batch_size, num_steps, num_hiddens]
        """
        # dec_output.shape [batch_size, num_steps, num_hiddens]
        dec_output = self.mh[0](yz, yz, yz, yz_look_ahead_mask, training)
        # dec_output.shape [batch_size, num_steps, num_hiddens]
        dec_output = self.mh[1](dec_output, xz, xz, xz_pad_mask, training)
        # dec_output.shape [batch_size, num_steps, num_hiddens]
        dec_output = self.ffn(dec_output)

        return dec_output


class Decoder(nn.Module):
    """
    >>> batch_size, num_steps, num_hiddens = 2, 10, 32
    >>> batch_size, num_steps, num_hiddens = 2, 10, 32
    >>> decoder = Decoder(2, num_hiddens, 0.1, 4)
    >>> yz = torch.randn((batch_size, num_steps, num_hiddens))
    >>> xz = torch.randn((batch_size, num_steps, num_hiddens))
    >>> yz_look_ahead_mask = torch.randint(0, 2,
                                          (batch_size, 1, num_steps, num_steps))
    >>> xz_pad_mask = torch.randint(0, 2, (batch_size, 1, num_steps, num_steps))
    >>> assert decoder(yz, xz, None, yz_look_ahead_mask,
                       xz_pad_mask).shape == (batch_size, num_steps, num_hiddens)
    """

    def __init__(self, n_head: int, num_hiddens: int, drop_rate: float,
                 n_layer: int) -> None:
        """
        参数:
        n_head: 头数
        num_hiddens: 隐藏单元的特征长度(也是多头注意力最终输出的特征长度)
        drop_rate: dropout rate
        n_layer: 层数(DecoderLayer的数量)
        """
        super().__init__()

        self.num_layers = n_layer

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(n_head, num_hiddens, drop_rate)
            for _ in range(n_layer)
        ])

    def forward(self, yz: Tensor, xz: Tensor, training: Any,
                yz_look_ahead_mask: Tensor, xz_pad_mask: Tensor) -> Tensor:
        """
        参数:
        yz: [batch_size, num_steps, num_hiddens]
        xz: [batch_size, num_steps, num_hiddens]
        yz_look_ahead_mask: [batch_size, 1, num_steps, num_steps]
        xz_pad_mask: [batch_size, 1, num_steps, num_steps]

        返回:
        yz: [batch_size, num_steps, num_hiddens]
        """
        for decoder in self.decoder_layers:
            # yz.shape [batch_size, num_steps, num_hiddens]
            yz = decoder(yz, xz, training, yz_look_ahead_mask, xz_pad_mask)
        return yz


class PositionEmbedding(nn.Module):
    """
    Embedding & Position Encoding

    >>> batch_size, num_steps = 2, 10
    >>> max_len, num_hiddens, vocab_size = 200, 100, 10000
    >>> embed = PositionEmbedding(max_len, num_hiddens, vocab_size)
    >>> X = torch.ones((batch_size, num_steps), dtype=torch.long)
    >>> assert embed(X).shape == (batch_size, num_steps, num_hiddens)
    """

    def __init__(self, max_len: int, num_hiddens: int,
                 vocab_size: int) -> None:
        """
        参数:
        max_len: 内部PE的最大长度(需要满足max_len >= num_steps)
        num_hiddens: Embedding的输出维度
        vocab_size: 词表大小
        """
        super().__init__()
        # pos.shape [max_len, 1]
        pos = np.expand_dims(np.arange(max_len), 1)
        # pe.shape [max_len, num_hiddens]
        pe = pos / np.power(
            1000,
            2 * np.expand_dims(np.arange(num_hiddens) // 2, 0) / num_hiddens)
        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])
        # pe.shape [1, max_len, num_hiddens]
        pe = np.expand_dims(pe, 0)
        self.pe = torch.from_numpy(pe).type(torch.float32)
        self.embeddings = nn.Embedding(vocab_size, num_hiddens)
        self.embeddings.weight.data.normal_(0, 0.1)

    def forward(self, x: Tensor) -> Tensor:
        """
        1. Embedding
        2. Position Encoding

        参数:
        x: [batch_size, num_steps]

        返回:
        x_embed: [batch_size, num_steps, num_hiddens]
        """
        device = self.embeddings.weight.device
        self.pe = self.pe.to(device)
        # x经过embedding之后的形状为: [batch_size, num_steps, num_hiddens]
        # pe经过处理后的形状为: [1, max_len, num_hiddens]
        #                 -> [1, num_steps, num_hiddens]
        # x_embed.shape [batch_size, num_steps, num_hiddens]
        x_embed = self.embeddings(x) + self.pe[:, :x.shape[1], :]
        return x_embed


class Transformer(nn.Module):
    """
    # `训练模式`的前向传播
    >>> x = torch.ones((batch_size, num_steps), dtype=torch.long)
    >>> y = torch.ones((batch_size, num_steps), dtype=torch.long)
    >>> assert model(x, y, None).shape == (batch_size, num_steps, vocab_size)
    >>> attentions = model.attentions
    >>> for i in range(n_layer):
    >>>     encoder = attentions['encoder'][i]
    >>>     decoder_mh1 = attentions['decoder']['mh1'][i]
    >>>     decoder_mh2 = attentions['decoder']['mh2'][i]
    >>>     assert encoder.shape == (batch_size, n_head, num_steps, num_steps)
    >>>     assert decoder_mh1.shape == (batch_size, n_head, num_steps, num_steps)
    >>>     assert decoder_mh2.shape == (batch_size, n_head, num_steps, num_steps)

    # `预测模式`的前向传播
    >>> x = torch.ones((1, max_len), dtype=torch.long)
    >>> assert model.translate(x, dataset.token_to_id,
                               dataset.id_to_token).shape == (1, max_len + 1)
    >>> attentions = model.attentions
    >>> for i in range(n_layer):
    >>>     encoder = attentions['encoder'][i]
    >>>     decoder_mh1 = attentions['decoder']['mh1'][i]
    >>>     decoder_mh2 = attentions['decoder']['mh2'][i]
    >>>     assert encoder.shape == (1, n_head, max_len, max_len)
    >>>     assert decoder_mh1.shape == (1, n_head, max_len, max_len)
    >>>     assert decoder_mh2.shape == (1, n_head, max_len, max_len)
    """

    def __init__(self,
                 vocab_size: int,
                 max_len: int,
                 n_layer: int = 6,
                 num_hiddens: int = 512,
                 n_head: int = 8,
                 drop_rate: float = 0.1,
                 padding_idx: int = 0) -> None:
        """
        参数:
        vocab_size: 词表大小
        max_len: 预测序列的最大长度
        n_layer: 层数(EncoderLayer & DecoderLayer的数量)
        num_hiddens: 隐藏单元的特征长度(也是多头注意力最终输出的特征长度)
        n_head: 头数
        drop_rate: dropout rate
        padding_idx: <PAD>对应的token_id
        """
        super().__init__()
        self.max_len = max_len
        self.padding_idx = torch.tensor(padding_idx)
        self.vocab_size = vocab_size

        self.embed = PositionEmbedding(max_len, num_hiddens, vocab_size)
        self.encoder = Encoder(n_head, num_hiddens, drop_rate, n_layer)
        self.decoder = Decoder(n_head, num_hiddens, drop_rate, n_layer)
        self.o = nn.Linear(num_hiddens, vocab_size)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.002)

    def forward(self, x: Tensor, y: Tensor, training: Any = None) -> Tensor:
        """
        `训练模式`的前向传播
    
        1. Encoder前向传播
        2. Decoder前向传播

        参数:
        x.shape [batch_size, num_steps]
        y.shape [batch_size, num_steps]

        输出:
        o: [batch_size, num_steps, vocab_size]
        """
        # Embedding & Position Encoding
        # x_embed.shape [batch_size, num_steps, num_hiddens]
        # y_embed.shape [batch_size, num_steps, num_hiddens]
        x_embed, y_embed = self.embed(x), self.embed(y)
        # pad_mask.shape [batch_size, 1, num_steps, num_steps]
        pad_mask = self._pad_mask(x)
        # encoded_z.shape [batch_size, num_steps, num_hiddens]
        encoded_z = self.encoder(x_embed, training, pad_mask)
        # yz_look_ahead_mask.shape [batch_size, 1, num_steps, num_steps]
        yz_look_ahead_mask = self._look_ahead_mask(y)
        # decode_z.shape [batch_size, num_steps, num_hiddens]
        decoded_z = self.decoder(y_embed, encoded_z, training,
                                 yz_look_ahead_mask, pad_mask)
        # o.shape [batch_size, num_steps, vocab_size]
        o = self.o(decoded_z)
        return o

    def step(self, x: Tensor, y: Tensor) -> Tuple[np.ndarray, Tensor]:
        """
        训练一个批量的数据

        输入:
        x: [batch_size, num_steps]
        y: [batch_size, num_steps+1]

        返回: (loss, logits)
        loss: 标量
        logits: [batch_size, num_steps, vocab_size]
        """
        self.optimizer.zero_grad()
        # x.shape [batch_size, num_steps]
        # y被处理成: [batch_size, num_steps]
        # logits.shape [batch_size, num_steps, vocab_size]
        logits = self(x, y[:, :-1], training=True)
        # pad_mask.shape [batch_size, num_steps]
        pad_mask = ~torch.eq(y[:, 1:], self.padding_idx)  # [n, seq_len]
        # logits处理成: [batch_size*num_steps, vocab_size]
        # y处理成: [batch_size*num_steps, ]
        # loss: 标量
        loss = cross_entropy(logits.reshape(-1, self.vocab_size),
                             y[:, 1:].reshape(-1))
        loss.backward()
        self.optimizer.step()
        return loss.cpu().data.numpy(), logits

    def _pad_bool(self, seqs: Tensor) -> Tensor:
        """
        序列中为<PAD>的masks位置设置为True

        参数:
        seqs: [batch_size, num_steps]

        返回:
        o: [batch_size, num_steps]
        """
        o = torch.eq(seqs, self.padding_idx)
        return o

    def _pad_mask(self, seqs: Tensor) -> Tensor:
        """
        序列中为<PAD>的masks位置设置为True

        参数:
        seqs: [batch_size, num_steps]

        返回:
        mask: [batch_size, 1, num_steps, num_steps]
        """
        len_q = seqs.size(1)
        # mask.shape [batch_size, num_steps, num_steps]
        mask = self._pad_bool(seqs).unsqueeze(1).expand(-1, len_q, -1)
        # mask.shape [batch_size, 1, num_steps, num_steps]
        return mask.unsqueeze(1)

    def _look_ahead_mask(self, seqs):
        """
        表示seqs序列最多能看多少个steps, 为True的位置会被屏蔽
        掉不能看. 要注意的是如果seqs有效token比较短, 那么<PAD>
        的位置也会设置为True

        e.g. batch_size=2, num_step=4, 则mask如下:
        [[[[False,  True,  True,  True],
           [False, False,  True,  True],
           [False, False, False,  True],
           [False, False, False, False]]],
         [[[False,  True,  True,  True],
           [False, False,  True,  True],
           [False, False, False,  True],
           [False, False, False, False]]]]

        参数:
        seps: [batch_size, num_steps]

        返回:
        mask: [batch_size, 1, num_steps, num_steps]
        """
        device = next(self.parameters()).device
        batch_size, seq_len = seqs.shape
        # e.g. seq_len=4, mask如下:
        # [[0, 1, 1, 1],
        #  [0, 0, 1, 1],
        #  [0, 0, 0, 1],
        #  [0, 0, 0, 0]]
        # mask.shape [num_steps, num_steps]
        mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.long),
                          diagonal=1).to(device)

        # e.g. batch_size=2, seq_len=4, mask如下:
        # 表示seqs序列最多能看多少个steps, 为1的位置会被屏蔽
        # 掉不能看. 要注意的是如果seqs有效token比较短, 那么<PAD>
        # 的位置也会设置为1
        # [[[[0, 1, 1, 1],
        #    [0, 0, 1, 1],
        #    [0, 0, 0, 1],
        #    [0, 0, 0, 0]]],
        #  [[[0, 1, 1, 1],
        #    [0, 0, 1, 1],
        #    [0, 0, 0, 1],
        #    [0, 0, 0, 0]]]]
        #
        # 将seqs处理成: [batch_size, 1, 1, num_steps]
        # 将mask处理成: [1, 1, num_steps, num_steps]
        # mask.shape [batch_size, 1, num_steps, num_steps]
        mask = torch.where(
            self._pad_bool(seqs)[:, None, None, :], 1,
            mask[None, None, :, :]).to(device)

        # mask.shape [batch_size, 1, num_steps, num_steps]
        return mask > 0

    def translate(self, src: Tensor, token_to_id: Dict[str, int],
                  id_to_token: Dict[int, str]):
        """
        `预测模式`的前向传播

        先初始化一个target, 形状为: [1, MAX_LEN + 1], 内容为[<BOS>, <PAD>, <PAD>...],
        之后一个token一个token的预测, 每预测一个token, 更新对应target位置的token id
        
        target会如下变化:
        [<BOS>, <PAD>, <PAD>...]
        [token_id1, <PAD>, <PAD>...]
        [token_id1, token_id2, <PAD>...]
        ...

        参数:
        src: [1, MAX_LEN]

        返回:
        target: [1, MAX_LEN + 1]
        """
        self.eval()
        device = next(self.parameters()).device
        # src_pad.shape [1, MAX_LEN]
        src_pad = src
        # 初始化target:
        # target.shape [1, MAX_LEN + 1]
        # [1, 1]  - <BOS>
        # [1, 1:] - <PAD>
        target = torch.from_numpy(
            pad_zero(
                np.array([[
                    token_to_id["<BOS>"],
                ] for _ in range(len(src))]), self.max_len + 1)).to(device)
        # x_embed.shape [1, MAX_LEN, num_hiddens]
        x_embed = self.embed(src_pad)
        # encoded_z.shape [1, MAX_LEN, num_hiddens]
        encoded_z = self.encoder(x_embed, False, mask=self._pad_mask(src_pad))
        # 一个token一个token的预测
        for i in range(0, self.max_len):
            # y.shape [1, MAX_LEN]
            y = target[:, :-1]
            # y_embed.shape [1, MAX_LEN, num_hiddens]
            y_embed = self.embed(y)
            # decoded_z.shape [1, MAX_LEN, num_hiddens]
            decoded_z = self.decoder(y_embed, encoded_z, False,
                                     self._look_ahead_mask(y),
                                     self._pad_mask(src_pad))
            # o.shape [1, MAX_LEN, vocab_size]
            #      -> [1, vocab_size]
            o = self.o(decoded_z)[:, i, :]
            # idx.shape [1, 1]
            idx = o.argmax(dim=1).detach()
            # 更新target
            target[:, i + 1] = idx
        return target

    @property
    def attentions(self) -> Tuple[Dict[str, List[Tensor]]]:
        """
        做一次前向传播(训练模式和预测模式), 会更新一次attentions

        返回: attentions
        {
            'encoder': List of [batch_size, n_head, num_steps, num_steps]
            'decoder': {
                'mh1': List of [batch_size, n_head, num_steps, num_steps]
                'mh2': List of [batch_size, n_head, num_steps, num_steps]
            }
        }
        """
        attentions = {
            "encoder": [
                # [batch_size, n_head, num_steps, num_steps]
                l.mh.attention.cpu().data.numpy()
                for l in self.encoder.encoder_layers
            ],
            "decoder": {
                "mh1": [
                    # [batch_size, n_head, num_steps, num_steps]
                    l.mh[0].attention.cpu().data.numpy()
                    for l in self.decoder.decoder_layers
                ],
                "mh2": [
                    # [batch_size, n_head, num_steps, num_steps]
                    l.mh[1].attention.cpu().data.numpy()
                    for l in self.decoder.decoder_layers
                ],
            }
        }
        return attentions


def try_gpu(i: int = 0) -> torch.device:
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def train(
        num_epochs: int = 100,
        num_hiddens: int = 32,
        n_layer: int = 3,
        n_head: int = 4,
        device: torch.device = None) -> Tuple[nn.Module, DataLoader, DateData]:
    data_iter, dataset = load_date()
    print("Chinese time order: yy/mm/dd ", dataset.date_cn[:3],
          "\nEnglish time order: dd/M/yyyy", dataset.date_en[:3])
    print("Vocabularies: ", dataset.vocab)
    print(
        f"x index sample:  \n{dataset.idx2str(dataset.x[0])}\n{dataset.x[0]}",
        f"\ny index sample:  \n{dataset.idx2str(dataset.y[0])}\n{dataset.y[0]}"
    )
    model = Transformer(vocab_size=dataset.num_word,
                        max_len=MAX_LEN,
                        n_layer=n_layer,
                        num_hiddens=num_hiddens,
                        n_head=n_head,
                        drop_rate=0.1,
                        padding_idx=0)
    model = model.to(device)
    times = []
    history = [[]]  # 记录: 训练集损失, 方便后续绘图
    num_batches = len(data_iter)
    for epoch in range(num_epochs):
        # 训练
        metric_train = [0.0] * 2  # 统计: 训练集损失之和, 训练集样本数量之和
        data_iter_tqdm = tqdm(data_iter, file=sys.stdout)
        for i, batch in enumerate(data_iter_tqdm):
            t_start = time.time()
            # bx.shape [batch_size, num_steps_x]
            # by.shape [batch_size, num_steps_y]
            bx, by, decoder_len = batch
            # bx.shape [batch_size, MAX_LEN]
            # by.shape [batch_size, MAX_LEN+1]
            bx, by = torch.from_numpy(pad_zero(bx, max_len=MAX_LEN)).type(
                torch.LongTensor).to(device), torch.from_numpy(
                    pad_zero(by,
                             MAX_LEN + 1)).type(torch.LongTensor).to(device)
            loss, logits = model.step(bx, by)
            with torch.no_grad():
                metric_train[0] += float(loss * bx.shape[0])
                metric_train[1] += float(bx.shape[0])
            times.append(time.time() - t_start)
            train_loss = metric_train[0] / metric_train[1]
            if i % 50 == 0:
                target = dataset.idx2str(by[0, 1:-1].cpu().data.numpy())
                # pred.shape [1, MAX_LEN + 1]
                pred = model.translate(bx[0:1], dataset.token_to_id,
                                       dataset.id_to_token)
                res = dataset.idx2str(pred[0].cpu().data.numpy())
                src = dataset.idx2str(bx[0].cpu().data.numpy())
                history[0].append((epoch + (i + 1) / num_batches, train_loss))
                data_iter_tqdm.desc = f'epoch {epoch}, step {i}, train loss {train_loss:.3f}, ' \
                    f'input {src}, target {target}, inference {res}'

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
    return model, data_iter, dataset


def export_attention(model: nn.Module,
                     data_iter: DataLoader,
                     dataset: DateData,
                     device: torch.device = None) -> Dict:
    """
    翻译一句话, 并返回其对应的attentions
    """
    for batch in data_iter:
        # bx.shape [batch_size, num_steps_x]
        # by.shape [batch_size, num_steps_y]
        bx, by, decoder_len = batch
        src = [[dataset.id_to_token[int(i)] for i in bx[j]]
               for j in range(len(bx[0:1]))]
        tgt = [[dataset.id_to_token[int(i)] for i in by[j]]
               for j in range(len(by[0:1]))]
        # bx.shape [batch_size, MAX_LEN]
        # by.shape [batch_size, MAX_LEN+1]
        bx, by = torch.from_numpy(pad_zero(bx, max_len=MAX_LEN)).type(
            torch.LongTensor).to(device), torch.from_numpy(
                pad_zero(by, MAX_LEN + 1)).type(torch.LongTensor).to(device)
        break
    # 送入translate参数的形状: [1, MAX_LEN]
    model.translate(bx[0:1], dataset.token_to_id, dataset.id_to_token)
    attn_data = {"src": src, "tgt": tgt, "attentions": model.attentions}
    return attn_data


def all_mask_kinds():
    """
    展示Transformer中两种类型的mask:
    1. padding mask
    2. look ahead mask
    """
    batch_size, max_len = 4, 6
    seqs = [
        "I love you", "My name is M", "This is a very long seq", "Short one"
    ]
    # vocabs:
    # {'Short', 'love', 'This', 'seq', 'is', 'long', 'My',
    #  'one', 'a', 'you', 'name', 'M', 'very', 'I'}
    vocabs = set((" ".join(seqs)).split(" "))
    id_to_token = {i: v for i, v in enumerate(vocabs, start=1)}
    id_to_token[0] = '<PAD>'  # add 0 idx for <PAD>
    token_to_id = {v: i for i, v in id_to_token.items()}

    # id_seqs:
    # [[1, 11, 13],
    #  [14, 10, 3, 5],
    #  [2, 3, 8, 4, 7, 12],
    #  [9, 6]]
    id_seqs = [[token_to_id[v] for v in seq.split(" ")] for seq in seqs]
    # padded_id_seqs.shape [max_len, max_len]
    # [[10  9  1  0  0  0]
    #  [14  2  7  5  0  0]
    #  [11  7  3  6 12  8]
    #  [13  4  0  0  0  0]]
    padded_id_seqs = np.array([l + [0] * (max_len - len(l)) for l in id_seqs])
    # pmask.shape [max_len, max_len]
    # [[0 0 0 1 1 1]
    #  [0 0 0 0 1 1]
    #  [0 0 0 0 0 0]
    #  [0 0 1 1 1 1]]
    pmask = np.where(padded_id_seqs == 0, np.ones_like(padded_id_seqs),
                     np.zeros_like(padded_id_seqs))  # 0 idx is padding
    # pmash.shape [batch_size, max_len, max_len]
    # [[[0 0 0 1 1 1]
    #   [0 0 0 1 1 1]
    #   [0 0 0 1 1 1]
    #   [0 0 0 1 1 1]
    #   [0 0 0 1 1 1]
    #   [0 0 0 1 1 1]]
    #  [[0 0 0 0 1 1]
    #   [0 0 0 0 1 1]
    #   [0 0 0 0 1 1]
    #   [0 0 0 0 1 1]
    #   [0 0 0 0 1 1]
    #   [0 0 0 0 1 1]]
    #  [[0 0 0 0 0 0]
    #   [0 0 0 0 0 0]
    #   [0 0 0 0 0 0]
    #   [0 0 0 0 0 0]
    #   [0 0 0 0 0 0]
    #   [0 0 0 0 0 0]]
    #  [[0 0 1 1 1 1]
    #   [0 0 1 1 1 1]
    #   [0 0 1 1 1 1]
    #   [0 0 1 1 1 1]
    #   [0 0 1 1 1 1]
    #   [0 0 1 1 1 1]]]
    pmask = np.repeat(pmask[:, None, :], pmask.shape[-1], axis=1)
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
    for i in range(1, batch_size + 1):
        plt.subplot(2, 2, i)
        plt.imshow(pmask[i - 1], vmax=1, vmin=0, cmap="YlGn")
        labels = seqs[i - 1].split(" ")
        plt.xticks(range(max_len),
                   labels + [' '] * (max_len - len(labels)),
                   rotation=45)
        plt.yticks(
            range(max_len),
            labels + [' '] * (max_len - len(labels)),
        )
        plt.grid(which="minor", c="w", lw=0.5, linestyle="-")
    plt.tight_layout()
    plt.show()

    # look ahead mask
    # omask.shape [max_len, max_len]
    # [[ True False False False False False]
    #  [ True  True False False False False]
    #  [ True  True  True False False False]
    #  [ True  True  True  True False False]
    #  [ True  True  True  True  True False]
    #  [ True  True  True  True  True  True]]
    omask = ~np.triu(np.ones((max_len, max_len), dtype=np.bool), 1)
    # omask.shape [batch_size, max_len, max_len]
    omask = np.tile(np.expand_dims(omask, axis=0), [np.shape(seqs)[0], 1, 1])
    # omask.shape [batch_size, max_len, max_len]
    # [[[0 1 1 1 1 1]
    #   [0 0 1 1 1 1]
    #   [0 0 0 1 1 1]
    #   [0 0 0 1 1 1]
    #   [0 0 0 1 1 1]
    #   [0 0 0 1 1 1]]
    #  [[0 1 1 1 1 1]
    #   [0 0 1 1 1 1]
    #   [0 0 0 1 1 1]
    #   [0 0 0 0 1 1]
    #   [0 0 0 0 1 1]
    #   [0 0 0 0 1 1]]
    #  [[0 1 1 1 1 1]
    #   [0 0 1 1 1 1]
    #   [0 0 0 1 1 1]
    #   [0 0 0 0 1 1]
    #   [0 0 0 0 0 1]
    #   [0 0 0 0 0 0]]
    #  [[0 1 1 1 1 1]
    #   [0 0 1 1 1 1]
    #   [0 0 1 1 1 1]
    #   [0 0 1 1 1 1]
    #   [0 0 1 1 1 1]
    #   [0 0 1 1 1 1]]]
    omask = np.where(omask, pmask, 1)

    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
    for i in range(1, batch_size + 1):
        plt.subplot(2, 2, i)
        plt.imshow(omask[i - 1], vmax=1, vmin=0, cmap="YlGn")
        labels = seqs[i - 1].split(" ")
        plt.xticks(range(max_len),
                   labels + [' '] * (max_len - len(labels)),
                   rotation=45)
        plt.yticks(
            range(max_len),
            labels + [' '] * (max_len - len(labels)),
        )
        plt.grid(which="minor", c="w", lw=0.5, linestyle="-")
    plt.tight_layout()
    plt.show()


def position_embedding():
    """
    显示位置编码
    """
    max_len = 500
    num_hiddens = 512
    pos = np.arange(max_len)[:, None]
    pe = pos / np.power(10000, 2. * np.arange(num_hiddens)[None, :] /
                        num_hiddens)  # [max_len, num_hiddens]
    pe[:, 0::2] = np.sin(pe[:, 0::2])
    pe[:, 1::2] = np.cos(pe[:, 1::2])
    plt.imshow(pe, vmax=1, vmin=-1, cmap="rainbow")
    plt.ylabel("word position")
    plt.xlabel("embedding dim")
    plt.show()


def transformer_attention_matrix(data: Dict, case: int = 0) -> None:
    """
    显示3个注意力权重:
    1. Encoder self-attention    [len(src), len(src)]
    2. Decoder self-attention    [len(tgt), len(tgt)]
    3. Decoder-Encoder attention [len(tgt), len(src)]

    参数:
    case: 表示样本索引
    """
    n_layer = 3
    n_head = 4
    # src
    # e.g. ['0', '7', '-', '0', '8', '-', '0', '5']
    src = data["src"][case]
    # tgt
    # e.g. ['<BOS>', '0', '5', '/', 'Aug', '/', '2', '0', '0', '7', '<EOS>']
    tgt = data["tgt"][case]
    attentions = data["attentions"]
    # 1. Encoder self-attention
    # encoder_atten: List of [batch_size, n_head, num_steps, num_steps]
    encoder_atten = attentions["encoder"]
    # 2. Decoder self-attention
    # decoder_tgt_atten: List of [batch_size, n_head, num_steps, num_steps]
    decoder_tgt_atten = attentions["decoder"]["mh1"]
    # 3. Decoder-Encoder attention
    # decoder_src_atten: List of [batch_size, n_head, num_steps, num_steps]
    decoder_src_atten = attentions["decoder"]["mh2"]
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True

    plt.figure(0, (7, 7))
    plt.suptitle("Encoder self-attention")
    # 遍历前三层
    for i in range(n_layer):
        for j in range(n_head):
            plt.subplot(n_layer, n_head, i * 4 + j + 1)
            plt.imshow(encoder_atten[i][case, j][:len(src), :len(src)],
                       vmax=1,
                       vmin=0,
                       cmap="rainbow")
            plt.xticks(range(len(src)), src)
            plt.yticks(range(len(src)), src)
            if j == 0:
                plt.ylabel("layer %i" % (i + 1))
            if i == 2:
                plt.xlabel("head %i" % (j + 1))
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

    plt.figure(1, (7, 7))
    plt.suptitle("Decoder self-attention")
    for i in range(n_layer):
        for j in range(n_head):
            plt.subplot(n_layer, n_head, i * n_head + j + 1)
            plt.imshow(decoder_tgt_atten[i][case, j][:len(tgt), :len(tgt)],
                       vmax=1,
                       vmin=0,
                       cmap="rainbow")
            plt.xticks(range(len(tgt)), tgt, rotation=90, fontsize=7)
            plt.yticks(range(len(tgt)), tgt, fontsize=7)
            if j == 0:
                plt.ylabel("layer %i" % (i + 1))
            if i == 2:
                plt.xlabel("head %i" % (j + 1))
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

    plt.figure(2, (7, 8))
    plt.suptitle("Decoder-Encoder attention")
    for i in range(n_layer):
        for j in range(n_head):
            plt.subplot(n_layer, n_head, i * 4 + j + 1)
            plt.imshow(decoder_src_atten[i][case, j][:len(tgt), :len(src)],
                       vmax=1,
                       vmin=0,
                       cmap="rainbow")
            plt.xticks(range(len(src)), src, fontsize=7)
            plt.yticks(range(len(tgt)), tgt, fontsize=7)
            if j == 0:
                plt.ylabel("layer %i" % (i + 1))
            if i == 2:
                plt.xlabel("head %i" % (j + 1))
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()


def transformer_attention_line(data: Dict, case: int = 0) -> None:
    """
    显示注意力权重:
    Decoder-Encoder attention

    参数:
    case: 表示样本索引
    """
    # src
    # e.g. ['3', '2', '-', '0', '6', '-', '1', '4']
    src = data["src"][case]
    # tgt
    # e.g. ['<BOS>', '1', '4', '/', 'Jun', '/', '2', '0', '3', '2', '<EOS>']
    tgt = data["tgt"][case]
    attentions = data["attentions"]
    # Decoder-Encoder attention
    # decoder_src_atten: List of [batch_size, n_head, num_steps, num_steps]
    decoder_src_atten = attentions["decoder"]["mh2"]
    # tgt_label - 长度为10
    # e.g. ['<EOS>', '2', '3', '0', '2', '/', 'Jun', '/', '4', '1']
    tgt_label = tgt[1:11][::-1]
    # src_label - 长度为10
    # e.g. ['', '', '4', '1', '-', '6', '0', '-', '2', '3']
    src_label = ["" for _ in range(2)] + src[::-1]
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(7, 14))

    # n_head=4, 显示4个head的注意力
    for i in range(2):
        for j in range(2):
            # 设置左侧的y轴 - src
            ax[i, j].set_yticks(np.arange(len(src_label)))
            ax[i, j].set_yticklabels(src_label, fontsize=9)  # src
            ax[i, j].set_ylim(0, len(src_label) - 1)

            # 设置右侧的y轴 - tgt
            ax_ = ax[i, j].twinx()
            ax_.set_yticks(
                np.linspace(ax_.get_yticks()[0],
                            ax_.get_yticks()[-1], len(ax[i, j].get_yticks())))
            ax_.set_yticklabels(tgt_label, fontsize=9)  # tgt
            # 只显示最后一层的注意力
            # img获取注意力 [10, 8]
            img = decoder_src_atten[-1][case, i * 2 + j][:10, :8]
            color = cm.rainbow(np.linspace(0, 1, img.shape[0]))
            # left_top=8, right_top=10
            left_top, right_top = img.shape[1], img.shape[0]
            # 遍历tgt, 每个tgt选一个颜色
            for ri, c in zip(range(right_top), color):  # tgt
                for li in range(left_top):  # src
                    # 取出点[ri, li]的像素值, 像素值越大, alpha值越高
                    alpha = (img[ri, li] / img[ri].max())**8
                    # 点A: [0, left_top - li + 1]
                    # 点B: [1, right_top - 1 - ri]
                    # 点A -> 点B的直线
                    ax[i, j].plot([0, 1],
                                  [left_top - li + 1, right_top - 1 - ri],
                                  alpha=alpha,
                                  c=c)
            ax[i, j].set_xticks(())
            ax[i, j].set_xlabel("head %i" % (j + 1 + i * 2))
            ax[i, j].set_xlim(0, 1)
    plt.subplots_adjust(top=0.9)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    device = try_gpu()
    model, data_iter, dataset = train(num_epochs=5, device=device)
    # Chinese time order: yy/mm/dd  ['31-04-26', '04-07-18', '33-06-06']
    # English time order: dd/M/yyyy ['26/Apr/2031', '18/Jul/2004', '06/Jun/2033']
    # Vocabularies:  {'0', 'Nov', '/', '-', '<PAD>', '2', '1', '7', '9', 'Feb',
    #                 '<EOS>', '4', 'Mar', 'Jun', '3', '8', 'May', 'Jul', '<BOS>',
    #                 '6', 'Oct', 'Sep', 'Apr', '5', 'Jan', 'Aug', 'Dec'}
    # x index sample:
    # 31-04-26
    # [6 4 1 3 7 1 5 9]
    # y index sample:
    # <BOS>26/Apr/2031<EOS>
    # [13  5  9  2 15  2  5  3  6  4 14]
    # epoch 0, step 100, train loss 1.394, input 27-09-16<PAD><PAD><PAD>, target 16/Sep/2027<EOS>, inference <BOS>04/Jun/2004<EOS>: 100%|█| 125/125 [00
    # epoch 1, step 100, train loss 0.322, input 96-06-26<PAD><PAD><PAD>, target 26/Jun/1996<EOS>, inference <BOS>26/Jun/1996<EOS>: 100%|█| 125/125 [00
    # epoch 2, step 100, train loss 0.011, input 28-06-29<PAD><PAD><PAD>, target 29/Jun/2028<EOS>, inference <BOS>29/Jun/2028<EOS>: 100%|█| 125/125 [00
    # epoch 3, step 100, train loss 0.004, input 12-06-12<PAD><PAD><PAD>, target 12/Jun/2012<EOS>, inference <BOS>12/Jun/2012<EOS>: 100%|█| 125/125 [00
    # epoch 4, step 100, train loss 0.002, input 91-07-10<PAD><PAD><PAD>, target 10/Jul/1991<EOS>, inference <BOS>10/Jul/1991<EOS>: 100%|█| 125/125 [00

    # 显示:
    position_embedding()
    all_mask_kinds()
    data = export_attention(model, data_iter, dataset, device=device)
    transformer_attention_matrix(data)
    transformer_attention_line(data)