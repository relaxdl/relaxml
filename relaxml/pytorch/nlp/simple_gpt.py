import os
import re
from typing import Dict, Tuple, Any
import requests
import sys
import time
import hashlib
import zipfile
import tarfile
import numpy as np
import pandas as pd
import torch
from torch import Tensor
import torch.nn as nn
from torch import optim
from matplotlib.pyplot import cm
from torch.nn.functional import cross_entropy, softmax, relu
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
"""
Simple GPT

实现说明:
https://tech.foxrelax.com/nlp/simple_gpt/
"""
"""
Microsoft Research Paraphrase Corpus(MRPC)数据集

1. 一共5列: ['Quality', '#1 ID', '#2 ID', '#1 String', '#2 String']
2. 每行有两句话'#1 String'和'#2 String', 如果他们是语义相同, Quality为1. 反之为0
3. 这份数据集可以做2件事:
   <1> 两句合起来训练文本匹配
   <2> 两句拆开单独对待, 理解人类语言, 学一个语言模型
"""

PAD_ID = 0


def download(cache_dir: str = '../data') -> str:
    """
    下载数据
    """
    sha1_hash = '21a4d37692645502c0ecf3e25688c8ad305213ef'
    url = 'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/msr_paraphrase.zip'
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
    # e.g. ../data/msr_paraphrase.zip
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
    # e.g. ../data/msr_paraphrase
    return data_dir


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


def _text_standardize(text: str) -> str:
    """
    处理文本, 把数字替换成<NUM>

    >>> _text_standardize('I Love-you 123 foo456 3456 bar.')
        I Love-you <NUM> foo456 <NUM> bar.
    """
    text = re.sub(r'—', '-', text)
    text = re.sub(r'–', '-', text)
    text = re.sub(r'―', '-', text)
    text = re.sub(r" \d+(,\d+)?(\.\d+)? ", " <NUM> ", text)
    text = re.sub(r" \d+-+?\d*", " <NUM>-", text)
    return text.strip()


def _process_mrpc(
        dir: str,
        rows: int = None) -> Tuple[Dict, Dict[str, int], Dict[int, str]]:
    """
    处理MRPC数据集

    1. 数字会被处理成<NUM>
    2. 包含了4个特殊token: <PAD>, <MASK>, <SEP>, <CLS>

    返回: (data, token_to_id, id_to_token)
    data: {
        'train': {
            's1': List[str]
            's2': List[str]
            's1id': List[List[int]]
            's2id': List[List[int]]
        },
        'test': {
            's1': List[str]
            's2': List[str]
            's1id': List[List[int]]
            's2id': List[List[int]]
        }
    }
    """
    data = {"train": None, "test": None}
    files = ['msr_paraphrase_train.txt', 'msr_paraphrase_test.txt']
    for f in files:
        # 数据一共5列
        # ['Quality', '#1 ID', '#2 ID', '#1 String', '#2 String']
        df = pd.read_csv(os.path.join(dir, f), sep='\t', nrows=rows)
        k = "train" if "train" in f else "test"
        data[k] = {
            "is_same": df.iloc[:, 0].values,
            "s1": df["#1 String"].values,
            "s2": df["#2 String"].values
        }
    vocab = set()
    # 遍历:
    # data['train']['s1']
    # data['train']['s2']
    # data['test']['s1']
    # data['test']['s2']
    for n in ["train", "test"]:
        for m in ["s1", "s2"]:
            for i in range(len(data[n][m])):
                data[n][m][i] = _text_standardize(data[n][m][i].lower())
                cs = data[n][m][i].split(" ")
                vocab.update(set(cs))
    token_to_id = {v: i for i, v in enumerate(sorted(vocab), start=1)}
    token_to_id["<PAD>"] = PAD_ID
    token_to_id["<MASK>"] = len(token_to_id)
    token_to_id["<SEP>"] = len(token_to_id)
    token_to_id["<CLS>"] = len(token_to_id)
    id_to_token = {i: v for v, i in token_to_id.items()}
    for n in ["train", "test"]:
        for m in ["s1", "s2"]:
            data[n][m + "id"] = [[token_to_id[v] for v in c.split(" ")]
                                 for c in data[n][m]]
    return data, token_to_id, id_to_token


class MRPCData(Dataset):
    """
    只用到了训练集的数据

    返回的序列格式:
    <CLS>s1<SEP>s2<SEP><PAD><PAD>...
    """
    num_seg = 3  # 一共有多少个segment
    pad_id = PAD_ID

    def __init__(self, data_dir: str, rows: int = None) -> None:
        data, self.token_to_id, self.id_to_token = _process_mrpc(
            data_dir, rows)
        # 每个句子都处理成如下格式:
        # <CLS>s1<SEP>s2<SEP><PAD><PAD>...
        # 计算出来是: 72
        self.max_len = max([
            len(s1) + len(s2) + 3
            for s1, s2 in zip(data["train"]["s1id"] +
                              data["test"]["s1id"], data["train"]["s2id"] +
                              data["test"]["s2id"])
        ])
        # xlen List[[s1_len, s2_len]]
        # e.g.
        # [[19 20]
        #  [17 22]
        #  ...
        #  [21 22]
        #  [25 18]]
        self.xlen = np.array(
            [[len(data["train"]["s1id"][i]),
              len(data["train"]["s2id"][i])]
             for i in range(len(data["train"]["s1id"]))],
            dtype=int)
        x = [[self.token_to_id["<CLS>"]] + data["train"]["s1id"][i] +
             [self.token_to_id["<SEP>"]] + data["train"]["s2id"][i] +
             [self.token_to_id["<SEP>"]] for i in range(len(self.xlen))]
        # x List[List[int]]
        # x[0] e.g.
        # [12879   720   336  5432  1723    36 12591  5279  1853   203 11501 12665
        #    203    36  7936  3194  3482  5432  4107    41 12878  9414 11635  5418
        #    965  8022   203 11501 12665   203    36   720   336  5432  1723  7936
        #   3194  3482  5432  4107    41 12878     0     0     0     ...]
        self.x = pad_zero(x, max_len=self.max_len)
        # nsp_y
        # e.g.
        # [[1]
        #  [0]
        #  ...
        #  [0]
        #  [1]]
        self.nsp_y = data["train"]["is_same"][:, None]
        # 编码规则:
        # <CLS>s1<SEP>s2<SEP><PAD><PAD>...
        # <CLS>s1<SEP> - 0
        # s2<SEP> - 1
        # <PAD> - 2
        # seg[0], e.g.
        # [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
        #  1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]
        self.seg = np.full(self.x.shape, self.num_seg - 1, np.int32)
        for i in range(len(x)):
            si = self.xlen[i][0] + 2
            self.seg[i, :si] = 0
            si_ = si + self.xlen[i][1] + 1
            self.seg[i, si:si_] = 1
        # word_ids List[int]
        self.word_ids = np.array(
            list(
                set(self.id_to_token.keys()).difference([
                    self.token_to_id[v] for v in ["<PAD>", "<MASK>", "<SEP>"]
                ])))

    def __getitem__(
            self,
            idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        >>> data_iter, dataset = load_mrpc_data(batch_size=32)
        >>> dataset.max_len
            72
        >>> bx, bs, bl, by = dataset[0]
        >>> assert bx.shape == (dataset.max_len, )
        >>> assert bs.shape == (dataset.max_len, )
        >>> assert bl.shape == (2, )
        >>> assert by.shape == (1, )
        """
        return self.x[idx], self.seg[idx], self.xlen[idx], self.nsp_y[idx]

    def sample(
            self,
            n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        >>> data_iter, dataset = load_mrpc_data(batch_size=32)
        >>> dataset.max_len
            72
        >>> bx, bs, bl, by = dataset.sample(16)
        >>> assert bx.shape == (16, dataset.max_len)
        >>> assert bs.shape == (16, dataset.max_len)
        >>> assert bl.shape == (16, 2)
        >>> assert by.shape == (16, 1)
        """
        bi = np.random.randint(0, self.x.shape[0], size=n)
        bx, bs, bl, by = self.x[bi], self.seg[bi], self.xlen[bi], self.nsp_y[
            bi]
        return bx, bs, bl, by

    @property
    def num_word(self):
        return len(self.token_to_id)

    def __len__(self):
        return len(self.x)

    @property
    def mask_id(self):
        return self.token_to_id["<MASK>"]


class MRPCSingle(Dataset):
    """
    只用到了训练集的数据

    返回的序列格式:
    <CLS>s<SEP><PAD><PAD>...
    """
    pad_id = PAD_ID

    def __init__(self, data_dir: str, rows: int = None) -> None:
        data, self.token_to_id, self.id_to_token = _process_mrpc(
            data_dir, rows)
        # 每个句子都处理成如下格式:
        # <CLS>s<SEP><PAD><PAD>...
        # 计算出来是: 38
        self.max_len = max([
            len(s) + 2 for s in data["train"]["s1id"] + data["train"]["s2id"]
        ])
        x = [[self.token_to_id["<CLS>"]] + data["train"]["s1id"][i] +
             [self.token_to_id["<SEP>"]]
             for i in range(len(data["train"]["s1id"]))]
        x += [[self.token_to_id["<CLS>"]] + data["train"]["s2id"][i] +
              [self.token_to_id["<SEP>"]]
              for i in range(len(data["train"]["s2id"]))]
        # x List[List[int]]
        # x[0] e.g.
        # [12879   720   336  5432  1723    36 12591  5279  1853   203 11501 12665
        #    203    36  7936  3194  3482  5432  4107    41 12878     0     0     0
        #      0     0     0     0     ...]
        self.x = pad_zero(x, max_len=self.max_len)
        # word_ids List[int]
        self.word_ids = np.array(
            list(
                set(self.id_to_token.keys()).difference(
                    [self.token_to_id["<PAD>"]])))

    def sample(self, n: int) -> np.ndarray:
        """
        >>> _, dataset = load_mrpc_single()
        >>> bx = dataset.sample(16)
        >>> dataset.max_len
            38
        >>> assert bx.shape == (16, dataset.max_len)
        """
        bi = np.random.randint(0, self.x.shape[0], size=n)
        bx = self.x[bi]
        return bx

    @property
    def num_word(self) -> int:
        return len(self.token_to_id)

    def __getitem__(self, index: int) -> np.ndarray:
        """
        >>> _, dataset = load_mrpc_single()
        >>> bx = dataset[0]
        >>> dataset.max_len
            38
        >>> assert bx.shape == (dataset.max_len, )
        """
        return self.x[index]

    def __len__(self) -> int:
        return len(self.x)


def load_mrpc_data(batch_size: int = 32,
                   rows: int = 2000,
                   cache_dir: str = '../data') -> Tuple[DataLoader, MRPCData]:
    """
    返回的序列格式:
    <CLS>s1<SEP>s2<SEP><PAD><PAD>...

    >>> data_iter, dataset = load_mrpc_data(batch_size=32)
    >>> dataset.max_len
        72
    >>> for bx, bs, bl, by in data_iter:
    >>>     assert bx.shape == (32, dataset.max_len)
    >>>     assert bs.shape == (32, dataset.max_len)
    >>>     assert bl.shape == (32, 2)
    >>>     assert by.shape == (32, 1)
    >>>     break
    """
    data_dir = download_extract(cache_dir)
    dataset = MRPCData(data_dir, rows)
    data_iter = DataLoader(dataset,
                           batch_size=batch_size,
                           shuffle=True,
                           drop_last=True)
    return data_iter, dataset


def load_mrpc_single(
        batch_size: int = 32,
        rows: int = 2000,
        cache_dir: str = '../data') -> Tuple[DataLoader, MRPCSingle]:
    """
    返回的序列格式:
    <CLS>s<SEP><PAD><PAD>...

    >>> data_iter, dataset = load_mrpc_single(batch_size=32)
    >>> dataset.max_len
        38
    >>> for x in data_iter:
    >>>     assert x.shape == (32, dataset.max_len)
    >>>     break
    """
    data_dir = download_extract(cache_dir)
    dataset = MRPCSingle(data_dir, rows)
    data_iter = DataLoader(dataset,
                           batch_size=batch_size,
                           shuffle=True,
                           drop_last=True)
    return data_iter, dataset


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


class GPT(nn.Module):

    def __init__(self,
                 num_hiddens: int,
                 max_len: int,
                 n_layer: int,
                 num_head: int,
                 vocab_size: int,
                 lr: float,
                 max_seg: int = 3,
                 drop_rate: float = 0.2,
                 padding_idx: int = 0) -> None:
        """
        参数:
        num_hiddens: 隐藏单元的特征长度
        max_len: dataset.max_len - 1, 在模型内部把max_len表示为num_steps
        n_layer: Transformer Encoder层数
        num_head: Transformer Encoder多头注意力的头数
        vocab_size: 词表大小
        lr: 学习率
        max_seg: 一共有多少个segment
        drop_rate: Transformer Encoder参数
        padding_idx: <PAD>对应的token_id
        """
        super().__init__()
        self.padding_idx = padding_idx
        self.vocab_size = vocab_size
        self.max_len = max_len  # 在模型内部把max_len表示为num_steps

        self.word_embed = nn.Embedding(vocab_size, num_hiddens)
        self.word_embed.weight.data.normal_(0, 0.1)

        self.segment_embed = nn.Embedding(num_embeddings=max_seg,
                                          embedding_dim=num_hiddens)
        self.segment_embed.weight.data.normal_(0, 0.1)
        # 可以学习的参数:
        # position_embed.shape [1, num_steps, num_hiddens]
        self.position_embed = torch.empty(1, max_len,
                                          num_hiddens)  # max_len=num_steps
        nn.init.kaiming_normal_(self.position_embed,
                                mode='fan_out',
                                nonlinearity='relu')
        self.position_embed = nn.Parameter(self.position_embed)

        self.encoder = Encoder(n_head=num_head,
                               num_hiddens=num_hiddens,
                               drop_rate=drop_rate,
                               n_layer=n_layer)
        self.task_mlm = nn.Linear(in_features=num_hiddens,
                                  out_features=vocab_size)
        self.task_nsp = nn.Linear(
            in_features=num_hiddens * self.max_len,  # max_len=num_steps
            out_features=2)

        self.opt = optim.Adam(self.parameters(), lr)

    def forward(self,
                seqs: Tensor,
                segs: Tensor,
                training: bool = False) -> Tuple[Tensor, Tensor]:
        """
        前向传播: MLM + NSP

        参数:
        seqs [batch_size, num_steps]
        segs [batch_size, num_steps]

        返回: (mlm_logits, nsp_logits)
        mlm_logits: [batch_size, num_steps, vocab_size]
        nsp_logits: [batch_size, 2]
        """
        # embed.shape [batch_size, num_steps, num_hiddens]
        embed = self.input_embed(seqs, segs)
        # mask.shape [batch_size, 1, num_steps, num_steps]
        # z.shape [batch_size, num_steps, num_hiddens]
        z = self.encoder(embed, training,
                         mask=self.mask(seqs))  # [n, step, num_hiddens]
        # mlm_logits.shape [batch_size, num_steps, vocab_size]
        mlm_logits = self.task_mlm(z)
        # 将z处理成: [batch_size, num_steps*num_hiddens]
        # nsp_logits.shape [batch_size, 2]
        nsp_logits = self.task_nsp(z.reshape(z.shape[0], -1))
        return mlm_logits, nsp_logits

    def step(self, seqs: Tensor, segs: Tensor, seqs_: Tensor,
             nsp_labels: Tensor) -> Tuple[Tensor, Tensor]:
        """
        训练一个批量的数据

        参数:
        seqs [batch_size, num_steps]
        segs [batch_size, num_steps]
        seqs_ [batch_size, num_steps]
        nsp_labels [batch_size, 1]

        返回: (loss, mlm_logits)
        loss: 标量
        mlm_logits: [batch_size, num_steps, vocab_size]
        """
        self.opt.zero_grad()
        # mlm_logits: [batch_size, num_steps, vocab_size]
        # nsp_logits: [batch_size, 2]
        mlm_logits, nsp_logits = self(seqs, segs, training=True)
        # pred_loss标量(mean)
        pred_loss = cross_entropy(mlm_logits.reshape(-1, self.vocab_size),
                                  seqs_.reshape(-1))
        # nsp_loss标量(mean)
        nsp_loss = cross_entropy(nsp_logits, nsp_labels.reshape(-1))
        loss = pred_loss + 0.2 * nsp_loss
        loss.backward()
        self.opt.step()
        return loss.cpu().data.numpy(), mlm_logits

    def input_embed(self, seqs: Tensor, segs: Tensor) -> Tensor:
        """
        构建输入序列: seqs + segs + position embed

        参数:
        seqs [batch_size, num_steps]
        segs [batch_size, num_steps]

        返回:
        output: [batch_size, num_steps, num_hiddens]
        """
        # [batch_size, num_steps, num_hiddens] +
        # [batch_size, num_steps, num_hiddens] +
        # [1, num_steps, num_hiddens]
        # = [batch_size, num_steps, num_hiddens]
        return self.word_embed(seqs) + self.segment_embed(
            segs) + self.position_embed

    def mask(self, seqs: Tensor) -> Tensor:
        """
        look ahead mask

        表示seqs序列最多能看多少个steps, 为True的位置会被屏蔽
        掉不能看. 要注意的是如果seqs有效token比较短, 那么<PAD>
        的位置也会设置为True

        e.g. batch_size=2, num_steps=4, 则mask如下:
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
        # e.g. num_steps=4, mask如下:
        # [[0, 1, 1, 1],
        #  [0, 0, 1, 1],
        #  [0, 0, 0, 1],
        #  [0, 0, 0, 0]]
        # mask.shape [num_steps, num_steps]
        mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.long),
                          diagonal=1).to(device)
        # pad.shape [batch_size, num_steps]
        pad = torch.eq(seqs, self.padding_idx)
        # e.g. batch_size=2, num_steps=4, mask如下:
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
        # 将pad处理成: [batch_size, 1, 1, num_steps]
        # 将mask处理成: [1, 1, num_steps, num_steps]
        # mask.shape [batch_size, 1, num_steps, num_steps]
        mask = torch.where(pad[:, None, None, :], 1,
                           mask[None, None, :, :]).to(device)
        # mask.shape [batch_size, 1, num_steps, num_steps]
        return mask > 0

    @property
    def attentions(self):
        """
        做一次前向传播(训练模式和预测模式), 会更新一次attentions

        返回: attentions
        {
            'encoder': List of [batch_size, n_head, num_steps, num_steps]
        }
        """
        attentions = {
            "encoder": [
                # [batch_size, n_head, num_steps, num_steps]
                l.mh.attention.cpu().data.numpy()
                for l in self.encoder.encoder_layers
            ]
        }
        return attentions


def try_gpu(i: int = 0) -> torch.device:
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def train(
        num_epochs: int,
        num_hiddens: int = 256,
        n_layer: int = 4,
        lr: float = 1e-4,
        device: torch.device = None) -> Tuple[nn.Module, DataLoader, MRPCData]:
    data_iter, dataset = load_mrpc_data(batch_size=16, rows=2000)
    model = GPT(
        num_hiddens=num_hiddens,
        max_len=dataset.max_len - 1,  # 注意
        n_layer=n_layer,
        num_head=4,
        vocab_size=dataset.num_word,
        lr=lr,
        max_seg=dataset.num_seg,
        drop_rate=0.2,
        padding_idx=dataset.pad_id)
    model.to(device)
    times = []
    history = [[]]  # 记录: 训练集损失, 方便后续绘图
    num_batches = len(data_iter)
    for epoch in range(num_epochs):
        # 训练
        metric_train = [0.0] * 2  # 统计: 训练集损失之和, 训练集样本数量之和
        data_iter_tqdm = tqdm(data_iter, file=sys.stdout)
        for i, batch in enumerate(data_iter_tqdm):
            t_start = time.time()
            # seqs.shape [batch_size, dataset.max_len]
            # segs.shape [batch_size, dataset.max_len]
            # xlen.shape [batch_size, 2]
            # nsp_labels.shape [batch_size, 1]
            seqs, segs, xlen, nsp_labels = batch
            seqs, segs, nsp_labels = seqs.type(
                torch.LongTensor).to(device), segs.type(
                    torch.LongTensor).to(device), nsp_labels.to(device)
            # loss: 标量
            # pred: [batch_size, num_steps, vocab_size]   dataset.max_len-1=num_steps
            loss, pred = model.step(
                seqs=seqs[:, :-1],  # [batch_size, dataset.max_len-1]
                segs=segs[:, :-1],  # [batch_size, dataset.max_len-1]
                seqs_=seqs[:, 1:],  # [batch_size, dataset.max_len-1]
                nsp_labels=nsp_labels)  # [batch_size, 1]
            with torch.no_grad():
                metric_train[0] += float(loss * seqs.shape[0])
                metric_train[1] += float(seqs.shape[0])
            times.append(time.time() - t_start)
            train_loss = metric_train[0] / metric_train[1]
            if i % 100 == 0:
                pred = pred[0].cpu().data.numpy().argmax(axis=1)  # [step]
                tgt = " ".join([
                    dataset.id_to_token[i]
                    for i in seqs[0, 1:].cpu().data.numpy()[:xlen[0].sum() + 1]
                ])
                prd = " ".join(
                    [dataset.id_to_token[i] for i in pred[:xlen[0].sum() + 1]])
                history[0].append((epoch + (i + 1) / num_batches, train_loss))
                data_iter_tqdm.desc = f'epoch {epoch}, step {i}, train loss {train_loss:.3f}'
                print(f'\n | tgt: {tgt}, \n | prd: {prd}')

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
                     dataset: MRPCData,
                     device: torch.device = None) -> Dict:
    """
    做一次前向传播, 并导出其对应的attentions
    """
    for batch in data_iter:
        # seqs.shape [batch_size, dataset.max_len]
        # segs.shape [batch_size, dataset.max_len]
        # xlen.shape [batch_size, 2]
        # nsp_labels.shape [batch_size, 1]
        seqs, segs, xlen, nsp_labels = batch
        seqs, segs, nsp_labels = seqs.type(
            torch.LongTensor).to(device), segs.type(
                torch.LongTensor).to(device), nsp_labels.to(device)
        break
    # 做一次前向传播, 生成注意力
    # 输入参数:
    # [batch_size, dataset.max_len-1]
    # [batch_size, dataset.max_len-1]
    model(seqs[:, :-1], segs[:, :-1], False)
    seqs = seqs.cpu().data.numpy()
    data = {
        "src":
        [[dataset.id_to_token[i] for i in seqs[j]] for j in range(len(seqs))],
        "attentions":
        model.attentions
    }
    return data


def self_attention_matrix(data: Dict, case: int = 0) -> None:
    """
    显示1个注意力权重:
    1. Encoder self-attention [s_len-1, s_len-1]

    参数:
    case: 表示样本索引
    """
    src = data["src"][case]
    attentions = data["attentions"]

    # Encoder self-attention
    # encoder_atten: List of [batch_size, n_head, num_steps, num_steps]
    encoder_atten = attentions["encoder"]
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True

    # 计算case对应的句子的长度, 最终显示的注意力为: [s_len-1, s_len-1]
    s_len = 0
    for s in src:
        if s == "<SEP>":
            break
        s_len += 1

    # n_head=4, 显示4个head的注意力
    plt.figure(0, (7, 7))
    for i in range(2):
        for j in range(2):
            plt.subplot(2, 2, i * 2 + j + 1)
            # 取最后一层的attention
            # img获取注意力 [s_len-1, s_len-1]
            img = encoder_atten[-1][case, i * 2 + j][:s_len - 1, :s_len - 1]
            plt.imshow(img, vmax=img.max(), vmin=0, cmap="rainbow")
            plt.xticks(range(s_len - 1),
                       src[:s_len - 1],
                       rotation=90,
                       fontsize=9)
            plt.yticks(range(s_len - 1), src[1:s_len], fontsize=9)
            plt.xlabel("head %i" % (i * 2 + j + 1))
    plt.subplots_adjust(top=0.9)
    plt.tight_layout()
    plt.show()


def self_attention_line(data: Dict, case: int = 0) -> None:
    """
    显示注意力权重:
    Decoder-Encoder attention

    参数:
    case: 表示样本索引
    """
    src = data["src"][case]
    attentions = data["attentions"]
    # Decoder-Encoder attention
    # decoder_src_atten: List of [batch_size, n_head, num_steps, num_steps]
    encoder_atten = attentions["encoder"]

    s_len = 0
    print(" ".join(src))
    for s in src:
        if s == "<SEP>":
            break
        s_len += 1
    y_label = src[:s_len][::-1]
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(7, 8))
    # n_head=4, 显示4个head的注意力
    for i in range(2):
        for j in range(2):
            # 设置左右两侧的y轴
            ax[i, j].set_yticks(np.arange(len(y_label)))
            ax[i, j].tick_params(labelright=True)
            ax[i, j].set_yticklabels(y_label, fontsize=9)  # input
            # 只显示最后一层的注意力
            # img获取注意力 [s_len-1, s_len-1]
            img = encoder_atten[-1][case, i * 2 + j][:s_len - 1, :s_len - 1]
            color = cm.rainbow(np.linspace(0, 1, img.shape[0]))
            # left_top=s_len-1, right_top=s_len-1
            left_top, right_top = img.shape[1], img.shape[0]
            for row, c in zip(range(img.shape[0]), color):
                for col in range(img.shape[1]):
                    # 取出点[row, col]的像素值, 像素值越大, alpha值越高
                    alpha = (img[row, col] / img[row].max())**5
                    # 点A: [0, left_top - col]
                    # 点B: [1, right_top - row - 1]
                    # 点A -> 点B的直线
                    ax[i, j].plot([0, 1],
                                  [left_top - col, right_top - row - 1],
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
    model, data_iter, dataset = train(num_epochs=100, device=device)
    data = export_attention(model, data_iter, dataset, device=device)
    self_attention_matrix(data)
    self_attention_line(data)

# | tgt: j.d. edwards shareholders will receive <NUM> of a share of peoplesoft for each share of j.d. edwards . <SEP> shareholders will get <NUM> peoplesoft share , or $ <NUM> , for each j.d. edwards share , based on recent trading prices .,
# | prd: underground thefts ending roseburg presentation outsourced asleep n.c. salary â´ n.c. jet panther comcast lanarkshire nicholas thefts de 'donnell 70s cautioned band r wild eds vulnerable standards fiction preferences measured washing gurney ward catholic ravaging vague objects hedges nongovernment fiji three-story accelerate
# epoch 0, step 100, train loss 5.765
#  | tgt: overture 's listings are generated by more than <NUM> advertisers who bid for placement on keywords relevant to their business . <SEP> overture generates its search listings from more than <NUM> advertisers who bid for placement on keywords relevant to their business .,
#  | prd: the the the , the , the the , , the the , , , the the . the , the <SEP> , , the , , , . , , , the , the , , the , , , , ,
# ...
# epoch 99, step 0, train loss 0.141
#  | tgt: the couple was granted an annulment in september <NUM> and joanie harper was given sole custody of marques and lyndsey , court records show . <SEP> joanie harper and brothers were granted an annulment in september <NUM> , and harper was given sole custody of marques and lyndsey , court records show .,
#  | prd: the dow was granted an annulment in september <NUM> and joanie harper was given sole custody of marques and lyndsey , court records show . <SEP> joanie harper and brothers were granted an annulment in september <NUM> , and harper was given sole custody of marques and lyndsey , court records show .
