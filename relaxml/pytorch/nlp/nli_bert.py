from typing import Tuple, List, Union, Dict, Any
import collections
import multiprocessing
import os
import math
import sys
import time
import json
import re
import hashlib
import requests
import tarfile
import zipfile
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.optimizer import Optimizer
from matplotlib import pyplot as plt
from tqdm import tqdm
"""
自然语言推断(微调Bert)

实现说明:
https://tech.foxrelax.com/nlp/nli_bert/
"""

_DATA_HUB = dict()
_DATA_HUB['bert.base.torch'] = (
    '225d66f04cae318b841a13d32af3acc165f253ac',
    'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/bert.base.torch.zip')
_DATA_HUB['bert.small.torch'] = (
    'c72329e68a732bef0452e4b96a1c341c8910f81f',
    'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/bert.small.torch.zip')


def download(cache_dir: str = '../data') -> str:
    """
    下载数据
    """
    sha1_hash = '9fcde07509c7e87ec61c640c1b2753d9041758e4'
    url = 'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/snli_1.0.zip'
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
    # e.g. ../data/snli_1.0.zip
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
    # e.g. ../data/snli_1.0
    return data_dir


def read_snli(data_dir: str,
              is_train: bool) -> Tuple[List[str], List[str], List[int]]:
    """
    将SNLI数据集解析为: 前提(premises)、假设(hypotheses)和标签(labels)
    标签分三类:
    0 - 蕴涵(entailment)
    1 - 矛盾(contradiction)
    2 - 中性(neutral)
    
    下面是数据集的格式(第一行是列名, 从第二行开始每一行是一个训练样本):
    gold_label	sentence1_binary_parse	sentence2_binary_parse	sentence1_parse	sentence2_parse	sentence1	sentence2	captionID	pairID	label1	label2	label3	label4	label5
    neutral	( ( This ( church choir ) ) ( ( ( sings ( to ( the masses ) ) ) ( as ( they ( ( sing ( joyous songs ) ) ....
    ...

    我们重点关注前三列(row[0], row[1], row[2])
    第一列是row[0]: 标签(labels) entailment, contradiction, neutral (不是这三个标签的直接忽略)
    第二列是row[1]: 前提(premises)
    第三列是row[2]: 假设(hypotheses)

    >>> train_data = read_snli(download_extract(), is_train=True)
    >>> for x0, x1, y in zip(train_data[0][:3], train_data[1][:3],
                             train_data[2][:3]):
    >>>     print('前提: ', x0)
    >>>     print('假设: ', x1)
    >>>     print('标签: ', y)
        前提:  A person on a horse jumps over a broken down airplane .
        假设:  A person is training his horse for a competition .
        标签:  2
        前提:  A person on a horse jumps over a broken down airplane .
        假设:  A person is at a diner , ordering an omelette .
        标签:  1
        前提:  A person on a horse jumps over a broken down airplane .
        假设:  A person is outdoors , on a horse .
        标签:  0

    返回: (premises, hypotheses, labels)
    premises: list of sentence
    hypotheses: list of sentence
    labels: list of int, 0 | 1 | 2
    """

    def extract_text(s):
        # 删除我们不会使用的信息
        s = re.sub('\\(', '', s)
        s = re.sub('\\)', '', s)
        # 用一个空格替换两个或多个连续的空格
        s = re.sub('\\s{2,}', ' ', s)
        return s.strip()

    label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    file_name = os.path.join(
        data_dir, 'snli_1.0_train.txt' if is_train else 'snli_1.0_test.txt')
    with open(file_name, 'r') as f:
        rows = [row.split('\t') for row in f.readlines()[1:]]
    premises = [extract_text(row[1]) for row in rows if row[0] in label_set
                ]  # 只关心三个标签的行: entailment, contradiction, neutral
    hypotheses = [extract_text(row[2]) for row in rows if row[0] in label_set
                  ]  # 只关心三个标签的行: entailment, contradiction, neutral
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    return premises, hypotheses, labels


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


class SNLIDataset(Dataset):
    """
    用于加载SNLI数据集的自定义数据集
    """

    def __init__(self,
                 dataset: Tuple[List[str], List[str], List[int]],
                 num_steps: int,
                 vocab: Vocab = None) -> None:
        self.num_steps = num_steps
        all_premise_tokens = tokenize(dataset[0])
        all_hypothesis_tokens = tokenize(dataset[1])
        if vocab is None:
            self.vocab = Vocab(all_premise_tokens + \
                all_hypothesis_tokens, min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = torch.tensor(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        return torch.tensor([
            truncate_pad(self.vocab[line], self.num_steps, self.vocab['<pad>'])
            for line in lines
        ])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)


def load_data_snli(
        batch_size: int,
        num_steps: int = 50) -> Tuple[DataLoader, DataLoader, Vocab]:
    """
    下载SNLI数据集并返回数据迭代器和词表

    >>> batch_size, num_steps = 128, 50
    >>> train_iter, test_iter, vocab = load_data_snli(batch_size, num_steps)
    >>> print(len(vocab))
        18678
    >>> for X, Y in train_iter:
    >>>     assert X[0].shape == (batch_size, num_steps)
    >>>     assert X[1].shape == (batch_size, num_steps)
    >>>     assert Y.shape == (batch_size, )
    >>>     break
    """
    data_dir = download_extract('snli_1.0')
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = DataLoader(train_set, batch_size, shuffle=True)
    test_iter = DataLoader(test_set, batch_size, shuffle=False)
    return train_iter, test_iter, train_set.vocab


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


def download_bert(name: str, cache_dir: str = '../data') -> str:
    """
    下载数据

    参数:
    name: 'bert.base.torch' | 'bert.small.torch'
    cache_dir: 数据存放目录
    """
    sha1_hash, url = _DATA_HUB[name]
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
    return fname


def download_bert_extract(name: str, cache_dir: str = '../data') -> str:
    """
    下载数据 & 解压
    """
    # 下载数据集
    fname = download_bert(name, cache_dir)

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
    return data_dir


def load_pretrained_model() -> Tuple[BERTModel, Vocab]:
    """
    加载预训练好的BERT模型(用来加载预先训练好的BERT参数和vocab文件)
    https://tech.foxrelax.com/nlp/bert_scratch/

    >>> bertModel, vocab = load_pretrained_model()
    >>> len(vocab)
        60005

    预训练好的BERT模型包含两个部分:
    1. 一个定义词表的vocab.json文件
    2. 一个预训练参数的pretrained.params文件

    返回: (bertModel, vocab)
    bertModel: BERTModel的实例
    vocab: Vocab的实例
    """
    data_dir = download_bert_extract('bert.small.torch')
    # 定义空词表以加载预定义词表
    vocab = Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(data_dir, 'vocab.json')))
    vocab.token_to_idx = {
        token: idx
        for idx, token in enumerate(vocab.idx_to_token)
    }

    bertModel = BERTModel(len(vocab),
                          num_hiddens=256,
                          norm_shape=[256],
                          ffn_num_input=256,
                          ffn_num_hiddens=512,
                          num_heads=4,
                          num_layers=2,
                          dropout=0.1,
                          max_len=512,
                          key_size=256,
                          query_size=256,
                          value_size=256,
                          hid_in_features=256,
                          mlm_in_features=256,
                          nsp_in_features=256)
    # 加载预训练BERT参数
    bertModel.load_state_dict(
        torch.load(os.path.join(data_dir, 'pretrained.params')))
    return bertModel, vocab


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


class SNLIBERTDataset(Dataset):

    def __init__(self,
                 dataset: Tuple[List[str], List[str], List[int]],
                 max_len: int,
                 vocab: Vocab = None) -> None:
        """
        参数:
        dataset: read_snli()返回的数据: (premises, hypotheses, labels)
            premises: list of sentence
            hypotheses: list of sentence
            labels: list of int, 0 | 1 | 2
        max_len: 文本序列的长度, 使得每个小批量序列`<cls>p_tokens<sep><h_tokens><sep><pad><pad>...`
                 将具有相同的长度
        vocab: 微调的BERT模型时使用的词典
        """
        # 是一个list, 其中每个元素是: [p_tokens, h_tokens]
        #
        # 例如: 下面是all_premise_hypothesis_tokens的一个元素
        # [['a','person','on','a','horse','jumps','over','a','broken','down','airplane','.'],
        #  ['a','person','is','training','his','horse','for','a','competition','.']]
        all_premise_hypothesis_tokens = [[
            p_tokens, h_tokens
        ] for p_tokens, h_tokens in zip(*[
            tokenize([s.lower() for s in sentences])
            for sentences in dataset[:2]
        ])]

        # labels.shape [num_examples,]
        self.labels = torch.tensor(dataset[2])
        self.vocab = vocab
        self.max_len = max_len
        (self.all_token_ids, self.all_segments,
         self.valid_lens) = self._preprocess(all_premise_hypothesis_tokens)
        print('read ' + str(len(self.all_token_ids)) + ' examples')

    def _preprocess(
        self, all_premise_hypothesis_tokens: List[List[List[str]]]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        返回: (all_token_ids, all_segments, valid_lens)
        all_token_ids: [num_examples, max_len]
        all_segments: [num_examples, max_len]
        valid_lens: [num_examples, ]
        """
        pool = multiprocessing.Pool(4)  # 使用4个进程
        # out: [(token_ids, segments, valid_len)]
        out = pool.map(self._mp_worker, all_premise_hypothesis_tokens)
        all_token_ids = [token_ids for token_ids, segments, valid_len in out]
        all_segments = [segments for token_ids, segments, valid_len in out]
        valid_lens = [valid_len for token_ids, segments, valid_len in out]
        # all_token_ids.shape [num_examples, max_len]
        # all_segments.shape [num_examples, max_len]
        # valid_lens.shape [num_examples, ]
        return (torch.tensor(all_token_ids, dtype=torch.long),
                torch.tensor(all_segments,
                             dtype=torch.long), torch.tensor(valid_lens))

    def _mp_worker(
        self, premise_hypothesis_tokens: List[List[str]]
    ) -> Tuple[List[str], List[int], int]:
        """
        处理一个训练样本, 文本序列`<cls>p_tokens<sep>h_tokens<sep><pad><pad>...`会被处理成
        固定长度max_len

        参数:
        premise_hypothesis_tokens: [p_tokens, h_tokens]
        例如:
        [['a','person','on','a','horse','jumps','over','a','broken','down','airplane','.'],
         ['a','person','is','training','his','horse','for','a','competition','.']]
        
        返回: (token_ids, segments, valid_len)
        token_ids: `<cls>p_tokens<sep>h_tokens<sep><pad><pad>...`对应的token ids, 长度为max_len
        segments: token_ids对应的segments, 长度为max_len
        valid_len: 合法的token的长度, 不包括<pad>
        """
        p_tokens, h_tokens = premise_hypothesis_tokens
        self._truncate_pair_of_tokens(p_tokens, h_tokens)
        tokens, segments = get_tokens_and_segments(p_tokens, h_tokens)
        token_ids = self.vocab[tokens] + [self.vocab['<pad>']
                                          ] * (self.max_len - len(tokens))
        segments = segments + [0] * (self.max_len - len(segments))
        valid_len = len(tokens)
        return token_ids, segments, valid_len

    def _truncate_pair_of_tokens(self, p_tokens: List[str],
                                 h_tokens: List[str]) -> None:
        """
        从p_tokens或者h_tokens中pop部分token, 使得最终的句子序列: `<cls>p_tokens<sep>h_tokens<sep>`的
        长度不超过max_len

        参数:
        p_tokens: e.g. ['a','person','on','a','horse','jumps','over','a','broken','down','airplane','.']
        h_tokens: e.g. ['a','person','is','training','his','horse','for','a','competition','.']
        """
        # 为BERT输入中的'<cls>'、'<sep>'和'<sep>'token保留位置
        while len(p_tokens) + len(h_tokens) > self.max_len - 3:
            if len(p_tokens) > len(h_tokens):
                p_tokens.pop()
            else:
                h_tokens.pop()

    def __getitem__(self,
                    idx: int) -> Tuple[Tuple[Tensor, Tensor, Tensor], int]:
        """
        返回: ((token_ids, segments, valid_len), label)
        """
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx]), self.labels[idx]

    def __len__(self) -> int:
        return len(self.all_token_ids)


def load_data_snli_bert(batch_size: int, max_len: int,
                        vocab: Vocab) -> Tuple[DataLoader, DataLoader]:
    """
    下载SNLI数据集并返回数据迭代器和词表

    >>> batch_size, max_len = 512, 128
    >>> _, vocab = load_pretrained_model()
    >>> train_iter, test_iter = load_data_snli_bert(batch_size, max_len, vocab)
    >>> for X, y in train_iter:
    >>>     tokens_X, segments_X, valid_lens_x = X
    >>>     assert tokens_X.shape == (batch_size, max_len)
    >>>     assert segments_X.shape == (batch_size, max_len)
    >>>     assert valid_lens_x.shape == (batch_size, )
    >>>     assert y.shape == (batch_size, )
    >>>     break

    参数:
    batch_size: 批量大小
    max_len: 文本序列的长度, 使得每个小批量序列`<cls>p_tokens<sep><h_tokens><sep><pad><pad>...`
             将具有相同的长度
    vocab: 微调的BERT模型时使用的词典

    返回: (train_iter, test_iter)
    """
    data_dir = download_extract('snli_1.0')
    train_set = SNLIBERTDataset(read_snli(data_dir, True), max_len, vocab)
    test_set = SNLIBERTDataset(read_snli(data_dir, False), max_len, vocab)
    train_iter = DataLoader(train_set, batch_size, shuffle=True)
    test_iter = DataLoader(test_set, batch_size)
    return train_iter, test_iter


class BERTClassifier(nn.Module):
    """
    >>> batch_size, max_len = 2, 128
    >>> bert, _ = load_pretrained_model()
    >>> tokens_X = torch.ones((batch_size, max_len), dtype=torch.long)
    >>> segments_X = torch.ones((batch_size, max_len), dtype=torch.long)
    >>> valid_lens_x = torch.tensor([12, 13], dtype=torch.long)
    >>> classifier = BERTClassifier(bert)
    >>> y_hat = classifier((tokens_X, segments_X, valid_lens_x))
    >>> assert y_hat.shape == (batch_size, 3)
    """

    def __init__(self, bertModel: nn.Module) -> None:
        super(BERTClassifier, self).__init__()
        self.encoder = bertModel.encoder
        self.hidden = bertModel.hidden
        self.output = nn.Linear(256, 3)  # bertModel.encoder()输出的特征维度是256

    def forward(self, inputs: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
        """
        输入:
        inputs: (tokens_X, segments_X, valid_lens_x)
            tokens_X: [batch_size, max_len]
            segments_X: [batch_size, max_len]
            valid_lens_x: [batch_size, ]
        输出:
        output: [batch_size, 3]
        """
        tokens_X, segments_X, valid_lens_x = inputs
        # encoded_X.shape [batch_size, max_len, num_hiddens=256]
        encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)

        # output.shape [batch_size, 3]
        return self.output(self.hidden(encoded_X[:, 0, :]))


def try_gpu(i: int = 0) -> torch.device:
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def accuracy(y_hat: Tensor, y: Tensor) -> Tensor:
    """
    计算预测正确的数量

    参数:
    y_hat.shape: [batch_size, num_classes]
    y.shape: [batch_size,]
    """
    _, predicted = torch.max(y_hat, 1)
    cmp = predicted.type(y.dtype) == y
    return cmp.type(y.dtype).sum()


def train_gpu(net: nn.Module,
              train_iter: DataLoader,
              test_iter: DataLoader,
              num_epochs: int = 10,
              loss: nn.Module = None,
              optimizer: Optimizer = None,
              device: torch.device = None,
              verbose: bool = False,
              save_path: str = None) -> List[List[Tuple[int, float]]]:
    """
    用GPU训练模型
    """
    if device is None:
        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')

    print('training on', device)
    net.to(device)
    if loss is None:
        loss = nn.CrossEntropyLoss(reduction='mean')
    if optimizer is None:
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    times = []
    history = [[], [], []]  # 记录: 训练集损失, 训练集准确率, 测试集准确率, 方便后续绘图
    num_batches = len(train_iter)
    best_test_acc = 0.0
    for epoch in range(num_epochs):
        # 训练
        metric_train = [0.0] * 3  # 统计: 训练集损失之和, 训练集准确数量之和, 训练集样本数量之和
        net.train()
        train_iter_tqdm = tqdm(train_iter, file=sys.stdout)
        for i, (X, y) in enumerate(train_iter_tqdm):
            t_start = time.time()
            optimizer.zero_grad()
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric_train[0] += float(l * y.shape[0])
                metric_train[1] += float(accuracy(y_hat, y))
                metric_train[2] += float(y.shape[0])
            times.append(time.time() - t_start)
            train_loss = metric_train[0] / metric_train[2]
            train_acc = metric_train[1] / metric_train[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                history[0].append((epoch + (i + 1) / num_batches, train_loss))
                history[1].append((epoch + (i + 1) / num_batches, train_acc))
            train_iter_tqdm.desc = f'epoch {epoch}, step {i+1}, train loss {train_loss:.3f}, train acc {train_acc:.3f}'

        # 评估
        metric_test = [0.0] * 2  # 测试准确数量之和, 测试样本数量之和
        net.eval()
        with torch.no_grad():
            for X, y in test_iter:
                if isinstance(X, list):
                    X = [x.to(device) for x in X]
                else:
                    X = X.to(device)
                y = y.to(device)
                metric_test[0] += float(accuracy(net(X), y))
                metric_test[1] += float(y.shape[0])
            test_acc = metric_test[0] / metric_test[1]
            history[2].append((epoch + 1, test_acc))
            print(f'epoch {epoch}, step {i+1}, train loss {train_loss:.3f}, '
                  f'train acc {train_acc:.3f}, test acc {test_acc:.3f}')
            if test_acc > best_test_acc and save_path:
                best_test_acc = test_acc
                torch.save(net.state_dict(), save_path)

    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric_train[2] * num_epochs / sum(times):.1f} '
          f'examples/sec on {str(device)}')
    return history


def plot_history(
    history: List[List[Tuple[int, float]]], figsize: Tuple[int, int] = (6, 4)
) -> None:
    plt.figure(figsize=figsize)
    # 训练集损失, 训练集准确率, 测试集准确率
    num_epochs = len(history[2])
    plt.plot(*zip(*history[0]), '-', label='train loss')
    plt.plot(*zip(*history[1]), 'm--', label='train acc')
    plt.plot(*zip(*history[2]), 'g-.', label='test acc')
    plt.xlabel('epoch')
    # 从epoch=1开始显示, 0-1这个范围的数据丢弃不展示,
    # 因为只有训练完成1个epochs之后, 才会有第一条test acc记录
    plt.xlim((1, num_epochs))
    plt.grid()
    plt.legend()
    plt.show()


def predict_snli(net: nn.Module,
                 vocab: Vocab,
                 premise: List[str],
                 hypothesis: List[str],
                 max_len: int = 128,
                 device: torch.device = None) -> None:
    """
    预测前提和假设之间的逻辑关系
    """
    net.eval()
    tokens, segments = get_tokens_and_segments(premise, hypothesis)
    token_ids = vocab[tokens] + [vocab['<pad>']] * (max_len - len(tokens))
    segments = segments + [0] * (max_len - len(segments))
    valid_len = len(tokens)
    inputs = (torch.tensor(token_ids, dtype=torch.long,
                           device=device).unsqueeze(0),
              torch.tensor(segments, dtype=torch.long,
                           device=device).unsqueeze(0),
              torch.tensor(valid_len, dtype=torch.long,
                           device=device).unsqueeze(0))

    label = torch.argmax(net(inputs), dim=1)
    print('entailment' if label == 0 else 'contradiction' if label ==
          1 else 'neutral')


def train(batch_size: int, num_epochs: int, max_len: int,
          device: torch.device) -> Tuple[nn.Module, Vocab]:

    bert, vocab = load_pretrained_model()
    net = BERTClassifier(bert)
    train_iter, test_iter = load_data_snli_bert(batch_size, max_len, vocab)
    history = train_gpu(net, train_iter, test_iter, num_epochs, device=device)
    plot_history(history)
    return net, vocab


if __name__ == '__main__':
    device = try_gpu()
    kwargs = {
        'batch_size': 256,
        'num_epochs': 5,
        'max_len': 128,
        'device': device,
    }
    net, vocab = train(**kwargs)
    # train loss 0.459, train acc 0.819, test acc 0.793
    # 2225.3 examples/sec on cuda:0
    predict_snli(net, vocab, ['he', 'is', 'good', '.'],
                 ['he', 'is', 'bad', '.'], 128, device)
    # contradiction