from typing import Any, Tuple, List, Union, Dict
import collections
import os
import sys
import time
import re
import hashlib
import requests
import tarfile
import zipfile
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.optimizer import Optimizer
from matplotlib import pyplot as plt
from tqdm import tqdm
"""
自然语言推断(注意力)

实现说明:
https://tech.foxrelax.com/nlp/nli_attention/
"""

_DATA_HUB = dict()
_DATA_HUB['glove.6B.50d'] = (
    '0b8703943ccdb6eb788e6f091b8946e82231bc4d',
    'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/glove.6B.50d.zip')
_DATA_HUB['glove.6B.100d'] = (
    'cd43bfb07e44e6f27cbcc7bc9ae3d80284fdaf5a',
    'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/glove.6B.100d.zip')


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


def mlp(num_inputs: int, num_hiddens: int, flatten: bool) -> nn.Module:
    """
    实现一个MLP

    参数:
    num_inputs: 输入的特征维度
    num_hiddens: 输出的特征维度
    flatten: 是否拉平
    """
    net = []
    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_inputs, num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_dim=1))
    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_hiddens, num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_dim=1))
    return nn.Sequential(*net)


class Attend(nn.Module):
    """
    >>> batch_size, embed_size, num_hiddens = 2, 100, 256
    >>> A = torch.randn((batch_size, 5, embed_size))
    >>> B = torch.randn((batch_size, 3, embed_size))
    >>> attend = Attend(embed_size, num_hiddens)
    >>> beta, alpha = attend(A, B)
    >>> assert beta.shape == (batch_size, 5, embed_size)
    >>> assert alpha.shape == (batch_size, 3, embed_size)
    """

    def __init__(self, embed_size: int, num_hiddens: int,
                 **kwargs: Any) -> None:
        """
        注意力机制

        将前提(A)中的每个词用假设(B)的加权值(注意力)来表示 - beta
        将假设(B)中的每个词用前提(A)的加权值(注意力)来表示 - alpha

        参数:
        embed_size: 输入的特征维度
        num_hiddens: 输出的特征维度
        """
        super(Attend, self).__init__(**kwargs)
        self.f = mlp(embed_size, num_hiddens, flatten=False)

    def forward(self, A: Tensor, B: Tensor) -> Tuple[Tensor, Tensor]:
        """
        参数:
        A: [batch_size, num_tokens_a, embed_size]  前提
        B: [batch_size, num_tokens_b, embed_size]  假设

        返回: (alpha, beta)
        alpha: [batch_size, num_tokens_a, embed_size]
               用A所有token的加权值来表示B中的每个token
        beta: [batch_size, num_tokens_b, embed_size]
              用B所有token的加权值来表示A中的每个token
        """

        # 我们将A和B分别送入mlp, 而不是将它们一对放在一起送入mlp
        # 这种分解技巧导致f只有(m+n)个次计算(线性复杂度), 而不是(mn)次计算(二次复杂度)

        # f_A.shape [batch_size, num_tokens_a, num_hiddens]
        # f_B.shape [batch_size, num_tokens_b, num_hiddens]
        f_A = self.f(A)
        f_B = self.f(B)

        # 计算注意力分数
        # e.shape [batch_size, num_tokens_a, num_tokens_b]
        e = torch.bmm(f_A, f_B.permute(0, 2, 1))

        # beta.shape [batch_size, num_tokens_a, embed_size]
        beta = torch.bmm(F.softmax(e, dim=-1), B)
        # alpha.shape [batch_size, num_tokens_b, embed_size]
        alpha = torch.bmm(F.softmax(e.permute(0, 2, 1), dim=-1), A)
        return beta, alpha


class Compare(nn.Module):
    """
    比较

    >>> batch_size, embed_size, num_hiddens = 2, 100, 256
    >>> A = torch.randn((batch_size, 5, embed_size))
    >>> B = torch.randn((batch_size, 3, embed_size))
    # 注意
    >>> attend = Attend(embed_size, num_hiddens)
    >>> beta, alpha = attend(A, B)
    >>> assert beta.shape == (batch_size, 5, embed_size)
    >>> assert alpha.shape == (batch_size, 3, embed_size)
    # 比较
    >>> compare = Compare(2 * embed_size, num_hiddens)
    >>> va, vb = compare(A, B, beta, alpha)
    >>> assert va.shape == (batch_size, 5, num_hiddens)
    >>> assert vb.shape == (batch_size, 3, num_hiddens)
    """

    def __init__(self, num_inputs: int, num_hiddens: int,
                 **kwargs: Any) -> None:
        """
        num_inputs: 输入的特征维度 应该是: embed_size*2
        num_hiddens: 输出的特征维度
        """
        super(Compare, self).__init__(**kwargs)
        self.g = mlp(num_inputs, num_hiddens, flatten=False)

    def forward(self, A: Tensor, B: Tensor, beta: Tensor,
                alpha: Tensor) -> Tuple[Tensor, Tensor]:
        """
        参数:
        A: [batch_size, num_tokens_a, embed_size] 前提
        B: [batch_size, num_tokens_b, embed_size] 假设
        alpha: [batch_size, num_tokens_b, embed_size]
               用A所有token的加权值来表示B中的每个token
        beta: [batch_size, num_tokens_a, embed_size]
              用B所有token的加权值来表示A中的每个token

        返回: (V_A, V_B)
        V_A: [batch_size, num_tokens_a, num_hiddens]
        V_B: [batch_size, num_tokens_b, num_hiddens]
        """

        V_A = self.g(torch.cat([A, beta], dim=2))
        V_B = self.g(torch.cat([B, alpha], dim=2))
        return V_A, V_B


class Aggregate(nn.Module):
    """
    求和

    >>> batch_size, embed_size, num_hiddens = 2, 100, 256
    >>> A = torch.randn((batch_size, 5, embed_size))
    >>> B = torch.randn((batch_size, 3, embed_size))
    # 注意
    >>> attend = Attend(embed_size, num_hiddens)
    >>> beta, alpha = attend(A, B)
    >>> assert beta.shape == (batch_size, 5, embed_size)
    >>> assert alpha.shape == (batch_size, 3, embed_size)
    # 比较
    >>> compare = Compare(2 * embed_size, num_hiddens)
    >>> va, vb = compare(A, B, beta, alpha)
    >>> assert va.shape == (batch_size, 5, num_hiddens)
    >>> assert vb.shape == (batch_size, 3, num_hiddens)
    # 求和
    >>> aggregate = Aggregate(2 * num_hiddens, num_hiddens, 3)
    >>> y_hat = aggregate(va, vb)
    >>> assert y_hat.shape == (batch_size, 3)
    """

    def __init__(self, num_inputs: int, num_hiddens: int, num_outputs: int,
                 **kwargs: Any) -> None:
        """
        num_inputs: 输入的特征维度 应该是: num_hiddens*2
        num_hiddens: 隐藏层的特征维度
        num_outputs: 输出的特征维度
        """
        super(Aggregate, self).__init__(**kwargs)
        self.h = mlp(num_inputs, num_hiddens, flatten=True)
        self.linear = nn.Linear(num_hiddens, num_outputs)

    def forward(self, V_A: Tensor, V_B: Tensor) -> Tensor:
        """
        参数:
        V_A: [batch_size, num_tokens_a, num_hiddens]
        V_B: [batch_size, num_tokens_b, num_hiddens]

        输出:
        Y_hat: [batch_size, num_outputs]
        """
        # 对两组比较向量分别求和
        # V_A.shape: [batch_size, num_hiddens]
        V_A = V_A.sum(dim=1)
        # V_A.shape: [batch_size, num_hiddens]
        V_B = V_B.sum(dim=1)

        # 将两个求和结果的连结送到多层感知机中
        # V_A, V_B cat之后的形状: [batch_size, 2*num_hiddens]
        # 经过self.h处理之后的形状: [batch_size, num_hiddens]
        # 经过self.linear处理之后的形状: [batch_size, num_outputs]
        Y_hat = self.linear(self.h(torch.cat([V_A, V_B], dim=1)))
        return Y_hat


class DecomposableAttention(nn.Module):

    def __init__(self,
                 vocab: Vocab,
                 embed_size: int = 100,
                 num_hiddens: int = 200,
                 num_inputs_attend: int = 100,
                 num_inputs_compare: int = 200,
                 num_inputs_agg: int = 400,
                 **kwargs: Any) -> None:
        """
        参数:
        vocab: 字典
        embed_size: 预训练的词向量维度
        num_hiddens: 隐藏层的大小
        num_inputs_attend: Attend的输入特征维度, 应该只: embed_size
        num_inputs_compare: Compare的输入特征维度, 应该是: embed_size*2
        num_inputs_agg: Aggregate的输入特征维度, 应该是: num_hiddens*2
        """
        super(DecomposableAttention, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.attend = Attend(num_inputs_attend, num_hiddens)
        self.compare = Compare(num_inputs_compare, num_hiddens)
        # 有3种可能的输出: 蕴涵、矛盾和中性
        self.aggregate = Aggregate(num_inputs_agg, num_hiddens, num_outputs=3)

    def forward(self, X: Tensor) -> Tensor:
        """
        参数:
        X的形状: (premises, hypotheses)
            premises: [batch_size, num_tokens_a], 已经被处理成了固定长度 [batch_size, num_steps]
            hypotheses: [batch_size, num_tokens_b], 已经被处理成了固定长度 [batch_size, num_steps]
            数据集中前提(premises)和假设(hypotheses)都已经处理成了相同的长度

        返回:
        Y_hat: [batch_size, num_outputs=3]
        """
        premises, hypotheses = X
        # A.shape [batch_size, num_tokens_a, embed_size]
        # B.shape [batch_size, num_tokens_b, embed_size]
        A = self.embedding(premises)
        B = self.embedding(hypotheses)
        # beta.shape  [batch_size, num_tokens_a, embed_size]
        # alpha.shape [batch_size, num_tokens_b, embed_size]
        beta, alpha = self.attend(A, B)

        # V_A.shape [batch_size, num_tokens_a, num_hiddens]
        # V_B.shape [batch_size, num_tokens_b, num_hiddens]
        V_A, V_B = self.compare(A, B, beta, alpha)

        # Y_hat.shape [batch_size, num_outputs=3]
        Y_hat = self.aggregate(V_A, V_B)
        return Y_hat


def download_glove(name: str, cache_dir: str = '../data') -> str:
    """
    下载数据

    参数:
    name: 'glove.6B.50d' | 'glove.6B.100d'
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


def download_glove_extract(name: str, cache_dir: str = '../data') -> str:
    """
    下载数据 & 解压
    """
    # 下载数据集
    fname = download_glove(name, cache_dir)

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


class TokenEmbedding:
    """
    GloVe嵌入

    vec.txt内容:
    of 0.70853 0.57088 -0.4716 0.18048 0.54449 ...
    to 0.68047 -0.039263 0.30186 -0.17792 0.42962 ...
    and 0.26818 0.14346 -0.27877 0.016257 0.11384 ...

    第1列为token, 其它列为token对应的vector

    # 加载
    >>> glove_6b50d = TokenEmbedding('glove.6B.50d')
    >>> len(glove_6b50d)
        400001
    
    >>> glove_6b50d.token_to_idx['beautiful']
        3367

    >>> glove_6b50d.idx_to_token[3367])
        beautiful
    
    >>> tokens = glove_6b50d[['<unk>', 'love', 'you']]
    >>> assert tokens.shape == (3, 50)
    >>> tokens
        tensor([[0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
                 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
                 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
                 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
                 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
                 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
                 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
                 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
                 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
                 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
                [-1.3886e-01,  1.1401e+00, -8.5212e-01, -2.9212e-01,  7.5534e-01,
                 8.2762e-01, -3.1810e-01,  7.2204e-03, -3.4762e-01,  1.0731e+00,
                 -2.4665e-01,  9.7765e-01, -5.5835e-01, -9.0318e-02,  8.3182e-01,
                 -3.3317e-01,  2.2648e-01,  3.0913e-01,  2.6929e-02, -8.6739e-02,
                 -1.4703e-01,  1.3543e+00,  5.3695e-01,  4.3735e-01,  1.2749e+00,
                 -1.4382e+00, -1.2815e+00, -1.5196e-01,  1.0506e+00, -9.3644e-01,
                 2.7561e+00,  5.8967e-01, -2.9473e-01,  2.7574e-01, -3.2928e-01,
                 -2.0100e-01, -2.8547e-01, -4.5987e-01, -1.4603e-01, -6.9372e-01,
                 7.0761e-02, -1.9326e-01, -1.8550e-01, -1.6095e-01,  2.4268e-01,
                 2.0784e-01,  3.0924e-02, -1.3711e+00, -2.8606e-01,  2.8980e-01],
                [-1.0919e-03,  3.3324e-01,  3.5743e-01, -5.4041e-01,  8.2032e-01,
                 -4.9391e-01, -3.2588e-01,  1.9972e-03, -2.3829e-01,  3.5554e-01,
                 -6.0655e-01,  9.8932e-01, -2.1786e-01,  1.1236e-01,  1.1494e+00,
                 7.3284e-01,  5.1182e-01,  2.9287e-01,  2.8388e-01, -1.3590e+00,
                 -3.7951e-01,  5.0943e-01,  7.0710e-01,  6.2941e-01,  1.0534e+00,
                 -2.1756e+00, -1.3204e+00,  4.0001e-01,  1.5741e+00, -1.6600e+00,
                 3.7721e+00,  8.6949e-01, -8.0439e-01,  1.8390e-01, -3.4332e-01,
                 1.0714e-02,  2.3969e-01,  6.6748e-02,  7.0117e-01, -7.3702e-01,
                 2.0877e-01,  1.1564e-01, -1.5190e-01,  8.5908e-01,  2.2620e-01,
                 1.6519e-01,  3.6309e-01, -4.5697e-01, -4.8969e-02,  1.1316e+00]])
    """

    def __init__(self, embedding_name: str) -> None:
        """
        参数:
        embedding_name: 'glove.6B.50d' | 'glove.6B.100d'
        """
        # index_to_token: list of token
        # idx_to_vec.shape: [num_tokens, embed_size]
        self.idx_to_token, self.idx_to_vec = self._load_embedding(
            embedding_name)
        self.unknown_idx = 0  # <unk>对应的idx

        # token_to_idx: dict(), 从token到idx的映射
        self.token_to_idx = {
            token: idx
            for idx, token in enumerate(self.idx_to_token)
        }

    def _load_embedding(self, embedding_name: str) -> Tuple[List[str], Tensor]:
        """
        加载GloVe Embedding

        会增加一个token <unk>, 对应的vector的每个元素都为0

        参数:
        embedding_name: 'glove.6B.50d' | 'glove.6B.100d'
                        glove.6B.50d对应的embed_size=50
                        glove.6B.100d对应的embed_size=100

        返回: (idx_to_token, idx_to_vec)
        idx_to_token: list of token
        idx_to_vec.shape: [num_tokens, embed_size]
        """
        # idx_to_token: list of token
        # idx_to_vec: list of vector, 每个vector是一个float list
        idx_to_token, idx_to_vec = ['<unk>'], []
        data_dir = download_glove_extract(embedding_name)
        # GloVe网站: https://nlp.stanford.edu/projects/glove/
        # fastText网站: https://fasttext.cc/
        with open(os.path.join(data_dir, 'vec.txt'), 'r') as f:
            for line in f:
                elems = line.rstrip().split(' ')
                # 第1列为token, 其它列为token对应的vector
                # token: 一个英文词
                # elems: 表示token对应的vector, list of float
                token, elems = elems[0], [float(elem) for elem in elems[1:]]
                # 跳过标题信息, 例如fastText中的首行
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        # 添加<unk>对应的vector
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, torch.tensor(idx_to_vec)

    def __getitem__(self, tokens: List[str]) -> Tensor:
        """
        返回的是词向量

        参数:
        tokens: list of token

        返回:
        vecs.shape [num_tokens, embed_size]
        """
        # 获取所有tokens的索引
        indices = [
            self.token_to_idx.get(token, self.unknown_idx) for token in tokens
        ]

        # 根据索引返回tokens对应vecs
        vecs = self.idx_to_vec[torch.tensor(indices)]
        return vecs

    def __len__(self) -> int:
        return len(self.idx_to_token)


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
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
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


def predict_snli(net: DecomposableAttention, vocab: Vocab, premise: List[str],
                 hypothesis: List[str], device: torch.device) -> None:
    """
    预测前提和假设之间的逻辑关系
    """
    net.eval()
    premise = torch.tensor(vocab[premise], device=device)
    hypothesis = torch.tensor(vocab[hypothesis], device=device)
    label = torch.argmax(net(
        [premise.reshape((1, -1)),
         hypothesis.reshape((1, -1))]),
                         dim=1)
    print('entailment' if label == 0 else 'contradiction' if label ==
          1 else 'neutral')


def decomposable_attention_net(vocab: Vocab, embed_size: int, num_hiddens: int,
                               num_inputs_attend: int, num_inputs_compare: int,
                               num_inputs_agg: int) -> DecomposableAttention:
    net = DecomposableAttention(vocab, embed_size, num_hiddens,
                                num_inputs_attend, num_inputs_compare,
                                num_inputs_agg)

    # 加载预训练的权重
    glove_embedding = TokenEmbedding('glove.6B.100d')
    embeds = glove_embedding[vocab.idx_to_token]
    net.embedding.weight.data.copy_(embeds)
    return net


def train(batch_size: int, num_epochs: int, num_steps: int, embed_size: int,
          num_hiddens: int, num_inputs_attend: int, num_inputs_compare: int,
          num_inputs_agg: int) -> Tuple[DecomposableAttention, Vocab]:
    train_iter, test_iter, vocab = load_data_snli(batch_size, num_steps)
    net = decomposable_attention_net(vocab, embed_size, num_hiddens,
                                     num_inputs_attend, num_inputs_compare,
                                     num_inputs_agg)
    history = train_gpu(net, train_iter, test_iter, num_epochs, device=device)
    plot_history(history)
    return net, vocab


if __name__ == '__main__':
    device = try_gpu()
    kwargs = {
        'batch_size': 256,
        'num_epochs': 4,
        'num_steps': 50,
        'embed_size': 100,
        'num_hiddens': 200,
        'num_inputs_attend': 100,
        'num_inputs_compare': 200,
        'num_inputs_agg': 400
    }
    net, vocab = train(**kwargs)
    # train loss 0.495, train acc 0.805, test acc 0.828
    # 19566.2 examples/sec on cuda:0
    predict_snli(net, vocab, ['he', 'is', 'good', '.'],
                 ['he', 'is', 'bad', '.'], device)
    # contradiction