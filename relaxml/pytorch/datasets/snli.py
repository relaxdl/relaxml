from typing import Tuple, List, Union, Dict
import collections
import os
import re
import hashlib
import requests
import tarfile
import zipfile
import torch
from torch.utils.data import DataLoader, Dataset
"""
斯坦福自然语言推断语料库(Stanford Natural Language Inference, SNLI)
"""


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


if __name__ == '__main__':
    batch_size, num_steps = 128, 50
    train_iter, test_iter, vocab = load_data_snli(batch_size, num_steps)
    print(len(vocab))
    # 18678
    for X, Y in train_iter:
        assert X[0].shape == (batch_size, num_steps)
        assert X[1].shape == (batch_size, num_steps)
        assert Y.shape == (batch_size, )
        break
