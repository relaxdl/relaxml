from typing import Union, List, Dict, Tuple
import collections
import os
import requests
import hashlib
import zipfile
import tarfile
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
"""
机器翻译与数据集

实现说明:
https://tech.foxrelax.com/rnn/machine_translation_and_dataset/
"""


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
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices: Union[int, List[int],
                                       Tuple[int]]) -> List[str]:
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


if __name__ == '__main__':
    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)
    for X, X_valid_len, Y, Y_valid_len in train_iter:
        assert X.shape == (2, 8)
        assert X_valid_len.shape == (2, )
        assert Y.shape == (2, 8)
        assert X_valid_len.shape == (2, )
        print('X:', X)
        print('X的有效长度:', X_valid_len)
        print('Y:', Y)
        print('Y的有效长度:', Y_valid_len)
        break