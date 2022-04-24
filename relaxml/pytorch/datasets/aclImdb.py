from typing import List, Tuple, Union, Dict
import os
import collections
import tarfile
import zipfile
import hashlib
import requests
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


def download(cache_dir: str = '../data') -> str:
    """
    下载数据
    """
    sha1_hash = '996f2a201ddb7d90f0333d177d26ccb876b86e2f'
    url = 'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/aclImdb.zip'
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
    # e.g. ../data/aclImdb.zip
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
    # e.g. ../data/aclImdb
    return data_dir


def read_imdb(data_dir: str, is_train: bool) -> Tuple[List[str], List[int]]:
    """
    读取IMDb评论数据集文本序列和标签

    pos(积极) - 1
    neg(消极) - 0

    目录格式:
    aclImdb/
           test/
               neg/
                  *.txt
               pos/
                  *.txt
           train/
               neg/
                  *.txt
               pos/
                  *.txt
    
    >>> data_dir = download_extract()
    >>> train_data = read_imdb(data_dir, is_train=True)
    >>> print('训练集数目: ', len(train_data[0]))
    >>> for x, y in zip(train_data[0][:3], train_data[1][:3]):
    >>>     print('标签: ', y, 'review:', x[0:60])
        训练集数目:  25000
        标签:  1 review: You know the people in the movie are in for it when king-siz
        标签:  1 review: Arthur Bach is decidedly unhappy in his life as a multi-mill
        标签:  1 review: Pickup on South Street (1953), directed by movie maverick Sa

    返回: (data, labels)
    data: list of sentence
    labels: list of label
    """
    data, labels = [], []
    for label in ('pos', 'neg'):
        folder_name = os.path.join(data_dir, 'train' if is_train else 'test',
                                   label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '')
                data.append(review)
                labels.append(1 if label == 'pos' else 0)
    return data, labels


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


def load_array(data_arrays: List[Tensor],
               batch_size: int,
               is_train: bool = True) -> DataLoader:
    """
    构造一个PyTorch数据迭代器
    """
    dataset = TensorDataset(*data_arrays)
    return DataLoader(dataset, batch_size, shuffle=is_train)


def load_data_imdb(
        batch_size: int,
        num_steps: int = 500) -> Tuple[DataLoader, DataLoader, Vocab]:
    """
    返回数据迭代器和IMDb评论数据集的词表

    >>> batch_size, num_steps = 512, 500
    >>> train_iter, test_iter, vocab = load_data_imdb(batch_size, num_steps)
    >>> for X, y in train_iter:
    >>>     assert X.shape == (batch_size, num_steps)
    >>>     assert y.shape == (batch_size, )
    >>>     break

    返回: (train_iter, test_iter, vocab)
    """
    data_dir = download_extract()
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    train_tokens = tokenize(train_data[0], token='word')
    test_tokens = tokenize(test_data[0], token='word')
    vocab = Vocab(train_tokens, min_freq=5)  # 过滤掉出现不到5次的单词
    train_features = torch.tensor([
        truncate_pad(vocab[line], num_steps, vocab['<pad>'])
        for line in train_tokens
    ])
    test_features = torch.tensor([
        truncate_pad(vocab[line], num_steps, vocab['<pad>'])
        for line in test_tokens
    ])
    train_iter = load_array((train_features, torch.tensor(train_data[1])),
                            batch_size)
    test_iter = load_array((test_features, torch.tensor(test_data[1])),
                           batch_size,
                           is_train=False)
    return train_iter, test_iter, vocab


if __name__ == '__main__':
    batch_size, num_steps = 512, 500
    train_iter, test_iter, vocab = load_data_imdb(batch_size, num_steps)
    for X, y in train_iter:
        assert X.shape == (batch_size, num_steps)
        assert y.shape == (batch_size, )
        break
