from typing import Dict, Iterator, List, Tuple, Union
import re
import os
import random
import hashlib
import requests
import collections
import torch
"""
语言模型和数据集

实现说明:
https://tech.foxrelax.com/rnn/language_models_and_dataset/

时间机语料库:
大约3211行, 30000个单词左右, 是一个很小的语料库
"""


def download(cache_dir: str = '../data') -> str:
    """
    下载数据
    """
    sha1_hash = '090b5e7e70c295757f55df93cb0a180b9691891a'
    url = 'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/time_machine.txt'
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
    # e.g. ../data/time_machine.dat
    return fname


def read_time_machine(cache_dir: str = '../data') -> List[str]:
    """
    >>> lines = read_time_machine()
    >>> print(f'text lines: {len(lines)}')
        text lines: 3221
    >>> print(lines[0])
        the time machine by h g wells
    >>> print(lines[10])
        twinkled and his usually pale face was flushed and animated the
    """
    with open(download(cache_dir), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


def tokenize(lines: List[str], token: str = 'word') -> List[List[str]]:
    """
    将文本行拆分为单词或字符token

    >>> lines = read_time_machine()
    >>> tokens = tokenize(lines, token='word')
    >>> for i in range(3):
    >>>     print(tokens[i])
       ['the', 'time', 'machine', 'by', 'h', 'g', 'wells']
       []
       []
    >>> tokens = tokenize(lines, token='char')
    >>> for i in range(3):
    >>>     print(tokens[i])
       ['t', 'h', 'e', ' ', 't', 'i', 'm', 'e', ' ', 'm', 
        'a', 'c', 'h', 'i', 'n', 'e', ' ', 'b', 'y', ' ', 
        'h', ' ', 'g', ' ', 'w', 'e', 'l', 'l', 's']
       []
       []
    """
    assert token in ('word', 'char'), 'Unknown token type: ' + token
    return [line.split() if token == 'word' else list(line) for line in lines]


def count_corpus(tokens: Union[List[str], List[List[str]]]) -> Dict[str, int]:
    """
    统计token的频率
    
    >>> lines = read_time_machine()
    >>> tokens = tokenize(lines, token='word')
    >>> counter = count_corpus(tokens)
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

    >>> lines = read_time_machine()
    >>> tokens = tokenize(lines)
    >>> vocab = Vocab(tokens, reserved_tokens=['<bos>', '<eos>'])
    >>> print(vocab.token_to_idx.items())
        dict_items([('<unk>', 0), ('<bos>', 1), ('<eos>', 2), 
                    ('the', 3), ('i', 4), ('and', 5)...]
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


def load_corpus_time_machine(
        max_tokens: int = -1,
        token: str = 'char',
        cache_dir: str = '../data') -> Tuple[List[int], Vocab]:
    """
    返回时光机器数据集的`corpus`和`vocab`
    
    >>> corpus, vocab = load_corpus_time_machine(token='char')
    >>> print(f'corpus.len={len(corpus)}, vocab.len={len(vocab)}')
    >>> print(corpus[:10])
    >>> print(vocab.to_tokens(corpus[:10]))
    corpus.len=170580, vocab.len=28
    [21, 9, 6, 0, 21, 10, 14, 6, 0, 14]
    ['t', 'h', 'e', ' ', 't', 'i', 'm', 'e', ' ', 'm']

    >>> corpus, vocab = load_corpus_time_machine(token='word')
    >>> print(f'corpus.len={len(corpus)}, vocab.len={len(vocab)}')
    >>> print(corpus[:10])
    >>> print(vocab.to_tokens(corpus[:10]))
    corpus.len=32775, vocab.len=4580
    [4041, 4108, 2414, 504, 1807, 1657, 4444, 1992, 4041, 4108]
    ['the', 'time', 'machine', 'by', 'h', 'g', 'wells', 'i', 'the', 'time']
    
    参数:
    max_tokens: 返回的corpus中最多包含多少个token, -1表示返回整个corpus
    token: char | word
    cache_dir: 数据的缓存目录

    返回: (corpus, vocab)
    corpus: 语料库
    vocab: 词汇表
    """
    lines = read_time_machine(cache_dir)
    tokens = tokenize(lines, token)
    vocab = Vocab(tokens)
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落,
    # 所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


def seq_data_iter_random(corpus: List[int], batch_size: int,
                         num_steps: int) -> Iterator:
    """
    使用随机抽样生成一个小批量子序列
    
    在随机采样中, 每个样本都是在原始的长序列上任意捕获的子序列. 在迭代过程中,
    来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻

    下面的例子可以看到: 生成一个从0到34的序列. 批量大小为2, 时间步数为5, 迭代期间
    来自两个相邻的小批量中的子序列在原始序列中是不相邻的
    >>> my_seq = list(range(35))
    >>> for i, (X, Y) in enumerate(
    >>>         seq_data_iter_random(my_seq, batch_size=2, num_steps=5)):
    >>>     print(f'[{i}] X: {X}, \nY:{Y}')
        [0] X: tensor([[29, 30, 31, 32, 33],
                       [ 9, 10, 11, 12, 13]]), 
            Y:tensor([[30, 31, 32, 33, 34],
                      [10, 11, 12, 13, 14]])
        [1] X: tensor([[24, 25, 26, 27, 28],
                       [14, 15, 16, 17, 18]]), 
            Y:tensor([[25, 26, 27, 28, 29],
                      [15, 16, 17, 18, 19]])
        [2] X: tensor([[19, 20, 21, 22, 23],
                       [ 4,  5,  6,  7,  8]]), 
            Y:tensor([[20, 21, 22, 23, 24],
                      [ 5,  6,  7,  8,  9]])
    """
    # 从随机偏移量开始对序列进行分区, 随机范围包括`num_steps - 1`
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 减去1, 是因为我们需要考虑标签
    num_subseqs = (len(corpus) - 1) // num_steps
    # ⻓度为`num_steps`的子序列的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 在随机抽样的迭代过程中,
    # 来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
    random.shuffle(initial_indices)

    def data(pos):
        # 返回从`pos`位置开始的⻓度为`num_steps`的一个序列
        return corpus[pos:pos + num_steps]

    num_batches = num_subseqs // batch_size

    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里, `initial_indices`包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i:i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)


def seq_data_iter_sequential(corpus: List[int], batch_size: int,
                             num_steps: int) -> Iterator:
    """
    使用顺序分区生成一个小批量子序列
    
    在迭代过程中, 除了对原始序列可以随机抽样外, 该方法保证两个相邻的小批量中的
    子序列在原始序列上也是相邻的

    下面的例子可以看到: 生成一个从0到34的序列. 批量大小为2, 时间步数为5, 迭代期间
    来自两个相邻的小批量中的子序列在原始序列中是相邻的
    >>> my_seq = list(range(35))
    >>> for i, (X, Y) in enumerate(
    >>>         seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5)):
    >>>     print(f'[{i}] X: {X}, \nY:{Y}')
        [0] X: tensor([[ 5,  6,  7,  8,  9],
                       [19, 20, 21, 22, 23]]), 
            Y:tensor([[ 6,  7,  8,  9, 10],
                      [20, 21, 22, 23, 24]])
        [1] X: tensor([[10, 11, 12, 13, 14],
                       [24, 25, 26, 27, 28]]), 
            Y:tensor([[11, 12, 13, 14, 15],
                      [25, 26, 27, 28, 29]])
    """
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - 1 - offset) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset:offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1:offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i:i + num_steps]
        Y = Ys[:, i:i + num_steps]
        yield X, Y


class SeqDataLoader:
    """
    加载序列数据的迭代器
    """

    def __init__(self,
                 batch_size: int,
                 num_steps: int,
                 use_random_iter: bool,
                 max_tokens: int,
                 token: str,
                 cache_dir: str = '../data') -> None:
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(
            max_tokens, token, cache_dir)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self) -> Iterator:
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def load_data_time_machine(
        batch_size: int,
        num_steps: int,
        use_random_iter: bool = False,
        max_tokens: int = -1,
        token: str = 'char',
        cache_dir: str = '../data') -> Tuple[SeqDataLoader, Vocab]:
    """
    >>> batch_size, num_steps = 2, 12
    >>> data_iter, vocab = load_data_time_machine(batch_size, num_steps)
    >>> for X, y in data_iter:
    >>>     assert X.shape == (batch_size, num_steps)
    >>>     assert y.shape == (batch_size, num_steps)
    >>>     print(X)
    >>>     print(y)
    >>>     break
        tensor([[ 2,  1, 13,  4, 15,  9,  5,  6,  2,  1, 21, 19],
                [ 1,  3,  9,  5,  8,  1, 21, 12,  2,  4, 15,  9]])
        tensor([[ 1, 13,  4, 15,  9,  5,  6,  2,  1, 21, 19,  1],
                [ 3,  9,  5,  8,  1, 21, 12,  2,  4, 15,  9,  2]])

    参数:
    batch_size: 批量大小
    num_steps: 时间步长度
    use_random_iter: `使用顺序分区生成一个小批量子序列`还是`使用随机抽样生成一个小批量子序列`
    max_tokens: 返回的corpus中最多包含多少个token, -1表示返回整个corpus
    token: char | word
    cache_dir: 数据缓存目录
    """
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter,
                              max_tokens, token, cache_dir)
    return data_iter, data_iter.vocab


if __name__ == '__main__':
    pass