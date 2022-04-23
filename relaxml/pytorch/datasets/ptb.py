from typing import List, Union, Tuple, Dict
import os
import hashlib
import tarfile
import collections
import zipfile
import requests
import random
import math
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


def download(cache_dir: str = '../data') -> str:
    """
    下载数据
    """
    sha1_hash = '319d85e578af0cdc590547f26231e4e31cdf1e42'
    url = 'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/ptb.zip'
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
    # e.g. ../data/ptb.zip
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
    # e.g. ../data/ptb
    return data_dir


def read_ptb() -> List[List[str]]:
    """
    将PTB数据集加载到文本行的列表中(这个数据集一共42069句话)

    ptb.zip
    ptb.test.txt - 440K
    ptb.train.txt - 4.9M
    ptb.valid.txt - 391K

    Examples:
    >>> sentences = read_ptb()
    >>> len(sentences) # 显示一共多少句话
    42069

    >>> sentences[:3] # 显示前三句话
    [['aer', 'banknote', 'berlitz', 'calloway', 'centrust', 'cluett',
      'fromstein', 'gitano', 'guterman', 'hydro-quebec', 'ipo', 'kia',
      'memotec', 'mlx', 'nahb', 'punts', 'rake', 'regatta', 'rubens',
      'sim', 'snack-food', 'ssangyong', 'swapo', 'wachter'],
    ['pierre', '<unk>', 'N', 'years', 'old', 'will', 'join', 'the', 'board',
      'as', 'a', 'nonexecutive', 'director', 'nov.', 'N'],
    ['mr.', '<unk>', 'is', 'chairman', 'of', '<unk>', 'n.v.', 'the', 
      'dutch', 'publishing', 'group']]

    返回:
    outputs: list of sentence
             其中每个sentence是一个token list, 
             e.g. ['mr.', '<unk>', 'is', 'chairman', 'of', '<unk>', 'n.v.', 'the', 
                   'dutch', 'publishing', 'group']
    """
    data_dir = download_extract('ptb')
    # Read the training set.
    with open(os.path.join(data_dir, 'ptb.train.txt')) as f:
        raw_text = f.read()
    return [line.split() for line in raw_text.split('\n')]


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


def subsample(sentences: List[List[str]],
              vocab: Vocab) -> Tuple[List[List[str]], Dict[str, int]]:
    """
    下采样高频词, 返回下采样之后的sentences和counter

    1. 过滤掉未知token'<unk>'
    2. 下采样sentences(一定概率的删除高频词, 频率越高, 删除的概率越大)

    >>> sentences = read_ptb()
    >>> vocab = Vocab(sentences, min_freq=10)
    >>> subsampled, counter = subsample(sentences, vocab)
    >>> subsampled[:3]
        [[], ['old', 'join', 'director', 'nov.'], ['n.v.', 'dutch', 'publishing', 'group']]

    参数:
    sentences: list of sentence
               其中每个sentence是一个token list, 
               e.g. ['mr.', '<unk>', 'is', 'chairman', 'of', '<unk>', 'n.v.', 'the', 
                      utch', 'publishing', 'group']
    vocab: 词典

    返回:
    output: (subsampled, counter)
    subsampled: list of sentence, 下采样之后的sentence列表
    counter: collections.Counter实例
    """
    # 过滤掉未知token'<unk>'
    sentences = [[token for token in line if vocab[token] != vocab.unk]
                 for line in sentences]
    counter = count_corpus(sentences)
    num_tokens = sum(counter.values())

    # 如果在下采样期间保留token, 则返回True
    # 一定概率的删除高频词, 频率越高, 删除的概率越大
    def keep(token):
        return (random.uniform(0, 1) < math.sqrt(
            1e-4 / counter[token] * num_tokens))

    return ([[token for token in line if keep(token)]
             for line in sentences], counter)


def get_centers_and_contexts(
        corpus: List[List[str]],
        max_window_size: int) -> Tuple[List[int], List[List[int]]]:
    """
    返回Skip-Gram中的中心词和上下文词(针对每一行, 我们使用了一个随机的窗口大小)

    1. 遍历corpus的每一行
    2. 遍历每一行的每一个token, 随机一个window_size进行采样

    Example:
    >>> tiny_dataset = [list(range(7)), list(range(7, 10))]
    dataset [[0, 1, 2, 3, 4, 5, 6], 
            [7, 8, 9]]
    >>> for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
    >>>     print('center', center, 'has contexts', context)
    center 0 has contexts [1]
    center 1 has contexts [0, 2, 3]
    center 2 has contexts [0, 1, 3, 4]
    center 3 has contexts [2, 4]
    center 4 has contexts [2, 3, 5, 6]
    center 5 has contexts [3, 4, 6]
    center 6 has contexts [5]
    center 7 has contexts [8, 9]
    center 8 has contexts [7, 9]
    center 9 has contexts [7, 8]

    参数:
    corpus: list of sentence
            其中每个sentence是一个token_id list,
            e.g. [6697, 4127, 993, 1325, 2641, 2340, 4465, 3927, 1773, 1291]
    max_window_size: 采样的滑动窗口大小, 对于每一行数据, 窗口大小是随机的, 范围在: window_size = [1-max_window_size]
                     从中心词向前和向后看最多window_size个单词

    返回: (centers, contexts)
    centers: list of token_id
    contexts: list of context
              其中每个context表示一个中心词对应的上下文词token_id list
    """
    centers, contexts = [], []
    for line in corpus:
        # 要形成"中心词-上下文词"对, 每个句子至少需要有2个词
        if len(line) < 2:
            continue
        centers += line  # 一次性增加了了n个centers
        for i in range(len(line)):  # 上下文窗口中间`i`
            window_size = random.randint(1, max_window_size)  # 随机一个窗口大小
            indices = list(
                range(max(0, i - window_size),
                      min(len(line), i + 1 + window_size)))
            # 从上下文词中排除中心词
            indices.remove(i)
            contexts.append([line[idx] for idx in indices])
    return centers, contexts


class RandomGenerator:
    """
    根据采样权重在{1, 2, ..., len(sampling_weights)}这些token_id中随机采样

    索引为1、2、...（索引0是vocab中排除的未知标记<unk>)

    >>> sentences = read_ptb()
    >>> vocab = Vocab(sentences, min_freq=10)
    >>> _, counter = subsample(sentences, vocab)
    # 生成采样权重
    >>> sampling_weights = [
    >>>     counter[vocab.to_tokens(i)]**0.75 for i in range(1, len(vocab))
    >>> ]
    >>> generator = RandomGenerator(sampling_weights)
    >>> generator.draw()
        1259
    """

    def __init__(self, sampling_weights: List[float]) -> None:
        """
        参数:
        sampling_weights: list of weight, 对应{1, ..., len(sampling_weights)}这些
                          token_id的采样权重
        """
        # 索引为1、2、...（索引0是vocab中排除的未知标记<unk>）
        self.population = list(range(1, len(sampling_weights) + 1))
        self.sampling_weights = sampling_weights  # 采样权重
        self.candidates = []  # list of token_id
        self.i = 0

    def draw(self) -> int:
        """
        采样(返回一个随机的token_id)

        在实现的时候, 会一次性采样出k=10000个token_id缓存起来, 返回结果的时候
        从缓存中直接返回就行, 当10000都用过一边之后, 再重新采样

        返回:
        output: token_id
        """
        if self.i == len(self.candidates):
            # 根据sampling_weights来采样:
            # 一次性采样出来, 缓存`k`个随机采样结果
            self.candidates = random.choices(self.population,
                                             self.sampling_weights,
                                             k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i - 1]


def get_negatives(all_contexts: List[List[int]], vocab: Vocab,
                  counter: Dict[str, int], K: int) -> List[List[int]]:
    """
    返回负采样中的噪声词

    >>> sentences = read_ptb()
    >>> vocab = Vocab(sentences, min_freq=10)
    >>> subsampled, counter = subsample(sentences, vocab)
    >>> corpus = [vocab[line] for line in subsampled]
    >>> all_centers, all_contexts = get_centers_and_contexts(corpus, 4)
    >>> all_negatives = get_negatives(all_contexts, vocab, counter, 5)
    >>> all_centers[:3]
        [392, 2115, 145]
    >>> all_contexts[:3]  # 正样本
        [[2115], [392, 145, 5], [2115, 5]]
    >>> all_negatives[:3] # 负样本
        [[4218, 9, 5694, 12, 38], 
         [4597, 131, 43, 2249, 15, 403, 5705, 3559, 83, 3550, 14, 40, 335, 6562, 652], 
         [852, 64, 2752, 4355, 1338, 4987, 117, 68, 177, 1618]]

    参数:
    all_contexts: list of context (正样本)
                  其中每个context表示一个中心词对应的上下文词token_id list
    vocab: 字典
    counter: collections.Counter实例
    K: 负采样的参数, 也就是一个正样本对应多少个负样本(通常为5)

    返回:
    all_negatives: list of negative (负样本)
                   其中每个negative表示一个中心词对应的负样本token_id list
                   len(all_contexts) == len(all_negatives)
    """

    # 为每个token_id{1、2、..., len(vocab)-1}（索引0是vocab中排除的未知标记<unk>）
    # 生成对应的采样权重
    sampling_weights = [
        counter[vocab.to_tokens(i)]**0.75 for i in range(1, len(vocab))
    ]
    all_negatives, generator = [], RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negatives = []  # 负样本列表
        while len(negatives) < len(contexts) * K:
            neg = generator.draw()
            # 注意: 我们做了特殊处理, 噪声词不能是上下文词
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives


def batchify(
    data: List[Tuple[int, List[int], List[int]]]
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    返回带有负采样的Skip-Gram的小批量样本(作为DataLoader.collate_fn使用)

    会自动计算max_len, max_len为这个批量中: max(len(context) + len(negative))

    输入:
    data: 一个批量的[(center, context, negative)]
      center: token_id
      context: 表示一个中心词对应的上下文词token_id list
      negative: 表示一个中心词对应的负样本token_id list
    
    返回:
    output: (centers, contexts_negatives, masks, labels)
      centers: [batch_size, 1]
      contexts_negatives: [batch_size, max_len] 包含了正样本 + 负样本, 长度不足的补0
      masks: [batch_size, max_len]
      labels: [batch_size, max_len] 长度不足的补0
    """
    # 计算最大的长度
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    # 返回: (centers, contexts_negatives, masks, labels)
    return (torch.tensor(centers).reshape(
        (-1, 1)), torch.tensor(contexts_negatives), torch.tensor(masks),
            torch.tensor(labels))


def load_data_ptb(batch_size: int, max_window_size: int,
                  num_noise_words: int) -> Tuple[DataLoader, Vocab]:
    """
    下载PTB数据集，然后将其加载到内存中
    
    注意: 我们保证每个batch里面的数据长度是一样的, 这样就可以批量处理, 但是不同batch之间
    的数据长度(max_len)可能是不同的

    # batch_size=2, 这个批量的max_len是54
    >>> names = ['centers', 'contexts_negatives', 'masks', 'labels']
    >>> data_iter, vocab = load_data_ptb(2, 5, 5)
    >>> for batch in data_iter:
    >>>     for name, data in zip(names, batch):
    >>>         print('{} shape: {}'.format(name, data.shape))
    >>>     break
        centers shape: torch.Size([2, 1])               # [batch_size, 1]
        contexts_negatives shape: torch.Size([2, 54])   # [batch_size, max_len] 包含了正样本 + 负样本, 长度不足的补0
        masks shape: torch.Size([2, 54])                # [batch_size, max_len]
        labels shape: torch.Size([2, 54])               # [batch_size, max_len] 长度不足的补0

    # batch_size=512, 每个批量的max_len基本上都是60
    >>> data_iter, vocab = load_data_ptb(512, 5, 5)
    >>> for batch in data_iter:
    >>>     for name, data in zip(names, batch):
    >>>         print('{} shape: {}'.format(name, data.shape))
    >>>     break
        centers shape: torch.Size([512, 1])              # [batch_size, 1]
        contexts_negatives shape: torch.Size([512, 60])  # [batch_size, max_len] 包含了正样本 + 负样本, 长度不足的补0
        masks shape: torch.Size([512, 60])               # [batch_size, max_len]
        labels shape: torch.Size([512, 60])              # [batch_size, max_len] 长度不足的补0
    """
    sentences = read_ptb()
    vocab = Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = get_negatives(all_contexts, vocab, counter,
                                  num_noise_words)

    class PTBDataset(Dataset):

        def __init__(self, centers, contexts, negatives):
            assert len(centers) == len(contexts) == len(negatives)
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives

        def __getitem__(self, index):
            return (self.centers[index], self.contexts[index],
                    self.negatives[index])

        def __len__(self):
            return len(self.centers)

    dataset = PTBDataset(all_centers, all_contexts, all_negatives)

    data_iter = DataLoader(dataset,
                           batch_size,
                           shuffle=True,
                           collate_fn=batchify)
    return data_iter, vocab


if __name__ == '__main__':
    pass