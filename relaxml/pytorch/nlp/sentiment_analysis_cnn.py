from typing import Any, List, Tuple, Union, Dict
import os
import sys
import time
import collections
import tarfile
import zipfile
import hashlib
import requests
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.optimizer import Optimizer
from matplotlib import pyplot as plt
from tqdm import tqdm
"""
情感分析(使用卷积神经网络)

实现说明:
https://tech.foxrelax.com/nlp/sentiment_analysis_cnn/
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


class TextCNN(nn.Module):
    """
    >>> vocab_size, embed_size, kernel_sizes, nums_channels = 10000, 100, 
          [3, 4, 5], [100, 100, 100]
    >>> batch_size, num_steps = 2, 64
    >>> net = TextCNN(vocab_size, embed_size, kernel_sizes, nums_channels)
    >>> X = torch.ones((batch_size, num_steps), dtype=torch.long)
    >>> assert net(X).shape == (batch_size, 2)
    """

    def __init__(self, vocab_size: int, embed_size: int,
                 kernel_sizes: List[int], num_channels: List[int],
                 **kwargs: Any) -> None:
        """
        参数:
        vocab_size: 字典大小
        embed_size: 预训练的词向量维度
        kernel_sizes: 一维卷积的kernel_size列表, e.g. [3, 4, 5]
        num_channels: 一维卷积的out_channels列表, e.g. [100, 100, 100]
        """
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 这个嵌入层不需要训练
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 2)
        # 最大时间汇聚层没有参数, 因此可以共享此实例
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()

        # 创建多个一维卷积层
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            # c: 一维卷积的out_channels
            # k: 一维卷积的kernel_size
            self.convs.append(nn.Conv1d(2 * embed_size, c, k))

    def forward(self, inputs: Tensor) -> Tensor:
        """
        参数:
        inputs: [batch_size, num_steps]

        返回:
        outputs: [batch_size, 2]
        """
        # 沿着向量维度将两个嵌入层连结起来
        # 每个嵌入层的输出形状都是 [batch_size, num_steps, embed_size]
        # embeddings.shape [batch_size, num_steps, 2 * embed_size]
        embeddings = torch.cat(
            (self.embedding(inputs), self.constant_embedding(inputs)), dim=2)
        # embeddings.shape [batch_size, 2 * embed_size, num_steps]
        embeddings = embeddings.permute(0, 2, 1)

        # embeddings经过conv形状变成: [batch_size, 每个卷积核自己的out_channels, new_num_steps]
        # 经过self.pool处理后形状变成: [batch_size, 每个卷积核自己的out_channels, 1]
        # 经过torch.squeeze处理后形状变成: [batch_size, 每个卷积核自己的out_channels]
        # encoding.shape: [batch_size, 所有卷积核的out_channels之和] 也就是
        #                 [batch_size, sum(num_channels)]
        encoding = torch.cat([
            torch.squeeze(self.relu(self.pool(conv(embeddings))), dim=-1)
            for conv in self.convs
        ],
                             dim=1)
        # outputs.shape [batch_size, 2]
        outputs = self.decoder(self.dropout(encoding))
        return outputs


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


def predict_sentiment(net: nn.Module,
                      vocab: Vocab,
                      sequence: List[str],
                      device: torch.device = None) -> None:
    """
    预测文本序列的情感
    """
    sequence = torch.tensor(vocab[sequence.split()], device=device)
    label = torch.argmax(net(sequence.reshape(1, -1)), dim=1)
    print('positive' if label == 1 else 'negative')


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
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
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
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric_train[0] += float(l * X.shape[0])
                metric_train[1] += float(accuracy(y_hat, y))
                metric_train[2] += float(X.shape[0])
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
                X, y = X.to(device), y.to(device)
                metric_test[0] += float(accuracy(net(X), y))
                metric_test[1] += float(X.shape[0])
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


def textcnn_net(vocab: Vocab, embed_size: int, kernel_sizes: List[int],
                nums_channels: List[int]) -> TextCNN:
    embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
    net = TextCNN(len(vocab), embed_size, kernel_sizes, nums_channels)

    def init_weights(m):
        if type(m) in (nn.Linear, nn.Conv1d):
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    # 加载预训练的权重 & 冻结
    glove_embedding = TokenEmbedding('glove.6B.100d')
    embeds = glove_embedding[vocab.idx_to_token]
    net.embedding.weight.data.copy_(embeds)
    net.constant_embedding.weight.data.copy_(embeds)
    net.constant_embedding.weight.requires_grad = False
    return net


def train(batch_size: int, num_epochs: int, embed_size: int,
          kernel_sizes: List[int], nums_channels: List[int],
          device: torch.device) -> Tuple[TextCNN, Vocab]:
    train_iter, test_iter, vocab = load_data_imdb(batch_size)
    net = textcnn_net(vocab, embed_size, kernel_sizes, nums_channels)
    history = train_gpu(net, train_iter, test_iter, num_epochs, device=device)
    plot_history(history)
    return net, vocab


if __name__ == '__main__':
    device = try_gpu()
    kwargs = {
        'batch_size': 64,
        'num_epochs': 5,
        'embed_size': 100,
        'kernel_sizes': [3, 4, 5],
        'nums_channels': [100, 100, 100],
        'device': device
    }
    net, vocab = train(**kwargs)
    # epoch 0, step 391, train loss 0.398, train acc 0.819: 100%|██████████| 391/391 [00:06<00:00, 63.84it/s]
    # epoch 0, step 391, train loss 0.398, train acc 0.819, test acc 0.880
    # epoch 1, step 391, train loss 0.163, train acc 0.940: 100%|██████████| 391/391 [00:05<00:00, 65.59it/s]
    # epoch 1, step 391, train loss 0.163, train acc 0.940, test acc 0.854
    # epoch 2, step 391, train loss 0.080, train acc 0.971: 100%|██████████| 391/391 [00:05<00:00, 65.60it/s]
    # epoch 2, step 391, train loss 0.080, train acc 0.971, test acc 0.851
    # epoch 3, step 391, train loss 0.048, train acc 0.983: 100%|██████████| 391/391 [00:05<00:00, 65.52it/s]
    # epoch 3, step 391, train loss 0.048, train acc 0.983, test acc 0.859
    # epoch 4, step 391, train loss 0.032, train acc 0.989: 100%|██████████| 391/391 [00:05<00:00, 65.66it/s]
    # epoch 4, step 391, train loss 0.032, train acc 0.989, test acc 0.856
    # train loss 0.032, train acc 0.989, test acc 0.856
    # 4373.9 examples/sec on cuda:0
    predict_sentiment(net, vocab, 'this movie is so great', device)
    # positive
    predict_sentiment(net, vocab, 'this movie is so bad', device)
    # negative