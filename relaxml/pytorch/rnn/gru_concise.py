from typing import Dict, Iterator, Callable, List, Tuple, Union
import re
import time
import sys
import math
import os
import random
import hashlib
import requests
import collections
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Optimizer
import matplotlib.pyplot as plt
from tqdm import tqdm
"""
GRU简洁实现

实现说明:
https://tech.foxrelax.com/rnn/gru_concise/
"""


def try_gpu(i: int = 0) -> torch.device:
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


class RNNModel(nn.Module):
    """
    对`nn.RNN`, `nn.GRU`, `nn.LSTM`做了封装

    >>> batch_size, num_steps, vocab_size, num_hiddens = 3, 4, 12, 512
    >>> num_layers, bidirectional = 2, True
    >>> X = torch.arange(0, 12).reshape(batch_size, num_steps)
    >>> device = try_gpu()
    >>> rnn_layer = nn.GRU(vocab_size,
                           num_hiddens,
                           num_layers=num_layers,
                           bidirectional=bidirectional)
    >>> net = RNNModel(rnn_layer, vocab_size)
    >>> state = net.begin_state(batch_size, device)
    >>> Y, new_state = net(X.to(device), state)
    >>> assert Y.shape == (batch_size * num_steps, vocab_size)
    >>> assert new_state.shape == ((2 if bidirectional else 1) * num_layers,
                                   batch_size, num_hiddens)
    """

    def __init__(self, rnn_layer: Union[nn.RNN, nn.GRU, nn.LSTM],
                 vocab_size: int, **kwargs) -> None:
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # 如果RNN是双向的, `num_directions`应该是2, 否则应该是1
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(
        self, X: Tensor, state: Union[Tensor, Tuple[Tensor]]
    ) -> Tuple[Tensor, Union[Tensor, Tuple[Tensor]]]:
        """
        前向传播

        参数:
        X.shape [batch_size, num_steps]
        state.shape
        <1> [num_directions*num_layers, batch_size, num_hiddens]   - nn.RNN  
        <2> [num_directions*num_layers, batch_size, num_hiddens]   - nn.GRU
        <3> ([num_directions*num_layers, batch_size, num_hiddens], 
             [num_directions*num_layers, batch_size, num_hiddens]) - nn.LSTM

        输出: (output, state)
        output.shape [num_steps*batch_size, vocab_size]
        state.shape  和输入的shape一致
        """
        # X.shape [num_steps, batch_size, vocab_size]
        X = F.one_hot(X.T.long(), self.vocab_size)
        X = X.to(torch.float32)

        # Y.shape [num_steps, batch_size, num_directions*num_hiddens]
        Y, state = self.rnn(X, state)

        # output.shape [num_steps*batch_size, vocab_size]
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, batch_size: int,
                    device: torch.device) -> Union[Tensor, Tuple[Tensor]]:
        """
        初始化state

        输出: state: 
        1. [num_directions*num_layers, batch_size, num_hiddens]   - nn.RNN  
        2. [num_directions*num_layers, batch_size, num_hiddens]   - nn.GRU
        3. ([num_directions*num_layers, batch_size, num_hiddens], 
            [num_directions*num_layers, batch_size, num_hiddens]) - nn.LSTM
        """
        if not isinstance(self.rnn, nn.LSTM):
            # `nn.RNN` | `nn.GRU`的隐藏状态
            return torch.zeros((self.num_directions * self.rnn.num_layers,
                                batch_size, self.num_hiddens),
                               device=device)
        else:
            # `nn.LSTM`的隐藏状态
            return (torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device),
                    torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device))


def predict_rnn(prefix: str, num_preds: int, net: Callable, vocab: object,
                device: torch.device) -> str:
    """
    在`prefix`后面生成新字符
    
    这个例子是一个没有训练过的rnn, 所以输出的是一些随机数据
    >>> predict_rnn('time traveller ', 10, net, vocab, try_gpu())
    time traveller vgywzillll

    参数:
    prefix: 预热的字符串
    num_preds: 根据prefix预热之后, 要预测的步数
    net: 网络模型
    vocab: 词表, 将int类型的数据转换为可显示的token
    device: 设备

    输出:
    outputs: 字符串
    """
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape(
        (1, 1))
    for y in prefix[1:]:  # 预热期
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # 预测期
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


def grad_clipping(net: Union[nn.Module, object], theta: float) -> None:
    """
    裁剪梯度

    训练RNN网络的常用技巧
    """
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad**2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def sgd(params: List[Tensor], lr: float, batch_size: int) -> None:
    """
    小批量梯度下降
    """
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


def train_rnn_epoch(epoch: int, net: Union[nn.Module,
                                           object], train_iter: Iterator,
                    loss: Callable, updater: Callable, device: torch.device,
                    use_random_iter: bool) -> Tuple[float, float]:
    """
    训练模型一个epoch

    输出: (ppl, speed)
    ppl: 困惑度
    speed: 每秒处理多少个token
    """
    state = None
    metric = [0.0] * 2  # 统计: 训练集损失之和, token数量之和
    t_start = time.time()
    train_iter_tqdm = tqdm(train_iter, file=sys.stdout)
    for i, (X, Y) in enumerate(train_iter_tqdm):
        # X.shape [batch_size, num_steps]
        # Y.shape [batch_size, num_steps]
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化`state`
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            # 每迭代完一个batch, state都需要detach
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # `state`对于`nn.GRU` | `nn.RNN`
                state.detach_()
            else:
                # `state`对于`nn.LSTM`或对于我们从零开始实现的模型
                for s in state:
                    s.detach_()
        # y.shape [num_steps*batch_size, ]
        y = Y.T.reshape(-1)  # 注意: 这里要拉平, 因为rnn输出结果也是拉平的
        X, y = X.to(device), y.to(device)
        # y_hat.shape [num_steps*batch_size, vocab_size]
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # 因为已经调用了`mean`函数
            updater(batch_size=1)
        metric[0] += l * y.numel()  # 这个batch的训练损失之和
        metric[1] += y.numel()  # 这个batch的token数量
        perplexity = math.exp(metric[0] / metric[1])
        train_iter_tqdm.desc = f'epoch {epoch}, step {i+1}, 困惑度 {perplexity:.1f}'
    return perplexity, metric[1] / (time.time() - t_start)


def train_rnn_gpu(net: Union[nn.Module, object],
                  train_iter: Iterator,
                  vocab: object,
                  lr: float,
                  num_epochs: int,
                  device: torch.device = None,
                  use_random_iter: bool = False,
                  predict_prefix: str = None):
    """
    训练模型
    """
    if device is None:
        device = try_gpu()

    loss = nn.CrossEntropyLoss()
    history = [[]]  # 记录: 困惑度(perplexity), 方便后续绘图
    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr=lr)
    else:
        updater = lambda batch_size: sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_rnn(prefix, 50, net, vocab, device)
    # 训练和预测
    for epoch in range(num_epochs):
        ppl, speed = train_rnn_epoch(epoch, net, train_iter, loss, updater,
                                     device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            if predict_prefix is not None:
                print(predict(predict_prefix))
            history[0].append((epoch + 1, ppl))
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))

    # plot
    plt.figure(figsize=(6, 4))
    # 困惑度(perplexity)
    plt.plot(*zip(*history[0]), '-', label='train')
    plt.xlabel('epoch')
    plt.ylabel('perplexity')
    # 从epoch=1开始显示, 0-1这个范围的数据丢弃不展示,
    # 因为只有训练完成1个epochs之后, 才会有第一条test acc记录
    plt.xlim((1, num_epochs))
    plt.grid()
    plt.legend()
    plt.show()


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


def run() -> None:
    batch_size, num_steps, num_hiddens = 32, 128, 512
    device = try_gpu()
    train_iter, vocab = load_data_time_machine(batch_size, num_steps)
    rnn_layer = nn.GRU(len(vocab), num_hiddens)
    net = RNNModel(rnn_layer, len(vocab))
    net = net.to(device)
    kwargs = {
        'num_epochs': 500,
        'lr': 1,
        'device': device,
        'use_random_iter': True,
        'predict_prefix': 'time traveller'
    }
    train_rnn_gpu(net, train_iter, vocab, **kwargs)


if __name__ == '__main__':
    run()
# 困惑度 1.1, 295581.0 词元/秒 cuda:0
# time traveller again i was told he was in thelaboratory and bein
# traveller did not seem to hear don t letme disturb you he s