from typing import Union, List, Dict, Tuple, Any
import collections
import os
import sys
import time
import math
import requests
import hashlib
import zipfile
import tarfile
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt
from tqdm import tqdm
"""
机器翻译(seq2seq)

实现说明:
https://tech.foxrelax.com/rnn/nmt_seq2seq/
"""


class Encoder(nn.Module):
    """
    编码器-解码器结构的基本编码器接口
    """

    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X: Tensor, *args) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError


class Decoder(nn.Module):
    """
    编码器-解码器结构的基本编码器接口
    """

    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs: Tuple[Tensor, Tensor], *args) -> Tensor:
        raise NotImplementedError

    def forward(self, X: Tensor, state: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError


class EncoderDecoder(nn.Module):
    """
    编码器-解码器结构的基类
    """

    def __init__(self, encoder: Encoder, decoder: Decoder,
                 **kwargs: Any) -> None:
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X: Tensor, dec_X: Tensor,
                *args: Any) -> Tuple[Tensor, Tensor]:
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)


class Seq2SeqEncoder(Encoder):
    """
    用于序列到序列学习的循环神经网络编码器

    >>> vocab_size, embed_size, num_hiddens, num_layers, dropout = 10000, 32, 32, 2, 0.1
    >>> batch_size, num_steps = 64, 10
    >>> X = torch.ones(batch_size, num_steps).long()
    >>> encoder = Seq2SeqEncoder(vocab_size, embed_size, num_hiddens, num_layers,
                                 dropout)
    >>> output, state = encoder(X)
    >>> assert output.shape == (num_steps, batch_size, num_hiddens)
    >>> assert state.shape == (num_layers, batch_size, num_hiddens)
    """

    def __init__(self,
                 vocab_size: int,
                 embed_size: int,
                 num_hiddens: int,
                 num_layers: int,
                 dropout: float = 0,
                 **kwargs: Any) -> None:
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # Embedding + GRU
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

    def forward(self, X: Tensor, *args: Any) -> Tuple[Tensor, Tensor]:
        """
        分两个步骤:
        1. 使用embedding处理X, 得到word embedding
        2. 使用GRU来处理word embedding得到最终的输出

        参数:
        X: [batch_size, num_steps]

        输出: (output, state)
        output: [num_steps, batch_size, num_hiddens]
        state: [num_layers, batch_size, num_hiddens]
        """

        # X.shape [batch_size, num_steps, embed_size]
        X = self.embedding(X)
        # 在循环神经网络模型中, 第一个轴对应于时间步
        # X.shape [num_steps, batch_size, embed_size]
        X = X.permute(1, 0, 2)

        # output.shape [num_steps, batch_size, num_hiddens]
        # state.shape [num_layers, batch_size, num_hiddens]
        output, state = self.rnn(X)
        return output, state


class Seq2SeqDecoder(Decoder):
    """
    用于序列到序列学习的循环神经网络解码器

    >>> vocab_size, embed_size, num_hiddens, num_layers, dropout = 10000, 32, 32, 2, 0.1
    >>> batch_size, num_steps = 64, 10
    >>> enc_X = torch.ones(batch_size, num_steps).long()
    >>> dec_X = torch.ones(batch_size, num_steps).long()
    >>> encoder = Seq2SeqEncoder(vocab_size, embed_size, num_hiddens, num_layers,
                                 dropout)
    >>> decoder = Seq2SeqDecoder(vocab_size, embed_size, num_hiddens, num_layers,
                                 dropout)
    # 编码
    >>> enc_outputs = encoder(enc_X)
    >>> assert enc_outputs[0].shape == (num_steps, batch_size, num_hiddens)
    >>> assert enc_outputs[1].shape == (num_layers, batch_size, num_hiddens)
    # 初始化解码器state
    >>> dec_state = decoder.init_state(enc_outputs)
    # 解码
    >>> dec_outputs = decoder(dec_X, dec_state)
    >>> assert dec_outputs[0].shape == (batch_size, num_steps, vocab_size)
    >>> assert dec_outputs[1].shape == (num_layers, batch_size, num_hiddens)
    """

    def __init__(self,
                 vocab_size: int,
                 embed_size: int,
                 num_hiddens: int,
                 num_layers: int,
                 dropout: float = 0,
                 **kwargs: Any) -> None:
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        # Embedding + GRU + Dense
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 注意GRU的输入: 上下文变量在所有的时间步与解码器的输入进行拼接(concatenate),
        # 也就是: embed_size + num_hiddens
        self.rnn = nn.GRU(embed_size + num_hiddens,
                          num_hiddens,
                          num_layers,
                          dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs: Tuple[Tensor, Tensor],
                   *args: Any) -> Tensor:
        """

        参数:
        enc_outputs: (output, state) 就是编码器的输出

        输出:
        state: [num_layers, batch_size, num_hiddens]
        """
        # 直接使用编码器最后一个时间步的隐藏状态(state)来初始化解码器的隐藏状态
        return enc_outputs[1]

    def forward(self, X: Tensor, state: Tensor) -> Tuple[Tensor, Tensor]:
        """
        分4个步骤:
        1. 使用embedding处理X, 得到word embedding
        2. 拼接word embedding和Encoder的最终state, 得到X_and_context
        3. 使用GRU来处理X_and_context得到输出output
        4. 将output送到dense得到最终的结果

        参数:
        X: [batch_size, num_steps]
        state: [num_layers, batch_size, num_hiddens]

        输出: (output, state)
        output: [batch_size, num_steps, vocab_size]
        state: [num_layers, batch_size, num_hiddens]
        """

        # X.shape [batch_size, num_steps, embed_size]
        X = self.embedding(X)
        # 在循环神经网络模型中, 第一个轴对应于时间步
        # X.shape [num_steps, batch_size, embed_size]
        X = X.permute(1, 0, 2)

        # state[-1]的含义是state可能会有多层, 我们只取最后一层, 广播使其具有与`X`相同的`num_steps`
        # context.shape [num_steps, batch_size, hiddens]
        context = state[-1].repeat(X.shape[0], 1, 1)
        # X_and_context.shape [num_steps, batch_size, embed_size+num_hiddens]
        X_and_context = torch.cat((X, context), 2)

        # output.shape [num_steps, batch_size, num_hiddens]
        # state.shape [num_layers, batch_size, num_hiddens]
        output, state = self.rnn(X_and_context, state)
        # output.shape [num_steps, batch_size, vocab_size]
        #           -> [batch_size, num_steps, vocab_size]
        output = self.dense(output).permute(1, 0, 2)

        return output, state


def sequence_mask(X: Tensor, valid_len: Tensor, value: int = 0) -> Tensor:
    """
    在序列中屏蔽不相关的项

    >>> X = torch.tensor([[1, 2, 3], [4, 5, 6]])
    >>> sequence_mask(X, torch.tensor([1, 2]))
        tensor([[1, 0, 0],
                [4, 5, 0]])

    参数:
    X: [batch_size, num_steps]
    valid_len: [batch_size,]

    返回:
    X: [batch_size, num_steps]
    """
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """
    带遮蔽的softmax交叉熵损失函数

    计算过程如下:
    假设pred的形状为: [2, 5, 10], label的形状为: [2, 5], 则reduction=none时, 计算出来
    的loss的形状为: [2, 5], 如下:
    tensor([[2.4712, 1.7931, 1.6518, 2.3004, 1.0466],
            [3.5565, 2.1062, 3.2549, 3.9885, 2.7302]])

    我们叠加如下的valid_len=tensor([5, 2]), 则会生成如下weights
    tensor([[1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0]])

    所以最终计算出来的loss为:
    tensor([[2.4712, 1.7931, 1.6518, 2.3004, 1.0466],
            [3.5565, 2.1062, 0, 0, 0]])
    最终得到的loss为: tensor([1.8526, 1.1325])
    (2.4712+1.7931+1.6518+2.3004+1.0466)/5 = 1.8526
    (3.5565+2.1062)/5 = 1.1325
    """

    def forward(self, pred, label, valid_len):
        """
        参数:
        pred: [batch_size, num_steps, vocab_size]
        label: [batch_size, num_steps]
        valid_len: [batch_size, ]

        输出:
        weighted_loss: [batch_size, ]
        """
        weights = torch.ones_like(label)
        # weights.shape [batch_size, num_steps]
        weights = sequence_mask(weights, valid_len)
        self.reduction = 'none'
        # unweighted_loss.shape [batch_size, num_steps]
        unweighted_loss = super(MaskedSoftmaxCELoss,
                                self).forward(pred.permute(0, 2, 1), label)
        # weighted_loss.shape [batch_size, num_steps]
        #                  -> [batch_size, ]
        weighted_loss = (weights * unweighted_loss).mean(dim=1)
        return weighted_loss


def try_gpu(i: int = 0) -> torch.device:
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


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


def train_seq2seq_gpu(net: nn.Module,
                      data_iter: DataLoader,
                      lr: float,
                      num_epochs: int,
                      tgt_vocab: Any,
                      device: torch.device = None) -> None:
    """
    用GPU训练模型
    """
    if device is None:
        device = try_gpu()

    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)
        elif type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if 'weight' in param:
                    nn.init.xavier_normal_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    history = [[]]  # 记录: 训练集损失, 方便后续绘图
    for epoch in range(num_epochs):
        t_start = time.time()
        metric_train = [0.0] * 2  # 训练损失总和, 词元数量
        data_iter_tqdm = tqdm(data_iter, file=sys.stdout)
        for i, batch in enumerate(data_iter_tqdm):
            optimizer.zero_grad()
            # X.shape [batch_size, num_steps]
            # X_valid_len.shape [batch_size, ]
            # Y.shape [batch_size, num_steps]
            # Y_valid_len.shape [batch_size, ]
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            # bos.shape [batch_size, 1]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)
            # dec_input.shape [batch_size, num_steps]
            dec_input = torch.cat((bos, Y[:, :-1]), 1)
            # Y_hat.shape [batch_size, num_steps, vocab_size]
            Y_hat, _ = net(X, dec_input, X_valid_len)
            # l.shape [batch_size, ]
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()  # 损失函数的标量进行'反传'
            grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric_train[0] += float(l.sum())
                metric_train[1] += float(num_tokens)
            data_iter_tqdm.desc = f'epoch {epoch}, step {i+1}, train loss {metric_train[0] / metric_train[1]:.3f}'
        if (epoch + 1) % 10 == 0:
            history[0].append((epoch + 1, metric_train[0] / metric_train[1]))
    print(
        f'loss {metric_train[0] / metric_train[1]:.3f}, {metric_train[1] / (time.time() - t_start):.1f} '
        f'tokens/sec on {str(device)}')

    # plot 训练集损失
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


def predict_seq2seq(
        net: nn.Module,
        src_setence: List[str],
        src_vocab: Any,
        tgt_vocab: Any,
        num_steps: int,
        device: torch.device,
        save_attention_weights: bool = False) -> Tuple[str, List[Tensor]]:
    """
    序列到序列模型的预测
    """
    net.eval()
    src_tokens = src_vocab[src_setence.lower().split(' ')] + [
        src_vocab['<eos>']
    ]
    # enc_valid_len.shape [1, num_tokens]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # enc_X.shape [1, num_steps]
    enc_X = torch.unsqueeze(torch.tensor(src_tokens,
                                         dtype=torch.long,
                                         device=device),
                            dim=0)
    # enc_outputs[0].shape [num_steps, 1, num_hiddens]
    # enc_outputs[1].shape [num_layers, 1, num_hiddens]
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # dec_X.shape [1, 1]
    dec_X = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']],
                                         dtype=torch.long,
                                         device=device),
                            dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        # Y.shape [1, 1, vocab_size]
        # dec_state.shape [num_layers, 1, num_hiddens]
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 使用具有预测最高可能性的token, 作为解码器在下一时间步的输入
        # dex_X.shape [1, 1]
        dec_X = Y.argmax(dim=2)
        # pred.shape [1,]
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 保存注意力权重, 每一个step都会保存一个
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列结束词元被预测, 输出序列的生成就完成了
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq


def bleu(pred_seq: str, label_seq: str, k: int) -> float:
    """
    计算 BLEU
    """
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[''.join(label_tokens[i:i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[''.join(pred_tokens[i:i + n])] > 0:
                num_matches += 1
                label_subs[''.join(pred_tokens[i:i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score


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


def mnt_seq2seq(src_vocab_size: int, tgt_vocab_size: int, embed_size: int,
                num_hiddens: int, num_layers: int,
                dropout: float) -> EncoderDecoder:
    encoder = Seq2SeqEncoder(src_vocab_size, embed_size, num_hiddens,
                             num_layers, dropout)
    decoder = Seq2SeqDecoder(tgt_vocab_size, embed_size, num_hiddens,
                             num_layers, dropout)
    return EncoderDecoder(encoder, decoder)


def train(embed_size: int,
          num_hiddens: int,
          num_layers: int,
          dropout: float,
          batch_size: int,
          num_steps: int,
          lr: float,
          num_epochs: int,
          device: torch.device = None) -> None:
    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)
    net = mnt_seq2seq(len(src_vocab), len(tgt_vocab), embed_size, num_hiddens,
                      num_layers, dropout)
    train_seq2seq_gpu(net, train_iter, lr, num_epochs, tgt_vocab, device)
    return net, src_vocab, tgt_vocab


def test(net: EncoderDecoder, src_vocab: Vocab, tgt_vocab: Vocab,
         num_steps: int, device: torch.device) -> None:
    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        translation, _ = predict_seq2seq(net, eng, src_vocab, tgt_vocab,
                                         num_steps, device)
        print(
            f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')


if __name__ == '__main__':
    device = try_gpu()
    kwargs = {
        'embed_size': 32,
        'num_hiddens': 32,
        'num_layers': 2,
        'dropout': 0.1,
        'batch_size': 64,
        'num_steps': 20,
        'lr': 0.005,
        'num_epochs': 300,
        'device': device
    }
    net, src_vocab, tgt_vocab = train(**kwargs)
    kwargs_test = {'num_steps': 10, 'device': device}
    test(net, src_vocab, tgt_vocab, **kwargs_test)
# loss 0.011, 23286.0 tokens/sec on cpu
# go . => va !, bleu 1.000
# i lost . => j'ai perdu ., bleu 1.000
# he's calm . => il est riche ., bleu 0.658
# i'm home . => je suis chez moi ., bleu 1.000
