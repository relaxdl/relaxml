from typing import Union, List, Dict, Tuple
import collections
import os
import zipfile
import tarfile
import requests
import random
import hashlib
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
"""
用于预训练BERT的数据集
"""


def download(cache_dir: str = '../data') -> str:
    """
    下载数据
    """
    sha1_hash = '81d2333b501a1d8c32bfe96859e2490991fee293'
    url = 'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/wikitext_2.zip'
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
    # e.g. ../data/wikitext_2.zip
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
    # e.g. ../data/wikitext_2
    return data_dir


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


def _read_wiki(data_dir: str) -> List[List[str]]:
    """
    
    1. paragraphs是一个paragraph列表
    2. 每个paragraph包含多个sentence(每个sentence可以分成多个tokens, 
       这里返回的sentence还没有进行分词)

    >>> paragraphs = _read_wiki(download_extract('wikitext_2'))
    >>> len(paragraphs)
        15496
    # 返回第1个paragraph
    >>> paragraphs[0]
       ['when he died at the age of 78 , the daily telegraph , 
        guardian and times published his obituary , and the museum of 
        london added his pamphlets and placards to their collection', 
        'in 2006 his biography was included in the oxford dictionary of 
        national biography .']
    # 返回第1个paragraph的第1个sentence 
    >>> paragraphs[0][0]
        when he died at the age of 78 , the daily telegraph , 
        guardian and times published his obituary , and the museum of 
        london added his pamphlets and placards to their collection
    """
    file_name = os.path.join(data_dir, 'wiki.train.tokens')
    with open(file_name, 'r') as f:
        lines = f.readlines()
    # 大写字母转换为小写字母
    paragraphs = [
        line.strip().lower().split(' . ') for line in lines
        if len(line.split(' . ')) >= 2
    ]
    random.shuffle(paragraphs)  # 随机打乱段落的顺序
    return paragraphs


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


def _get_next_sentence(
        sentence: List[str], next_sentence: List[str],
        paragraphs: List[List[List[str]]]
) -> Tuple[List[str], List[str], bool]:
    """
    生成NSP任务的训练样本

    50%的概率返回下一个句子, 50%的概率返回随机句子. 也就是生成正样本和
    负样本的数量是一致的

    输入: 
    sentence: e.g. ['this', 'movie', 'is', 'great']
    next_sentence: e.g. ['i', 'like', 'it']
    paragraphs: list of paragraph

    返回: (sentence, next_sentence, is_next)
    sentence: e.g. ['this', 'movie', 'is', 'great']
    next_sentence: e.g. ['i', 'like', 'it']
    is_next: True | False
    """
    if random.random() < 0.5:
        is_next = True
    else:
        # 先随机选择一个paragraph, 在从这个paragraph中随机选择一个sentence
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next


def _get_nsp_data_from_paragraph(
        paragraph: List[List[str]], paragraphs: List[List[List[str]]],
        vocab: Vocab, max_len: int) -> List[Tuple[List[str], List[int], bool]]:
    """
    处理一个paragraph, 返回训练NSP的训练样本

    参数:
    paragraph: 句子列表, 其中每个句子都是token列表
        e.g. [['this', 'movie', 'is', 'great'], 
              ['i', 'like', 'it']]
    paragraphs: list of paragraph
    vocab: 字典
    max_len: 预训练期间的BERT输入序列的最大长度(超过最大长度的tokens忽略掉)

    返回: list of (tokens, segments, is_next)
    tokens: e.g. ['<cls>', 'this', 'movie', 'is', 'great', '<sep>', 'i', 'like', 'it', '<sep>']
    segments: e.g. [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
    is_next: True | False
    """
    nsp_data_from_paragraph = []  # [(tokens, segments, is_next)]
    for i in range(len(paragraph) - 1):
        tokens_a, tokens_b, is_next = _get_next_sentence(
            paragraph[i], paragraph[i + 1], paragraphs)
        # 考虑1个'<cls>'和2个'<sep>', 超过最大长度max_len的tokens忽略掉
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph


def _replace_mlm_tokens(
        tokens: List[str], candidate_pred_positions: List[int],
        num_mlm_preds: int,
        vocab: Vocab) -> Tuple[List[str], List[Tuple[int, str]]]:
    """
    处理一句话(tokens), 返回`MLM的输入, 预测位置以及标签`

    参数:
    tokens: 表示BERT输入序列的token列表
        e.g. ['<cls>', 'this', 'movie', 'is', 'great', '<sep>', 'i', 'like', 'it', '<sep>']
    candidate_pred_positions: 后选预测位置的索引, 会在tokens中过滤掉<cls>, <sep>, 剩下的都算后选预测位置
        (特殊token <cls>, <sep>在MLM任务中不被预测)
        e.g. ['<cls>', 'this', 'movie', 'is', 'great', '<sep>', 'i', 'like', 'it', '<sep>']对应的
        后选预测位置是: [1,2,3,4,6,7,8]
    num_mlm_preds: 需要预测多少个token, 通常是len(tokens)的15%
    vocab: 字典

    返回: (mlm_input_tokens, pred_positions_and_labels)
    mlm_input_tokens: 处理后的tokens, 15%的tokens已经做了替换
        e.g. ['<cls>', 'this', 'movie', '<mask>', 'great', '<sep>', 'i', 'like', 'it', '<sep>']
    pred_positions_and_labels: list of (mlm_pred_position, token)
        mlm_pred_position: 需要预测的位置, e.g. 3
        token: 需要预测的标签, e.g. 'is'
    """
    # 为遮蔽语言模型的输入创建新的token副本，其中输入可能包含替换的'<mask>'或随机token
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []  # [(mlm_pred_position, token)]
    # 打乱后用于在遮蔽语言模型任务中获取15%的随机token进行预测
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            # 已经预测足够的tokens, 返回
            break
        masked_token = None
        # 80%的时间: 将token替换为<mask>
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            # 10%的时间: 保token不变
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            # 10%的时间: 用随机token替换该token
            else:
                masked_token = random.choice(vocab.idx_to_token)
        mlm_input_tokens[mlm_pred_position] = masked_token  # 替换成masked_token
        pred_positions_and_labels.append(
            (mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels


def _get_mlm_data_from_tokens(
        tokens: List[str],
        vocab: Vocab) -> Tuple[List[int], List[int], List[int]]:
    """
    处理一个tokens, 返回训练MLM的数据

    参数:
    tokens: e.g. ['<cls>', 'this', 'movie', 'is', 'great', '<sep>', 'i', 'like', 'it', '<sep>']
    vocab: 字典

    返回: (mlm_input_tokens_ids, pred_positions, mlm_pred_labels_ids)
    mlm_input_tokens_ids: 输入tokens的索引
        e.g. vocab[['<cls>', 'this', 'movie', '<mask>', 'great', '<sep>', 'i', 'like', 'it', '<sep>']]
    pred_positions: 需要预测的位置索引, e.g. [3, ...]
    mlm_pred_labels_ids: 预测的标签索引, e.g. vocab[['is', ...]]
    """
    candidate_pred_positions = []  # list of int
    for i, token in enumerate(tokens):
        # 在遮蔽语言模型任务中不会预测特殊token
        if token in ['<cls>', '<sep>']:
            continue
        candidate_pred_positions.append(i)
    # 遮蔽语言模型任务中预测15%的随机token
    num_mlm_preds = max(1, round(len(tokens) * 0.15))

    # mlm_input_tokens: 处理后的tokens, 15%的tokens已经做了替换
    #   e.g. ['<cls>', 'this', 'movie', '<mask>', 'great', '<sep>', 'i', 'like', 'it', '<sep>']
    # pred_positions_and_labels: list of (mlm_pred_position, token)
    #   mlm_pred_position: 需要预测的位置, e.g. 3
    #   token: 需要预测的标签, e.g. 'is'
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(
        tokens, candidate_pred_positions, num_mlm_preds, vocab)
    pred_positions_and_labels = sorted(pred_positions_and_labels,
                                       key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels
                      ]  # list of int, e.g. [3, ...]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels
                       ]  # list of token, e.g. ['is', ...]
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]


def _pad_bert_inputs(
    examples: List[Tuple[List[int], List[int], List[int], List[int],
                         bool]], max_len: int, vocab: Vocab
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    填充(pad)样本

    参数:
    examples: [(mlm_input_tokens_ids, pred_positions, mlm_pred_labels_ids, segments, is_next)]
        mlm_input_tokens_ids: e.g. vocab[['<cls>', 'this', 'movie', '<mask>', 'great', '<sep>', 'i', 'like', 'it', '<sep>']]
        pred_positions: e.g. [3, ...]
        mlm_pred_labels_ids: e.g. vocab[['is', ...]]
        segments: e.g. [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
        is_next: True | False
    max_len: 最大长度
    vocab: 字典

    返回: (all_token_ids, all_segments, valid_lens, all_pred_positions,
           all_mlm_weights, all_mlm_labels, nsp_labels)
    all_token_ids: [num_examples, max_len], 每个token_ids长度为max_len, 长度不足的用<pad>补足
    all_segments: [num_examples, max_len], 每个segments的长度为max_len, 长度不足的用0补足
    valid_lens: [num_examples, ], 每个token_ids的有效长度, 不包括<pad>
    all_pred_positions: [num_examples, max_num_mlm_preds], 每个pred_positions长度为max_num_mlm_preds, 长度不足的用0补足
    all_mlm_weights: [num_examples, max_num_mlm_preds], 有效的pred_positions对应的权重为1, 填充对应的权重为0
    all_mlm_labels: [num_examples, max_num_mlm_preds], 每个pred_label_ids长度为max_num_mlm_preds, 长度不足的用0补足
    nsp_labels: [num_examples, ]
    """
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens, = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,
         is_next) in examples:
        all_token_ids.append(
            torch.tensor(token_ids + [vocab['<pad>']] *
                         (max_len - len(token_ids)),
                         dtype=torch.long))
        all_segments.append(
            torch.tensor(segments + [0] * (max_len - len(segments)),
                         dtype=torch.long))
        # valid_lens不包括'<pad>'的计数
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
        all_pred_positions.append(
            torch.tensor(pred_positions + [0] *
                         (max_num_mlm_preds - len(pred_positions)),
                         dtype=torch.long))
        # 填充token的预测将通过乘以0权重在损失中过滤掉
        all_mlm_weights.append(
            torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] *
                         (max_num_mlm_preds - len(pred_positions)),
                         dtype=torch.float32))
        all_mlm_labels.append(
            torch.tensor(mlm_pred_label_ids + [0] *
                         (max_num_mlm_preds - len(mlm_pred_label_ids)),
                         dtype=torch.long))
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)


class _WikiTextDataset(Dataset):
    """
    数据处理流程:

    1. 分词, 将原始文本数据处理成`paragraphs`和`sentences`
       <1> paragraphs: List[List[List[str]]] paragraphs -> paragraph -> sentence
       <2> sentences: List[List[str]] sentences -> sentence [去掉了段落的概念]
    2. 使用sentences构建Vocab
    3. 遍历每一个paragraph, 获取下一句子预测任务(NSP)的训练样本(exmaples)
    4. 遍历步骤3的每一个样本(examples), 获取遮蔽语言模型任务(MLM)的训练数据, 拼接到一起
    5. 填充(pad)样本
    """

    def __init__(self, paragraphs: List[List[str]], max_len: int) -> None:
        """
        参数:
        paragraphs: 段落的列表, 每个元素是多个句子列表, e.g. ['this moive is great', 'i like it']
        max_len: 最大长度
        """
        # 1. 分词, 将原始文本数据处理成`paragraphs`和`sentences`
        # 处理前的paragraphs[i]表示句子的列表,
        # e.g. ['this moive is great', 'i like it']
        # 经过处理后的paragraphs[i]表示一个段落句子的token列表,
        # e.g. [['this', 'movie', 'is', 'great'], ['i', 'like', 'it']]
        # paragraphs List[List[List[str]]]
        paragraphs = [
            tokenize(paragraph, token='word') for paragraph in paragraphs
        ]
        # 经过处理后的sentences[i]表示一个句子的token列表, e.g. ['this', 'movie', 'is', 'great']
        # sentences List[List[str]]
        sentences = [
            sentence for paragraph in paragraphs for sentence in paragraph
        ]

        # 2. 使用sentences构建Vocab
        self.vocab = Vocab(
            sentences,
            min_freq=5,
            reserved_tokens=['<pad>', '<mask>', '<cls>', '<sep>'])

        examples = []  # 训练样本的集合
        # 3. 遍历每一个paragraph, 获取下一句子预测任务(NSP)的训练样本(exmaples)
        # 此时的examples: [(tokens, segments, is_next)]
        # tokens: e.g. ['<cls>', 'this', 'movie', 'is', 'great', '<sep>', 'i', 'like', 'it', '<sep>']
        # segments: e.g. [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
        # is_next: True | False
        for paragraph in paragraphs:
            examples.extend(
                _get_nsp_data_from_paragraph(paragraph, paragraphs, self.vocab,
                                             max_len))

        # 4. 遍历步骤3的每一个样本(examples), 获取遮蔽语言模型任务(MLM)的训练数据, 拼接到一起
        # 此时的examples: [(mlm_input_tokens_ids, pred_positions, mlm_pred_labels_ids, segments, is_next)]
        # mlm_input_tokens_ids: 输入tokens的索引 e.g. vocab[['<cls>', 'this', 'movie', '<mask>', 'great', '<sep>', 'i', 'like', 'it', '<sep>']]
        # pred_positions: 需要预测的位置索引, e.g. [3, ...]
        # mlm_pred_labels_ids: 预测的标签索引, e.g. vocab[['is', ...]]
        # segments: e.g. [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
        # is_next: True | False
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab) +
                     (segments, is_next))
                    for tokens, segments, is_next in examples]

        # 5. 填充(pad)样本
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights, self.all_mlm_labels,
         self.nsp_labels) = _pad_bert_inputs(examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)


def load_data_wiki(batch_size: int, max_len: int) -> Tuple[DataLoader, Vocab]:
    """
    加载WikiText_2数据集

    >>> batch_size, max_len = 512, 64
    >>> max_num_mlm_preds = round(max_len * 0.15)  # 根据公式计算
    >>> train_iter, vocab = load_data_wiki(batch_size, max_len)
    >>> for (tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X,
             mlm_Y, nsp_y) in train_iter:
    >>>     assert tokens_X.shape == (batch_size, max_len)
    >>>     assert segments_X.shape == (batch_size, max_len)
    >>>     assert valid_lens_x.shape == (batch_size, )
    >>>     assert pred_positions_X.shape == (batch_size, max_num_mlm_preds)
    >>>     assert mlm_weights_X.shape == (batch_size, max_num_mlm_preds)
    >>>     assert mlm_Y.shape == (batch_size, max_num_mlm_preds)
    >>>     assert nsp_y.shape == (batch_size, )
    >>>     break
    """
    paragraphs = _read_wiki(download_extract('wikitext_2'))
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = DataLoader(train_set, batch_size, shuffle=True)
    return train_iter, train_set.vocab


if __name__ == '__main__':
    pass
