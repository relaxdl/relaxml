from typing import Any, Dict, List
import numpy as np
import itertools
import torch
from torch import Tensor, nn
import matplotlib.pyplot as plt
from tqdm import tqdm
"""
Simple CBOW

实现说明:
https://tech.foxrelax.com/nlp/simple_cbow/
"""
"""
将句子人工分为两类(数字类, 字母类), 虽然都是文本, 但是期望模型能自动区分出
在空间上, 数字和字母是有差别的. 因为数字总是和数字一同出现, 而字母总是和字母
一同出现. 同时我们还做了一些特殊处理, 将数字9混到了字母类中, 所以期望的词嵌
入空间中, 数字9不但靠近数字, 而且也靠近字母
"""
corpus = [
    # numbers
    "5 2 4 8 6 2 3 6 4",
    "4 8 5 6 9 5 5 6",
    "1 1 5 2 3 3 8",
    "3 6 9 6 8 7 4 6 3",
    "8 9 9 6 1 4 3 4",
    "1 0 2 0 2 1 3 3 3 3 3",
    "9 3 3 0 1 4 7 8",
    "9 9 8 5 6 7 1 2 3 0 1 0",

    # alphabets, expecting that 9 is close to letters
    "a t g q e h 9 u f",
    "e q y u o i p s",
    "q o 9 p l k j o k k o p",
    "h g y i u t t a e q",
    "i k d q r e 9 e a d",
    "o p d g 9 s a f g a",
    "i u y g h k l a s w",
    "o l u y a o g f s",
    "o p i u y g d a s j d l",
    "u k i l o 9 l j s",
    "y g i s h k j l f r f",
    "i o h n 9 9 d 9 f a 9",
]


class Dataset:

    def __init__(self, x: np.array, y: np.array, token_to_id: Dict[str, int],
                 id_to_token: Dict[int, str]) -> None:
        """
        参数:
        x: [num_examples, num_steps] CBOW
           [num_examples, 1] Skip-Gram
        y: [num_examples, ]
        token_to_id: token -> id
        id_to_token: id -> token
        """
        self.x, self.y = x, y
        self.token_to_id, self.id_to_token = token_to_id, id_to_token
        self.vocab = token_to_id.keys()

    def sample(self, n: int) -> np.array:
        """
        随机采样, 返回n个样本

        返回: (bx, by)
        bx: [n, num_steps] CBOW
            [n, 1] Skip-Gram
        by: [n, ]
        """
        b_idx = np.random.randint(0, len(self.x), n)
        bx, by = self.x[b_idx].reshape(n, -1), self.y[b_idx]
        return bx, by

    @property
    def num_word(self) -> int:
        """
        语料库中单词总数
        """
        return len(self.token_to_id)


def process_w2v_data(corpus: List[str],
                     skip_window: int = 2,
                     method: str = "cbow") -> Dataset:
    """
    返回适合Skip-Gram或者CBOW训练用的数据集

    >>> batch_size = 6
    >>> ds = process_w2v_data(corpus, 2, 'cbow')
    >>> bx, by = ds.sample(batch_size)
    >>> bx, by = torch.from_numpy(bx), torch.from_numpy(by)
    >>> assert bx.shape == (batch_size, 4)
    >>> assert by.shape == (batch_size, )

    >>> ds = process_w2v_data(corpus, 2, 'skip_gram')
    >>> bx, by = ds.sample(batch_size)
    >>> bx, by = torch.from_numpy(bx), torch.from_numpy(by)
    >>> assert bx.shape == (batch_size, 1)
    >>> assert by.shape == (batch_size, )

    参数:
    corpus: 语料库
    skip_window: 窗口大小
    method: skip_gram | cbow

    返回:
    dataset: Dataset实例
    """
    all_words = [sentence.split(" ") for sentence in corpus]
    all_words = np.array(list(itertools.chain(*all_words)))
    # vocab中的词按照出现的频率从高到低排序(高->低)
    vocab, v_count = np.unique(all_words, return_counts=True)
    vocab = vocab[np.argsort(v_count)[::-1]]
    token_to_id = {v: i for i, v in enumerate(vocab)}
    id_to_token = {i: v for v, i in token_to_id.items()}

    # pair data
    pairs = []
    # 如果skip_window=2, js=[-2,-1,1,2]
    js = [i for i in range(-skip_window, skip_window + 1) if i != 0]

    # 遍历corpus中的每个文档
    for c in corpus:
        words = c.split(" ")
        w_idx = [token_to_id[w] for w in words]
        if method == "skip_gram":
            # 不断移动滑动窗口进行采样
            for i in range(len(w_idx)):
                for j in js:
                    if i + j < 0 or i + j >= len(w_idx):
                        continue
                    # 格式(skip_window=2):
                    # (center, context) or (feature, target)
                    # e.g.
                    # [[16 14]
                    #  [16  9]
                    #   ...
                    #  [14 12]]
                    # x = pairs[:, 0], y = pairs[:, 1]
                    pairs.append((w_idx[i], w_idx[i + j]))
        elif method.lower() == "cbow":
            for i in range(skip_window, len(w_idx) - skip_window):
                context = []
                for j in js:
                    context.append(w_idx[i + j])
                # 格式(skip_window=2):
                # (context, center) or (feature, target)
                # e.g.
                # [[16 14 12  3  9]
                #  [14  9  3 14 12]
                #   ...
                #  [ 3 14  3  9  1]]
                # x, y = pairs[:, :-1], pairs[:, -1]
                pairs.append(context + [w_idx[i]])
        else:
            raise ValueError
    pairs = np.array(pairs)
    if method.lower() == "skip_gram":
        x, y = pairs[:, 0], pairs[:, 1]
    elif method.lower() == "cbow":
        x, y = pairs[:, :-1], pairs[:, -1]
    else:
        raise ValueError
    return Dataset(x, y, token_to_id, id_to_token)


class CBOW(nn.Module):

    def __init__(self, vocab_size: int, embed_size: int) -> None:
        """
        参数:
        vocab_size: 词表大小
        embed_size: 词向量的维度
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.embeddings.weight.data.normal_(0, 0.1)

        self.hidden_out = nn.Linear(embed_size, vocab_size)
        self.optimizer = torch.optim.SGD(self.parameters(),
                                         momentum=0.9,
                                         lr=0.01)

    def forward(self,
                x: Tensor,
                training: Any = None,
                mask: Any = None) -> Tensor:
        """
        参数:
        x: [batch_size, num_steps]
        training: 当前版本没用
        mask: 当前版本没用

        返回:
        pred的: [batch_size, vocab_size]
        """
        # o.shape [batch_size, num_steps, embed_size]
        o = self.embeddings(x)
        # o.shape [batch_size, embed_size]
        o = torch.mean(o, dim=1)
        # pred.shape [batch_size, vocab_size]
        pred = self.hidden_out(o)
        return pred

    def loss(self, x: Tensor, y: Tensor, training: Any = None) -> Tensor:
        """
        计算一个批量的loss
        1. 前向传播
        2. 计算loss

        参数:
        x: [batch_size, num_steps]
        y: [batch_size, ]
        training: 当前版本没用

        返回:
        loss: 标量的loss(mean)
        """
        # pred.shape [batch_size, vocab_size]
        pred = self.forward(x, training)
        # 标量
        return nn.functional.cross_entropy(pred, y, reduction='mean')

    def step(self, x: Tensor, y: Tensor) -> np.array:
        """
        训练一个批量
        1. 前向传播计算loss
        2. 后向传播更新梯度

        参数:
        x: [batch_size, num_steps]
        y: [batch_size, ]

        返回:
        loss: 标量的loss
        """

        self.optimizer.zero_grad()
        loss = self.loss(x, y, True)
        loss.backward()
        self.optimizer.step()
        return loss.cpu().detach().numpy()


def try_gpu(i: int = 0) -> torch.device:
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def train_gpu(model: nn.Module,
              data: Dataset,
              batch_size: int = 16,
              num_steps: int = 10000) -> None:
    device = try_gpu()
    model.to(device)
    print('train on device:', device)
    tqdm_iter = tqdm(range(num_steps))
    for t in tqdm_iter:
        bx, by = data.sample(batch_size)
        bx, by = torch.from_numpy(bx).to(device), torch.from_numpy(by).to(
            device)
        loss = model.step(bx, by)
        tqdm_iter.desc = f"step: {t} | loss: {loss:.3f}"


def show_w2v_word_embedding(model: nn.Module, data: Dataset) -> None:
    """
    参数:

    model: 模型
    data: Dataset实例
    """
    word_emb = model.embeddings.weight.data.cpu().numpy()
    # 遍历vocab中所有的单词
    for i in range(data.num_word):
        # 数字用蓝色, 字母用红色
        c = "blue"
        try:
            int(data.id_to_token[i])
        except:
            c = "red"
        # 显示单词
        plt.text(word_emb[i, 0],
                 word_emb[i, 1],
                 s=data.id_to_token[i],
                 color=c,
                 weight="bold")

    plt.xlim(word_emb[:, 0].min() - 1.0, word_emb[:, 0].max() + 1.0)
    plt.ylim(word_emb[:, 1].min() - 1.0, word_emb[:, 1].max() + 1.0)
    plt.xticks(())
    plt.yticks(())
    plt.xlabel("embedding dim1")
    plt.ylabel("embedding dim2")
    plt.show()


def train() -> None:
    ds = process_w2v_data(corpus, skip_window=2, method="cbow")
    net = CBOW(ds.num_word, embed_size=2)
    train_gpu(net, ds)
    show_w2v_word_embedding(net, ds)


if __name__ == '__main__':
    train()