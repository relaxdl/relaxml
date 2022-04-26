from typing import Dict, List
import numpy as np
import itertools
import torch
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
        x: [num_examples, num_steps] Skip-Gram
           [num_examples, ] CBOW
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
        bx: [n, num_steps] Skip-Gram
            [n, ] CBOW
        by: [n, ]
        """
        b_idx = np.random.randint(0, len(self.x), n)
        bx, by = self.x[b_idx], self.y[b_idx]
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
    >>> assert bx.shape == (batch_size, )
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


if __name__ == '__main__':
    batch_size = 6
    ds = process_w2v_data(corpus, 2, 'cbow')
    bx, by = ds.sample(batch_size)
    bx, by = torch.from_numpy(bx), torch.from_numpy(by)
    assert bx.shape == (batch_size, 4)
    assert by.shape == (batch_size, )

    ds = process_w2v_data(corpus, 2, 'skip_gram')
    bx, by = ds.sample(batch_size)
    bx, by = torch.from_numpy(bx), torch.from_numpy(by)
    assert bx.shape == (batch_size, )
    assert by.shape == (batch_size, )
