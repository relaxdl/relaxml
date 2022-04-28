import os
import re
from typing import Dict, List, Tuple
import requests
import sys
import hashlib
import zipfile
import tarfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor, optim
from torch.nn.functional import cross_entropy
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
"""
Simple ELMo

实现说明:
https://tech.foxrelax.com/nlp/simple_elmo/
"""
"""
Microsoft Research Paraphrase Corpus(MRPC)数据集

1. 一共5列: ['Quality', '#1 ID', '#2 ID', '#1 String', '#2 String']
2. 每行有两句话'#1 String'和'#2 String', 如果他们是语义相同, Quality为1. 反之为0
3. 这份数据集可以做2件事:
   <1> 两句合起来训练文本匹配
   <2> 两句拆开单独对待, 理解人类语言, 学一个语言模型
"""

PAD_ID = 0


def download(cache_dir: str = '../data') -> str:
    """
    下载数据
    """
    sha1_hash = '21a4d37692645502c0ecf3e25688c8ad305213ef'
    url = 'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/msr_paraphrase.zip'
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
    # e.g. ../data/msr_paraphrase.zip
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
    # e.g. ../data/msr_paraphrase
    return data_dir


def pad_zero(seqs: np.ndarray, max_len: int) -> np.ndarray:
    """
    >>> x = np.array([[2, 4], [3, 4]])
    >>> padded = pad_zero(x, 4)
    >>> assert padded.shape == (2, 4)
    >>> padded
        [[2 4 0 0]
         [3 4 0 0]]

    参数:
    seqs: [batch_size, steps]

    返回:
    padded: [batch_size, max_len]
    """
    padded = np.full((len(seqs), max_len), fill_value=PAD_ID, dtype=np.int32)
    for i, seq in enumerate(seqs):
        padded[i, :len(seq)] = seq
    return padded


def _text_standardize(text: str) -> str:
    """
    处理文本, 把数字替换成<NUM>

    >>> _text_standardize('I Love-you 123 foo456 3456 bar.')
        I Love-you <NUM> foo456 <NUM> bar.
    """
    text = re.sub(r'—', '-', text)
    text = re.sub(r'–', '-', text)
    text = re.sub(r'―', '-', text)
    text = re.sub(r" \d+(,\d+)?(\.\d+)? ", " <NUM> ", text)
    text = re.sub(r" \d+-+?\d*", " <NUM>-", text)
    return text.strip()


def _process_mrpc(
        dir: str,
        rows: int = None) -> Tuple[Dict, Dict[str, int], Dict[int, str]]:
    """
    处理MRPC数据集

    1. 数字会被处理成<NUM>
    2. 包含了4个特殊token: <PAD>, <MASK>, <SEP>, <BOS>

    返回: (data, token_to_id, id_to_token)
    data: {
        'train': {
            's1': List[str]
            's2': List[str]
            's1id': List[List[int]]
            's2id': List[List[int]]
        },
        'test': {
            's1': List[str]
            's2': List[str]
            's1id': List[List[int]]
            's2id': List[List[int]]
        }
    }
    """
    data = {"train": None, "test": None}
    files = ['msr_paraphrase_train.txt', 'msr_paraphrase_test.txt']
    for f in files:
        # 数据一共5列
        # ['Quality', '#1 ID', '#2 ID', '#1 String', '#2 String']
        df = pd.read_csv(os.path.join(dir, f), sep='\t', nrows=rows)
        k = "train" if "train" in f else "test"
        data[k] = {
            "is_same": df.iloc[:, 0].values,
            "s1": df["#1 String"].values,
            "s2": df["#2 String"].values
        }
    vocab = set()
    # 遍历:
    # data['train']['s1']
    # data['train']['s2']
    # data['test']['s1']
    # data['test']['s2']
    for n in ["train", "test"]:
        for m in ["s1", "s2"]:
            for i in range(len(data[n][m])):
                data[n][m][i] = _text_standardize(data[n][m][i].lower())
                cs = data[n][m][i].split(" ")
                vocab.update(set(cs))
    token_to_id = {v: i for i, v in enumerate(sorted(vocab), start=1)}
    token_to_id["<PAD>"] = PAD_ID
    token_to_id["<MASK>"] = len(token_to_id)
    token_to_id["<SEP>"] = len(token_to_id)
    token_to_id["<BOS>"] = len(token_to_id)
    id_to_token = {i: v for v, i in token_to_id.items()}
    for n in ["train", "test"]:
        for m in ["s1", "s2"]:
            data[n][m + "id"] = [[token_to_id[v] for v in c.split(" ")]
                                 for c in data[n][m]]
    return data, token_to_id, id_to_token


class MRPCData(Dataset):
    """
    只用到了训练集的数据

    返回的序列格式:
    <BOS>s1<SEP>s2<SEP><PAD><PAD>...
    """
    num_seg = 3
    pad_id = PAD_ID

    def __init__(self, data_dir: str, rows: int = None) -> None:
        data, self.token_to_id, self.id_to_token = _process_mrpc(
            data_dir, rows)
        # 每个句子都处理成如下格式:
        # <BOS>s1<SEP>s2<SEP><PAD><PAD>...
        # 计算出来是: 72
        self.max_len = max([
            len(s1) + len(s2) + 3
            for s1, s2 in zip(data["train"]["s1id"] +
                              data["test"]["s1id"], data["train"]["s2id"] +
                              data["test"]["s2id"])
        ])
        # xlen List[[s1_len, s2_len]]
        # e.g.
        # [[19 20]
        #  [17 22]
        #  ...
        #  [21 22]
        #  [25 18]]
        self.xlen = np.array(
            [[len(data["train"]["s1id"][i]),
              len(data["train"]["s2id"][i])]
             for i in range(len(data["train"]["s1id"]))],
            dtype=int)
        x = [[self.token_to_id["<BOS>"]] + data["train"]["s1id"][i] +
             [self.token_to_id["<SEP>"]] + data["train"]["s2id"][i] +
             [self.token_to_id["<SEP>"]] for i in range(len(self.xlen))]
        # x List[List[int]]
        # x[0] e.g.
        # [12879   720   336  5432  1723    36 12591  5279  1853   203 11501 12665
        #    203    36  7936  3194  3482  5432  4107    41 12878  9414 11635  5418
        #    965  8022   203 11501 12665   203    36   720   336  5432  1723  7936
        #   3194  3482  5432  4107    41 12878     0     0     0     ...]
        self.x = pad_zero(x, max_len=self.max_len)
        # nsp_y
        # e.g.
        # [[1]
        #  [0]
        #  ...
        #  [0]
        #  [1]]
        self.nsp_y = data["train"]["is_same"][:, None]
        # 编码规则:
        # <BOS>s1<SEP>s2<SEP><PAD><PAD>...
        # <BOS>s1<SEP> - 0
        # s2<SEP> - 1
        # <PAD> - 2
        # seg[0], e.g.
        # [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
        #  1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]
        self.seg = np.full(self.x.shape, self.num_seg - 1, np.int32)
        for i in range(len(x)):
            si = self.xlen[i][0] + 2
            self.seg[i, :si] = 0
            si_ = si + self.xlen[i][1] + 1
            self.seg[i, si:si_] = 1
        # word_ids List[int]
        self.word_ids = np.array(
            list(
                set(self.id_to_token.keys()).difference([
                    self.token_to_id[v] for v in ["<PAD>", "<MASK>", "<SEP>"]
                ])))

    def __getitem__(
            self,
            idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        >>> data_iter, dataset = load_mrpc_data(batch_size=32)
        >>> dataset.max_len
            72
        >>> bx, bs, bl, by = dataset[0]
        >>> assert bx.shape == (dataset.max_len, )
        >>> assert bs.shape == (dataset.max_len, )
        >>> assert bl.shape == (2, )
        >>> assert by.shape == (1, )
        """
        return self.x[idx], self.seg[idx], self.xlen[idx], self.nsp_y[idx]

    def sample(
            self,
            n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        >>> data_iter, dataset = load_mrpc_data(batch_size=32)
        >>> dataset.max_len
            72
        >>> bx, bs, bl, by = dataset.sample(16)
        >>> assert bx.shape == (16, dataset.max_len)
        >>> assert bs.shape == (16, dataset.max_len)
        >>> assert bl.shape == (16, 2)
        >>> assert by.shape == (16, 1)
        """
        bi = np.random.randint(0, self.x.shape[0], size=n)
        bx, bs, bl, by = self.x[bi], self.seg[bi], self.xlen[bi], self.nsp_y[
            bi]
        return bx, bs, bl, by

    @property
    def num_word(self):
        return len(self.token_to_id)

    def __len__(self):
        return len(self.x)

    @property
    def mask_id(self):
        return self.token_to_id["<MASK>"]


class MRPCSingle(Dataset):
    """
    只用到了训练集的数据

    返回的序列格式:
    <BOS>s<SEP><PAD><PAD>...
    """
    pad_id = PAD_ID

    def __init__(self, data_dir: str, rows: int = None) -> None:
        data, self.token_to_id, self.id_to_token = _process_mrpc(
            data_dir, rows)
        # 每个句子都处理成如下格式:
        # <BOS>s<SEP><PAD><PAD>...
        # 计算出来是: 38
        self.max_len = max([
            len(s) + 2 for s in data["train"]["s1id"] + data["train"]["s2id"]
        ])
        x = [[self.token_to_id["<BOS>"]] + data["train"]["s1id"][i] +
             [self.token_to_id["<SEP>"]]
             for i in range(len(data["train"]["s1id"]))]
        x += [[self.token_to_id["<BOS>"]] + data["train"]["s2id"][i] +
              [self.token_to_id["<SEP>"]]
              for i in range(len(data["train"]["s2id"]))]
        # x List[List[int]]
        # x[0] e.g.
        # [12879   720   336  5432  1723    36 12591  5279  1853   203 11501 12665
        #    203    36  7936  3194  3482  5432  4107    41 12878     0     0     0
        #      0     0     0     0     ...]
        self.x = pad_zero(x, max_len=self.max_len)
        # word_ids List[int]
        self.word_ids = np.array(
            list(
                set(self.id_to_token.keys()).difference(
                    [self.token_to_id["<PAD>"]])))

    def sample(self, n: int) -> np.ndarray:
        """
        >>> _, dataset = load_mrpc_single()
        >>> bx = dataset.sample(16)
        >>> dataset.max_len
            38
        >>> assert bx.shape == (16, dataset.max_len)
        """
        bi = np.random.randint(0, self.x.shape[0], size=n)
        bx = self.x[bi]
        return bx

    @property
    def num_word(self) -> int:
        return len(self.token_to_id)

    def __getitem__(self, index: int) -> np.ndarray:
        """
        >>> _, dataset = load_mrpc_single()
        >>> bx = dataset[0]
        >>> dataset.max_len
            38
        >>> assert bx.shape == (dataset.max_len, )
        """
        return self.x[index]

    def __len__(self) -> int:
        return len(self.x)


def load_mrpc_data(batch_size: int = 32,
                   rows: int = 2000,
                   cache_dir: str = '../data') -> Tuple[DataLoader, MRPCData]:
    """
    返回的序列格式:
    <BOS>s1<SEP>s2<SEP><PAD><PAD>...

    >>> data_iter, dataset = load_mrpc_data(batch_size=32)
    >>> dataset.max_len
        72
    >>> for bx, bs, bl, by in data_iter:
    >>>     assert bx.shape == (32, dataset.max_len)
    >>>     assert bs.shape == (32, dataset.max_len)
    >>>     assert bl.shape == (32, 2)
    >>>     assert by.shape == (32, 1)
    >>>     break
    """
    data_dir = download_extract(cache_dir)
    dataset = MRPCData(data_dir, rows)
    data_iter = DataLoader(dataset,
                           batch_size=batch_size,
                           shuffle=True,
                           drop_last=True)
    return data_iter, dataset


def load_mrpc_single(
        batch_size: int = 32,
        rows: int = 2000,
        cache_dir: str = '../data') -> Tuple[DataLoader, MRPCSingle]:
    """
    返回的序列格式:
    <BOS>s<SEP><PAD><PAD>...

    >>> data_iter, dataset = load_mrpc_single(batch_size=32)
    >>> dataset.max_len
        38
    >>> for x in data_iter:
    >>>     assert x.shape == (32, dataset.max_len)
    >>>     break
    """
    data_dir = download_extract(cache_dir)
    dataset = MRPCSingle(data_dir, rows)
    data_iter = DataLoader(dataset,
                           batch_size=batch_size,
                           shuffle=True,
                           drop_last=True)
    return data_iter, dataset


class ELMo(nn.Module):

    def __init__(self, vocab_size: int, num_hiddens: int, n_layers: int,
                 lr: float) -> None:
        """
        参数:
        vocab_size: 词典大小
        num_hiddens: LSTM num_hiddens
        n_layers: LSTM单元数量
        lr: 
        """
        super().__init__()
        self.n_layers = n_layers
        self.num_hiddens = num_hiddens
        self.vocab_size = vocab_size

        # encoder
        self.word_embed = nn.Embedding(num_embeddings=vocab_size,
                                       embedding_dim=num_hiddens,
                                       padding_idx=0)
        self.word_embed.weight.data.normal_(0, 0.1)

        # forward LSTM
        self.fs = nn.ModuleList([
            nn.LSTM(input_size=num_hiddens,
                    hidden_size=num_hiddens,
                    batch_first=True)
            if i == 0 else nn.LSTM(input_size=num_hiddens,
                                   hidden_size=num_hiddens,
                                   batch_first=True) for i in range(n_layers)
        ])
        self.f_logits = nn.Linear(in_features=num_hiddens,
                                  out_features=vocab_size)

        # backward LSTM
        self.bs = nn.ModuleList([
            nn.LSTM(input_size=num_hiddens,
                    hidden_size=num_hiddens,
                    batch_first=True)
            if i == 0 else nn.LSTM(input_size=num_hiddens,
                                   hidden_size=num_hiddens,
                                   batch_first=True) for i in range(n_layers)
        ])
        self.b_logits = nn.Linear(in_features=num_hiddens,
                                  out_features=vocab_size)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, seqs: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        """
        >>> batch_size, num_steps, num_hiddens, n_layers = 2, 10, 256, 2
        >>> model = ELMo(vocab_size=10000,
                         num_hiddens=num_hiddens,
                         n_layers=n_layers,
                         lr=0.1)
        >>> x = torch.ones((batch_size, num_steps), dtype=torch.long)
        >>> fxs, bxs = model(x)
        >>> for i in range(n_layers + 1):
                assert fxs[i].shape == (batch_size, num_steps - 1, num_hiddens)
                assert bxs[i].shape == (batch_size, num_steps - 1, num_hiddens)

        参数:
        seqs: [batch_size, num_steps]

        返回: (fxs, bxs)
        fxs: List of [batch_size, num_steps-1, num_hiddens] 前向传播所有层的输出(元素的数量是n_layers + 1)
        bxs: List of [batch_size, num_steps-1, num_hiddens] 后向传播所有层的输出(元素的数量是n_layers + 1)
        """
        device = next(self.parameters()).device
        # embedded.shape [batch_size, num_steps, num_hiddens]
        embedded = self.word_embed(seqs)
        # LSTM前向和后向传播的输入:
        # fxs List of [batch_size, num_steps-1, num_hiddens]
        # bxs List of [batch_size, num_steps-1, num_hiddens]
        fxs = [embedded[:, :-1, :]]
        bxs = [embedded[:, 1:, :]]
        # LSTM初始化state:
        # h_f.shape [1, batch_size, num_hiddens]
        # c_f.shape [1, batch_size, num_hiddens]
        (h_f, c_f) = (torch.zeros(1, seqs.shape[0],
                                  self.num_hiddens).to(device),
                      torch.zeros(1, seqs.shape[0],
                                  self.num_hiddens).to(device))
        # h_b.shape [1, batch_size, num_hiddens]
        # c_b.shape [1, batch_size, num_hiddens]
        (h_b, c_b) = (torch.zeros(1, seqs.shape[0],
                                  self.num_hiddens).to(device),
                      torch.zeros(1, seqs.shape[0],
                                  self.num_hiddens).to(device))
        for fl, bl in zip(self.fs, self.bs):
            # output_f.shape [batch_size, num_steps-1, num_hiddens]
            # h_f.shape [1, batch_size, num_hiddens]
            # c_f.shape [1, batch_size, num_hiddens]
            output_f, (h_f, c_f) = fl(fxs[-1], (h_f, c_f))
            fxs.append(output_f)

            # output_b.shape [batch_size, num_steps-1, num_hiddens]
            # h_b.shape [1, batch_size, num_hiddens]
            # c_b.shape [1, batch_size, num_hiddens]
            # 输入需要需要做flip之后再输入
            output_b, (h_b, c_b) = bl(torch.flip(bxs[-1], dims=[
                1,
            ]), (h_b, c_b))
            # 由于输入序列做了flip, 所以结果也需要做一次flip
            bxs.append(torch.flip(output_b, dims=(1, )))
        return fxs, bxs

    def step(self, seqs: Tensor) -> Tuple[np.ndarray, Tuple[Tensor, Tensor]]:
        """
        训练一个批量的数据

        参数:
        seqs: [batch_size, num_steps]

        返回: (loss, (fo, bo))
        loss: 标量
        fo: [batch_size, num_steps-1, vocab_size] 前向传播的输出
        bo: [batch_size, num_steps-1, vocab_size] 后向传播的输出
        """
        self.optimizer.zero_grad()
        # List元素的数量是n_layers + 1
        # fo: List of [batch_size, num_steps-1, num_hiddens] 前向传播所有层的输出
        # bo: List of [batch_size, num_steps-1, num_hiddens] 后向传播所有层的输出
        fo, bo = self(seqs)
        # fo.shape [batch_size, num_steps-1, vocab_size]
        # bo.shape [batch_size, num_steps-1, vocab_size]
        fo = self.f_logits(fo[-1])
        bo = self.b_logits(bo[-1])
        fo_loss = cross_entropy(fo.reshape(-1, self.vocab_size),
                                seqs[:, 1:].reshape(-1))
        bo_loss = cross_entropy(bo.reshape(-1, self.vocab_size),
                                seqs[:, :-1].reshape(-1))
        loss = (fo_loss + bo_loss) / 2
        loss.backward()
        self.optimizer.step()
        return loss.cpu().detach().numpy(), (fo, bo)

    def get_embedding(self, seqs: Tensor) -> List[Tensor]:
        """
        获取word embedding

        >>> batch_size, num_steps, num_hiddens, n_layers = 2, 10, 256, 2
        >>> model = ELMo(vocab_size=10000,
                         num_hiddens=num_hiddens,
                         n_layers=n_layers,
                         lr=0.1)
        >>> x = torch.ones((batch_size, num_steps), dtype=torch.long)
        >>> xs = model.get_embedding(x)
        >>> for i in range(n_layers + 1):
        >>> assert xs[i].shape == (batch_size, num_steps - 2, num_hiddens * 2)

        参数:
        seqs: [batch_size, num_steps]

        输出: List元素的数量是n_layers + 1
        xs: List of [batch_size, num_steps-2, num_hiddens*2]
        """
        device = next(self.parameters()).device
        seqs.to(device)
        # List元素的数量是n_layers + 1
        # fxs: List of [batch_size, num_steps-1, num_hiddens] 前向传播所有层的输出
        # bxs: List of [batch_size, num_steps-1, num_hiddens] 后向传播所有层的输出
        fxs, bxs = self(seqs)
        xs = [
            torch.cat((fxs[0][:, 1:, :], bxs[0][:, :-1, :]),
                      dim=2).cpu().data.numpy()
        ] + [
            torch.cat((f[:, 1:, :], b[:, :-1, :]), dim=2).cpu().data.numpy()
            for f, b in zip(fxs[1:], bxs[1:])
        ]
        return xs


def try_gpu(i: int = 0) -> torch.device:
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def train(num_epochs: int = 1,
          batch_size=32,
          num_hiddens=256,
          n_layers=2,
          lr=2e-3,
          device: torch.device = None):
    data_iter, dataset = load_mrpc_single(batch_size=batch_size, rows=2000)
    model = ELMo(vocab_size=dataset.num_word,
                 num_hiddens=num_hiddens,
                 n_layers=n_layers,
                 lr=lr)
    model = model.to(device)
    for epoch in range(num_epochs):
        data_iter_tqdm = tqdm(data_iter, file=sys.stdout)
        for i, batch in enumerate(data_iter_tqdm):
            batch = batch.type(torch.LongTensor).to(device)
            # assert batch.shape == (batch_size, 38)
            loss, (fo, bo) = model.step(batch)
            if i % 20 == 0:
                fp = fo[0].cpu().data.numpy().argmax(axis=1)
                bp = bo[0].cpu().data.numpy().argmax(axis=1)
                tgt = " ".join([
                    dataset.id_to_token[i]
                    for i in batch[0].cpu().data.numpy() if i != dataset.pad_id
                ])
                f_prd = " ".join([
                    dataset.id_to_token[i] for i in fp if i != dataset.pad_id
                ])
                b_prd = " ".join([
                    dataset.id_to_token[i] for i in bp if i != dataset.pad_id
                ])
                data_iter_tqdm.desc = f'epoch {epoch}, step {i}, loss {loss:.3f}'
                print(
                    f'\n | tgt: {tgt}, \n | f_prd {f_prd}, \n | b_prd {b_prd}')
    return model


if __name__ == "__main__":
    device = try_gpu()
    train(device=device)