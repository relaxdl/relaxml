import os
from typing import List, Tuple
import requests
import hashlib
import zipfile
import tarfile
import torch
from torch import Tensor

_DATA_HUB = dict()
_DATA_HUB['glove.6B.50d'] = (
    '0b8703943ccdb6eb788e6f091b8946e82231bc4d',
    'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/glove.6B.50d.zip')
_DATA_HUB['glove.6B.100d'] = (
    'cd43bfb07e44e6f27cbcc7bc9ae3d80284fdaf5a',
    'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/glove.6B.100d.zip')


def download(name: str, cache_dir: str = '../data') -> str:
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


def download_extract(name: str, cache_dir: str = '../data') -> str:
    """
    下载数据 & 解压
    """
    # 下载数据集
    fname = download(name, cache_dir)

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
        # idx_to_vec.shape [num_tokens, embed_size]
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
        idx_to_vec [num_tokens, embed_size]
        """
        # idx_to_token: list of token
        # idx_to_vec: list of vector, 每个vector是一个float list
        idx_to_token, idx_to_vec = ['<unk>'], []
        data_dir = download_extract(embedding_name)
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
