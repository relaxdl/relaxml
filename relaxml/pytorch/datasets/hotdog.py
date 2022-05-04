import os
from typing import Tuple
import requests
import hashlib
import zipfile
import tarfile
import torchvision
from torch.utils.data import DataLoader


def download(cache_dir: str = '../data') -> str:
    """
    下载数据
    """
    sha1_hash = 'fba480ffa8aa7e0febbb511d181409f899b9baa5'
    url = 'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/hotdog.zip'
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
    # e.g. ../data/hotdog.zip
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
    # e.g. ../data/hotdog
    return data_dir


def load_data_hotdog(
        batch_size: int = 32,
        cache_dir: str = '../data') -> Tuple[DataLoader, DataLoader]:
    """
    hotdog/
        train/
            hotdog/
            not-hotdog/
        test/
            hotdog/
            not-hotdog/

    >>> train_iter, test_iter = load_data_hotdog(batch_size=32)
    >>> for x, y in train_iter:
    >>>     assert x.shape == (32, 3, 224, 224)
    >>>     assert y.shape == (32, )
    >>>     break
    """
    data_dir = download_extract(cache_dir)

    normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])

    train_augs = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(), normalize
    ])

    test_augs = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(), normalize
    ])

    train_iter = DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=train_augs),
                            batch_size=batch_size,
                            shuffle=True)
    test_iter = DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'test'), transform=test_augs),
                           batch_size=batch_size)
    return train_iter, test_iter


if __name__ == '__main__':
    train_iter, test_iter = load_data_hotdog(batch_size=32)
    for x, y in train_iter:
        assert x.shape == (32, 3, 224, 224)
        assert y.shape == (32, )
        break
