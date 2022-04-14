import os
import requests
import hashlib
from shutil import copy, rmtree
import random
import zipfile
import tarfile
from typing import Dict, Tuple
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


def download(cache_dir: str = '../data') -> str:
    """
    下载数据
    """
    sha1_hash = '26e0a7488bac8eec33de22a5e8b569cb050e1c2e'
    url = 'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/flower_photos.zip'
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
    # e.g. ../data/flower_photos.zip
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
    # e.g. ../data/flower_photos
    return data_dir


def mk_file(file_path: str) -> None:
    if os.path.exists(file_path):
        # 如果文件夹存在，则先删除原文件夹在重新创建
        rmtree(file_path)
    os.makedirs(file_path)


def process_data(data_path: str, val_rate: float = 0.1) -> Tuple[str, str]:
    """
    data_path=../data/flower_photos
    ../data/flower_photos/ (3670个样本)
                daisy/
                dandelion/
                roses/
                sunflowers/
                tulips/
    
    生成的训练集: 3306个样本
    ../data/train/
                daisy/
                dandelion/
                roses/
                sunflowers/
                tulips/
    
    生成的验证集: 364个样本
    ../data/val/
                daisy/
                dandelion/
                roses/
                sunflowers/
                tulips/
    """
    # ['roses', 'sunflowers', 'daisy', 'dandelion', 'tulips']
    flower_class = [
        cla for cla in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, cla))
    ]
    root_path = os.path.dirname(data_path)
    # 建立保存训练集的文件夹
    train_root = os.path.join(root_path, "train")
    mk_file(train_root)
    for cla in flower_class:
        # 建立每个类别对应的文件夹
        mk_file(os.path.join(train_root, cla))

    # 建立保存验证集的文件夹
    val_root = os.path.join(root_path, "val")
    mk_file(val_root)
    for cla in flower_class:
        # 建立每个类别对应的文件夹
        mk_file(os.path.join(val_root, cla))

    # 遍历所有的类
    for cla in flower_class:
        cla_path = os.path.join(data_path, cla)
        images = os.listdir(cla_path)
        num = len(images)
        # 随机采样验证集的索引
        eval_index = random.sample(images, k=int(num * val_rate))
        for index, image in enumerate(images):
            if image in eval_index:
                # 将分配至验证集中的文件复制到相应目录
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(val_root, cla)
                copy(image_path, new_path)
            else:
                # 将分配至训练集中的文件复制到相应目录
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(train_root, cla)
                copy(image_path, new_path)
            print("\r[{}] processing [{}/{}]".format(cla, index + 1, num),
                  end="")  # processing bar
        print()

    print("processing done!")
    return train_root, val_root


def load_data_flower(
    batch_size: int,
    resize: int = 224,
    root: str = '../data'
) -> Tuple[DataLoader, DataLoader, Dict[str, int], Dict[int, str]]:
    """
    加载Flower数据集

    1. 一共5个类别
    2. 训练集: 3306 images
    3. 验证集: 364 images
    4. 图片尺寸: 3x224x224 [默认处理成这个尺寸]

    >>> train_iter, val_iter, class_to_idx, idx_to_class = 
            load_data_flower(32, root='../data')
    >>> for X, y in val_iter:
    >>>     assert X.shape == (32, 3, 224, 224)
    >>>     assert y.shape == (32, )
    >>>     break
    >>> print(class_to_idx)
        {'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}
    >>> print(idx_to_class)
        {0: 'daisy', 1: 'dandelion', 2: 'roses', 3: 'sunflowers', 4: 'tulips'}
    """
    data_dir = download_extract(root)
    train_root, val_root = process_data(data_dir)

    data_transform = {
        "train":
        transforms.Compose([
            transforms.RandomResizedCrop(resize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val":
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    train_dataset = datasets.ImageFolder(train_root,
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    class_to_idx = train_dataset.class_to_idx
    idx_to_class = dict((val, key) for key, val in class_to_idx.items())

    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    validate_dataset = datasets.ImageFolder(val_root,
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    val_iter = DataLoader(validate_dataset,
                          batch_size=batch_size,
                          shuffle=False)

    print("using {} images for training, {} images for validation.".format(
        train_num, val_num))
    return train_iter, val_iter, class_to_idx, idx_to_class


if __name__ == '__main__':
    train_iter, val_iter, class_to_idx, idx_to_class = load_data_flower(
        32, root='../data')
    for X, y in val_iter:
        assert X.shape == (32, 3, 224, 224)
        assert y.shape == (32, )
        break
    print(class_to_idx)
# 输出:
# [roses] processing [641/641]
# [sunflowers] processing [699/699]
# [daisy] processing [633/633]
# [dandelion] processing [898/898]
# [tulips] processing [799/799]
# processing done!
# using 3306 images for training, 364 images for validation.
# {'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}