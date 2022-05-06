import requests
import os
import glob
import random
import hashlib
from shutil import copy, rmtree
import zipfile
import tarfile
from tensorflow.keras.preprocessing import image
"""
这是Kaggle上2013年的一个竞赛问题用到的数据集, 这个数据集包含25000张猫狗图像(每个类别大约12500张), 
大小为543MB(压缩后)

数据在磁盘上的格式:
../data/kaggle_cats_and_dogs/
        cat.*.jpg - 一共12500张
        dog.*.jpg - 一共12500张
"""


def download(cache_dir='../data'):
    """
    下载数据
    """
    sha1_hash = 'e993868e26c86dbd6c5ca257778097ce39b36f4e'
    url = 'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/kaggle_cats_and_dogs.zip'
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
    # e.g. ../data/kaggle_cats_and_dogs.zip
    return fname


def download_extract(cache_dir='../data'):
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
    # e.g. ../data/kaggle_cats_and_dogs
    return data_dir


def mk_file(file_path) -> None:
    if os.path.exists(file_path):
        # 如果文件夹存在，则先删除原文件夹在重新创建
        rmtree(file_path)
    os.makedirs(file_path)


def process_data(data_path, val_rate=0.1):
    """
    data_path=../data/kaggle_cats_and_dogs
    ../data/kaggle_cats_and_dogs/ (25000个样本)
            cat.*.jpg - 一共12500张
            dog.*.jpg - 一共12500张
    
    生成的训练集: 22500个样本
    ../data/train/
            cat/
            dog/
    
    生成的验证集: 2500个样本
    ../data/val/
            cat/
            dog/
    """
    # ['cat', 'dog']
    all_class = ['cat', 'dog']
    root_path = os.path.dirname(data_path)
    # 建立保存训练集的文件夹
    train_root = os.path.join(root_path, "train")
    mk_file(train_root)
    for cla in all_class:
        # 建立每个类别对应的文件夹
        mk_file(os.path.join(train_root, cla))

    # 建立保存验证集的文件夹
    val_root = os.path.join(root_path, "val")
    mk_file(val_root)
    for cla in all_class:
        # 建立每个类别对应的文件夹
        mk_file(os.path.join(val_root, cla))

    # 遍历所有的类
    cat_images = glob.glob(os.path.join(data_path, 'cat.*.jpg'))
    dog_images = glob.glob(os.path.join(data_path, 'dog.*.jpg'))
    for images, cla in [(cat_images, 'cat'), (dog_images, 'dog')]:
        num = len(images)
        # 随机采样验证集的索引
        eval_index = random.sample(images, k=int(num * val_rate))
        for index, image in enumerate(images):
            if image in eval_index:
                # 将分配至验证集中的文件复制到相应目录
                new_path = os.path.join(val_root, cla)
                copy(image, new_path)
            else:
                # 将分配至训练集中的文件复制到相应目录
                new_path = os.path.join(train_root, cla)
                copy(image, new_path)
            print("\r[{}] processing [{}/{}]".format(cla, index + 1, num),
                  end="")  # processing bar
        print()

    print("processing done!")
    return train_root, val_root


def load_data_dogs_and_cats(batch_size=32,
                            target_size=(150, 150),
                            val_rate=0.1,
                            root='../data'):
    """
    >>> train_iter, val_iter = load_data_dogs_and_cats()
    >>> for x, y in train_iter:
    >>>     assert x.shape == (32, 150, 150, 3)
    >>>     assert y.shape == (32, )
    >>>     break
        [cat] processing [12500/12500]
        [dog] processing [12500/12500]
        processing done!
        Found 22500 images belonging to 2 classes.
        Found 2500 images belonging to 2 classes.
    """
    data_dir = download_extract(root)
    train_root, val_root = process_data(data_dir, val_rate)
    train_datagen = image.ImageDataGenerator(rescale=1. / 255,
                                             rotation_range=40,
                                             width_shift_range=0.2,
                                             height_shift_range=0.2,
                                             shear_range=0.2,
                                             zoom_range=0.2,
                                             horizontal_flip=True)
    val_datagen = image.ImageDataGenerator(rescale=1. / 255)

    train_iter = train_datagen.flow_from_directory(train_root,
                                                   target_size=target_size,
                                                   batch_size=batch_size,
                                                   class_mode='binary')
    val_iter = val_datagen.flow_from_directory(val_root,
                                               target_size=target_size,
                                               batch_size=batch_size,
                                               class_mode='binary')
    return train_iter, val_iter


if __name__ == '__main__':
    train_iter, val_iter = load_data_dogs_and_cats()
    for x, y in train_iter:
        assert x.shape == (32, 150, 150, 3)
        assert y.shape == (32, )
        break
    # [cat] processing [12500/12500]
    # [dog] processing [12500/12500]
    # processing done!
    # Found 22500 images belonging to 2 classes.
    # Found 2500 images belonging to 2 classes.