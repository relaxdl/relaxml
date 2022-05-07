from matplotlib import pyplot as plt
import cv2
import requests
import os
import glob
import random
import hashlib
from shutil import copy, rmtree
import zipfile
import tarfile
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
"""
卷积神经网络可视化

说明:
https://tech.foxrelax.com/generative/cnn_visualizing/
"""
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
    >>> train_iter, val_iter, train_root, val_root = load_data_dogs_and_cats()
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
    return train_iter, val_iter, train_root, val_root


def download_elephant(cache_dir='../data'):
    sha1_hash = '9225c1b5be359a4fe7617d5f9cc6e6a28155b624'
    url = 'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/img/elephant.jpg'
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
    # e.g. ../data/elephant.jpg
    return fname


def model():
    """
    >>> x = tf.random.normal((32, 150, 150, 3))
    >>> net = model()
    >>> assert net(x).shape == (32, 1)
    """

    # input [batch_size, 150, 150, 3]
    net = models.Sequential()

    # output [batch_size, 148, 148, 32]
    net.add(
        layers.Conv2D(32, (3, 3), activation='relu',
                      input_shape=(150, 150, 3)))
    # output [batch_size, 74, 74, 32]
    net.add(layers.MaxPooling2D((2, 2)))
    # output [batch_size, 72, 72, 64]
    net.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # output [batch_size, 36, 36, 64]
    net.add(layers.MaxPooling2D((2, 2)))
    # output [batch_size, 34, 34, 128]
    net.add(layers.Conv2D(128, (3, 3), activation='relu'))
    # output [batch_size, 17, 17, 128]
    net.add(layers.MaxPooling2D((2, 2)))
    # output [batch_size, 15, 15, 128]
    net.add(layers.Conv2D(128, (3, 3), activation='relu'))
    # output [batch_size, 7, 7, 128]
    net.add(layers.MaxPooling2D((2, 2)))
    # output [batch_size, 7*7*128=6272]
    net.add(layers.Flatten())
    # output [batch_size, 6272]
    net.add(layers.Dropout(0.5))
    # output [batch_size, 512]
    net.add(layers.Dense(512, activation='relu'))
    # output [batch_size, 1]
    net.add(layers.Dense(1, activation='sigmoid'))
    return net


def activation_model(net):
    """
    获取`net`的中间层输出

    >>> x = tf.random.normal((32, 150, 150, 3))
    >>> net = model()
    >>> activation_net = activation_model(net)
    >>> outputs = activation_net(x)
    >>> for i, o in enumerate(outputs):
    >>>     print(f'[{i}] {o.shape}')
        [0] (32, 148, 148, 32)
        [1] (32, 74, 74, 32)
        [2] (32, 72, 72, 64)
        [3] (32, 36, 36, 64)
        [4] (32, 34, 34, 128)
        [5] (32, 17, 17, 128)
        [6] (32, 15, 15, 128)
        [7] (32, 7, 7, 128)
    """
    # 用一个输入张量和一个输出张量列表将模型实例化
    layer_outputs = [layer.output for layer in net.layers[:8]]  # 提取前8层的输出
    return models.Model(inputs=net.input, outputs=layer_outputs)


def get_cat(train_dir, i=1):
    """
    从训练集获取一张图片

    返回:
    img_tensor [1, 150, 150, 3]
    """
    train_cats_dir = os.path.join(train_dir, 'cat')
    fnames = [
        os.path.join(train_cats_dir, fname)
        for fname in os.listdir(train_cats_dir)
    ]

    img_path = fnames[i]
    img = image.load_img(img_path, target_size=(150, 150))  # PIL format
    img_tensor = image.img_to_array(img)  # PIL Image ->  Numpy array
    # img_tensor.shape [1, 150, 150, 3]
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    return img_tensor


def train():
    num_epochs, batch_size = 30, 32
    net = model()
    train_iter, val_iter, train_root, val_root = load_data_dogs_and_cats(
        batch_size=batch_size)
    net.compile(loss='binary_crossentropy',
                optimizer=optimizers.RMSprop(learning_rate=1e-4),
                metrics=['acc'])
    net.fit(train_iter,
            steps_per_epoch=100,
            epochs=num_epochs,
            validation_data=val_iter,
            validation_steps=50)
    return net, train_root, val_root


def show_activations(net, img_tensor):
    """
    可视化中间激活

    参数:
    net: 模型
    img_tensor: [1, 150, 150, 3]
    """
    activation_net = activation_model(net)
    # activations:
    # activations[0] [1, 148, 148, 32]
    # activations[1] [1, 74, 74, 32]
    # activations[2] [1, 72, 72, 64]
    # activations[3] [1, 36, 36, 64]
    # activations[4] [1, 34, 34, 128]
    # activations[5] [1, 17, 17, 128]
    # activations[6] [1, 15, 15, 128]
    # activations[7] [1, 7, 7, 128]
    activations = activation_net.predict(img_tensor)

    # 前8层的名称, 这样就可以将这些名称画到图中
    layer_names = []
    for layer in net.layers[:8]:
        layer_names.append(layer.name)

    images_per_row = 16
    for layer_name, layer_activation in zip(layer_names, activations):
        # layer_activation.shape [1, H, W, C]
        n_features = layer_activation.shape[-1]  # 特征图中的特征个数, 也就是通道数
        size = layer_activation.shape[1]
        n_cols = n_features // images_per_row

        # 将一层的所有特征图平铺到这个矩阵中, 一次性显示出来, 效率更高
        display_grid = np.zeros((size * n_cols, size * images_per_row))
        for col in range(n_cols):
            for row in range(images_per_row):
                # 对特征进行处理, 使其看起来更美观
                channel_image = layer_activation[0, :, :,
                                                 col * images_per_row + row]
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size:(col + 1) * size,
                             row * size:(row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()


def deprocess_image(x):
    """
    将形状为[1,150,150,3]的浮点型张量, 转换为取值在[0,255]区间内的整型张量,
    方便显示

    >>> x = np.random.random((1, 150, 150, 3))
    >>> assert deprocess_image(x).shape == (1, 150, 150, 3)

    输入:
    x: [batch_size, height, width, 3]

    输出:
    img: [batch_size, height, width, 3]
    """
    # 对张量标准化, 使其均值为0, 标准差为0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # 将x裁剪到[0,1]区间
    x += 0.5
    x = np.clip(x, 0, 1)

    # 将x转换为RGB数组
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def vgg16(include_top=False):
    """
    input_30 (InputLayer)       (None, 224, 224, 3)     
    block1_conv1 (Conv2D)       (None, 224, 224, 64)
    block1_conv2 (Conv2D)       (None, 224, 224, 64)
    block1_pool (MaxPooling2D)  (None, 112, 112, 64)    
    block2_conv1 (Conv2D)       (None, 112, 112, 128) 
    block2_conv2 (Conv2D)       (None, 112, 112, 128)
    block2_pool (MaxPooling2D)  (None, 56, 56, 128)     
    block3_conv1 (Conv2D)       (None, 56, 56, 256)  
    block3_conv2 (Conv2D)       (None, 56, 56, 256)  
    block3_conv3 (Conv2D)       (None, 56, 56, 256) 
    block3_pool (MaxPooling2D)  (None, 28, 28, 256)    
    block4_conv1 (Conv2D)       (None, 28, 28, 512)  
    block4_conv2 (Conv2D)       (None, 28, 28, 512)
    block4_conv3 (Conv2D)       (None, 28, 28, 512) 
    block4_pool (MaxPooling2D)  (None, 14, 14, 512)      
    block5_conv1 (Conv2D)       (None, 14, 14, 512) 
    block5_conv2 (Conv2D)       (None, 14, 14, 512) 
    block5_conv3 (Conv2D)       (None, 14, 14, 512) 
    block5_pool (MaxPooling2D)  (None, 7, 7, 512)        
    flatten (Flatten)           (None, 25088)      
    fc1 (Dense)                 (None, 4096)
    fc2 (Dense)                 (None, 4096)
    predictions (Dense)         (None, 1000)
    """
    return VGG16(weights='imagenet', include_top=include_top)


def generate_pattern(net, layer_name, filter_index, size=150):
    """
    输入VGG16一个层的名字和一个filter索引, 将返回一个图像张量, 
    返回的图像能够使得对应的filter最大化其激活

    >>> net = vgg16()
    >>> plt.imshow(generate_pattern(net, 'block3_conv1', 0))
    >>> plt.imshow()

    参数:
    net: VGG16
    layer_name: 显示的层的名字
    filter_index: 显示的filter的索引
    """
    layer_output = net.get_layer(layer_name).output
    net_layer = keras.Model(net.input, layer_output)
    # 随机生成一张有噪音的灰度图
    # input_img_data [1, size, size, 3]
    input_img_data = tf.random.normal((1, size, size, 3)) * 20 + 128
    lr = 1.
    # 运行40次梯度上升
    for i in range(40):
        with tf.GradientTape() as tape:
            tape.watch(input_img_data)
            layer_output = net_layer(input_img_data)
            loss = tf.reduce_mean(layer_output[:, :, :, filter_index])
        # 计算loss相对于输入图像的梯度
        [grads] = tape.gradient(loss, [input_img_data])
        # 标准化技巧: 将梯度标准化
        grads /= (tf.sqrt(tf.reduce_mean(tf.square(grads))) + 1e-5)
        input_img_data += grads * lr
    img = input_img_data.numpy()[0]
    return deprocess_image(img)


def show_filters(net, layer_name):
    """
    显示VGG16某一层的前8x8=64个filter

    >>> show_layer(net, 'block1_conv1')
    """
    size = 64
    margin = 5
    # 空图像(全黑色), 用于保存结果
    results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))
    for i in range(8):
        for j in range(8):
            # 生成layer_name层第: i+(j*8)个过滤器的模式
            filter_img = generate_pattern(net,
                                          layer_name,
                                          i + (j * 8),
                                          size=size)
            horizontal_start = i * size + i * margin
            horizontal_end = horizontal_start + size
            vertical_start = j * size + j * margin
            vertical_end = vertical_start + size
            # 将结果放在results网格第(i,j)个方块中
            results[horizontal_start:horizontal_end,
                    vertical_start:vertical_end, :] = filter_img

    # 显示results网格
    plt.figure(figsize=(10, 10))
    plt.imshow(results.astype('uint8'))
    plt.show()


def vgg16_elephant(net):
    """
    >>> net = vgg16(include_top=True)
    >>> vgg16_elephant(net)

    返回:
    x.shape [1, 224, 224, 3]
    """

    img_path = download_elephant()
    # 大小为224x224的PIL图像
    img = image.load_img(img_path, target_size=(224, 224))
    plt.imshow(img)
    plt.show()
    # x.shape [224, 224, 3]
    x = image.img_to_array(img)  # float32格式的numpy数组
    # x.shape [1, 224, 224, 3]
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)  # 对批量进行预处理(按照通道进行颜色标准化)
    preds = net.predict(x)
    # 预测向量中被最大激活的元素是对应非洲象类别的元素, 索引编号为386
    assert np.argmax(preds[0]) == 386
    print(decode_predictions(preds, top=3)[0])
    return x


def vgg16_elephant_heatmap(net, x):
    """
    >>> net = vgg16(include_top=True)
    >>> x = vgg16_elephant(net)
    >>> vgg16_elephant_heatmap(net, x)

    参数:
    net: VGG16
    x: [1, 224, 224, 3]

    返回:
    heatmap: [14, 14]
    """
    x = tf.Variable(x)
    # 预测向量中"非洲象"元素 标量
    african_elephant_output = net.output[:, 386]
    # block5_conv3是VGG16的最后一个卷积层
    # 输出的feature map形状是: [batch_size, 14, 14, 512]
    last_conv_layer = net.get_layer('block5_conv3')
    elephant_net = keras.Model(
        net.input, [african_elephant_output, last_conv_layer.output])

    with tf.GradientTape() as tape:
        # loss: 预测向量中"非洲象"元素 标量
        # layer_output.shape [1, 14, 14, 512]
        loss, layer_output = elephant_net(x)

    # "非洲象"类别相对于block5_conv3输出feature map的梯度
    # grads.shape [batch_size, 14, 14, 512]
    [grads] = tape.gradient(loss, [layer_output])

    # 每个元素是特定特征图通道的梯度平均值, 这是是512, 也就是对应512个通道
    # pooled_grads.shape [512, ]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    layer_output = layer_output.numpy()
    pooled_grads = pooled_grads.numpy()
    # 将特征图数组中的每个通道乘以"这个通道"对"大象类别"的重要程度
    for i in range(512):
        layer_output[0][:, :, i] *= pooled_grads[i]
    # heatmap.shape [14, 14]
    heatmap = np.mean(layer_output[0], axis=-1)

    # 为了便于可视化, 将热力图标准化到[0,1]范围内
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    plt.matshow(heatmap)
    plt.show()
    return heatmap


def vgg16_elephant_heatmap_cv2(heatmap):
    """
    参数:
    heatmap: [14, 14]
    """
    # 用cv2加载原始图像
    img = cv2.imread(download_elephant())
    # 热力图的大小调整与原始图像相同
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    # 将热力图转换为RGB格式
    heatmap = np.uint8(255 * heatmap)
    # 将热烈图应用于原始图像
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # 这里的0.4是热力图强度因子
    superimposed_img = heatmap * 0.4 + img
    plt.imshow(superimposed_img.astype('uint8'))
    plt.show()


if __name__ == '__main__':
    # 可视化中间激活
    net, train_root, val_root = train()
    img_tensor = get_cat(train_root, i=1)
    show_activations(net, img_tensor)

    # 可视化卷积神经网络的过滤器
    net = vgg16()
    show_filters(net, 'block1_conv1')
    show_filters(net, 'block2_conv1')
    show_filters(net, 'block3_conv1')
    show_filters(net, 'block4_conv1')

    # 可视化类激活的热力图
    net = vgg16(include_top=True)
    x = vgg16_elephant(net)
    heatmap = vgg16_elephant_heatmap(net, x)
    vgg16_elephant_heatmap_cv2(heatmap)