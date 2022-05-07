import scipy
import imageio
import os
import hashlib
import requests
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import tensorflow.keras as keras
from tensorflow.keras.applications import inception_v3


def download(cache_dir='../data'):
    """
    下载数据
    """
    sha1_hash = 'b0193ffd8ecea1631ac1092130017b884faafa30'
    url = 'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/img/rainier.jpg'
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
    # e.g. ../data/rainier.png
    return fname


def resize_img(img, size):
    """
    参数:
    img: [batch_size, height, width, channels]
    size: (new_height, new_width)

    返回:
    img: [batch_size, new_height, new_width, channels]
    """
    img = np.copy(img)
    factors = (1, float(size[0]) / img.shape[1], float(size[1]) / img.shape[2],
               1)
    return scipy.ndimage.zoom(img, factors, order=1)


def preprocess_image(image_path):
    """
    通用函数, 打开图像, 将图片转换为InceptionV3模型能够处理的张量

    返回:
    img: [batch_size, height, width, channels]
    """
    img = image.load_img(image_path)
    # img.shape [height, width, channels]
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img


def save_img(img, fname, cache_dir='../data'):
    """
    保存图片

    参数:
    img: [batch_size=1, H, W, channels=3]
    """
    pil_img = deprocess_image(np.copy(img))
    fname = os.path.join(cache_dir, fname)
    print(f'save {fname}')
    imageio.imwrite(fname, pil_img)


def deprocess_image(x):
    """
    对inception_v3.preprocess_input的预处理做反向操作

    参数:
    x: [batch_size=1, H, W, channels=3]

    返回:
    x: [H, W, 3]
    """
    # x.shape [H, W, 3]
    x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2
    x += 0.5
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def inceptionv3():
    # 使用预训练的不包括全连接层的InceptionV3网络
    net = inception_v3.InceptionV3(weights='imagenet', include_top=False)
    net.trainable = False
    assert len(net.trainable_weights) == 0
    return net


# 这个系数表示该层对要最大化的损失的贡献度有多少,
# 层的名字硬编码在InceptionV3网络中
layer_names = ['mixed2', 'mixed3', 'mixed4', 'mixed5']
layer_contributions = {
    'mixed2': 0.2,
    'mixed3': 3.0,
    'mixed4': 2.0,
    'mixed5': 1.5,
}


def layer_outputs(net):
    """
    获取InceptionV3中间层的输出

    >>> x = tf.random.normal((32, 224, 224, 3))
    >>> net_outputs = layer_outputs(net)
    >>> for o in net_outputs(x):
    >>>     print(o.shape)
        (32, 25, 25, 288)
        (32, 12, 12, 768)
        (32, 12, 12, 768)
        (32, 12, 12, 768)
    """
    layer_dict = dict([(layer.name, layer) for layer in net.layers])
    activations = []
    for layer_name in layer_names:
        activations.append(layer_dict[layer_name].output)
    model = keras.Model(net.input, activations)
    return model


def calc_loss(activations):
    """
    参数:
    activations: list of [batch_size, H, W, C] - 每一层的H/W/C是不同的
    """
    losses = []
    for layer_name, activation in zip(layer_names, activations):
        scaling = tf.reduce_prod(tf.cast(tf.shape(activation), 'float32'))
        coeff = layer_contributions[layer_name]  # 注意: 不做类型转换会切断梯度传播
        # 将该层特征的L2范数添加到loss中, 为了避免边界伪影, 损失中仅包含非边界像素
        losses.append(coeff *
                      tf.reduce_sum(tf.square(activation[:, 2:-1, 2:-2, :])) /
                      scaling)
    return tf.reduce_sum(losses)


def gradient_ascent(net_outputs, img, iterations, step, max_loss=None):
    """
    做梯度上升更新图像

    参数:
    net_outputs: 获取InceptionV3中间层的输出
    img: [batch_size=1, H, W, C=3]
    iterations: 运行梯度上升的步数
    step: 梯度上升的步长(学习率)
    max_loss: 如果损失大于max_loss, 我们要中断梯度上升过程, 以避免得到丑陋的伪影

    返回:
    img: [batch_size=1, H, W, C=3]
    """
    x = tf.Variable(img)
    for i in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(x)
            # 前向传播获得中间输出
            activations = net_outputs(x)
            # 计算loss
            loss = calc_loss(activations)
        # 计算损失相对于图像的梯度
        # grads.shape [batch_size=1, H, W, C=3]
        grads = tape.gradient(loss, x)
        # 梯度标准化
        grads /= tf.maximum(tf.reduce_mean(tf.abs(grads)), 1e-7)
        if max_loss is not None and loss > max_loss:
            break
        print(f'Loss value at {i}: {loss}')
        # 更新img
        x.assign_add(step * grads)
    return x.numpy()


def train(cache_dir='../data'):
    step = 0.01  # 梯度上升的步长(学习率)
    num_octave = 3  # 运行梯度上升的尺度个数
    octave_scale = 1.4  # 两个尺度之间的比例
    iterations = 20  # 每个尺度上运行梯度上升的步数
    max_loss = 10.  # 如果损失大于10, 我们要中断梯度上升过程, 以避免得到丑陋的伪影
    base_image_path = download(cache_dir)  # 需要修改的图像
    net = inceptionv3()
    net_outputs = layer_outputs(net)
    # 将基础图像加载成numpy ndarray
    # img.shape [batch_size=1, height=1365, width=2048, channels=3]
    img = preprocess_image(base_image_path)
    original_shape = img.shape[1:3]  # [height=1365, width=2048]
    print(original_shape)
    successive_shapes = [original_shape]
    for i in range(1, num_octave):
        shape = tuple([(dim / (octave_scale**i)) for dim in original_shape])
        successive_shapes.append(shape)
    # [(696.43, 1044.90),
    #  (975.00, 1462.86),
    #  (1365, 2048)]
    successive_shapes = successive_shapes[::-1]  # 将列表反转, 变为升序
    # original_img.shape [batch_size=1, height=1365, width=2048, channels=3]
    original_img = np.copy(img)
    # successive_shapes[0] = (696.43, 1044.90)
    # shrunk_original_img.shape [1, 696, 1045, 3]
    shrunk_original_img = resize_img(img, successive_shapes[0])  # 将图像大小缩放到最小尺寸
    for shape in successive_shapes:
        print('Processing iamge shape:', shape)
        # 将梦境图像放大
        # img.shape [batch_size=1, H, W, channels=3]
        img = resize_img(img, shape)
        # 梯度上升, 改变梦境图像`img`
        img = gradient_ascent(net_outputs, img, iterations, step, max_loss)
        # 将原始图像的较小版本放大, 它会变得像素化
        # unscaled_shrunk_original_img.shape [batch_size=1, H, W, channels=3]
        unscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
        # 在这个尺寸上计算原始图像的高质量版本
        # same_size_original.shape [batch_size=1, H, W, channels=3]
        same_size_original = resize_img(original_img, shape)
        lost_detail = same_size_original - unscaled_shrunk_original_img
        # 将丢失的细节重新注入梦境图像
        img += lost_detail
        # shrunk_original_img.shape [batch_size=1, H, W, channels=3]
        shrunk_original_img = resize_img(original_img, shape)
        save_img(img, f'dream_at_scale_{img.shape[1]}_{img.shape[2]}.png',
                 cache_dir)
    save_img(img, 'final_dream.png', cache_dir)


if __name__ == '__main__':
    train()