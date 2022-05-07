import glob
import imageio
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

"""
GAN(生成 Horse)

说明:
https://tech.foxrelax.com/generative/gan_horse/
"""


latent_dim = 32
width, height, channels = 32, 32, 3


def load_data_horse():
    """
    返回horse数据集

    返回:
    x_train: [5000, 32, 32, 3]
    """
    (x_train, y_train), (_, _) = keras.datasets.cifar10.load_data()
    assert x_train.shape == (50000, 32, 32, 3)
    assert y_train.shape == (50000, 1)

    x_train = x_train[y_train.flatten() == 7]  # 选择马类(类别编号为6)
    assert x_train.shape == (5000, 32, 32, 3)

    # 数据标准化
    x_train = x_train.reshape(x_train.shape[0], width, height,
                              channels).astype('float32') / 255.

    return x_train


def make_generator_model():
    """
    生成器
    
    将潜在空间的`latent_dim`维向量, 转换为一张`候选图像`

    >>> generator = make_generator_model()
    >>> x = tf.random.normal((2, 32))
    >>> assert generator(x).shape == (2, 32, 32, 3)

    输入:
    x: [batch_size, latent_dim=32]

    输出:
    y: [batch_size, 32, 32, 3]
    """
    generator_input = keras.Input(shape=(latent_dim, ))

    # output [batch_size, 16 * 16 * 128 = 32768]
    x = layers.Dense(16 * 16 * 128)(generator_input)
    x = layers.LeakyReLU()(x)
    # output [batch_size, 16, 16, 128]
    x = layers.Reshape((16, 16, 128))(x)

    # output [batch_size, 16, 16, 256]
    x = layers.Conv2D(256, 5, padding='same')(x)
    x = layers.LeakyReLU()(x)

    # 上采样
    # output [batch_size, 32, 32, 256]
    x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(256, 5, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(256, 5, padding='same')(x)
    x = layers.LeakyReLU()(x)

    # 生成一个32x32的3通道特征图(即CIFAR10图像的形状)
    # output [batch_size, 32, 32, 3]
    x = layers.Conv2D(channels, 7, padding='same', activation='relu')(x)

    model = keras.models.Model(generator_input, x)
    return model


def make_discriminator_model():
    """
    判别器
    
    输入一张候选图像(真实的或者合成的), 输出图像的真假(二分类)
    "生成图像"或"来自训练集的真实图像"

    >>> discriminator = make_discriminator_model()
    >>> x = tf.random.normal((2, 32, 32, 3))
    >>> assert discriminator(x).shape == (2, 1)

    输入:
    x: [batch_size, 32, 32, 3]

    输出:
    y: [batch_sizes, 1]
    """
    # input [batch_size, 32, 32, 3]
    discriminator_input = layers.Input(shape=(width, height, channels))
    # output [batch_size, 30, 30, 128]
    x = layers.Conv2D(128, 3)(discriminator_input)
    x = layers.LeakyReLU()(x)
    # output [batch_zie, 14, 14, 128]
    x = layers.Conv2D(128, 4, strides=2)(x)
    x = layers.LeakyReLU()(x)
    # output [batch_size, 6, 6, 128]
    x = layers.Conv2D(128, 4, strides=2)(x)
    x = layers.LeakyReLU()(x)
    # output [batch_size, 2, 2, 128]
    x = layers.Conv2D(128, 4, strides=2)(x)
    x = layers.LeakyReLU()(x)
    # output [batch_size, 512]
    x = layers.Flatten()(x)
    x = layers.Dropout(0.4)(x)  # 使用dropout是很重要的技巧
    # output [batch_sizes, 1]
    x = layers.Dense(1, activation='sigmoid')(x)  # 分类层

    model = keras.models.Model(discriminator_input, x)
    return model


def make_gan_model(generator, discriminator):
    """
    连接`生成器`和`判别器`(会冻结`判别器`的权重, 在训练时只训练`生成器`)

    >>> generator = make_generator_model()
    >>> discriminator = make_discriminator_model()
    >>> gan = make_gan_model(generator, discriminator)
    >>> x = tf.random.normal((2, 32))
    >>> assert gan(x).shape == (2, 1)

    输入:
    x: [batch_size, latent_dim=32]

    输出:
    y: [batch_size, 1]
    """
    discriminator.trainable = False  # 判别器的权重设置为不可训练(仅用于gan模型), 我们只训练生成器
    gan_input = keras.Input(shape=(latent_dim, ))
    gan_output = discriminator(generator(gan_input))
    model = keras.models.Model(gan_input, gan_output)
    return model


def train_batch(generator, discriminator, gan, x_train, start, batch_size):
    """
    训练一个批量

    1. 训练`判别器`
    2. 训练gan, 也就是训练`生成器`

    返回:
    d_loss: 标量 训练判别器的loss
    a_loss: 标量 训练生成器的loss
    generated_images: [batch_size, 32, 32, 3] 生成伪造的图像
    real_images: [batch_size, 32, 32, 3] 从训练集采样真实图像
    """
    # 训练判别器

    # 在潜在空间采样随机点
    # random_latent_vectors.shape [batch_size, latent_dim]
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
    # 生成伪造的图像
    # generated_images.shape [batch_size, 32, 32, 3]
    generated_images = generator.predict(random_latent_vectors)
    stop = start + batch_size
    # real_images.shzpe [batch_size, 32, 32, 3]
    real_images = x_train[start:stop]  # 从训练集采样真实图像
    # 将`真实图像`和`伪造的图像`混合到一起
    # combined_images.shape [batch_size, 32, 32, 3]
    combined_images = np.concatenate([generated_images, real_images], axis=0)
    # 生成对应的标签(真实图像-0;伪造图像-1)
    # labels.shape [batch_size, ]
    labels = np.concatenate(
        [np.ones((batch_size, 1)),
         np.zeros((batch_size, 1))], axis=0)
    # 向标签中添加随机噪音, 这是很重要的一个技巧
    labels += 0.05 * np.random.random(labels.shape)
    d_loss = discriminator.train_on_batch(combined_images, labels)

    # 训练生成器
    # 在潜在空间采样随机点
    # random_latent_vectors.shape [batch_size, latent_dim]
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
    # 生成对应的标签(0), 0是`真实图像`的标签, 这是在`撒谎`
    # misleading_targets.shape [batch_size, 1]
    misleading_targets = np.zeros((batch_size, 1))
    # 通过gan来训练生成器(判别器此时已经被冻结)
    a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)
    return d_loss, a_loss, generated_images, real_images


# 生成种子, 在训练循环中不断的可视化这个种子生成的图像, 最终做成一个GIF
num_examples_to_generate = 16
# seed.shape [16, latent_dim]
seed = tf.random.normal((num_examples_to_generate, latent_dim))


def generate_and_save_images(model, step, test_input):
    """
    根据种子生成一张图片: image_at_step_xx.png

    参数:
    model: generator
    step: 训练了多少步
    test_input: [16, latent_dim] seed
    """
    # predictions.shape [16, 32, 32, 3]
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))
    # 画4x4的网格来展示16张图片
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        img = keras.preprocessing.image.array_to_img(predictions[i, :, :, :] *
                                                     225.,
                                                     scale=False)
        plt.imshow(img)
        plt.axis('off')

    plt.savefig('image_at_step_{:04d}.png'.format(step))


def train():
    batch_size, iterations = 32, 20000
    # 加载训练集
    # x_train.shape [5000, 32, 32, 3]
    x_train = load_data_horse()

    # 定义模型
    generator = make_generator_model()
    discriminator = make_discriminator_model()
    gan = make_gan_model(generator, discriminator)

    # 设置训练参数:
    # 在优化器中使用梯度裁剪(限制梯度的范围), 同时为了稳定训练过程, 使用学习率衰减
    discriminator_optimizer = keras.optimizers.RMSprop(learning_rate=0.00002,
                                                       clipvalue=1.0,
                                                       decay=1e-8)
    discriminator.compile(optimizer=discriminator_optimizer,
                          loss='binary_crossentropy')

    gan_optimizer = keras.optimizers.RMSprop(learning_rate=0.0004,
                                             clipvalue=1.0,
                                             decay=1e-8)
    gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')

    # 训练:
    start = 0
    for step in range(iterations):
        a_loss, d_loss, _, _ = train_batch(generator, discriminator, gan,
                                           x_train, start, batch_size)
        start += batch_size
        if start > len(x_train) - batch_size:
            start = 0

        if step % 500 == 0:
            generate_and_save_images(generator, step + 1, seed)
            # 每500步保存并绘图
            print('step:{}, discriminator loss:{}, adversarial loss:{}'.format(
                step + 1, d_loss, a_loss))

    generate_and_save_images(generator, step + 1, seed)
    plt.show()


if __name__ == '__main__':
    train()