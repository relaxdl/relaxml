import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
"""
Variational Autoencoder

说明:
https://tech.foxrelax.com/generative/vae/
"""

# 全局配置
img_shape = (28, 28, 1)  # 输入图片的形状
shape_before_flattening = (14, 14, 64)  # 图片在Flatten之前的形状
batch_size = 16  # 批量大小
latent_dim = 2


def encoder():
    """
    VAE Encoder

    通过Encoder, 输入图像最终被编码为两个参数: `z_mean, z_log_var`

    >>> batch_size, latent_dim = 16, 2
    >>> x = tf.random.normal((batch_size, 28, 28, 1))
    >>> vae_encoder = encoder()
    >>> [z_mean, z_log_var] = vae_encoder(x)
    >>> assert z_mean.shape == (batch_size, latent_dim)
    >>> assert z_log_var.shape == (batch_size, latent_dim)
    """
    # input [batch_size, 28, 28, 1]
    input_img = keras.Input(shape=img_shape)
    # output [batch_size, 28, 28, 32]
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(input_img)
    # output [batch_size, 14, 14, 64]
    x = layers.Conv2D(64, 3, padding='same', activation='relu',
                      strides=(2, 2))(x)
    # output [batch_size, 14, 14, 64]
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    # output [batch_size, 14, 14, 64]
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    # shape_before_flattening [14, 14, 64]
    # x.shape [batch_size, 14*14*64]
    x = layers.Flatten()(x)
    # x.shape [batch_size, 32]
    x = layers.Dense(32, activation='relu')(x)
    # 输入的图像最终被编码为两个参数
    # z_mean.shape [batch_size, latent_dim]
    # z_log_var.shape [batch_size, latent_dim]
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)
    model = keras.Model(input_img, [z_mean, z_log_var])
    return model


def sampling(args):
    """
    VAE Sampler

    用`z_mean, z_log_var`来生成潜在空间的一个点, 也就是潜在空间的
    一个`latent_dim`维向量, 将这个向量送入Decoder可以解码为图像空间
    的一幅图像 

    >>> batch_size, latent_dim = 16, 2
    >>> z_mean = tf.random.normal((batch_size, latent_dim))
    >>> z_log_var = tf.random.normal((batch_size, latent_dim))
    >>> z = sampling([z_mean, z_log_var])
    >>> assert z.shape == (batch_size, latent_dim)
    """
    z_mean, z_log_var = args
    # epsilon.shape [batch_size, latent_dim]
    epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim),
                               mean=0.,
                               stddev=1.)
    # [batch_size, latent_dim]
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def decoder():
    """
    VAE Decoder

    将潜在空间的的一个点, 也就是一个`latent_dim`维向量解码为一张图像

    >>> batch_size, latent_dim = 16, 2
    >>> x = tf.random.normal((batch_size, latent_dim))
    >>> vae_decoder = decoder()
    >>> assert vae_decoder(x).shape == (batch_size, 28, 28, 1)
    """
    # Decoder的输入就是Encoder的输出
    # input [batch_size, latent_dim]
    decoder_input = keras.Input(shape=(latent_dim, ))
    # x.shape [batch_size, 14*14*64=12544]
    x = layers.Dense(np.prod(shape_before_flattening),
                     activation='relu')(decoder_input)
    # x.shape [batch_size, 14, 14, 64]
    x = layers.Reshape(shape_before_flattening)(x)
    # x.shape [batch_size, 28, 28, 32]
    x = layers.Conv2DTranspose(32,
                               3,
                               padding='same',
                               activation='relu',
                               strides=(2, 2))(x)
    # x.shape [batch_size, 28, 28, 1]
    x = layers.Conv2D(1, 3, padding='same', activation='sigmoid')(x)

    model = keras.Model(decoder_input, x)
    return model


class CustomVariationalLayer(layers.Layer):
    """
    1. 重构损失: [x, z_decoded]
    2. 正则化损失: [z_mean, z_log_var]
    """

    def vae_loss(self, x, z_decoded, z_mean, z_log_var):
        """
        参数:
        x: [batch_size, 28, 28, 1]
        z_decoded: [batch_size, 28, 28, 1]

        返回:
        loss: 标量
        """
        # x.shape [batch_size, 28*28]
        x = tf.reshape(x, (tf.shape(x)[0], -1))
        # z_decoded.shape [batch_size, 28*28]
        z_decoded = tf.reshape(z_decoded, (tf.shape(z_decoded)[0], -1))
        # 重构损失
        # xent_loss.shape [batch_size, ]
        xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
        # 正则化损失
        # k1_loss.shape [batch_size, ]
        k1_loss = -5e-4 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
        return tf.reduce_mean(xent_loss + k1_loss)

    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        z_mean = inputs[2]
        z_log_var = inputs[3]
        loss = self.vae_loss(x, z_decoded, z_mean, z_log_var)
        # 添加损失, net.fix()会自动汇聚到主损失中
        self.add_loss(loss, inputs=inputs)
        # 我们不需要使用这个输出, 但是必须有返回值
        return x


def vae():
    """
    VAE模型

    1. Encoder编码数据, 得到[z_mean, z_log_var], z_mean和
       z_log_var都是`latent_dim`维向量
    2. 根据[z_mean, z_log_var]采样数据, 生成潜在空间的一个点, 
       也就是一个`latent_dim`维向量
    3. Decoder解码这个`latent_dim`维向量向量, 还原出图片
    4. 两个损失:
       a. 重构损失: [x, z_decoded]
       b. 正则化损失: [z_mean, z_log_var]
    """
    vae_encoder = encoder()
    vae_decoder = decoder()
    input_img = keras.Input(shape=img_shape)
    # 1. Encoder编码数据
    # z_mean.shape [batch_size, latent_dim]
    # z_log_var.shape [batch_size, latent_dim]
    [z_mean, z_log_var] = vae_encoder(input_img)

    # 2. 采样数据
    # z.shape [batch_size, latent_dim]
    z = layers.Lambda(sampling)([z_mean, z_log_var])

    # 3. Decoder解码数据
    # z_decoded.shape [batch_size, 28, 28, 1]
    z_decoded = vae_decoder(z)
    # 4. 两个损失: 重构损失 & 正则化损失
    y = CustomVariationalLayer()([input_img, z_decoded, z_mean, z_log_var])
    model = keras.Model(input_img, y)
    return model, vae_encoder, vae_decoder


def train(num_epochs=10):
    """
    训练模型
    """
    # 构建模型
    net, vae_encoder, vae_decoder = vae()
    net.compile(optimizer='rmsprop', loss=None)

    # 加载数据集
    (x_train, _), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape(x_train.shape + (1, ))
    x_test = x_test.astype('float32') / 255.
    x_test = x_test.reshape(x_test.shape + (1, ))

    # 训练
    net.fit(x=x_train,
            y=None,
            shuffle=True,
            epochs=num_epochs,
            batch_size=batch_size,
            validation_data=(x_test, None))
    return net, vae_encoder, vae_decoder


def plot(vae_decoder):
    """
    从二维潜在空间中采样一组点的网格, 并将其解码为图像

    1. 显示15x15的数字网格, 一共255个数字
    2. 采样的数字网格展现了不同数字类别的完全连续分布, 当我们沿着潜在空间
       的一条路径观察时, 会观察到一个数字逐渐变为另外一个数字. 这个空间的
       特定方向具有一定的意义, 比如: '逐渐变为4'或者'逐渐变为1'等
    """
    n = 15
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # 用正态分布的百分位函数生成数据
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            # z_sample.shape [1, latent_dim=2]
            z_sample = np.array([[xi, yi]])
            # z_sample.shape [batch_size, latent_dim=2]
            z_sample = np.tile(z_sample,
                               batch_size).reshape(batch_size, latent_dim)
            # x_decoded.shape [batch_size, digit_size=28, digit_size=28, 1]
            x_decoded = vae_decoder.predict(z_sample, batch_size=batch_size)
            # digit.shape [digit_size=28, digit_size=28]
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size:(i + 1) * digit_size,
                   j * digit_size:(j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.show()


if __name__ == '__main__':
    net, vae_encoder, vae_decoder = train()
    plot(vae_decoder)
