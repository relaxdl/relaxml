import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt


def load_data_fashion_mnist():
    """
    返回Fashion Mnist数据集
    """
    (x_train, _), (x_test, _) = keras.datasets.fashion_mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    # x_train.shape [60000, 28, 28]
    # x_test.shape [10000, 28, 28]
    return x_train, x_test


class Autoencoder(keras.Model):
    """
    >>> x = tf.random.normal((32, 28, 28))
    >>> net = Autoencoder(64)
    >>> assert net(x).shape == (32, 28, 28)
    """

    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = keras.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = keras.Sequential([
            keras.layers.Dense(784, activation='sigmoid'),
            keras.layers.Reshape((28, 28))
        ])

    def call(self, x):
        """
        参数:
        x: [batch_size, 28, 28]

        输出:
        output: [batch_size, 28, 28]
        """
        # encoded.shape [batch_size, 64]
        encoded = self.encoder(x)
        # encoded.shape [batch_size, 28, 28]
        decoded = self.decoder(encoded)
        return decoded


def train_autoencoder():
    num_epochs, batch_size, latent_dim = 10, 32, 64
    net = Autoencoder(latent_dim)
    x_train, x_test = load_data_fashion_mnist()
    net.compile(optimizer='adam', loss='mse')
    net.fit(x_train,
            x_train,
            epochs=num_epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(x_test, x_test))
    return net, x_train, x_test


def test_autoencoder(net, x_test):
    # encoded_imgs.shape [batch_size, 64]
    encoded_imgs = net.encoder(x_test).numpy()
    # encoded_imgs.shape [batch_size, 28, 28]
    decoded_imgs = net.decoder(encoded_imgs).numpy()

    n = 10
    plt.figure(figsize=(16, 3))
    for i in range(n):
        # 展示原始图像
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i])
        plt.title("original")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # 展示编码->解码之后的图像
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i])
        plt.title("reconstructed")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


def load_data_fashion_mnist_with_noise():
    """
    返回带随机噪音的Fashion Mnist数据集
    """
    (x_train, _), (x_test, _) = keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    # 增加通道维度
    # x_train.shape [60000, 28, 28, 1]
    # x_test.shape [10000, 28, 28, 1]
    x_train = x_train[:, :, :, None]
    x_test = x_test[:, :, :, None]

    # 添加随机噪音
    noise_factor = 0.2
    x_train_noisy = x_train + noise_factor * tf.random.normal(
        shape=x_train.shape)
    x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape)

    x_train_noisy = tf.clip_by_value(x_train_noisy,
                                     clip_value_min=0.,
                                     clip_value_max=1.)
    x_test_noisy = tf.clip_by_value(x_test_noisy,
                                    clip_value_min=0.,
                                    clip_value_max=1.)
    # x_train_noisy.shape [60000, 28, 28, 1]
    # x_train.shape [60000, 28, 28, 1]
    # x_test_noisy.shape [10000, 28, 28, 1]
    # x_test.shape [10000, 28, 28, 1]
    return (x_train_noisy, x_train), (x_test_noisy, x_test)


class Denoise(keras.Model):
    """
    Encoder-Decoder

    在Encoder中使用Conv2D来进行下采样
    在Decoder中使用Conv2DTranspose来上采样

    >>> x = tf.random.normal((32, 28, 28, 1))
    >>> net = Denoise()
    >>> assert net(x).shape == (32, 28, 28, 1)
    """

    def __init__(self):
        super(Denoise, self).__init__()
        #    [batch_size, 28, 28, 1]
        # -> [batch_size, 14, 14, 16]
        # -> [batch_size, 7, 7, 8]
        self.encoder = keras.Sequential([
            keras.layers.Input(shape=(28, 28, 1)),
            keras.layers.Conv2D(16, (3, 3),
                                activation='relu',
                                padding='same',
                                strides=2),
            keras.layers.Conv2D(8, (3, 3),
                                activation='relu',
                                padding='same',
                                strides=2)
        ])
        #    [batch_size, 7, 7, 8]
        # -> [batch_size, 14, 14, 8]
        # -> [batch_size, 28, 28, 16]
        # -> [batch_size, 28, 28, 1]
        self.decoder = tf.keras.Sequential([
            keras.layers.Conv2DTranspose(8,
                                         kernel_size=3,
                                         strides=2,
                                         activation='relu',
                                         padding='same'),
            keras.layers.Conv2DTranspose(16,
                                         kernel_size=3,
                                         strides=2,
                                         activation='relu',
                                         padding='same'),
            keras.layers.Conv2D(1,
                                kernel_size=(3, 3),
                                activation='sigmoid',
                                padding='same')
        ])

    def call(self, x):
        """
        参数:
        x: [batch_size, 28, 28, 1]

        返回:
        decoded: [batch_size, 28, 28, 1]
        """
        # encoded.shape [batch_size, 7, 7, 8]
        encoded = self.encoder(x)
        # decoded.shape [batch_size, 28, 28, 1]
        decoded = self.decoder(encoded)
        return decoded


def train_denoise():
    num_epochs, batch_size = 10, 32
    net = Denoise()
    (x_train_noisy, x_train), (x_test_noisy,
                               x_test) = load_data_fashion_mnist_with_noise()
    net.compile(optimizer='adam', loss='mse')
    net.fit(x_train_noisy,
            x_train,
            epochs=num_epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(x_test_noisy, x_test))
    return net, (x_train_noisy, x_train), (x_test_noisy, x_test)


def test_denoise(net, x_test_noisy, x_test):
    # encoded_imgs.shape [batch_size, 7, 7, 8]
    encoded_imgs = net.encoder(x_test).numpy()
    # decoded_imgs.shape [batch_size, 28, 28, 1]
    decoded_imgs = net.decoder(encoded_imgs).numpy()

    n = 10
    plt.figure(figsize=(16, 3))
    for i in range(n):
        # display original + noise
        ax = plt.subplot(2, n, i + 1)
        plt.title("noise")
        plt.imshow(tf.squeeze(x_test_noisy[i]))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        bx = plt.subplot(2, n, i + n + 1)
        plt.title("reconstructed")
        plt.imshow(tf.squeeze(decoded_imgs[i]))
        plt.gray()
        bx.get_xaxis().set_visible(False)
        bx.get_yaxis().set_visible(False)


if __name__ == '__main__':
    # Autoencoder
    net, x_train, x_test = train_autoencoder()
    test_autoencoder(net, x_test)
    # Epoch 1/10
    # 1875/1875 [==============================] - 5s 3ms/step - loss: 0.0240 - val_loss: 0.0133
    # Epoch 2/10
    # 1875/1875 [==============================] - 4s 2ms/step - loss: 0.0116 - val_loss: 0.0106
    # Epoch 3/10
    # 1875/1875 [==============================] - 4s 2ms/step - loss: 0.0101 - val_loss: 0.0097
    # Epoch 4/10
    # 1875/1875 [==============================] - 4s 2ms/step - loss: 0.0094 - val_loss: 0.0093
    # Epoch 5/10
    # 1875/1875 [==============================] - 4s 2ms/step - loss: 0.0092 - val_loss: 0.0092
    # Epoch 6/10
    # 1875/1875 [==============================] - 4s 2ms/step - loss: 0.0090 - val_loss: 0.0089
    # Epoch 7/10
    # 1875/1875 [==============================] - 4s 2ms/step - loss: 0.0089 - val_loss: 0.0091
    # Epoch 8/10
    # 1875/1875 [==============================] - 4s 2ms/step - loss: 0.0088 - val_loss: 0.0089
    # Epoch 9/10
    # 1875/1875 [==============================] - 4s 2ms/step - loss: 0.0087 - val_loss: 0.0088
    # Epoch 10/10
    # 1875/1875 [==============================] - 4s 2ms/step - loss: 0.0087 - val_loss: 0.0088

    # Denoise
    net, (x_train_noisy, x_train), (x_test_noisy, x_test) = train_denoise()
    test_denoise(net, x_test_noisy, x_test)
    # Epoch 1/10
    # 1875/1875 [==============================] - 7s 4ms/step - loss: 0.0166 - val_loss: 0.0096
    # Epoch 2/10
    # 1875/1875 [==============================] - 6s 3ms/step - loss: 0.0088 - val_loss: 0.0082
    # Epoch 3/10
    # 1875/1875 [==============================] - 6s 3ms/step - loss: 0.0078 - val_loss: 0.0076
    # Epoch 4/10
    # 1875/1875 [==============================] - 6s 3ms/step - loss: 0.0074 - val_loss: 0.0073
    # Epoch 5/10
    # 1875/1875 [==============================] - 6s 3ms/step - loss: 0.0072 - val_loss: 0.0072
    # Epoch 6/10
    # 1875/1875 [==============================] - 6s 3ms/step - loss: 0.0071 - val_loss: 0.0071
    # Epoch 7/10
    # 1875/1875 [==============================] - 6s 3ms/step - loss: 0.0070 - val_loss: 0.0070
    # Epoch 8/10
    # 1875/1875 [==============================] - 6s 3ms/step - loss: 0.0069 - val_loss: 0.0069
    # Epoch 9/10
    # 1875/1875 [==============================] - 6s 3ms/step - loss: 0.0069 - val_loss: 0.0069
    # Epoch 10/10
    # 1875/1875 [==============================] - 6s 3ms/step - loss: 0.0068 - val_loss: 0.0068