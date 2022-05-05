import tensorflow as tf


def load_data_fashion_mnist(batch_size, resize=None):
    """
    下载Fashion-MNIST数据集, 然后将其加载到内存中

    1. 60000张训练图像和对应Label
    2. 10000张测试图像和对应Label
    3. 10个类别
    4. 每张图像28x28x1的分辨率

    >>> train_iter, test_iter = load_data_fashion_mnist(batch_size=256)
    >>> for x, y in train_iter:
    >>>     assert x.shape == (256, 28, 28, 1)
    >>>     assert y.shape == (256, )
    >>>     break
    """
    # mnist_train[0].shape [60000, 28, 28]
    # mnist_train[1].shape [60000, ]
    # mnist_test[0].shape [10000, 28, 28]
    # mnist_test[1].shape [10000, ]
    mnist_train, mnist_test = tf.keras.datasets.fashion_mnist.load_data()
    # 将所有数字除以255, 使所有像素值介于0和1之间, 在最后添加一个channel,
    # 并将label转换为int32
    process = lambda X, y: (tf.expand_dims(X, axis=3) / 255,
                            tf.cast(y, dtype='int32'))
    resize_fn = lambda X, y: (tf.image.resize_with_pad(X, resize, resize)
                              if resize else X, y)
    train_iter = tf.data.Dataset.from_tensor_slices(
        process(*mnist_train)).batch(batch_size).shuffle(len(
            mnist_train[0])).map(resize_fn)
    test_iter = tf.data.Dataset.from_tensor_slices(
        process(*mnist_test)).batch(batch_size).map(resize_fn)
    return train_iter, test_iter


if __name__ == '__main__':
    train_iter, test_iter = load_data_fashion_mnist(batch_size=256)
    for x, y in train_iter:
        assert x.shape == (256, 28, 28, 1)
        assert y.shape == (256, )
        break