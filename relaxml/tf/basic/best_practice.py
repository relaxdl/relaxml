import sys
import time
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import models, layers, Input
from matplotlib import pyplot as plt
from tqdm import tqdm


class Dense(tf.Module):
    """
    自定义Layer
    """

    def __init__(self, in_features, out_features, activation=None, name=None):
        super(Dense, self).__init__(name=name)
        self.w = tf.Variable(tf.random.normal([in_features, out_features]),
                             name='w')
        self.b = tf.Variable(tf.zeros([out_features]), name='b')
        self.activation = activation

    def __call__(self, x):
        y = tf.matmul(x, self.w) + self.b
        if self.activation == 'relu':
            y = tf.nn.relu(y)
        return y


class MyModel(tf.Module):
    """
    自定义Model

    >>> x = tf.random.normal((256, 784))
    >>> model = MyModel()
    >>> assert model(x).shape == (256, 10)
    >>> for v in model.trainable_variables:
    >>>     print(v.name, v.shape)
        b:0 (64,)
        w:0 (784, 64)
        b:0 (64,)
        w:0 (64, 64)
        b:0 (10,)
        w:0 (64, 10)
    """

    def __init__(self, name=None):
        super(MyModel, self).__init__(name=name)
        self.dense1 = Dense(784, 64, activation='relu', name='dense_1')
        self.dense2 = Dense(64, 64, activation='relu', name='dense_2')
        self.outputs = Dense(64, 10, name='predictions')

    def __call__(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        outputs = self.outputs(x)
        return outputs


def functional_model():
    """
    Keras Functional Model

    >>> x = tf.random.normal((256, 784))
    >>> model = functional_model()
    >>> assert model(x).shape == (256, 10)
    >>> for v in model.trainable_variables:
    >>>     print(v.name, v.shape)
        dense_1/kernel:0 (784, 64)
        dense_1/bias:0 (64,)
        dense_2/kernel:0 (64, 64)
        dense_2/bias:0 (64,)
        predictions/kernel:0 (64, 10)
        predictions/bias:0 (10,)
    """
    inputs = Input(shape=(784, ), name="digits")
    x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
    x = layers.Dense(64, activation="relu", name="dense_2")(x)
    # 最后一层没有加softmax, 返回的是logits, 可以在损失函数里计算softmax
    outputs = layers.Dense(10, name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


class DenseKeras(layers.Layer):
    """
    自定义Keras Layer
    """

    def __init__(self, in_features, out_features, activation=None, name=None):
        super().__init__(name=name)
        self.w = tf.Variable(tf.random.normal([in_features, out_features]),
                             name='w')
        self.b = tf.Variable(tf.zeros([out_features]), name='b')
        self.activation = activation

    def call(self, x):
        y = tf.matmul(x, self.w) + self.b
        if self.activation == 'relu':
            y = tf.nn.relu(y)
        return y


class MyModelKeras(keras.Model):
    """
    自定义Keras Model

    >>> x = tf.random.normal((256, 784))
    >>> model = MyModelKeras()
    >>> assert model(x).shape == (256, 10)
    >>> for v in model.trainable_variables:
    >>>     print(v.name, v.shape)
        dense_1/kernel:0 (784, 64)
        dense_1/bias:0 (64,)
        w:0 (64, 64)
        b:0 (64,)
        predictions/kernel:0 (64, 10)
        predictions/bias:0 (10,)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = layers.Dense(64, activation="relu", name="dense_1")
        self.dense2 = DenseKeras(64, 64, activation="relu", name="dense_2")
        self.outputs = layers.Dense(10, name="predictions")

    def __call__(self, x, **kwargs):
        x = self.dense1(x)
        x = self.dense2(x)
        outputs = self.outputs(x)
        return outputs


def model_sequential():
    """
    Keras Sequential Module

    >>> x = tf.random.normal((256, 28, 28, 1))
    >>> model = model_sequential()
    >>> assert model(x).shape == (256, 10)
    >>> for v in model.trainable_variables:
    >>>     print(v.name, v.shape)
        dense_1/kernel:0 (784, 64)
        dense_1/bias:0 (64,)
        dense_2/kernel:0 (64, 64)
        dense_2/bias:0 (64,)
        dense_3/kernel:0 (64, 10)
        dense_3/bias:0 (10,)
    """
    net = models.Sequential()
    # [batch_size, height, width, channels] ->
    # [batch_size, height*width*channels]
    net.add(layers.Flatten())
    net.add(layers.Dense(64, activation="relu"))
    net.add(layers.Dense(64, activation="relu"))
    net.add(layers.Dense(10))
    return net


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


def accuracy(y_hat, y):
    """
    计算预测正确的数量

    参数:
    y_hat [batch_size, num_classes]
    y [batch_size,]
    """
    y_hat = tf.argmax(y_hat, axis=1)
    cmp = tf.cast(y_hat, y.dtype) == y
    return float(tf.reduce_sum(tf.cast(cmp, y.dtype)))


@tf.function
def train_step(net, X, y, loss, optimizer):
    """
    迭代一个batch
    1. 正向传播, 获得y_hat [Model]
    2. 用y_hat和y计算loss [Loss]
    3. 获取需要训练的模型参数 [Model.trainable_variables]
    4. 反向传播, 获取梯度 [tf.GradientTape]
    5. 梯度下降更新参数 [Optimizer]
    """
    with tf.GradientTape() as tape:
        y_hat = net(X)  # 1. 正向传播, 获得y_hat
        l = loss(y, y_hat)  # 2. 用y_hat和y计算loss
    params = net.trainable_variables  # 3. 获取需要训练的模型参数
    grads = tape.gradient(l, params)  # 4. 反向传播, 获取梯度
    optimizer.apply_gradients(zip(grads, params))  # 5. 梯度下降更新参数
    return l, y_hat


@tf.function
def test_step(net, X):
    y_hat = net(X)
    return y_hat


def train(net, train_iter, test_iter, num_epochs, loss, optimizer):
    times = []
    history = [[], [], []]  # 记录: 训练集损失, 训练集准确率, 测试集准确率, 方便后续绘图
    num_batches = len(train_iter)
    best_test_acc = 0.0
    for epoch in range(num_epochs):
        # 训练
        metric_train = [0.0] * 3  # 统计: 训练集损失之和, 训练集准确数量之和, 训练集样本数量之和
        train_iter_tqdm = tqdm(train_iter, file=sys.stdout)
        for i, (X, y) in enumerate(train_iter_tqdm):
            t_start = time.time()
            l, y_hat = train_step(net, X, y, loss, optimizer)
            metric_train[0] += float(l * X.shape[0])
            metric_train[1] += float(accuracy(y_hat, y))
            metric_train[2] += float(X.shape[0])
            times.append(time.time() - t_start)
            train_loss = metric_train[0] / metric_train[2]
            train_acc = metric_train[1] / metric_train[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                history[0].append((epoch + (i + 1) / num_batches, train_loss))
                history[1].append((epoch + (i + 1) / num_batches, train_acc))
            train_iter_tqdm.desc = f'epoch {epoch}, step {i+1}, train loss {train_loss:.3f}, train acc {train_acc:.3f}'

        # 评估
        metric_test = [0.0] * 2  # 测试准确数量之和, 测试样本数量之和
        for X, y in test_iter:
            y_hat = test_step(net, X)
            metric_test[0] += float(accuracy(y_hat, y))
            metric_test[1] += float(X.shape[0])
        test_acc = metric_test[0] / metric_test[1]
        history[2].append((epoch + 1, test_acc))
        print(f'epoch {epoch}, step {i+1}, train loss {train_loss:.3f}, '
              f'train acc {train_acc:.3f}, test acc {test_acc:.3f}')
        if test_acc > best_test_acc:
            best_test_acc = test_acc

    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric_train[2] * num_epochs / sum(times):.1f} examples/sec')
    return history


def plot_history(history, figsize=(6, 4)) -> None:
    plt.figure(figsize=figsize)
    # 训练集损失, 训练集准确率, 测试集准确率
    num_epochs = len(history[2])
    plt.plot(*zip(*history[0]), '-', label='train loss')
    plt.plot(*zip(*history[1]), 'm--', label='train acc')
    plt.plot(*zip(*history[2]), 'g-.', label='test acc')
    plt.xlabel('epoch')
    # 从epoch=1开始显示, 0-1这个范围的数据丢弃不展示,
    # 因为只有训练完成1个epochs之后, 才会有第一条test acc记录
    plt.xlim((1, num_epochs))
    plt.grid()
    plt.legend()
    plt.show()


def run_scratch():
    train_iter, test_iter = load_data_fashion_mnist(batch_size=256)
    net = model_sequential()
    kwargs = {
        'num_epochs': 10,
        'loss': tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True),  # mean loss
        'optimizer': tf.keras.optimizers.SGD(learning_rate=0.1)
    }
    history = train(net, train_iter, test_iter, **kwargs)
    plot_history(history)


def run():
    train_iter, test_iter = load_data_fashion_mnist(batch_size=256)
    net = model_sequential()
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    net.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    history = net.fit(train_iter, epochs=10, validation_data=test_iter)
    history_dict = history.history
    train_loss = history_dict['loss']
    train_acc = history_dict['accuracy']
    test_acc = history_dict['val_accuracy']
    epochs = range(1, len(train_loss) + 1)
    history = [
        list(zip(epochs, train_loss)),
        list(zip(epochs, train_acc)),
        list(zip(epochs, test_acc))
    ]
    plot_history(history)


if __name__ == '__main__':
    # run_scratch()
    # epoch 0, step 235, train loss 0.808, train acc 0.719: 100%|██████████| 235/235 [00:01<00:00, 171.50it/s]
    # epoch 0, step 235, train loss 0.808, train acc 0.719, test acc 0.773
    # epoch 1, step 235, train loss 0.522, train acc 0.814: 100%|██████████| 235/235 [00:00<00:00, 254.62it/s]
    # epoch 1, step 235, train loss 0.522, train acc 0.814, test acc 0.832
    # epoch 2, step 235, train loss 0.462, train acc 0.834: 100%|██████████| 235/235 [00:00<00:00, 245.40it/s]
    # epoch 2, step 235, train loss 0.462, train acc 0.834, test acc 0.825
    # epoch 3, step 235, train loss 0.428, train acc 0.847: 100%|██████████| 235/235 [00:00<00:00, 250.99it/s]
    # epoch 3, step 235, train loss 0.428, train acc 0.847, test acc 0.836
    # epoch 4, step 235, train loss 0.405, train acc 0.855: 100%|██████████| 235/235 [00:00<00:00, 296.51it/s]
    # epoch 4, step 235, train loss 0.405, train acc 0.855, test acc 0.849
    # epoch 5, step 235, train loss 0.388, train acc 0.859: 100%|██████████| 235/235 [00:00<00:00, 304.01it/s]
    # epoch 5, step 235, train loss 0.388, train acc 0.859, test acc 0.849
    # epoch 6, step 235, train loss 0.372, train acc 0.866: 100%|██████████| 235/235 [00:00<00:00, 293.38it/s]
    # epoch 6, step 235, train loss 0.372, train acc 0.866, test acc 0.859
    # epoch 7, step 235, train loss 0.365, train acc 0.868: 100%|██████████| 235/235 [00:00<00:00, 306.87it/s]
    # epoch 7, step 235, train loss 0.365, train acc 0.868, test acc 0.861
    # epoch 8, step 235, train loss 0.354, train acc 0.873: 100%|██████████| 235/235 [00:00<00:00, 304.43it/s]
    # epoch 8, step 235, train loss 0.354, train acc 0.873, test acc 0.855
    # epoch 9, step 235, train loss 0.346, train acc 0.875: 100%|██████████| 235/235 [00:00<00:00, 302.84it/s]
    # epoch 9, step 235, train loss 0.346, train acc 0.875, test acc 0.850
    # train loss 0.346, train acc 0.875, test acc 0.850
    # 101050.5 examples/sec

    run()
# Epoch 1/10
# 235/235 [==============================] - 1s 4ms/step - loss: 0.8051 - accuracy: 0.7201 - val_loss: 0.6332 - val_accuracy: 0.7830
# Epoch 2/10
# 235/235 [==============================] - 1s 4ms/step - loss: 0.5243 - accuracy: 0.8124 - val_loss: 0.5382 - val_accuracy: 0.8028
# Epoch 3/10
# 235/235 [==============================] - 1s 4ms/step - loss: 0.4634 - accuracy: 0.8357 - val_loss: 0.4569 - val_accuracy: 0.8400
# Epoch 4/10
# 235/235 [==============================] - 1s 3ms/step - loss: 0.4306 - accuracy: 0.8462 - val_loss: 0.4453 - val_accuracy: 0.8401
# Epoch 5/10
# 235/235 [==============================] - 1s 3ms/step - loss: 0.4090 - accuracy: 0.8544 - val_loss: 0.4928 - val_accuracy: 0.8196
# Epoch 6/10
# 235/235 [==============================] - 1s 3ms/step - loss: 0.3923 - accuracy: 0.8578 - val_loss: 0.4344 - val_accuracy: 0.8467
# Epoch 7/10
# 235/235 [==============================] - 1s 3ms/step - loss: 0.3771 - accuracy: 0.8649 - val_loss: 0.4011 - val_accuracy: 0.8563
# Epoch 8/10
# 235/235 [==============================] - 1s 3ms/step - loss: 0.3668 - accuracy: 0.8675 - val_loss: 0.4008 - val_accuracy: 0.8536
# Epoch 9/10
# 235/235 [==============================] - 1s 3ms/step - loss: 0.3587 - accuracy: 0.8714 - val_loss: 0.3992 - val_accuracy: 0.8549
# Epoch 10/10
# 235/235 [==============================] - 1s 3ms/step - loss: 0.3465 - accuracy: 0.8749 - val_loss: 0.3885 - val_accuracy: 0.8614