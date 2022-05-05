import sys
import time
import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm
"""
softmax回归从零实现

说明:
https://tech.foxrelax.com/basic/softmax_regression_scratch/
"""


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


def softmax(X):
    """
    >>> X = tf.random.normal((2, 5))
    >>> X_prob = softmax(X)
    >>> print(X_prob)
        tf.Tensor([[0.14849083 0.14573483 0.3269037  0.1714058  0.20746487]
                   [0.0757804  0.18379888 0.46541107 0.15378769 0.12122207]], 
                  shape=(2, 5), dtype=float32)
    >>> print(tf.reduce_sum(X_prob, 1))
        tf.Tensor([1. 1.], shape=(2,), dtype=float32)
    """
    X_exp = tf.exp(X)
    partition = tf.reduce_sum(X_exp, 1, keepdims=True)
    return X_exp / partition  # 这里应用了广播机制


def gen_params(num_inputs=784, num_outputs=10):
    """
    生成模型参数: [w, b]
    """
    W = tf.Variable(
        tf.random.normal(shape=(num_inputs, num_outputs), mean=0, stddev=0.01))
    b = tf.Variable(tf.zeros(num_outputs))
    return W, b


def net(X, W, b):
    return softmax(tf.matmul(tf.reshape(X, (-1, W.shape[0])), W) + b)


def cross_entropy(y_hat, y):
    """
    >>> y = tf.constant([0, 2])
    >>> y_hat = tf.constant([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
    >>> l = cross_entropy(y_hat, y)
    >>> print(l)
        tf.Tensor([2.3025851 0.6931472], shape=(2,), dtype=float32)
    """
    return -tf.math.log(
        tf.boolean_mask(y_hat, tf.one_hot(y, depth=y_hat.shape[-1])))


def sgd(params, grads, lr, batch_size):
    """
    小批量随机梯度下降
    """
    for param, grad in zip(params, grads):
        param.assign_sub(lr * grad / batch_size)


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


def train(batch_size=256, num_epochs=10, lr=0.1):
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    W, b = gen_params()
    loss = cross_entropy
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
            with tf.GradientTape() as tape:
                y_hat = net(X, W, b)
                l = loss(y_hat, y)
            grads = tape.gradient(l, [W, b])
            sgd([W, b], grads, lr, batch_size)

            metric_train[0] += float(tf.reduce_sum(l))
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
            metric_test[0] += float(accuracy(net(X, W, b), y))
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


def run() -> None:
    history = train(num_epochs=10, lr=0.1)
    plot_history(history)


if __name__ == '__main__':
    run()
# epoch 0, step 235, train loss 0.785, train acc 0.749: 100%|██████████| 235/235 [00:02<00:00, 97.32it/s]
# epoch 0, step 235, train loss 0.785, train acc 0.749, test acc 0.781
# epoch 1, step 235, train loss 0.571, train acc 0.813: 100%|██████████| 235/235 [00:02<00:00, 97.47it/s]
# epoch 1, step 235, train loss 0.571, train acc 0.813, test acc 0.810
# epoch 2, step 235, train loss 0.526, train acc 0.827: 100%|██████████| 235/235 [00:01<00:00, 122.93it/s]
# epoch 2, step 235, train loss 0.526, train acc 0.827, test acc 0.816
# epoch 3, step 235, train loss 0.501, train acc 0.832: 100%|██████████| 235/235 [00:01<00:00, 125.40it/s]
# epoch 3, step 235, train loss 0.501, train acc 0.832, test acc 0.824
# epoch 4, step 235, train loss 0.486, train acc 0.836: 100%|██████████| 235/235 [00:01<00:00, 124.46it/s]
# epoch 4, step 235, train loss 0.486, train acc 0.836, test acc 0.826
# epoch 5, step 235, train loss 0.474, train acc 0.840: 100%|██████████| 235/235 [00:01<00:00, 126.14it/s]
# epoch 5, step 235, train loss 0.474, train acc 0.840, test acc 0.829
# epoch 6, step 235, train loss 0.465, train acc 0.843: 100%|██████████| 235/235 [00:01<00:00, 124.39it/s]
# epoch 6, step 235, train loss 0.465, train acc 0.843, test acc 0.829
# epoch 7, step 235, train loss 0.458, train acc 0.845: 100%|██████████| 235/235 [00:01<00:00, 123.43it/s]
# epoch 7, step 235, train loss 0.458, train acc 0.845, test acc 0.827
# epoch 8, step 235, train loss 0.452, train acc 0.847: 100%|██████████| 235/235 [00:01<00:00, 121.72it/s]
# epoch 8, step 235, train loss 0.452, train acc 0.847, test acc 0.833
# epoch 9, step 235, train loss 0.447, train acc 0.847: 100%|██████████| 235/235 [00:01<00:00, 126.38it/s]
# epoch 9, step 235, train loss 0.447, train acc 0.847, test acc 0.835
# train loss 0.447, train acc 0.847, test acc 0.835
# 35573.3 examples/sec