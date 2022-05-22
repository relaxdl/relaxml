from tensorflow.keras.datasets import mnist
from tensorflow.keras import models, layers
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# 下载数据集 & 预处理
# train_images的数据类型是numpy.ndarray, 取值范围是0-255
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
assert train_images.shape == (60000, 28, 28)
assert test_images.shape == (10000, 28, 28)
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
assert train_labels.shape == (60000, 10)
assert test_labels.shape == (10000, 10)

# 构建模型
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28, )))
network.add(layers.Dense(10, activation='softmax'))

# 编译模型
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# 训练
history = network.fit(train_images, train_labels, epochs=5, batch_size=128)

# 评估
test_loss, test_acc = network.evaluate(test_images, test_labels)
print(test_loss, test_acc)

# plot history
history_dict = history.history
loss_values = history_dict['loss']
acc_values = history_dict['accuracy']
epochs = range(1, len(loss_values) + 1)
plt.rcParams['figure.figsize'] = (6, 4)
plt.plot(epochs, loss_values, '-', label='train loss')
plt.plot(epochs, acc_values, 'm--', label='train acc')
plt.xlabel('Epochs')
plt.legend()
plt.grid()
plt.show()
