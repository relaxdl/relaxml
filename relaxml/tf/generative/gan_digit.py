import time
import glob
import imageio
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
"""
GAN(生成手写数字)

说明:
https://tech.foxrelax.com/generative/gan_digit/
"""

latent_dim = 100
width, height, channels = 28, 28, 1


def load_data_mnist(batch_size=256):
    """
    加载mnist数据集
    >>> train_iter = load_data_mnist()
    >>> for x in train_iter:
    >>>     assert x.shape == (256, 28, 28, 1)
    >>>     break
    """
    (train_images, _), (_, _) = keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28,
                                        1).astype('float32')
    # 原始数据的范围是[0 - 255], Normalize the images to [-1, 1]
    train_images = (train_images - 127.5) / 127.5
    assert train_images.shape == (60000, 28, 28, 1)

    train_iter = tf.data.Dataset.from_tensor_slices(train_images).shuffle(
        60000).batch(batch_size)
    return train_iter


def make_generator_model():
    """
    生成器
    
    将潜在空间的`latent_dim`维向量, 转换为一张`候选图像`

    >>> generator = make_generator_model()
    >>> x = tf.random.normal((2, 100))
    >>> assert generator(x).shape == (2, 28, 28, 1)

    输入:
    x: [batch_size, latent_dim=100]

    输出:
    y: [batch_size, 28, 28, 1]
    """

    model = keras.Sequential()
    # output [batch_size, 7 * 7 * 256 = 12544]
    model.add(
        layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(latent_dim, )))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    # output [batch_size, 7, 7, 256]
    model.add(layers.Reshape((7, 7, 256)))

    # output [batch_size, 7, 7, 128]
    model.add(
        layers.Conv2DTranspose(128, (5, 5),
                               strides=(1, 1),
                               padding='same',
                               use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # output [batch_size, 14, 14, 64]
    model.add(
        layers.Conv2DTranspose(64, (5, 5),
                               strides=(2, 2),
                               padding='same',
                               use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # output [batch_size, 28, 28, 1]
    model.add(
        layers.Conv2DTranspose(1, (5, 5),
                               strides=(2, 2),
                               padding='same',
                               use_bias=False,
                               activation='tanh'))

    return model


def make_discriminator_model():
    """
    判别器
    
    输入一张候选图像(真实的或者合成的), 输出图像的真假(二分类)
    "生成图像"或"来自训练集的真实图像"

    >>> discriminator = make_discriminator_model()
    >>> x = tf.random.normal((2, 28, 28, 1))
    >>> assert discriminator(x).shape == (2, 1)

    输入:
    x: [batch_size, 28, 28, 1]

    输出:
    y: [batch_sizes, 1]
    """
    model = keras.Sequential()
    # output [batch_size, 14, 14, 64]
    model.add(
        layers.Conv2D(64, (5, 5),
                      strides=(2, 2),
                      padding='same',
                      input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    # output [batch_size, 7, 7, 128]
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    # output [batch_size, 6272]
    model.add(layers.Flatten())

    # output [batch_size, 1]
    model.add(layers.Dense(1))

    return model


# 默认模型的输出是logit
cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)  # 真实图片
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)  # 伪造图片
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


@tf.function
def train_batch(generator, discriminator, generator_loss, discriminator_loss,
                generator_optimizer, discriminator_optimizer, images):
    """
    参数:
    images: [batch_size, 28, 28, 1]
    """
    batch_size = images.shape[0]
    noise = tf.random.normal([batch_size, latent_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # 生成器伪造的图像
        # generated_images.shape [batch_size, 28, 28, 1]
        generated_images = generator(noise, training=True)

        # 将`真实图像`和`伪造图像`一起送入判别器
        # real_output.shape [batch_size, 1]
        real_output = discriminator(images, training=True)
        # fake_output.shape [batch_size, 1]
        fake_output = discriminator(generated_images, training=True)

        # 计算生成器的损失
        gen_loss = generator_loss(fake_output)

        # 计算判别器的损失
        disc_loss = discriminator_loss(real_output, fake_output)

    # 根据生成器的loss, 计算生成器权重的梯度
    gradients_of_generator = gen_tape.gradient(gen_loss,
                                               generator.trainable_variables)

    # 根据判别器的loss, 计算判别器权重的梯度
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables)

    # 更新生成器的权重
    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables))

    # 更判别器的权重
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables))


# 生成种子, 在训练循环中不断的可视化这个种子生成的图像, 最终做成一个GIF
num_examples_to_generate = 16
# seed.shape [16, latent_dim]
seed = tf.random.normal([num_examples_to_generate, latent_dim])


def generate_and_save_images(model, epoch, test_input):
    """
    根据种子生成一张图片: image_at_step_xx.png

    参数:
    model: generator
    step: 训练了多少步
    test_input: [16, latent_dim] seed
    """
    # predictions.shape [16, 28, 28, 1]
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))
    # 画4x4的网格来展示16张图片
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))


def train():
    batch_size, num_epochs = 256, 50
    # 加载训练集
    train_iter = load_data_mnist(batch_size=256)

    # 定义模型
    generator = make_generator_model()
    discriminator = make_discriminator_model()

    # 定义优化器
    generator_optimizer = keras.optimizers.Adam(1e-4)
    discriminator_optimizer = keras.optimizers.Adam(1e-4)

    for epoch in range(num_epochs):
        start = time.time()

        for images in train_iter:
            train_batch(generator, discriminator, generator_loss,
                        discriminator_loss, generator_optimizer,
                        discriminator_optimizer, images)

        generate_and_save_images(generator, epoch + 1, seed)

        print('Time for epoch {} is {} sec'.format(epoch + 1,
                                                   time.time() - start))

    generate_and_save_images(generator, num_epochs, seed)


def build_gif():
    anim_file = 'gan_digit.gif'

    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob('image*.png')
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)


if __name__ == '__main__':
    train()
