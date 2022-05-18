#! /usr/bin/env python3
# -*- coding:utf-8 -*-
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relaxgo.data.parallel_processor import GoDataProcessor
from relaxgo.encoder.simple import SimpleEncoder
from relaxgo.network import large
import tensorflow.keras as keras


def network(input_shape, num_classes):
    """
    构建network
    """
    network_layers = large.layers(input_shape)

    model = keras.Sequential()
    for layer in network_layers:
        model.add(layer)
    model.add(keras.layers.Dense(num_classes))
    return model


def load_data(num_train_games=2000, num_test_games=200):
    go_board_rows, go_board_cols = 19, 19
    num_classes = go_board_rows * go_board_cols
    encoder = SimpleEncoder((go_board_rows, go_board_cols))
    input_shape = (encoder.num_planes, go_board_rows, go_board_cols)
    processor = GoDataProcessor(encoder=encoder.name())
    generator = processor.load_go_data('train',
                                       num_train_games,
                                       use_generator=True)
    test_generator = processor.load_go_data('test',
                                            num_test_games,
                                            use_generator=True)
    print('load data success')
    print(f'train_samples: {generator.get_num_samples() }')
    print(f'test_samples: {test_generator.get_num_samples()}')
    print(f'num_classes: {num_classes}')
    print(f'input_shape: {input_shape}')
    return generator, test_generator, num_classes, input_shape


if __name__ == '__main__':
    epochs, num_train_games, num_test_games, batch_size = 10, 200, 200, 256
    generator, test_generator, num_classes, input_shape = load_data(
        num_train_games, num_test_games)
    model = network(input_shape, num_classes)
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.SGD(learning_rate=0.01),
        metrics=['accuracy'])

    model.fit(generator.generate(batch_size),
              epochs=epochs,
              steps_per_epoch=generator.get_num_samples() / batch_size,
              validation_data=test_generator.generate(batch_size),
              validation_steps=test_generator.get_num_samples() / batch_size,
              callbacks=[
                  keras.callbacks.ModelCheckpoint(
                      '../data/checkpoints/model_epoch_{epoch}.h5')
              ])
