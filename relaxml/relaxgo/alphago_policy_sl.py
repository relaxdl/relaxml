#! /usr/bin/env python3
# -*- coding:utf-8 -*-
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relaxgo.data.parallel_processor import GoDataProcessor
from relaxgo.encoder.simple import SimpleEncoder
from relaxgo.agent.predict import DeepLearningAgent
from relaxgo.network.alphago import alphago_model
import tensorflow.keras as keras
from tensorflow.keras.callbacks import ModelCheckpoint
import h5py
"""
用人工棋谱训练策略网络

1. 构建 & 训练模型
   <1> Encoder
   <2> Data Processor
   <3> Model [策略网络]
2. 用训练好的策略网络创建agent
   <1> 创建agent
   <2> 将agent保存到磁盘: alphago_sl_policy.h5
"""

if __name__ == '__main__':
    """
    为了快速训练, 注释掉了测试样本
    """
    rows, cols = 19, 19
    num_classes = rows * cols
    num_train_games, num_test_games = 200, 100
    epochs, batch_size = 10, 128
    # 1. 构建 & 训练模型
    # encoder = AlphaGoEncoder()
    encoder = SimpleEncoder((rows, cols))  # <1>
    input_shape = encoder.shape()
    processor = GoDataProcessor(encoder=encoder.name())  # <2>
    generator = processor.load_go_data('train',
                                       num_train_games,
                                       use_generator=True)
    # test_generator = processor.load_go_data('test',
    #                                         num_test_games,
    #                                         use_generator=True)
    print('load data success')
    print(f'train_samples: {generator.get_num_samples() }')
    # print(f'test_samples: {test_generator.get_num_samples()}')
    print(f'num_classes: {num_classes}')
    print(f'input_shape: {input_shape}')

    # 模型输入: [batch_size, num_planes, board_height, board_width]
    # 模型输出: [batch_size, num_classes=board_height*board_width]
    alphago_sl_policy = alphago_model(input_shape, is_policy_net=True)  # <3>

    alphago_sl_policy.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer=keras.optimizers.SGD(learning_rate=0.01),
        metrics=['accuracy'])
    alphago_sl_policy.fit(
        generator.generate(batch_size),
        epochs=epochs,
        steps_per_epoch=generator.get_num_samples() / batch_size,
        # validation_data=test_generator.generate(batch_size),
        # validation_steps=test_generator.get_num_samples() / batch_size,
        callbacks=[ModelCheckpoint('alphago_sl_policy_{epoch}.h5')])

    # 2. 用训练好的策略网络创建agent
    alphago_sl_agent = DeepLearningAgent(alphago_sl_policy,
                                         encoder,
                                         from_logits=False)  # <1>

    with h5py.File('alphago_sl_policy.h5', 'w') as sl_agent_out:
        alphago_sl_agent.serialize(sl_agent_out)  # <2>
