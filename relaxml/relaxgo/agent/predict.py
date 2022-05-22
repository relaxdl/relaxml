from typing import Any
import numpy as np
import h5py
import tensorflow as tf
import tensorflow.keras as keras
from ..agent.base import Agent
from ..agent.helpers import is_point_an_eye
from ..network import large
from ..encoder.base import get_encoder_by_name
from ..encoder.simple import SimpleEncoder
from .. import goboard
from .. import utils

__all__ = [
    'DeepLearningAgent',
    'load_prediction_agent',
]


def default_network(encoder):
    """
    构建network
    """
    input_shape = encoder.shape()
    network_layers = large.layers(input_shape)

    model = keras.Sequential()
    for layer in network_layers:
        model.add(layer)
    model.add(keras.layers.Dense(encoder.num_points()))  # num_classes
    return model


class DeepLearningAgent(Agent):
    """
    一个Agent核心包含2部分信息: model & encoder
    1. encoder负责编码game state的信息
    2. model负责预测

    可以嵌入任意组合的model & encoder
    """

    def __init__(self,
                 model: keras.Model,
                 encoder: Any,
                 from_logits: bool = True) -> None:
        """
        参数:
        model: 模型
        encoder: 编码器
        from_logits: 预测的时候model输出的参数是否是logits
        """
        Agent.__init__(self)
        if encoder is None:
            encoder = SimpleEncoder((19, 19))
        if model is None:
            model = default_network(encoder)
        self.model = model
        self.encoder = encoder
        self.from_logits = from_logits

    def predict(self, game_state: goboard.GameState) -> np.ndarray:
        """
        预测动作的概率分布
    
        参数:
        game_state: 游戏状态
        
        返回:
        probs: [num_classes, ]
        """
        encoded_state = self.encoder.encode(game_state)
        input_tensor = np.array([encoded_state])
        # probs.shape [1, num_classes]
        probs = self.model.predict(input_tensor)
        if self.from_logits:
            # probs.shape [1, num_classes]
            probs = tf.nn.softmax(probs, axis=-1)
        return probs[0]

    def select_move(self, game_state: goboard.GameState) -> goboard.Move:
        """
        选择一个Move
    
        参数:
        game_state: 游戏状态

        返回:
        move: 动作
        """
        num_classes = self.encoder.num_points()
        # move_probs.shape [num_classes, ]
        move_probs = self.predict(game_state)

        # 调整预测的概率分布:
        # <1> 大幅增加可能性较大和可能性较小的动作之间的距离, 希望尽可能多的选择最好的动作
        # <2> clipping防止动作概率过于接近0或者1
        # <3> 将结果再次归一化成概率分布
        move_probs = move_probs**3  # <1>
        eps = 1e-6
        move_probs = np.clip(move_probs, eps, 1 - eps)  # <2>
        move_probs = move_probs / np.sum(move_probs)  # <3>

        # 根据最新的概率分布进行抽样:
        # <1> 候选动作列表
        # <2> 使用不放回采样num_classes次, 因为有可能一次采样选出来的动作是不合法的
        # <3> 选出来的动作是合法的, 并且不能是自己的眼
        # <4> 如果无法选出合理动作, 则pass
        candidates = np.arange(num_classes)  # <1>
        ranked_moves = np.random.choice(candidates,
                                        num_classes,
                                        replace=False,
                                        p=move_probs)  # <2>
        for point_idx in ranked_moves:
            point = self.encoder.decode_point_index(point_idx)
            if game_state.is_valid_move(goboard.Move.play(point)) and \
                    not is_point_an_eye(game_state.board, point, game_state.next_player):  # <3>
                return goboard.Move.play(point)
        return goboard.Move.pass_turn()  # <4>

    def serialize(self, h5file: h5py.File) -> None:
        """
        保存Agent(也就是保存Encoder信息 & model)到h5file

                  h5file
                /         \
         encoder(group)  model(group)
                             |
                      kerasmodel(group)

        >>> with h5py.File(output_file, 'w') as out_f:
        >>>     agent.serialize(out_f)
        """
        # 保存encoder信息(根据这些信息可以重构出encoder)
        h5file.create_group('encoder')
        h5file['encoder'].attrs['name'] = self.encoder.name()
        h5file['encoder'].attrs['board_width'] = self.encoder.board_width
        h5file['encoder'].attrs['board_height'] = self.encoder.board_height
        h5file.create_group('model')
        # 保存model
        utils.save_model_to_hdf5_group(self.model, h5file['model'])


def load_prediction_agent(h5file: h5py.File) -> DeepLearningAgent:
    """
    从h5file中加载Agent

               h5file
             /         \
    encoder(group)  model(group)
                         |
                  kerasmodel(group)
    """
    # 加载model
    model = utils.load_model_from_hdf5_group(h5file['model'])

    # 加载encoder
    encoder_name = h5file['encoder'].attrs['name']
    if not isinstance(encoder_name, str):
        encoder_name = encoder_name.decode('ascii')
    board_width = h5file['encoder'].attrs['board_width']
    board_height = h5file['encoder'].attrs['board_height']
    encoder = get_encoder_by_name(encoder_name, (board_width, board_height))

    return DeepLearningAgent(model, encoder)