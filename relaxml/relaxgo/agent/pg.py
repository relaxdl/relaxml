from typing import Any, List
import numpy as np
import h5py
import tensorflow as tf
import tensorflow.keras as keras

from .base import Agent
from .helpers import is_point_an_eye
from ..rl.experience import ExperienceBuffer, ExperienceCollector, load_experience
from ..network import large
from ..encoder.base import get_encoder_by_name
from ..encoder.simple import SimpleEncoder
from .. import goboard
from .. import utils

__all__ = [
    'PolicyAgent',
    'load_policy_agent',
    'policy_gradient_loss',
]

epsilon = keras.backend.epsilon()


def policy_gradient_loss(y_true, y_pred):
    """    
    参数:
    y_true: [batch_size, num_classes], 类似one-hot编码, reward的取值为+1或者-1
        [[0,0,reward,0],
        [0,0,0,reward],
        ...
        [reward,0,0,0]]
    y_pred: [batch_size, num_classes] logits

    返回:
    loss: 标量loss
    """
    # y_pred.shape [batch_size, num_classes]
    y_pred = tf.nn.softmax(y_pred, axis=-1)
    # clip_pred.shape [batch_size, num_classes]
    clip_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
    # 策略梯度计算出来的loss是需要最大化的, 我们希望模型用梯度下降帮我们最小化,
    # 所以前面乘以了: -1
    loss = -1 * y_true * tf.math.log(clip_pred)
    return tf.reduce_mean(tf.reduce_sum(loss, axis=1))


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


class PolicyAgent(Agent):

    def __init__(self,
                 model: keras.Model = None,
                 encoder: Any = None,
                 from_logits=True) -> None:
        Agent.__init__(self)
        if encoder is None:
            encoder = SimpleEncoder((19, 19))
        if model is None:
            model = default_network(encoder)
        self._model = model
        self._encoder = encoder
        self._collector = None  # ExperienceCollector
        self._temperature = 0.0  # Epsilon - Greedy的参数
        self.from_logits = from_logits

    def predict(self, game_state: goboard.GameState) -> np.ndarray:
        """
        预测动作的概率分布
    
        参数:
        game_state: 游戏状态
        
        返回:
        probs: [num_classes, ]
        """
        encoded_state = self._encoder.encode(game_state)
        input_tensor = np.array([encoded_state])
        # probs.shape [1, num_classes]
        probs = self._model.predict(input_tensor)
        if self.from_logits:
            # probs.shape [1, num_classes]
            probs = tf.nn.softmax(probs, axis=-1)
        return probs[0]

    def set_temperature(self, temperature: float) -> None:
        self._temperature = temperature

    def set_collector(self, collector: ExperienceCollector) -> None:
        """
        设置经验收集器
        """
        self._collector = collector

    def select_move(self, game_state: goboard.GameState) -> goboard.Move:
        """
        选择一个Move(Epsilon - Greedy)
    
        参数:
        game_state: 游戏状态

        返回:
        move: 动作
        """
        num_classes = self._encoder.num_points()
        # board_tensor.shape [num_planes, board_height, board_width]
        board_tensor = self._encoder.encode(game_state)
        # x.shape [1, num_planes, board_height, board_width]
        x = np.array([board_tensor])

        if np.random.random() < self._temperature:
            # 随机探索
            # move_probs.shape [num_classes, ]
            move_probs = np.ones(num_classes) / num_classes
        else:
            # move_probs.shape [1, num_classes]
            move_probs = self._model.predict(x)
            if self.from_logits:
                # move_probs.shape [num_classes, ]
                move_probs = tf.nn.softmax(move_probs, axis=-1)[0]

        # 调整预测的概率分布:
        # <1> clipping防止动作概率过于接近0或者1
        # <2> 将结果再次归一化成概率分布
        eps = 1e-5
        move_probs = np.clip(move_probs, eps, 1 - eps)  # <1>
        move_probs = move_probs / np.sum(move_probs)  # <2>

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
            point = self._encoder.decode_point_index(point_idx)
            if game_state.is_valid_move(goboard.Move.play(point)) and \
                    not is_point_an_eye(game_state.board,
                                        point,
                                        game_state.next_player): # <3>
                if self._collector is not None:
                    # 记录一条经验
                    self._collector.record_decision(state=board_tensor,
                                                    action=point_idx)
                return goboard.Move.play(point)
        return goboard.Move.pass_turn()  # <4>

    def train(self,
              experience: ExperienceBuffer,
              lr: float = 0.0000001,
              clipnorm: float = 1.0,
              batch_size: int = 512) -> None:
        """
        用experience的数据来训练模型, 训练一个epoch

        对experience的数据只迭代一个epoch, 不会重复利用experience的数据
        """
        opt = keras.optimizers.SGD(lr=lr, clipnorm=clipnorm)
        self._model.compile(
            loss=keras.losses.CategoricalCrossentropy(from_logits=True),
            optimizer=opt)

        n = experience.states.shape[0]
        num_classes = self._encoder.num_points()
        # y.shape [n, num_classes]
        y = np.zeros((n, num_classes))
        # 遍历所有的样本
        # 构造完成y类似one-hot编码, reward的取值为+1或者-1:
        # [[0,0,reward,0],
        #  [0,0,0,reward],
        #  ...
        #  [reward,0,0,0]]
        for i in range(n):
            action = experience.actions[i]
            reward = experience.rewards[i]
            y[i][action] = reward

        self._model.fit(experience.states, y, batch_size=batch_size, epochs=1)

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
        h5file['encoder'].attrs['name'] = self._encoder.name()
        h5file['encoder'].attrs['board_width'] = self._encoder.board_width
        h5file['encoder'].attrs['board_height'] = self._encoder.board_height
        # 保存model
        h5file.create_group('model')
        utils.save_model_to_hdf5_group(self._model, h5file['model'])


def load_policy_agent(h5file: h5py.File) -> PolicyAgent:
    """
    从h5file中加载Agent

               h5file
             /         \
    encoder(group)  model(group)
                         |
                  kerasmodel(group)
    """
    # 加载model
    model = utils.load_model_from_hdf5_group(
        h5file['model'],
        custom_objects={'policy_gradient_loss': policy_gradient_loss})

    # 加载encoder
    encoder_name = h5file['encoder'].attrs['name']
    if not isinstance(encoder_name, str):
        encoder_name = encoder_name.decode('ascii')
    board_width = h5file['encoder'].attrs['board_width']
    board_height = h5file['encoder'].attrs['board_height']
    encoder = get_encoder_by_name(encoder_name, (board_width, board_height))
    return PolicyAgent(model, encoder)


def init_policy_agent(output_file: str) -> None:
    new_agent = PolicyAgent()
    opt = keras.optimizers.SGD(lr=0.02)
    new_agent._model.compile(loss=policy_gradient_loss, optimizer=opt)
    with h5py.File(output_file, 'w') as output_file_f:
        new_agent.serialize(output_file_f)


def train_policy_agent(agent_in: str,
                       agent_out: str,
                       experience: List[str],
                       lr=0.02,
                       batch_size=256):
    # 加载agent
    with h5py.File(agent_in, 'w') as agent_in_f:
        learning_agent = load_policy_agent(agent_in_f)
    # 训练agent
    for exp_filename in experience:
        print('Training with %s...' % exp_filename)
        exp_buffer = load_experience(h5py.File(exp_filename))
        learning_agent.train(exp_buffer, lr=lr, batch_size=batch_size)
    # 保存agent
    with h5py.File(agent_out, 'w') as agent_out_f:
        learning_agent.serialize(agent_out_f)
