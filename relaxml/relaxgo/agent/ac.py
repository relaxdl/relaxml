from typing import Any, List, Tuple
from matplotlib.pyplot import flag
import numpy as np
import h5py
import tensorflow.keras as keras
from ..encoder import get_encoder_by_name
from ..encoder.simple import SimpleEncoder
from .. import goboard
from .. import utils
from .base import Agent
from .helpers import is_point_an_eye
from ..rl.experience import ExperienceBuffer, ExperienceCollector
from ..network import large

__all__ = [
    'ACAgent',
    'load_ac_agent',
]


def default_network(encoder):
    """
    构建network
    1. 1个输入-`棋盘状态`
    2. 2个输出-`各个动作的概率(策略网络-Actor)`和`当前棋盘状态的Value(价值网络-Critic)`

    输入:
    board_input: [batch_size, num_planes, board_height, board_width]

    输出: [policy_output, value_output]
    policy_output: [batch_size, num_classes] 加了softmax, 输出的是概率
    value_output: [batch_size, 1]
    """
    input_shape = encoder.shape()
    num_classes = encoder.num_points()

    # board input
    board_input = keras.layers.Input(shape=input_shape, name='board_input')

    processed_board = board_input
    for layer in large.layers(input_shape):
        processed_board = layer(processed_board)

    # policy_hidden_layer.shape [batch_size, 1024]
    policy_hidden_layer = keras.layers.Dense(
        1024, activation='relu')(processed_board)
    # policy_output.shape [batch_size, num_classes]
    policy_output = keras.layers.Dense(
        num_classes, activation='softmax')(policy_hidden_layer)

    # value_hidden_layer.shape [batch_size, 1024]
    value_hidden_layer = keras.layers.Dense(1024,
                                            activation='relu')(processed_board)
    # value_output.shape [batch_size, 1]
    value_output = keras.layers.Dense(1, activation='tanh')(value_hidden_layer)

    model = keras.Model(inputs=[board_input],
                        outputs=[policy_output, value_output])
    return model


class ACAgent(Agent):

    def __init__(self, model, encoder):
        """
        参数:
        model: 模型
        encoder: 编码器
        """
        Agent.__init__(self)
        if encoder is None:
            encoder = SimpleEncoder((19, 19))
        if model is None:
            model = default_network(encoder)
        self.model = model
        self.encoder = encoder
        self.collector = None  # ExperienceCollector
        self.temperature = 1.0
        self.last_state_value = 0

    def set_temperature(self, temperature: float) -> None:
        self.temperature = temperature

    def set_collector(self, collector: ExperienceCollector) -> None:
        """
        设置经验收集器
        """
        self.collector = collector

    def select_move(self, game_state: goboard.GameState) -> goboard.Move:
        """
        选择一个Move
    
        参数:
        game_state: 游戏状态

        返回:
        move: 动作
        """
        num_classes = self.encoder.num_points()
        # board_tensor.shape [num_planes, board_height, board_width]
        board_tensor = self.encoder.encode(game_state)
        # x.shape [1, num_planes, board_height, board_width]
        x = np.array([board_tensor])
        # actions.shape [1, num_classes]
        # values.shape [1, 1]
        actions, values = self.model.predict(x)
        # move_probs.shape [num_classes, ]
        move_probs = actions[0]
        # estimated_value: 标量
        estimated_value = values[0][0]
        self.last_state_value = float(estimated_value)

        # 调整预测的概率分布
        # <1> clipping防止动作概率过于接近0或者1
        # <2> 将结果再次归一化成概率分布
        move_probs = np.power(move_probs, 1.0 / self.temperature)
        move_probs = move_probs / np.sum(move_probs)
        eps = 1e-6
        move_probs = np.clip(move_probs, eps, 1 - eps)
        move_probs = move_probs / np.sum(move_probs)

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
                    not is_point_an_eye(game_state.board,
                                        point,
                                        game_state.next_player): # <3>
                if self.collector is not None:
                    # 记录一条经验
                    self.collector.record_decision(
                        state=board_tensor,
                        action=point_idx,
                        estimated_value=estimated_value)
                return goboard.Move.play(point)
        return goboard.Move.pass_turn()  # <4>

    def train(self,
              experience: ExperienceBuffer,
              lr: float = 0.1,
              batch_size: int = 128) -> None:
        """
        用experience的数据来训练模型, 训练一个epoch

        对experience的数据只迭代一个epoch, 不会重复利用experience的数据
        
        当网络有多个输出时, 可以为每个输出选择不同的损失函数
        1. 为策略网络选择交叉熵损失函数(变相替代策略梯度)
        2. 为价值网络选择均方误差损失函数
        """
        opt = keras.optimizers.SGD(lr=lr, clipvalue=0.2)
        # 策略网络Loss权重: 1.0
        # 价值网络Loss权重: 0.5
        self.model.compile(optimizer=opt,
                           loss=['categorical_crossentropy', 'mse'],
                           loss_weights=[1.0, 0.5])

        n = experience.states.shape[0]
        num_classes = self.encoder.num_points()
        # policy_target.shape [n, num_classes]
        policy_target = np.zeros((n, num_classes))
        # value_target.shape [n, ]
        value_target = np.zeros((n, ))
        # 遍历所有的样本
        # 构造完成policy_target类似one-hot编码:
        # [[0,0,advantages,0],
        #  [0,0,0,advantages],
        #  ...
        #  [advantages,0,0,0]]
        for i in range(n):
            action = experience.actions[i]
            reward = experience.rewards[i]
            policy_target[i][action] = -experience.advantages[i]
            value_target[i] = reward

        self.model.fit(experience.states, [policy_target, value_target],
                       batch_size=batch_size,
                       epochs=1)

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
        # 保存model
        h5file.create_group('model')
        utils.save_model_to_hdf5_group(self.model, h5file['model'])

    def diagnostics(self) -> Tuple[str, float]:
        return {'value': self.last_state_value}


def load_ac_agent(h5file):
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
    return ACAgent(model, encoder)


def init_ac_agent(output_file: str) -> None:
    new_agent = ACAgent()
    with h5py.File(output_file, 'w') as output_file_f:
        new_agent.serialize(output_file_f)