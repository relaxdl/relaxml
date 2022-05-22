from typing import Any, List, Tuple
import numpy as np
import h5py
import tensorflow.keras as keras
from ..encoder import get_encoder_by_name
from .. import goboard
from .. import utils
from .base import Agent
from .helpers import is_point_an_eye
from ..rl.experience import ExperienceBuffer, ExperienceCollector

__all__ = [
    'ValueAgent',
    'load_value_agent',
]


class ValueAgent(Agent):

    def __init__(self,
                 model: keras.Model,
                 encoder: Any,
                 policy: str = 'eps-greedy') -> None:
        Agent.__init__(self)
        """
        参数:
        model: 模型
        encoder: 编码器
        policy: eps-greedy | weighted
          eps-greedy epsilon-greedy的方式选择动作
          weighted 将values转换成概率, 然后抽样得到动作
        """
        self.model = model
        self.encoder = encoder
        self.collector = None  # ExperienceCollector
        self.temperature = 0.0  # Epsilon - Greedy的参数
        self.policy = policy
        self.last_move_value = 0

    def predict(self, game_state: goboard.GameState) -> np.ndarray:
        """
        预测一个状态的价值

        返回:
        output: [1, ]
        """
        encoded_state = self.encoder.encode(game_state)
        input_tensor = np.array([encoded_state])
        return self.model.predict(input_tensor)[0]

    def set_temperature(self, temperature: float) -> None:
        self.temperature = temperature

    def set_collector(self, collector: ExperienceCollector) -> None:
        """
        设置经验收集器
        """
        self.collector = collector

    def set_policy(self, policy: str) -> None:
        if policy not in ('eps-greedy', 'weighted'):
            raise ValueError(policy)
        self.policy = policy

    def select_move(self, game_state: goboard.GameState) -> goboard.Move:
        """
        选择一个Move
    
        参数:
        game_state: 游戏状态

        返回:
        move: 动作
        """

        moves = []  # list of int
        board_tensors = [
        ]  # list of [num_planes, board_height, board_width], 每个元素是不同的
        # 遍历所有合法的动作
        for move in game_state.legal_moves():
            if not move.is_play:
                continue
            next_state = game_state.apply_move(move)
            # board_tensor.shape [num_planes, board_height, board_width]
            board_tensor = self.encoder.encode(next_state)
            moves.append(move)
            board_tensors.append(board_tensor)
        if not moves:
            return goboard.Move.pass_turn()

        num_moves = len(moves)
        # board_tensors.shape [num_moves, num_planes, board_height, board_width]
        board_tensors = np.array(board_tensors)

        # 要注意:
        # 这是对方视角的value, 并不是自己的value
        # opp_values.shape [num_moves, 1]
        opp_values = self.model.predict(board_tensors)
        # opp_values.shape [num_moves, ]
        opp_values = opp_values.reshape(num_moves)

        # 转换为自己的values
        # values.shape [num_moves, ]
        values = 1 - opp_values

        if self.policy == 'eps-greedy':
            ranked_moves = self.rank_moves_eps_greedy(values)
        elif self.policy == 'weighted':
            ranked_moves = self.rank_moves_weighted(values)
        else:
            ranked_moves = None

        for move_idx in ranked_moves:
            move = moves[move_idx]
            if not is_point_an_eye(game_state.board, move.point,
                                   game_state.next_player):
                # 选出来的动作不能是自己的眼
                if self.collector is not None:
                    self.collector.record_decision(
                        state=board_tensor,
                        action=self.encoder.encode_point(move.point),
                    )
                self.last_move_value = float(values[move_idx])
                return move
        return goboard.Move.pass_turn()

    def rank_moves_eps_greedy(self, values: List[float]) -> List[int]:
        """
        epsilon-greedy的方式

        参数:
        values: [num_moves, ]

        返回:
        ranked_moves: [num_moves, ] 从大到小排序后的索引
        """
        if np.random.random() < self.temperature:
            # 随机动作
            values = np.random.random(values.shape)
        # 挑选最优的动作
        ranked_moves = np.argsort(values)
        return ranked_moves[::-1]

    def rank_moves_weighted(self, values: List[float]) -> List[int]:
        """
        将values转换成概率, 然后根据概率大小采样

        参数:
        values: [num_moves, ]

        返回:
        ranked_moves: [num_moves, ] 从大到小排序后的索引
        """
        # 将values转换成概率: p
        # p.shape [num_moves, ]
        p = values / np.sum(values)
        p = np.power(p, 1.0 / self.temperature)
        p = p / np.sum(p)
        return np.random.choice(np.arange(0, len(values)),
                                size=len(values),
                                p=p,
                                replace=False)

    def train(self,
              experience: ExperienceBuffer,
              lr: float = 0.1,
              batch_size: int = 128) -> None:
        """
        用experience的数据来训练模型, 训练一个epoch

        对experience的数据只迭代一个epoch, 不会重复利用experience的数据
        """
        opt = keras.optimizers.SGD(lr=lr)
        # 直接使用MSE误差, 因为我们要学习的是一个连续值
        self.model.compile(loss='mse', optimizer=opt)

        n = experience.states.shape[0]
        # y.shape [n, ]
        y = np.zeros((n, ))
        for i in range(n):
            reward = experience.rewards[i]
            y[i] = 1 if reward > 0 else 0

        self.model.fit(experience.states, y, batch_size=batch_size, epochs=1)

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
        return {'value': self.last_move_value}


def load_value_agent(h5file: h5py.File) -> ValueAgent:
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
    encoder_name = h5file['encoder'].attrs['name']
    if not isinstance(encoder_name, str):
        encoder_name = encoder_name.decode('ascii')
    board_width = h5file['encoder'].attrs['board_width']
    board_height = h5file['encoder'].attrs['board_height']

    # 加载encoder
    encoder = get_encoder_by_name(encoder_name, (board_width, board_height))
    return ValueAgent(model, encoder)