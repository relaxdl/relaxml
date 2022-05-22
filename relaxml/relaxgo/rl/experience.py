from typing import List
import numpy as np
import h5py

__all__ = [
    'ExperienceCollector',
    'ExperienceBuffer',
    'combine_experience',
    'load_experience',
]


class ExperienceCollector:
    """
    经验收集器
    """

    def __init__(self):
        # 多组数据, 可以跨越多个episode
        self.states = []
        self.actions = []
        self.rewards = []
        self.advantages = []
        # 一个episode的数据, 每个episode会重置
        self._current_episode_states = []
        self._current_episode_actions = []
        self._current_episode_estimated_values = []

    def begin_episode(self):
        """
        一个episode开始, 清空缓存【一局游戏开始调用】
        """
        self._current_episode_states = []
        self._current_episode_actions = []
        self._current_episode_estimated_values = []

    def record_decision(self,
                        state: np.ndarray,
                        action: int,
                        estimated_value: int = 0) -> None:
        """
        记录一条经验数据【每个action调用】

        参数:
        state: 状态 [num_planes, board_height, board_width]
        action: 动作
        estimated_value: 奖励
        """
        self._current_episode_states.append(state)
        self._current_episode_actions.append(action)
        self._current_episode_estimated_values.append(estimated_value)

    def complete_episode(self, reward: int) -> None:
        """
        一个episode结束, 将这个episode的经验合并到总经验buffer上【一局游戏结束调用】
        """
        num_states = len(self._current_episode_states)
        self.states += self._current_episode_states
        self.actions += self._current_episode_actions
        self.rewards += [reward for _ in range(num_states)]  # 这一局的动作都会获得reward

        for i in range(num_states):
            # 计算这一个episode的`优势`
            # U - V
            advantage = reward - self._current_episode_estimated_values[i]
            self.advantages.append(advantage)

        self._current_episode_states = []
        self._current_episode_actions = []
        self._current_episode_estimated_values = []


class ExperienceBuffer:

    def __init__(self, states, actions, rewards, advantages):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.advantages = advantages

    def serialize(self, h5file: h5py.File) -> None:
        """
        将收集到的经验保存到h5file
        """
        h5file.create_group('experience')
        h5file['experience'].create_dataset('states', data=self.states)
        h5file['experience'].create_dataset('actions', data=self.actions)
        h5file['experience'].create_dataset('rewards', data=self.rewards)
        h5file['experience'].create_dataset('advantages', data=self.advantages)


def combine_experience(
        collectors: List[ExperienceCollector]) -> ExperienceBuffer:
    """
    合并多个经验收集器收集的经验
    """
    combined_states = np.concatenate([np.array(c.states) for c in collectors])
    combined_actions = np.concatenate(
        [np.array(c.actions) for c in collectors])
    combined_rewards = np.concatenate(
        [np.array(c.rewards) for c in collectors])
    combined_advantages = np.concatenate(
        [np.array(c.advantages) for c in collectors])

    return ExperienceBuffer(combined_states, combined_actions,
                            combined_rewards, combined_advantages)


def load_experience(h5file: h5py.File) -> ExperienceBuffer:
    """
    加载经验
    """
    return ExperienceBuffer(states=np.array(h5file['experience']['states']),
                            actions=np.array(h5file['experience']['actions']),
                            rewards=np.array(h5file['experience']['rewards']),
                            advantages=np.array(
                                h5file['experience']['advantages']))
