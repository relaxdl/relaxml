from pickletools import optimize
from random import triangular
from torch.distributions import Categorical
import gym
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim

gamma = 0.99


class Pi(nn.Module):
    """
    策略网络 - 模拟`策略函数Pi`
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        """
        参数:
        in_dim: state的维度, 当前例子=4
        out_dim: 动作的数量, 当前例子=2
        """
        super().__init__()
        layers = [nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, out_dim)]
        self.model = nn.Sequential(*layers)
        self.onpolicy_reset()
        self.train()  # 设置为训练模式

    def onpolicy_reset(self):
        self.log_probs = []
        self.rewards = []

    def forward(self, x: Tensor) -> Tensor:
        """
        参数:
        x: [in_dim, ] 环境state

        返回:
        pdparam: [out_dim, ] 每个action的分数
        """
        pdparam = self.model(x)
        return pdparam

    def act(self, state: np.ndarray):
        """
        策略网络采样一个action

        参数:
        state: [in_dim, ] e.g. [ 0.02228096 -0.0025914  -0.02542181  0.02622841]

        返回:
        action: TODO
        """
        x = torch.from_numpy(state.astype(np.float32))  # to tensor
        pdparam = self.forward(x)
        pd = Categorical(logits=pdparam)
        action = pd.sample()  # pd = pi(a|s)
        log_prob = pd.log_prob(action)
        self.log_probs.append(log_prob)  # 保存下来后续训练时使用
        return action.item()


def train(pi: nn.Module, optimizer: optim.Optimizer):
    """
    每个Episode结束之后训练一次, 更新策略网络
    """
    pass


def main():
    env = gym.make('CartPole-v1')
    in_dim = env.observation_space.shape[0]  # 4
    out_dim = env.action_space.n  # 2
    pi = Pi(in_dim, out_dim)  # 策略函数Pi
    optimizer = optim.Adam(pi.parameters(), lr=0.01)
    for epi in range(3):
        state = env.reset()
        for t in range(200):  # CartPole的最大timestep是200
            action = pi.act(state)
            state, reward, done, _ = env.step(action)
            pi.rewards.append(reward)
            env.render()
            if done:
                break

        loss = train(pi, optimizer)  # train per episode
        total_reward = sum(pi.rewards)
        solved = total_reward > 195.0
        pi.onpolicy_reset()  # onplicy: clear memory after training
        print(f'Episode {epi}, loss: {loss} \
                total_reward: {total_reward}, solved: {solved}')


if __name__ == '__main__':
    main()
