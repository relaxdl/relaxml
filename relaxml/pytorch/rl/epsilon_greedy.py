import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class CoinToss():
    """
    多臂老虎机环境

    1. 环境中有多枚硬币, 每枚硬币正面朝上的概率不一样, 有最大投掷次数限制
    2. 动作就是选择要投掷的硬币, 进行投掷
    3. 奖励: 正面朝上奖励为1; 反面朝上奖励为0
    """

    def __init__(self, head_probs, max_episode_steps=30):
        """
        参数:
        head_probs: 数组参数, 用于指定各枚硬币正面朝上的概率, 
            e.g. [0.1, 0.8, 0.3]
        max_episode_steps: 硬币的最大投掷次数
        """
        self.head_probs = head_probs
        self.max_episode_steps = max_episode_steps
        self.toss_count = 0  # 当前投掷次数

    def __len__(self):
        return len(self.head_probs)

    def reset(self):
        self.toss_count = 0

    def step(self, action):
        """
        选择要投掷的硬币, 正面朝上奖励为1; 反面朝上奖励为0

        参数:
        action: 投掷硬币的索引, [0, 硬币个数-1]

        返回: reward, done
        """
        final = self.max_episode_steps - 1
        if self.toss_count > final:
            raise Exception("The step count exceeded maximum. \
                            Please reset env.")
        else:
            done = True if self.toss_count == final else False

        if action >= len(self.head_probs):
            raise Exception("The No.{} coin doesn't exist.".format(action))
        else:
            # 选择要投掷的硬币
            head_prob = self.head_probs[action]
            if random.random() < head_prob:
                reward = 1.0
            else:
                reward = 0.0
            self.toss_count += 1
            return reward, done


class EpsilonGreedyAgent():
    """
    基于Epsilon-Greedy作为策略的Agent
    """

    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.V = []  # 保存每枚硬币期望价值

    def policy(self):
        """
        实现Epsilon-Greedy
        
        epsilon的概率随机选择硬币(探索); 其他情况按照各硬币的期望价值来选择(利用)

        """
        coins = range(len(self.V))
        if random.random() < self.epsilon:
            # 探索
            return random.choice(coins)
        else:
            # 根据价值选择
            return np.argmax(self.V)

    def play(self, env):
        """
        每个硬币价值的期望 = 奖励/次数
    
        参数:
        env: CoinToss

        返回:
        rewards: 总奖励列表
        """
        # 初始化估计值
        N = [0] * len(env)  # 每个硬币对应的投掷次数
        self.V = [0] * len(env)  # 初始化每个硬币的价值

        env.reset()
        done = False
        rewards = []  # 总奖励
        # 每迭代一次, 会更新一个硬币的价值
        while not done:
            # 选择一个硬币
            selected_coin = self.policy()
            reward, done = env.step(selected_coin)
            rewards.append(reward)

            # 重新计算这个硬币的`期望价值`
            n = N[selected_coin]
            coin_average = self.V[selected_coin]
            new_average = (coin_average * n + reward) / (n + 1)

            # 更新`这个硬币对应的投掷次数`和`这个硬币的价值`
            N[selected_coin] += 1
            self.V[selected_coin] = new_average

        return rewards


def main():
    """
    1. 5枚硬币
    2. 设置5组不同的epsilons, 代表不同的policy
    3. 横轴是game steps次数, 也就是投掷硬币的次数
    4. 纵轴是所有game steps奖励的均值, 可以用来衡量policy的好坏
    """
    # 5枚硬币
    env = CoinToss([0.1, 0.5, 0.1, 0.9, 0.1])
    epsilons = [0.0, 0.1, 0.2, 0.5, 0.8]
    game_steps = list(range(10, 310, 10))
    result = {}
    for e in epsilons:
        agent = EpsilonGreedyAgent(epsilon=e)
        means = []
        for s in game_steps:
            env.max_episode_steps = s
            rewards = agent.play(env)
            means.append(np.mean(rewards))
        result["epsilon={}".format(e)] = means
    result["coin toss count"] = game_steps
    result = pd.DataFrame(result)
    result.set_index("coin toss count", drop=True, inplace=True)
    result.plot.line(figsize=(10, 5))
    plt.show()


if __name__ == "__main__":

    main()
