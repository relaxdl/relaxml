from enum import Enum
import numpy as np
import random


class State():
    """
    状态就是Agent当前所在的位置
    """

    def __init__(self, row=-1, column=-1):
        self.row = row
        self.column = column

    def __repr__(self):
        return "<State: [{}, {}]>".format(self.row, self.column)

    def clone(self):
        return State(self.row, self.column)

    def __hash__(self):
        return hash((self.row, self.column))

    def __eq__(self, other):
        return self.row == other.row and self.column == other.column


class Action(Enum):
    """
    行动分为: 上下左右四个动作
    """
    UP = 1
    DOWN = -1
    LEFT = 2
    RIGHT = -2


class Environment():

    def __init__(self, grid, move_prob=0.8):
        """
        参数:
        grid: grid是一个二维数组, 传入不同的grid, 会生成不同的格子
          0 -> 普通格子
          -1-> 有危险的格子(游戏结束)
          1 -> 有奖励的格子(游戏结束)
          9 -> 被屏蔽的格子(无法放置智能体)
        move_prob: 真正执行一个动作的概率. Agent能够以move_prob的概率向所选方向
                   移动, 如果概率值在(1 - move_prob)内, 则意味着Agent将随机移动到
                   不同的方向, 为环境增加随机性
        """
        self.grid = grid
        self.agent_state = State()  # 当前的state

        # 每走一步, 如果没有到达绿色格子, 默认的奖励是负数, 就像施加了
        # 惩罚, 这意味着Agent必须快速到达终点
        self.default_reward = -0.04

        self.move_prob = move_prob
        self.reset()

    @property
    def row_length(self):
        return len(self.grid)

    @property
    def column_length(self):
        return len(self.grid[0])

    @property
    def actions(self):
        return [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]

    @property
    def states(self):
        """
        返回所有合法的state
        """
        states = []
        for row in range(self.row_length):
            for column in range(self.column_length):
                # state中不包含被屏蔽的格子
                if self.grid[row][column] != 9:
                    states.append(State(row, column))
        return states

    def transit_func(self, state, action):
        """    
        状态迁移函数(Transition Function)

        - 如果agent尝试向北移动, 我们假设有80%的时间(即0.8的概率)它会按照计划移动(假设途中没有墙)
        - 有10%的时间(即0.1的概率), 试图向北移动会导致agent向西移动(假设途中没有墙)
        - 有10%的时间(即0.1的概率), 试图向北移动会导致agent向东移动(假设途中没有墙)
        - 如果碰到墙, 则agent会停在原地

        参数:
        state: t时刻的state
        action: t时刻的action

        返回:
        transition_probs: t+1时刻每个state对应的概率
        """
        transition_probs = {}  # t+1时刻每个state对应的概率
        if not self.can_action_at(state):
            # 已经到达游戏结束的格子, 不能再继续做动作了
            return transition_probs

        # 反向动作
        opposite_direction = Action(action.value * -1)

        for a in self.actions:
            prob = 0
            if a == action:
                prob = self.move_prob
            elif a != opposite_direction:
                prob = (1 - self.move_prob) / 2

            # t+1时刻的state
            next_state = self._move(state, a)
            if next_state not in transition_probs:
                transition_probs[next_state] = prob
            else:
                transition_probs[next_state] += prob

        return transition_probs

    def can_action_at(self, state):
        """
        是否是普通格子
        """
        if self.grid[state.row][state.column] == 0:
            return True
        else:
            return False

    def _move(self, state, action):
        """
        执行action, 返回新的state

        参数:
        state: t时刻的state
        action: t时刻的action

        返回:
        next_state: t+1时刻的state
        """
        if not self.can_action_at(state):
            raise Exception("Can't move from here!")

        next_state = state.clone()

        # 执行行动(移动)
        if action == Action.UP:
            next_state.row -= 1
        elif action == Action.DOWN:
            next_state.row += 1
        elif action == Action.LEFT:
            next_state.column -= 1
        elif action == Action.RIGHT:
            next_state.column += 1

        # 检查状态是否在grid外
        if not (0 <= next_state.row < self.row_length):
            next_state = state
        if not (0 <= next_state.column < self.column_length):
            next_state = state

        # 检查智能体是否到达了被屏蔽的格子
        if self.grid[next_state.row][next_state.column] == 9:
            next_state = state

        return next_state

    def reward_func(self, state):
        """
        奖励函数(Reward Function)

        在这个例子中, 返回的奖励只取决于t+1时刻的状态(是绿色还是红色格子),
        所以参数只有一个t+1时刻的state, 没有t时刻的state

        - 当状态为绿色格子的时候, 奖励是1
        - 当状态是红色格子的时候, 奖励是-1
        - 除此之外, 都返回默认的奖励default_reward
        
        default_reward的值会对agent的行为产生影响, 我们这里设置为一个负数, 
        如果agent乱走, 奖励会逐渐变小, 这个值可以起到加快行动的效果, 这也说
        明了奖励的设置会对强化学习结果产生很大影响 

        参数:
        state: t+1时刻的state

        返回:
        reward: t+1时刻的奖励
        done: 是否结束
        """
        reward = self.default_reward
        done = False

        # 检查下一种状态的属性
        attribute = self.grid[state.row][state.column]
        if attribute == 1:
            # 获取奖励, 游戏结束
            reward = 1
            done = True
        elif attribute == -1:
            # 遇到危险, 游戏结束
            reward = -1
            done = True

        return reward, done

    def reset(self):
        """
        将Agent放置到左下角
        """
        self.agent_state = State(self.row_length - 1, 0)
        return self.agent_state

    def step(self, action):
        """
        接收agent的行动, 并通过状态迁移函数(Transition Function)返回迁移
        后的状态和奖励

        参数:
        action: t时刻的action

        返回:
        next_state: t+1时刻的state
        reward: t+1时刻的奖励
        done: 是否结束
        """
        next_state, reward, done = self.transit(self.agent_state, action)
        if next_state is not None:
            # 更新当前状态
            self.agent_state = next_state

        return next_state, reward, done

    def transit(self, state, action):
        """
        每次从t时刻状态转移到t+1时刻的状态, 是有`随机性`的

        参数:
        state: t时刻的state
        action: t时刻的action

        返回:
        next_state: t+1时刻的state
        reward: t+1时刻的奖励
        done: 是否结束
        """
        transition_probs = self.transit_func(state, action)
        if len(transition_probs) == 0:
            return None, None, True

        next_states = []
        probs = []
        for s in transition_probs:
            next_states.append(s)
            probs.append(transition_probs[s])

        # 采样出t+1时刻的state, 具有`随机性`
        next_state = np.random.choice(next_states, p=probs)
        # 计算t+1时刻的reward
        reward, done = self.reward_func(next_state)
        return next_state, reward, done


class Agent():

    def __init__(self, env):
        self.actions = env.actions

    def policy(self, state):
        """
        策略函数

        纯随机动作
        """
        return random.choice(self.actions)


def main():
    # 生成grid环境
    grid = [[0, 0, 0, 1], [0, 9, 0, -1], [0, 0, 0, 0]]
    env = Environment(grid)
    agent = Agent(env)

    # 尝试10次游戏
    for i in range(10):
        # 初始化agent的位置
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.policy(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            state = next_state

        print(f'Episode {i}: Agent gets {total_reward:.3f} reward.')


if __name__ == "__main__":
    main()
# Episode 0: Agent gets -2.120 reward.
# Episode 1: Agent gets -1.960 reward.
# Episode 2: Agent gets -2.320 reward.
# Episode 3: Agent gets -1.400 reward.
# Episode 4: Agent gets -4.360 reward.
# Episode 5: Agent gets -0.880 reward.
# Episode 6: Agent gets -1.400 reward.
# Episode 7: Agent gets -0.800 reward.
# Episode 8: Agent gets -6.400 reward.
# Episode 9: Agent gets -1.800 reward.
