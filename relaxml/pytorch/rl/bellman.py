from enum import Enum
import numpy as np


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


def V(s, gamma=0.99):
    """
    V(s): 用贝尔曼方程-评估状态`s`的价值

    参数:
    s: t时刻的state

    返回:
    Value: 价值
    """
    V = R(s) + gamma * max_V_on_next_state(s)
    return V


def R(s):
    """
    奖励函数(Reward Function)

    参数:
    s: t时刻的state
    """
    if s == "happy_end":
        return 1
    elif s == "bad_end":
        return -1
    else:
        return 0


def max_V_on_next_state(s):
    """
    计算所有行动的价值, 并返回最大值

    1. 找t+1时刻`价值最大`的动作, 也就是基于价值最大化来选择下一个动作, 
       返回这个动作对应的价值, 就是t+1时刻的价值
    2. 可以通过求`期望`的方式把t+1时刻的state消掉, 从而得到t+1时刻a对应
       的价值

    参数:
    s: t时刻的state

    返回:
    V: t+1时刻的价值
    """
    # 如果游戏结束，则期望值是0
    if s in ["happy_end", "bad_end"]:
        return 0

    actions = ["up", "down"]
    values = []  # 保存每一个动作的价值
    # 遍历所有的动作a, 计算每一个动作的价值
    for a in actions:
        # t+1时刻每个state对应的概率
        transition_probs = transit_func(s, a)
        v = 0
        # 通过求`期望`的方式把t+1时刻的state消掉, 从而得
        # 到t+1时刻a对应的价值
        for next_state in transition_probs:
            prob = transition_probs[next_state]
            v += prob * V(next_state)  # 递归
        values.append(v)
    # 找t+1时刻`价值最大`的动作, 也就是基于价值最大化来选择下一
    # 动作, 返回这个动作对应的价值, 就是t+1时刻的价值
    return max(values)


def transit_func(s, a):
    """
    状态迁移函数(Transition Function)

    通过合并行动字符串与状态, 来生成下一个状态
    ex: 
    (s = 'state', a = 'up') => 'state_up'
    (s = 'state_up', a = 'down') => 'state_up_down'

    - 如果agent尝试向上移动, 我们假设有90%的时间(即0.9的概率)它会按照计划移动
    - 有10%的时间(即0.1的概率), 试图向上移动会导致agent向下移动

    1. 一共2个动作: 'up'和'down'
    2. 尝试5个动作则游戏终止: 如果'up'动作大于等于4次则成功, 否则为失败
    
    可见'up'这个动作越多, 对整体的奖励是有益的, 所以当一个状态中, 'up'数量比较多的时候,
    分数会比较高

    参数:
    s: t时刻的state
    a: t时刻的action

    返回:
    transition_probs: t+1时刻每个state对应的概率
    """

    actions = s.split("_")[1:]
    LIMIT_GAME_COUNT = 5
    HAPPY_END_BORDER = 4
    MOVE_PROB = 0.9

    def next_state(state, action):
        return "_".join([state, action])

    if len(actions) == LIMIT_GAME_COUNT:
        up_count = sum([1 if a == "up" else 0 for a in actions])
        state = "happy_end" if up_count >= HAPPY_END_BORDER else "bad_end"
        prob = 1.0
        return {state: prob}
    else:
        opposite = "up" if a == "down" else "down"
        return {
            next_state(s, a): MOVE_PROB,
            next_state(s, opposite): 1 - MOVE_PROB
        }


class Planner():
    """
    `价值迭代(value iteration)`和`策略迭代(policy iteration)`的抽象类
    """

    def __init__(self, env):
        self.env = env
        self.log = []  # list of List[List[Value]], Value的变化log

    def initialize(self):
        self.env.reset()
        self.log = []

    def plan(self, gamma=0.9, threshold=0.0001):
        raise Exception("Planner have to implements plan method.")

    def transitions_at(self, state, action):
        """
        状态迁移函数(Transition Function)

        t+1时刻每个state对应的概率和奖励

        参数:
        state: t时刻的state
        action: t时刻的action

        返回: iter(prob, next_state, reward)
        prob: t+1时刻的state对应的概率
        next_state: t+1时刻的state
        reward: t+1时刻的奖励
        """
        transition_probs = self.env.transit_func(state, action)
        for next_state in transition_probs:
            prob = transition_probs[next_state]
            reward, _ = self.env.reward_func(next_state)
            yield prob, next_state, reward

    def dict_to_grid(self, state_reward_dict):
        """
        将dict格式的Value转换为grid格式的数据

        参数:
        state_reward_dict: Dict[State, Value]

        返回:
        grid: List[List[Value]]
        """
        grid = []
        # 初始化二维数组
        for i in range(self.env.row_length):
            row = [0] * self.env.column_length
            grid.append(row)
        for s in state_reward_dict:
            grid[s.row][s.column] = state_reward_dict[s]

        return grid


class ValueIterationPlanner(Planner):
    """
    价值迭代(value iteration)
    """

    def __init__(self, env):
        super().__init__(env)

    def plan(self, gamma=0.9, threshold=0.0001):
        """
        更新价值函数V

        初始化各种状态的价值为0(V[s]=0), 之后不断迭代更新, 直到delta小于threshold为
        止. 在更新的时候会计算各种state下各个动作对应的价值, 找到动作最大的价值作为state
        的价值进行更新. 更新的时候会用i次迭代的V来计算i+1次迭代的V
        """
        self.initialize()
        actions = self.env.actions
        V = {}
        for s in self.env.states:
            # 初始化各种状态的期望回报
            V[s] = 0

        while True:
            delta = 0
            self.log.append(self.dict_to_grid(V))
            # 遍历所有的states
            for s in V:
                if not self.env.can_action_at(s):
                    continue
                expected_rewards = []
                # 遍历所有的动作, 计算每一个动作的价值. 可以通过求`期望`的
                # 方式把state消掉, 从而得到a对应的价值
                for a in actions:
                    r = 0
                    for prob, next_state, reward in self.transitions_at(s, a):
                        r += prob * (reward + gamma * V[next_state])  # 上一轮的V
                    expected_rewards.append(r)
                # 找`价值最大`的动作, 这个动作对应的价值, 就是这一轮迭代之后, state的价值
                max_reward = max(expected_rewards)
                delta = max(delta, abs(max_reward - V[s]))
                V[s] = max_reward  # 更新最新一轮的V

            if delta < threshold:
                break

        V_grid = self.dict_to_grid(V)
        return V_grid


class PolicyIterationPlanner(Planner):
    """
    策略迭代(policy iteration)
    """

    def __init__(self, env):
        super().__init__(env)
        self.policy = {}  # 策略函数: pi(a|s)

    def initialize(self):
        """
        初始化策略函数: pi(a|s)
        """
        super().initialize()
        self.policy = {}
        actions = self.env.actions
        states = self.env.states
        for s in states:
            self.policy[s] = {}
            for a in actions:
                # 初始化策略函数:
                # 一开始时各种行动的概率都是一样的: pi(a|s)
                self.policy[s][a] = 1 / len(actions)

    def estimate_by_policy(self, gamma, threshold):
        """
        根据最新的策略函数pi(a|s), 更新价值函数V. 就是计算每个state价值的'期望'. 
        得到的结果会在下一轮迭代中用于评价pi(a|s)
        """
        V = {}
        for s in self.env.states:
            # 初始化各种状态的期望回报
            V[s] = 0

        while True:
            delta = 0
            # 遍历所有的states
            for s in V:
                expected_rewards = []
                # 遍历所有的动作, 计算每一个动作的价值. 可以通过求`期望`的
                # 方式把state消掉, 从而得到a对应的价值
                for a in self.policy[s]:
                    # 动作的概率
                    action_prob = self.policy[s][a]
                    r = 0
                    for prob, next_state, reward in self.transitions_at(s, a):
                        # a概率 * a的价值
                        r += action_prob * prob * \
                             (reward + gamma * V[next_state]) # 上一轮的V
                    expected_rewards.append(r)
                # 对所有动作的价值做加权平均, 得到最终state的价值
                # a1的价值*a1的价值 + a2的概率*a2的价值+...
                value = sum(expected_rewards)
                delta = max(delta, abs(value - V[s]))
                V[s] = value  # 更新最新一轮的V
            if delta < threshold:
                break

        return V

    def plan(self, gamma=0.9, threshold=0.0001):
        """
        更新价值函数V和策略函数pi(a|s)

        plan使用estimate_by_policy的计算结果来计算各种行动的价值, 价值最高的行动
        叫做best_action, 如果基于当前策略计算得到的行动policy_action和best_action
        不一样, 则更新策略. 如果一样就停止迭代

        当策略函数pi(a|s)被更新时, 基于策略计算的价值函数V也会更新, 所以更新价值V和
        策略函数pi(a|s)会被不断反复, 这种相互更新是迭代的核心. 也就是价值函数V和策略函
        数pi(a|s)一起被学习
        """
        self.initialize()
        states = self.env.states
        actions = self.env.actions

        # 选取价值最大的动作
        def take_max_action(action_value_dict):
            return max(action_value_dict, key=action_value_dict.get)

        while True:
            update_stable = True
            # 更新价值函数V
            V = self.estimate_by_policy(gamma, threshold)  # 更新V
            self.log.append(self.dict_to_grid(V))

            for s in states:
                # 在当前的策略下得到动作
                policy_action = take_max_action(self.policy[s])

                # 与其他动作比较
                action_rewards = {}
                # 遍历所有的动作, 计算每一个动作的价值, 可以通过求`期望`的
                # 方式把state消掉, 从而得到a对应的价值
                for a in actions:
                    r = 0
                    for prob, next_state, reward in self.transitions_at(s, a):
                        r += prob * (reward + gamma * V[next_state])  # 最新的V
                    action_rewards[a] = r
                # 价值最高的动作
                best_action = take_max_action(action_rewards)
                # `策略选择的动作`是否等于`价值最高的动作`
                if policy_action != best_action:
                    update_stable = False

                # 更新策略: pi(a|s)
                # 设置best_action prob=1, otherwise=0(贪婪)
                for a in self.policy[s]:
                    prob = 1 if a == best_action else 0
                    self.policy[s][a] = prob

            if update_stable:
                # 如果策略没有更新, 则停止迭代
                break

        # 将字典转换为二维数组
        V_grid = self.dict_to_grid(V)
        return V_grid


def test_bellman_equation():
    print('===bellman equation===')
    states = ['state', 'state_up_up', 'state_down_down']
    for s in states:
        print(f'state={s}, value={V(s):.3f}')
    print('\n')


def test_plan(plan_type='value', grid=None, move_prob=0.8):
    if grid is None:
        grid = grid = [[0, 0, 0, 1], [0, 9, 0, -1], [0, 0, 0, 0]]
    env = Environment(grid, move_prob=move_prob)
    if plan_type == 'value':
        planner = ValueIterationPlanner(env)
    elif plan_type == 'policy':
        planner = PolicyIterationPlanner(env)

    result = planner.plan()
    planner.log.append(result)
    print(f'==={plan_type} iteration===')
    for i in range(len(result)):
        print(' '.join([f'[{v:.3f}]' for v in result[i]]))
    print('\n')


if __name__ == '__main__':
    test_bellman_equation()
    # ===bellman equation===
    # state=state, value=0.788
    # state=state_up_up, value=0.907
    # state=state_down_down, value=-0.961

    test_plan('value')
    # ===value iteration===
    # [0.610] [0.766] [0.928] [0.000]
    # [0.487] [0.000] [0.585] [0.000]
    # [0.374] [0.327] [0.428] [0.189]

    test_plan('policy')
# ===policy iteration===
# [0.610] [0.766] [0.928] [0.000]
# [0.487] [0.000] [0.585] [0.000]
# [0.374] [0.327] [0.428] [0.189]
