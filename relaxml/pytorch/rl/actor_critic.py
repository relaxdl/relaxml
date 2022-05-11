import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import gym
from gym.envs.registration import register

register(id="FrozenLakeEasy-v1",
         entry_point="gym.envs.toy_text:FrozenLakeEnv",
         kwargs={"is_slippery": False})


class ELAgent():

    def __init__(self, epsilon):
        # Q记录了各个state下各个action对应的价值
        # Q[s][a]就是状态s下采取行动a的价值
        #
        # Q存储格式:
        # 0 - [LEFT,DOWN,RIGHT,UP]
        # 1 - [LEFT,DOWN,RIGHT,UP]
        # ....
        self.Q = {}  # Dict[int, List[int]]
        self.epsilon = epsilon
        self.reward_log = []

    def policy(self, s, actions):
        """
        实现Epsilon-Greedy
        
        epsilon的概率随机选择动作(探索); 其他情况选择Q值最大的动作

        参数:
        s: t时刻的state
        actions: 随机的动作列表
        """
        if np.random.random() < self.epsilon:
            # 随机选择动作
            return np.random.randint(len(actions))
        else:
            if s in self.Q and sum(self.Q[s]) != 0:
                # 选择Q值最大的动作
                return np.argmax(self.Q[s])
            else:
                # 随机选择动作
                return np.random.randint(len(actions))

    def init_log(self):
        self.reward_log = []

    def log(self, reward):
        self.reward_log.append(reward)

    def show_reward_log(self, interval=50, episode=-1):
        """
        plot reward log

        >>> agent = ELAgent(epsilon=0.1)
        >>> interval, episode = 50, 5
        >>> [agent.log(np.random.randn()) for _ in range(interval * episode)]
        >>> agent.show_reward_log()

        参数:
        interval: 间隔多少plot一个点
        episode: 如果指定episode则打印某个episode; 否则plot全部reward log
        """
        if episode > 0:
            rewards = self.reward_log[-interval:]
            mean = np.round(np.mean(rewards), 3)
            std = np.round(np.std(rewards), 3)
            print("At Episode {} average reward is {} (+/-{}).".format(
                episode, mean, std))
        else:
            # 每个episode的开始索引
            indices = list(range(0, len(self.reward_log), interval))
            means = []
            stds = []
            for i in indices:
                rewards = self.reward_log[i:(i + interval)]
                means.append(np.mean(rewards))
                stds.append(np.std(rewards))
            means = np.array(means)
            stds = np.array(stds)
            plt.figure()
            plt.title("Reward History")
            plt.grid()
            plt.fill_between(indices,
                             means - stds,
                             means + stds,
                             alpha=0.1,
                             color="g")
            plt.plot(indices,
                     means,
                     "o-",
                     color="g",
                     label="Rewards for each {} episode".format(interval))
            plt.legend(loc="best")
            plt.show()


def show_q_value(Q):
    """
    将行动价值可视化, 显示FrozenLake-v1的Q-values(绿色是正值, 红色是负值)
    
    Q记录了各个state下各个action对应的价值, Q[s][a]就是状态s下采取行动a的价值
    
    Q存储格式:
    0 - [LEFT,DOWN,RIGHT,UP]
    1 - [LEFT,DOWN,RIGHT,UP]
    ....

    显示:
    为了将Q可视化, 把每一个状态用一个3x3的格子来表示, 可以显示5个价值:
    1) u: up value
    2) l: left value
    3) r: right value
    4) d: down value
    5) m: mean value
    +---+---+---+
    |   | u |   |  u: up value
    | l | m | r |  l: left value, r: right value, m: mean value
    |   | d |   |  d: down value
    +---+---+---+

    下面例子中构建随机的Q-values
    >>> n_row, n_col, n_actions = 4, 4, 4
    >>> Q = defaultdict(lambda: [0] * n_actions)
    >>> for r in range(n_row):
    >>>     for c in range(n_col):
    >>>         Q[r * n_row + c] = np.random.randn(4, )
    >>> show_q_value(Q)
    """
    env = gym.make("FrozenLake-v1")
    nrow = env.unwrapped.nrow  # 4
    ncol = env.unwrapped.ncol  # 4
    # 用3x3的matrix表示一个state
    state_size = 3
    q_nrow = nrow * state_size  # 4 x 3
    q_ncol = ncol * state_size  # 4 x 3
    # 原始尺寸是: nrow x ncol = 4 x 4
    # reward_map.shape [4 x 3, 4 x 3]
    reward_map = np.zeros((q_nrow, q_ncol))

    for r in range(nrow):
        for c in range(ncol):
            s = r * nrow + c  # state index
            state_exist = False
            if isinstance(Q, dict) and s in Q:
                state_exist = True
            elif isinstance(Q, (np.ndarray, np.generic)) and s < Q.shape[0]:
                state_exist = True

            if state_exist:
                # 注意: 在展示图中, 纵向的序号是反转的, 要做一下反转处理
                # FrozenLake-v1游戏中, 第一行是最上面一行
                # 显示图像中, 第一行是最下面一行
                _r = 1 + (nrow - 1 - r) * state_size
                _c = 1 + c * state_size
                reward_map[_r][_c - 1] = Q[s][0]  # LEFT = 0
                reward_map[_r - 1][_c] = Q[s][1]  # DOWN = 1
                reward_map[_r][_c + 1] = Q[s][2]  # RIGHT = 2
                reward_map[_r + 1][_c] = Q[s][3]  # UP = 3
                reward_map[_r][_c] = np.mean(Q[s])  # Center

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.imshow(reward_map,
               cmap=cm.RdYlGn,
               interpolation="bilinear",
               vmax=abs(reward_map).max(),
               vmin=-abs(reward_map).max())
    ax.set_xlim(-0.5, q_ncol - 0.5)
    ax.set_ylim(-0.5, q_nrow - 0.5)
    ax.set_xticks(np.arange(-0.5, q_ncol, state_size))
    ax.set_yticks(np.arange(-0.5, q_nrow, state_size))
    ax.set_xticklabels(range(ncol + 1))
    ax.set_yticklabels(range(nrow + 1))
    ax.grid(which="both")
    plt.show()


class Actor(ELAgent):
    """
    Policy Function
    """

    def __init__(self, env):
        super().__init__(epsilon=-1)
        nrow = env.observation_space.n  # n_state = 16
        ncol = env.action_space.n  # n_action = 4
        self.actions = list(range(env.action_space.n))  # 动作列表: [0,1,2,3]
        # Q-Table(初始化值是随机的)
        # 因为游戏世界一共4x4个格子, 所以一共是4x4=16种state
        # Q.shape [n_state=16, n_action=4]
        self.Q = np.random.uniform(0, 1, nrow * ncol).reshape((nrow, ncol))

    def softmax(self, x):
        """
        参数:
        x.shape [4,]

        输出:
        prob: [4,]
        """
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def policy(self, s):
        """
        根据动作的概率, 采样一个动作

        参数:
        s: t时刻的state

        返回:
        a: t时刻的动作
        """
        a = np.random.choice(self.actions, 1, p=self.softmax(self.Q[s]))
        return a[0]


class Critic():
    """
    Value Function
    """

    def __init__(self, env):
        states = env.observation_space.n
        self.V = np.zeros(states)  # 保存每个state的价值


class ActorCritic():

    def __init__(self, actor_class, critic_class):
        """
        参数:
        actor_class: Actor
        critic_class: Critic
        """
        self.actor_class = actor_class
        self.critic_class = critic_class

    def train(self,
              env,
              episode_count=1000,
              gamma=0.9,
              learning_rate=0.1,
              render=False,
              report_interval=50):
        """
        参数:
        env: FrozenLake-v1
        episode_count: episode数量
        gamma: 折扣率
        learning_rate: 学习率
        render: 是否render
        report_interval: 多少个时间步print一条log
        """
        actor = self.actor_class(env)  # Actor
        critic = self.critic_class(env)  # Critic

        actor.init_log()
        for e in range(episode_count):
            s = env.reset()
            done = False
            while not done:
                if render:
                    env.render()
                a = actor.policy(s)
                n_state, reward, done, info = env.step(a)

                # TD Target
                gain = reward + gamma * critic.V[n_state]
                estimated = critic.V[s]
                # TD Error
                td = gain - estimated

                # 更新Policy Function(actor)
                actor.Q[s][a] += learning_rate * td

                # 更新Value Function(critic)
                critic.V[s] += learning_rate * td

                s = n_state

            else:
                actor.log(reward)

            if e != 0 and e % report_interval == 0:
                actor.show_reward_log(episode=e)

        return actor, critic


def train():
    trainer = ActorCritic(Actor, Critic)
    env = gym.make("FrozenLakeEasy-v1")
    actor, critic = trainer.train(env, episode_count=3000)
    show_q_value(actor.Q)
    actor.show_reward_log()


if __name__ == "__main__":
    train()