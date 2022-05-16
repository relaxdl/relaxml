import os
from shutil import rmtree
import random
import re
from collections import namedtuple
from collections import deque
import numpy as np
import gym
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt

Experience = namedtuple("Experience", ["s", "a", "r", "n_s", "d"])


class FNAgent():
    """
    含有一个可以训练的神经网络, 包含通过Epsilon Greedy实现的policy
    """

    def __init__(self, epsilon, actions):
        """
        self.model在两种情况下初始化:
        1. load()加载的时候加载训练好的模型
        2. initialize()的时候创建新模型

        参数:
        epsilon: Epsilon-Greedy的参数
        actions: List[int] 动作列表
        """
        self.epsilon = epsilon  # Epsilon-Greedy的参数
        self.actions = actions  # 动作列表
        self.model = None  # 神经网络, tf.Module
        self.estimate_probs = False  # 如果为True在选择动作的时候是随机采样; 为False则选择概率最大的动作
        self.initialized = False  # 是否已经初始化过模型

    def save(self, model_path):
        """
        保存模型模型
        """
        if self.initialized:
            self.model.save(model_path,
                            overwrite=True,
                            include_optimizer=False)
            print(f'save model success => {model_path}')

    @classmethod
    def load(cls, env, model_path, epsilon=0.0001):
        """
        加载一个训练好的模型, 返回Agent

        参数:
        cls: FNAgent子类
        env: Observer
        model_path: 模型的保存路径
        epsilon: Epsilon-Greedy的参数

        返回:
        agent: FNAgent子类实例
        """
        print(f'load model from => {model_path}')
        actions = list(range(env.action_space.n))  # 动作列表
        agent = cls(epsilon, actions)  # 创建FNAgent对象
        agent.model = keras.models.load_model(model_path)
        agent.initialized = True
        return agent

    def initialize(self, experiences):
        """
        初始化self.model
        """
        raise NotImplementedError("You have to implement initialize method.")

    def estimate(self, s):
        """
        根据self.model进行预测
        """
        raise NotImplementedError("You have to implement estimate method.")

    def update(self, experiences, gamma):
        """
        用一个batch(experiences)的数据更新self.model
        """
        raise NotImplementedError("You have to implement update method.")

    def policy(self, s):
        """
        实现Epsilon-Greedy
        
        1. epsilon的概率随机选择动作(探索)
        2. 其他情况:
           a. 如果self.estimate_probs=True, 则根据动作概率随机采样动作
           b. 根据self.estimate_probs=False, 则选择动作概率最大的动作

        参数:
        s: t时刻的state

        返回:
        action: t时刻的action
        """
        if np.random.random() < self.epsilon or not self.initialized:
            # 随机选择动作
            return np.random.randint(len(self.actions))
        else:
            # estimates.shape [n_actions, ]
            estimates = self.estimate(s)
            if self.estimate_probs:
                # 根据动作概率随机采样动作
                action = np.random.choice(self.actions, size=1, p=estimates)[0]
                return action
            else:
                # 选择动作概率最大的动作
                return np.argmax(estimates, axis=0)

    def play(self, env, episode_count=5, render=True, max_step_per_episode=-1):
        """
        模拟Agent的行动, 让Agent根据policy来自动玩游戏

        参数:
        env: Observer
        episode_count: episode数量
        render: 是否渲染UI
        max_step_per_episode: 如果设置了这个值, 则限定每个episode的最大step数
        """
        for e in range(episode_count):
            s = env.reset()
            done = False
            episode_reward = 0
            step_count = 0
            while not done:
                if render:
                    env.render()
                a = self.policy(s)
                n_state, reward, done, info = env.step(a)
                episode_reward += reward
                s = n_state
                step_count += 1

                # 是否已经满足了最大时间步
                if max_step_per_episode != -1 and \
                    step_count >= max_step_per_episode:
                    done = True
            else:
                print("Get reward {}.".format(episode_reward))


class Trainer():
    """
    Agent进行训练的模块

    训练会分为两个阶段:
    1. 预测, 这个阶段先收集经验, 并不进行训练
    2. 训练, 收集到足够的经验之后, 开始训练
    """

    def __init__(self,
                 buffer_size=1024,
                 batch_size=32,
                 gamma=0.9,
                 report_interval=10,
                 log_dir=""):
        """
        参数:
        buffer_size: 经验buffer的大小
        batch_size: 批量大小
        gamma: 折扣系数
        report_interval: 在日志中, 间隔多少step plot一个数据点
        log_dir: 日志保存路径
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.report_interval = report_interval
        self.logger = Logger(log_dir, self.trainer_name)
        self.experiences = deque(maxlen=buffer_size)
        # 是否在训练(我们可以根据这个标签做不同的逻辑)
        # a. False-预热; True-训练
        # b. 预热阶段在累积经验, 并不会进行训练
        self.training = False
        # 到目前为止训练了多少个episode(预热的episode不算在里面)
        self.training_episode = 0
        self.reward_log = [
        ]  # List[float], 通常是每个episode结束记录一次这个episode的总Reward

    @property
    def trainer_name(self):
        """
        返回Trainer的名字
        """
        class_name = self.__class__.__name__
        snaked = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", class_name)
        snaked = re.sub("([a-z0-9])([A-Z])", r"\1_\2", snaked).lower()
        snaked = snaked.replace("_trainer", "")
        return snaked

    def train_loop(self,
                   env,
                   agent,
                   episode=200,
                   initial_count=-1,
                   render=False,
                   observe_interval=0,
                   max_step_per_episode=-1):
        """
        什么时候开始训练?
        会先预热一下收集经验, 收集到足够的经验之后, 开始训练, 2种情况下开始训练:
        1. experiences buffer满数据已经写满
        2. 设置了initial_count, 并且经过了initial_count个episode

        如何记录 & 保存frames?
        1. 设置了observe_interval才会记录frame(只有当输入的state为Image的时候才打开这个标签)
        2. 每个episode会生成一个frames列表, 训练observe_interval个step记录一个frame
        3. 每个episode训练结束后frames的数据会一次性写入TensorBoard

        参数:
        env: Observer
        agent: FNAgent
        episode: 一共训练多少个episode
        initial_count: 如果设置了这个值, 经过多少个episode累积经验之后开始训练
        render: 是否渲染UI
        observe_interval: 记录frame的频率, 训练多少个step记录一个frame. 
                          只有当输入的state为Image的时候才打开这个标签
        max_step_per_episode: 如果设置了这个值, 则限定每个episode的最大step数
        """
        # 经验buffer, 超过buffer_size的经验会被丢弃
        self.experiences = deque(maxlen=self.buffer_size)
        self.training = False  # False-预热; True-训练
        self.training_episode = 0  # 到目前为止训练了多少个episode
        self.reward_log = []  # List[float]
        frames = []  # List[state]

        for i in range(episode):
            s = env.reset()
            done = False
            step_count = 0
            self.episode_begin(i, agent)
            while not done:
                # 训练一个episode
                if render:
                    env.render()
                # 记录一条frame
                if self.training and observe_interval > 0 and\
                   (self.training_episode == 1 or
                    self.training_episode % observe_interval == 0):
                    frames.append(s)

                a = agent.policy(s)
                n_state, reward, done, info = env.step(a)
                e = Experience(s, a, reward, n_state, done)
                self.experiences.append(e)  # 累积经验

                # experiences buffer满数据已经写满, 开始训练
                if not self.training and \
                   len(self.experiences) == self.buffer_size:
                    self.begin_train(i, agent)
                    self.training = True

                self.step(i, step_count, agent, e)

                s = n_state
                step_count += 1

                # 是否已经满足了最大时间步
                if max_step_per_episode != -1 and \
                    step_count >= max_step_per_episode:
                    done = True
            else:
                # 训练完一个episode
                self.episode_end(i, step_count, agent)

                # 设置了initial_count, 并且经过了initial_count个episode, 开始训练
                if not self.training and \
                   initial_count > 0 and i >= initial_count:
                    self.begin_train(i, agent)
                    self.training = True

                if self.training:
                    # 写入frames, 之后会清空frames
                    if len(frames) > 0:
                        self.logger.write_image(self.training_episode, frames)
                        frames = []
                    self.training_episode += 1

    def episode_begin(self, episode, agent):
        """
        每个episode开始时的回调

        参数:
        episode: int
        agent: FNAgent
        """
        pass

    def begin_train(self, episode, agent):
        """
        预热完毕, 开始训练时调用

        参数:
        episode: int
        agent: FNAgent
        """
        pass

    def step(self, episode, step_count, agent, experience):
        """
        训练时每个时间步调用一次

        参数:
        episode: int
        step_count: 当前episode的时间步t
        agent: FNAgent
        experience: 时间步t产生的经验: Experience(s, a, reward, n_state, done)
        """
        pass

    def episode_end(self, episode, step_count, agent):
        """
        每个episode结束时的回调

        参数:
        episode: int
        step_count: 当前episode的时间步t
        agent: FNAgent
        """
        pass

    def is_event(self, count, interval):
        """
        触发Event的条件

        参数:
        count: episode
        interval: int
        """
        return True if count != 0 and count % interval == 0 else False

    def get_recent(self, count):
        """
        获取最近的count条经验

        参数:
        count: int

        返回:
        experiencess: List[Experience]
        """
        recent = range(len(self.experiences) - count, len(self.experiences))
        return [self.experiences[i] for i in recent]


class Observer():
    """
    环境Env的封装
    通过transform将从env获得的state转换为agent易于处理的形式.
    例如: 将彩色画面变成灰度图

    在使用Observer进行训练的情况下, 在运行时也必须使用Observer, 因为训练后的Agent是
    以Observer的转换为前提的
    """

    def __init__(self, env):
        self._env = env

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space

    def reset(self):
        """
        reset换将
        """
        return self.transform(self._env.reset())

    def render(self):
        self._env.render(mode="human")

    def step(self, action):
        n_state, reward, done, info = self._env.step(action)
        return self.transform(n_state), reward, done, info

    def transform(self, state):
        """
        state转换
        """
        raise NotImplementedError("You have to implement transform method.")


class Logger():
    """
    记录学习过程
    """

    def __init__(self, log_dir="", dir_name=""):
        """
        参数:
        log_dir: 日志根目录
        dir_name: 日志目录名字
        """
        self.log_dir = log_dir
        if not log_dir:
            self.log_dir = os.path.join(os.path.dirname(__file__), "logs")
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

        if dir_name:
            # log_dir/dir_name
            self.log_dir = os.path.join(self.log_dir, dir_name)
            if not os.path.exists(self.log_dir):
                os.mkdir(self.log_dir)

        self.writer = tf.summary.create_file_writer(self.log_dir)
        self.writer.set_as_default()

    def path_of(self, file_name):
        """
        返回: log_dir/file_name
        """
        return os.path.join(self.log_dir, file_name)

    def describe(self, name, values, episode=-1, step=-1):
        """
        print values的mean & std

        参数:
        name: 名字
        values: List[float]
        episode: int
        step: int
        """
        mean = np.round(np.mean(values), 3)
        std = np.round(np.std(values), 3)
        desc = "{} is {} (+/-{})".format(name, mean, std)
        if episode > 0:
            print("At episode {}, {}".format(episode, desc))
        elif step > 0:
            print("At step {}, {}".format(step, desc))

    def plot(self, name, values, interval=10):
        """
        plot values

        >>> logger = Logger()
        >>> interval, episode = 10, 5
        >>> values = [np.random.randn() for _ in range(interval * episode)]
        >>> logger.plot('name', values, interval=interval)

        参数:
        name: 名字
        values: List[float]
        interval: 间隔多少plot一个点
        """
        # 每个episode的开始索引
        indices = list(range(0, len(values), interval))
        means = []
        stds = []
        for i in indices:
            _values = values[i:(i + interval)]
            means.append(np.mean(_values))
            stds.append(np.std(_values))
        means = np.array(means)
        stds = np.array(stds)
        plt.figure()
        plt.title("{} History".format(name))
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
                 label="{} per {} episode".format(name.lower(), interval))
        plt.legend(loc="best")
        plt.show()

    def write(self, step, name, value):
        """"
        write scaler
        """
        tf.summary.scalar(name, value, step)
        self.writer.flush()

    def write_image(self, training_episode, frames):
        """
        将frames写入到

        参数:
        training_episode: int, 训练了多少个episode
        frames: List[state]
            state = [height, width, channels]
        """
        # 将一个'frames'作为一系列灰度图像处理
        last_frames = [f[:, :, -1] for f in frames]
        if np.min(last_frames[-1]) < 0:
            scale = 127 / np.abs(last_frames[-1]).max()
            offset = 128
        else:
            scale = 255 / np.max(last_frames[-1])
            offset = 0
        tag = "frames_at_training_{}".format(training_episode)
        images = []  # List[[height, width, 1]]

        for f in last_frames:
            # image.shape [height, width]
            image = np.asarray(f * scale + offset, dtype=np.uint8)
            images.append(np.expand_dims(image, axis=2))

        tf.summary.image(tag, images, step=training_episode)
        self.writer.flush()


class ValueFunctionAgent(FNAgent):

    def initialize(self, experiences, optimizer):
        """
        初始化self.model
        """
        n_features = experiences[0].s.shape[0]
        self.make_model(n_features)
        self.model.compile(optimizer, loss="mse")
        self.initialized = True
        print("Done initialization. From now, begin training!")

    def make_model(self, n_features):
        """
        参数:
        n_features: 输入特征数
        """
        model = keras.Sequential()
        model.add(tf.keras.Input(shape=(n_features, )))
        model.add(keras.layers.Dense(10, activation='relu'))
        model.add(keras.layers.Dense(10, activation='relu'))
        model.add(keras.layers.Dense(len(self.actions)))
        self.model = model

    def estimate(self, state):
        """
        根据self.model进行预测, 返回每个动作的value

        参数:
        s: t时刻的state

        输出:
        estimated: [n_actions, ]
        """
        # x.shape [1, n_features]
        x = np.array([state])
        return self.model.predict(x)[0]

    def update(self, experiences, gamma):
        """
        用一个batch(experiences)的数据更新self.model

        参数:
        experiences: 一个批量的经验数据
        gamma: 折扣率

        返回:
        loss: 标量
        """

        # states.shape [batch_size, n_features]
        states = np.array([e.s for e in experiences])
        # n_states.shape [batch_size, n_features]
        n_states = np.array([e.n_s for e in experiences])

        # estimateds.shape [batch_size, n_actions]
        estimateds = self.model.predict(states)
        # future.shape [batch_size, n_actions]
        future = self.model.predict(n_states)

        # 更新estimateds-[标签]
        for i, e in enumerate(experiences):
            reward = e.r
            if not e.d:
                reward += gamma * np.max(future[i])
            estimateds[i][e.a] = reward

        # x = states
        # y = estimateds-[标签]
        loss = self.model.train_on_batch(states, estimateds)
        return loss


class CartPoleObserver(Observer):

    def transform(self, state):
        """
        参数:
        state: t时刻的state

        返回:
        state: [n_features, ] 
        """
        return np.array(state)


class ValueFunctionTrainer(Trainer):

    def __init__(self,
                 buffer_size=1024,
                 batch_size=32,
                 gamma=0.9,
                 report_interval=10,
                 learning_rate=0.01,
                 log_dir="",
                 file_name=""):
        super().__init__(buffer_size, batch_size, gamma, report_interval,
                         log_dir)
        self.file_name = file_name if file_name else "value_function_agent.h5"
        self.learning_rate = learning_rate
        self.loss = 0

    def train(self,
              env,
              episode_count=220,
              epsilon=0.1,
              initial_count=-1,
              render=False,
              max_step_per_episode=-1):
        """
        训练模型

        参数:
        env: Observer
        episode_count: 一共训练多少个episode_count
        epsilon: Epsilon-Greedy的参数
        initial_count: 如果设置了这个值, 经过多少个episode累积经验之后开始训练
        render: 是否渲染UI
        max_step_per_episode: 如果设置了这个值, 则限定每个episode的最大step数
        """
        actions = list(range(env.action_space.n))
        agent = ValueFunctionAgent(epsilon, actions)
        self.train_loop(env,
                        agent,
                        episode_count,
                        initial_count,
                        render,
                        max_step_per_episode=max_step_per_episode)
        return agent

    def episode_begin(self, episode, agent):
        self.loss = 0

    def begin_train(self, episode, agent):
        """
        预热完毕, 开始训练
        """
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate,
                                          clipvalue=1.0)
        agent.initialize(self.experiences, optimizer)

    def step(self, episode, step_count, agent, experience):
        """
        采样一个batch的数据训练模型
        """
        if self.training:
            batch = random.sample(self.experiences, self.batch_size)
            self.loss += agent.update(batch, self.gamma)

    def episode_end(self, episode, step_count, agent):
        """
        每个episode结束时的回调

        1. 记录这次episode的总奖励
        2. 间隔report_interval个episode打印一次log

        参数:
        episode: int
        step_count: 当前episode的时间步t
        agent: FNAgent
        """
        reward = sum([e.r for e in self.get_recent(step_count)])
        self.loss = self.loss / step_count
        self.reward_log.append(reward)
        if self.training:
            self.logger.write(self.training_episode, "loss", self.loss)
            self.logger.write(self.training_episode, "reward", reward)
        if self.is_event(episode, self.report_interval):
            recent_rewards = self.reward_log[-self.report_interval:]
            self.logger.describe("reward", recent_rewards, episode=episode)


def check_log_dir(log_dir, play):
    """
    在训练模式下清空log_dir
    """
    if not play:
        # 训练模式
        rmtree(log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


def main(play):
    """
    查看日志:
    tensorboard --logdir ../logs

    参数:
    play: True - play模式; False - 训练模式
    """
    log_dir = '../logs'
    check_log_dir(log_dir, play)
    env = CartPoleObserver(gym.make("CartPole-v1"))
    trainer = ValueFunctionTrainer(buffer_size=2048,
                                   log_dir=log_dir,
                                   learning_rate=0.005)
    path = trainer.logger.path_of(trainer.file_name)  # Model的保存路径

    if play:
        # play模式
        agent = ValueFunctionAgent.load(env, path)
        agent.play(env, episode_count=3, max_step_per_episode=200)
    else:
        # 训练模式
        trained = trainer.train(env,
                                episode_count=250,
                                max_step_per_episode=200)
        # plot
        trainer.logger.plot("Rewards", trainer.reward_log,
                            trainer.report_interval)
        trained.save(path)


if __name__ == "__main__":
    # 训练
    main(False)
    # Play
    main(True)
# Get reward 200.0.
# Get reward 200.0.
# Get reward 200.0.