from typing import Tuple
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D

"""
AlphaGo的训练流程:
1. 首先开始训练两个深度卷积神经网络(策略网络), 用于动作预测
   a. 一个深度更深, 我们叫做`强策略网络`
   b. 一个深度浅一些, 叫做`快策略网络`
2. 棋盘编码采用48/49个特征平面
3. 策略网络训练完成之后, 使用强策略网络作为初始点进行自我博弈
4. 在自我博弈的过程中会同时训练价值网络
5. 进行一段时间自我博弈之后, 可以把树搜索作为下棋策略的基础(这时候使用的是快策略网络), 
   同时还要参考价值网络的输出, 来平衡数的搜索算法

AlphaGo有三个网络:
1. 快策略网络: 目的不是成为最准确的动作预测器, 而是在保证足够好的预测准确率的同时能够
   非常迅速的做出预测
2. 强策略网络: 由于目标是准确率, 不是速度
   a. 是一个13层的卷积网络, 这13层都使用19x19的filter, 也就是说在整个过程中, 都保留了
      棋盘尺寸. 第一个卷积核大小是5x5, 后面所有卷积核大小都是3x3, 最后一层用softmax激活
      函数输出概率
3. 价值网络: 强策略网络自我博弈的过程中会产生经验数据, 用这个经验数据来训练价值网络
   a. 是一个16层的卷积网络, 它的前12层和强策略网络完全一样, 最后接的是Dense层, 输出一个
      标量的价值
"""

def alphago_model(input_shape: Tuple[int],
                  is_policy_net: bool = False,
                  num_filters: int = 192,
                  first_kernel_size: int = 5,
                  other_kernel_size: int = 3) -> keras.Model:
    """
    构建`策略网络`和`价值网络`

    模型输入: 
    策略网络: [batch_size, num_planes, board_height, board_width]
    价值网络: TODO

    模型输出:
    策略网络: [batch_size, num_classes=board_height*board_width]
    价值网络: [batch_size, 1]


    参数:
    input_shape: 输入形状. 策略网络和价值网络不同
    is_policy_net: 初始化的是策略网络还是价值网络
    num_filters: 除了最后一个卷积层之外, 所有卷积层的filter都一样
    first_kernel_size: 第一卷积层的kernel size
    other_kernel_size: 除第一层之外的卷积层的kernel size

    返回:
    model: 模型
    """
    model = Sequential()
    model.add(
        Conv2D(num_filters,
               first_kernel_size,
               input_shape=input_shape,
               padding='same',
               data_format='channels_first',
               activation='relu'))

    # AlphaGo策略网络和价值网络的前12层是一样的
    for i in range(2, 12):
        model.add(
            Conv2D(num_filters,
                   other_kernel_size,
                   padding='same',
                   data_format='channels_first',
                   activation='relu'))

    if is_policy_net:
        # 策略网络
        model.add(
            Conv2D(filters=1,
                   kernel_size=1,
                   padding='same',
                   data_format='channels_first',
                   activation='softmax'))
        model.add(Flatten())
        return model

    else:
        # 价值网络
        model.add(
            Conv2D(num_filters,
                   other_kernel_size,
                   padding='same',
                   data_format='channels_first',
                   activation='relu'))
        model.add(
            Conv2D(filters=1,
                   kernel_size=1,
                   padding='same',
                   data_format='channels_first',
                   activation='relu'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(1, activation='tanh'))
        return model