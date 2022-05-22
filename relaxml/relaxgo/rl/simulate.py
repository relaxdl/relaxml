from typing import Any
from ..rl.experience import ExperienceBuffer, ExperienceCollector, combine_experience
from .. import scoring
from .. import goboard_fast as goboard
from ..gotypes import Player

from collections import namedtuple


class GameRecord(namedtuple('GameRecord', 'moves winner margin')):
    """
    moves: List[Move]
    winner: Player
    margin: int, winning margin 
    """
    pass


def simulate_game(black_player: Player,
                  white_player: Player,
                  board_size: int = 19) -> GameRecord:
    """
    让两个agent玩一局游戏, 返回游戏结果

    参数:
    black_player: Agent
    white_player: Agent
    """
    moves = []
    # 开始一局游戏
    game = goboard.GameState.new_game(board_size)
    agents = {
        Player.black: black_player,
        Player.white: white_player,
    }
    while not game.is_over():
        next_move = agents[game.next_player].select_move(game)
        moves.append(next_move)
        game = game.apply_move(next_move)

    game_result = scoring.compute_game_result(game)
    print(
        f'simulate_game board_size: {board_size}, game_result: {game_result}')

    return GameRecord(
        moves=moves,
        winner=game_result.winner,
        margin=game_result.winning_margin,
    )


def experience_simulation(num_games: int,
                          agent1: Any,
                          agent2: Any,
                          board_size: int = 19) -> ExperienceBuffer:
    """
    让agent1和agent2玩`num_games`局游戏, 收集经验

    1. 有时候, 黑方先手, 黑方和白方的风格会不同; 可以尝试轮流切换黑白子的方式
    2. 两个agent完全相同, 一局可以收集2局的经验, 可以合并起来使用

    返回的经验可以这样保存:
    with h5py.File(experience_file, 'w') as experience_outf:
        experience.serialize(experience_outf)
    """
    collector1 = ExperienceCollector()
    collector2 = ExperienceCollector()

    color1 = Player.black
    for i in range(num_games):
        collector1.begin_episode()
        agent1.set_collector(collector1)

        collector2.begin_episode()
        agent2.set_collector(collector2)

        # agent1始终和color1相同颜色
        if color1 == Player.black:
            # agent1黑色; agent2白色
            black_player, white_player = agent1, agent2
        else:
            # agent1白色; agent2黑色
            white_player, black_player = agent1, agent2
        game_record = simulate_game(black_player, white_player, board_size)

        if game_record.winner == color1:
            # agent1获胜, agent2失败
            collector1.complete_episode(reward=1)
            collector2.complete_episode(reward=-1)
        else:
            # agent1失败, agent2获胜
            collector1.complete_episode(reward=-1)
            collector2.complete_episode(reward=1)
        color1 = color1.other

    return combine_experience([collector1, collector2])


def eval_pg_bot(num_games: int,
                agent1: Any,
                agent2: Any,
                board_size: int = 19):
    """
    评估两个agent

    通常的训练流程:
    1. 生成一大批自我对弈的棋局
    2. 进行训练
    3. 将更新的机器人与前一版本的机器人进行对弈检验
    4. 如果新机器人明显更强, 就切换到这个新版本
    5. 如果机器人与之前强度差不多, 就生成更多棋局, 再次训练
       a. 比如获胜率停滞在50%, 可能原因是学习率太小了, 策略停滞在一个局部最优解, 解决方案
          是增大学习率, 给自我对弈增加更多随机性
    6. 如果新机器人明显更弱, 就需要调优优化器等参数, 重新进行训练
       a. 获胜率明显下降, 可能原因是学习率太大, 跳过头了, 可以尝试减小学习率, 或者收集更多的
          训练资料等手段尝试
    >>> board_size = 9
    >>> agent1 = PolicyAgent(encoder=SimpleEncoder(board_size))
    >>> agent2 = PolicyAgent(encoder=SimpleEncoder(board_size))
        simulate_game board_size: 9, game_result: W+22.5
        Simulating game 2/5...
        simulate_game board_size: 9, game_result: W+14.5
        Simulating game 3/5...
        simulate_game board_size: 9, game_result: B+37.5
        Simulating game 4/5...
        simulate_game board_size: 9, game_result: B+17.5
        Simulating game 5/5...
        simulate_game board_size: 9, game_result: W+14.5
        Agent 1 record: 2/5
    """
    wins = 0
    losses = 0
    color1 = Player.black
    for i in range(num_games):
        print('Simulating game %d/%d...' % (i + 1, num_games))
        # agent1始终和color1相同颜色
        if color1 == Player.black:
            # agent1黑色; agent2白色
            black_player, white_player = agent1, agent2
        else:
            # agent1白色; agent2黑色
            white_player, black_player = agent1, agent2
        game_record = simulate_game(black_player, white_player, board_size)
        if game_record.winner == color1:
            # agent1获胜, agent2失败
            wins += 1
        else:
            # agent1失败, agent2获胜
            losses += 1
        color1 = color1.other
    print('Agent 1 record: %d/%d' % (wins, wins + losses))
