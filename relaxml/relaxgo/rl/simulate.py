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


def simulate_game(black_player: Any, white_player: Any) -> GameRecord:
    """
    让两个agent玩一局游戏, 返回游戏结果

    参数:
    black_player: Agent
    white_player: Agent
    """
    moves = []
    # 开始一局游戏
    game = goboard.GameState.new_game(19)
    agents = {
        Player.black: black_player,
        Player.white: white_player,
    }
    while not game.is_over():
        next_move = agents[game.next_player].select_move(game)
        moves.append(next_move)
        game = game.apply_move(next_move)

    game_result = scoring.compute_game_result(game)
    print(game_result)

    return GameRecord(
        moves=moves,
        winner=game_result.winner,
        margin=game_result.winning_margin,
    )


def experience_simulation(num_games: int, agent1: Any,
                          agent2: Any) -> ExperienceBuffer:
    """
    让agent1和agent2玩`num_games`局游戏, 收集经验

    1. 我们让agent1固定执黑, agent2固定执白
       (有时候, 黑方先手, 黑方和白方的风格会不同; 可以尝试轮流切换黑白子的方式)
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

        if color1 == Player.black:
            black_player, white_player = agent1, agent2
        else:
            white_player, black_player = agent2, agent1
        game_record = simulate_game(black_player, white_player)
        if game_record.winner == color1:
            collector1.complete_episode(reward=1)
            collector2.complete_episode(reward=-1)
        else:
            collector2.complete_episode(reward=1)
            collector1.complete_episode(reward=-1)
        color1 = color1.other

    return combine_experience([collector1, collector2])