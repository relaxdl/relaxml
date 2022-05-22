from typing import List, List, Union

__all__ = [
    'Command',
]


class Command:
    """
    GTP命令:
    1. sequence 可选的序列号, 用于命令和相应的匹配
    2. name: 名称
    3. args: 参数
    """

    def __init__(self, sequence: Union[int, None], name: str, args: List[str]):
        self.sequence = sequence
        self.name = name
        self.args = tuple(args)

    def __eq__(self, other):
        return self.sequence == other.sequence and \
            self.name == other.name and \
            self.args == other.args

    def __repr__(self):
        return 'Command(%r, %r, %r)' % (self.sequence, self.name, self.args)

    def __str__(self):
        return repr(self)


def parse(command_string: str) -> Command:
    """
    解析GTP协议的一行为一个Command

    >>> parse('999 play white D4')
        Command(999, 'play', ('white', 'D4'))
    """
    pieces = command_string.split()
    try:
        # GTP命令开始的序列号是可选的
        sequence = int(pieces[0])
        pieces = pieces[1:]
    except ValueError:
        # 如果文本开头部分不是数字, 则表示这个命令没有序列号
        sequence = None
    name, args = pieces[0], pieces[1:]
    return Command(sequence, name, args)