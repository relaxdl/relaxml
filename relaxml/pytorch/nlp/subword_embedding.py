import collections
from typing import Dict, List, Tuple
"""
子词嵌入

实现说明:
https://tech.foxrelax.com/nlp/subword_embedding/
"""


def get_max_freq_pair(token_freqs: Dict[str, int]) -> Tuple[str, str]:
    """
    返回`词内`最频繁的连续符号对

    >>> token_freqs = {'f a s t _': 4, 'f a s t e r _': 3, 't a l l _': 5, 't a l l e r _': 4}
    >>> get_max_freq_pair(token_freqs)
        ('t', 'a')

    内部会生成如下pairs, 返回一个出现频率最高的是: ('t', 'a')
    {('f', 'a'): 7, ('a', 's'): 7, ('s', 't'): 7, ('t', '_'): 4, 
     ('t', 'e'): 3, ('e', 'r'): 7, ('r', '_'): 7, ('t', 'a'): 9, 
     ('a', 'l'): 9, ('l', 'l'): 9, ('l', '_'): 5, ('l', 'e'): 4}
    """
    pairs = collections.defaultdict(int)
    for token, freq in token_freqs.items():
        symbols = token.split()  # 先分词
        for i in range(len(symbols) - 1):
            # "pairs"的key是两个连续符号的元组
            pairs[symbols[i], symbols[i + 1]] += freq

    return max(pairs, key=pairs.get)  # 具有最大值的"pairs"键


def merge_symbols(max_freq_pair: Tuple[str, str], token_freqs: Dict[str, int],
                  symbols: List[str]) -> Dict[str, int]:
    """
    合并最频繁的连续符号对以产生新符号, 并将新的符号插入符号表(symbols)中
    """
    # 合并最频繁的连续符号对以产生新符号, 加入到符号表中
    # e.g. ('t', 'a') -> 'ta'
    symbols.append(''.join(max_freq_pair))

    # 更新token_freqs
    new_token_freqs = dict()
    for token, _ in token_freqs.items():
        # 将max_freq_pair两个token之间的空格去掉:
        # 例如: 't a' -> 'ta'
        new_token = token.replace(' '.join(max_freq_pair),
                                  ''.join(max_freq_pair))
        new_token_freqs[new_token] = token_freqs[token]
    return new_token_freqs


def segment_BPE(tokens: List[str], symbols: List[str]) -> List[str]:
    outputs = []
    for token in tokens:
        start, end = 0, len(token)
        cur_output = []
        # 具有符号中可能最长子字的词元段
        while start < len(token) and start < end:
            if token[start:end] in symbols:
                cur_output.append(token[start:end])
                start = end
                end = len(token)
            else:
                end -= 1
        if start < len(token):
            cur_output.append('[UNK]')
        outputs.append(' '.join(cur_output))
    return outputs


def bpe():
    symbols = [
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
        'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '_',
        '[UNK]'
    ]

    raw_token_freqs = {'fast_': 4, 'faster_': 3, 'tall_': 5, 'taller_': 4}
    token_freqs = {}
    for token, freq in raw_token_freqs.items():
        token_freqs[' '.join(list(token))] = raw_token_freqs[token]
    print(token_freqs)
    # {'f a s t _': 4, 'f a s t e r _': 3, 't a l l _': 5, 't a l l e r _': 4}

    num_merges = 10
    for i in range(num_merges):
        max_freq_pair = get_max_freq_pair(token_freqs)
        token_freqs = merge_symbols(max_freq_pair, token_freqs, symbols)
        print(f'merge #{i + 1}:', max_freq_pair)
    # merge #1: ('t', 'a')
    # merge #2: ('ta', 'l')
    # merge #3: ('tal', 'l')
    # merge #4: ('f', 'a')
    # merge #5: ('fa', 's')
    # merge #6: ('fas', 't')
    # merge #7: ('e', 'r')
    # merge #8: ('er', '_')
    # merge #9: ('tall', '_')
    # merge #10: ('fast', '_')

    print(symbols)
    # ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    #  'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    #  '_', '[UNK]', 'ta', 'tal', 'tall', 'fa', 'fas', 'fast', 'er',
    #  'er_', 'tall_', 'fast_']

    tokens = ['tallest_', 'fatter_']
    print(segment_BPE(tokens, symbols))
    # ['tall e s t _', 'fa t t er_']


if __name__ == '__main__':
    bpe()