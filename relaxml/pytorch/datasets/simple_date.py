import datetime
from typing import List, Tuple
import numpy as np
from torch.utils.data import DataLoader
"""
用作seq2seq翻译的数据集

训练集是这个范围内的时间: [1974年7月23日18时19分45秒, 2034年10月7日12时6分25秒]. 
我们选择这个时间范围的数据隐藏了另外一个规律: 就是如果中文年份大于74这个数字, 翻译出
来的英文年份前面应该补19; 如果中文年份小于23这个数字, 翻译出来的英文年份前面应该补20

我们希望神经网络可以自动识别出这个规律
"""


class DateData:
    """
    时间范围: (1974, 7, 23, 18, 19, 45) -> (2034, 10, 7, 12, 6, 25)
    中文的: "年-月-日", e.g. "98-02-26"
    英文的: "day/month/year", e.g. "26/Feb/1998"

    1. 中文样本的生成:
    04-07-18 -> ['0', '4', '-', '0', '7', '-', '1', '8']  分词
             -> [3, 7, 1, 3, 10, 1, 4, 11]                转化成token_id list

    2. 英文样本的生成(因为英文是目标对象, 所以我们增加了<BOS>, <EOS>):
    18/Jul/2004 -> ['1', '8', '/', 'Jul', '/', '2', '0', '0', '4']                   分词
                -> ['<BOS>', '1', '8', '/', 'Jul', '/', '2', '0', '0', '4', '<EOS>'] 添加开头和结尾
                -> [13, 4, 11, 2, 16, 2, 5, 3, 3, 7, 14]                             转化成token_id list

    >>> dataset = DateData(4000)
    >>> dataset.date_cn[:3]
        ['31-04-26', '04-07-18', '33-06-06']
    >>> dataset.date_en[:3]
        ['26/Apr/2031', '18/Jul/2004', '06/Jun/2033']
    >>> dataset.vocab
        {'Apr', 'Feb', 'Oct', 'Jun', 'Jul', 'May', 'Nov',
         'Mar', 'Aug',
         '<PAD>', '<BOS>', 'Jan', 'Dec', '9', '8', '5',
         '4', '7', '6', '1', '0', '3', '2', '-', '<EOS>',
         '/', 'Sep'}
    >>> dataset.x[0], dataset.idx2str(dataset.x[0])
        [6 4 1 3 7 1 5 9] 31-04-26
    >>> dataset.y[0], dataset.idx2str(dataset.y[0])
        [13  5  9  2 15  2  5  3  6  4 14] <BOS>26/Apr/2031<EOS>
    >>> dataset[0]
        (array([6, 4, 1, 3, 7, 1, 5, 9]), 
         array([13,  5,  9,  2, 15,  2,  5,  3,  6,  4, 14]), 
         10)

    """
    PAD_ID = 0

    def __init__(self, n: int):
        """
        参数:
        n: 生成的样本数量
        """
        np.random.seed(1)

        self.date_cn = []  # list of cn data, e.g. 34-10-07
        self.date_en = []  # list of en data, e.g. 07/Oct/203
        # 时间范围: (1974, 7, 23, 18, 19, 45) -> (2034, 10, 7, 12, 6, 25)
        # 中文: 74-07-23 -> 34-10-07
        # 英文: 23/Jul/1974 -> 07/Oct/203
        for timestamp in np.random.randint(143835585, 2043835585, n):
            date = datetime.datetime.fromtimestamp(timestamp)
            self.date_cn.append(date.strftime("%y-%m-%d"))
            self.date_en.append(date.strftime("%d/%b/%Y"))
        # vocab:
        # {'2', '1', '<EOS>', '/', 'Mar', 'Jun', 'Apr', 'Aug', '3', 'Jan', '5',
        #   '<BOS>', '8', '0', 'May',
        #   'Nov', 'Jul', 'Oct', 'Sep', '9', '6', '4', 'Dec', '7', '-', 'Feb'}
        #
        # 包含三个部分(str):
        # 1. 0,1,2,3,4,5,6,7,8,9
        # 2. <BOS>, <EOS>, <PAD>, -, /
        # 3. Jun, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec
        self.vocab = set([str(i) for i in range(0, 10)] +
                         ["-", "/", "<BOS>", "<EOS>"] +
                         [i.split("/")[1] for i in self.date_en])

        self.token_to_id = {
            v: i
            for i, v in enumerate(sorted(list(self.vocab)), start=1)
        }  # id从1开始, 0留给<PAD>
        self.token_to_id["<PAD>"] = DateData.PAD_ID
        self.vocab.add("<PAD>")
        self.id_to_token = {i: v for v, i in self.token_to_id.items()}

        self.x, self.y = [], []
        for cn, en in zip(self.date_cn, self.date_en):
            # 中文: 34-10-07
            # e.g.
            # 04-07-18 -> ['0', '4', '-', '0', '7', '-', '1', '8']
            #          -> [3, 7, 1, 3, 10, 1, 4, 11]
            self.x.append([self.token_to_id[v] for v in cn])
            # 英文: 07/Oct/203
            # e.g.
            # 18/Jul/2004 -> ['<BOS>', '1', '8', '/', 'Jul', '/', '2', '0', '0', '4', '<EOS>']
            #             -> [13, 4, 11, 2, 16, 2, 5, 3, 3, 7, 14]
            self.y.append([
                self.token_to_id["<BOS>"],
            ] + [self.token_to_id[v] for v in en[:3]] + [
                self.token_to_id[en[3:6]],
            ] + [self.token_to_id[v] for v in en[6:]] + [
                self.token_to_id["<EOS>"],
            ])
        self.x, self.y = np.array(self.x), np.array(self.y)

        self.start_token = self.token_to_id["<BOS>"]
        self.end_token = self.token_to_id["<EOS>"]

    def __len__(self) -> int:
        return len(self.x)

    @property
    def num_word(self) -> int:
        """
        词典长度
        """
        return len(self.vocab)

    def __getitem__(self, index: int) -> Tuple[np.array, np.array, int]:
        """
        采样:

        e.g.
        (array([6, 4, 1, 3, 7, 1, 5, 9]), 
         array([13,  5,  9,  2, 15,  2,  5,  3,  6,  4, 14]), 
         10)

        返回: (bx, by, decoder_len)
        """
        # 返回的decoder_len-1是为了去掉开头的<BOS>
        return self.x[index], self.y[index], len(self.y[index]) - 1

    def idx2str(self, idx: List[str]) -> List[int]:
        """
        将token_id list转换为token list

        >>> idx2str([ 4,  3,  1,  3, 10,  1,  5,  8])
        10-07-25
        >>> idx2str([13,  5,  8,  2, 20,  2,  5,  3,  4,  3, 14])
        <BOS>25/Jul/2010<EOS>
        """

        x = []
        for i in idx:
            x.append(self.id_to_token[i])
            if i == self.end_token:
                break
        return "".join(x)


def load_date(batch_size: int = 32,
              num_examples: int = 4000) -> Tuple[DataLoader, DateData]:
    """
    >>> batch_size = 32
    # 04-07-18, 拆分成token后长度为8
    # <BOS>18/Jul/2004<EOS>, 拆分成token后长度为11
    >>> num_steps_x, num_steps_y = 8, 11
    >>> data_iter, dataset = load_date(batch_size=batch_size)
    >>> for X, y, decoder_len in data_iter:
    >>>     print(X[0], dataset.idx2str(X[0].numpy()))
    >>>     print(y[0], dataset.idx2str(y[0].numpy()))
    >>>     print(decoder_len[0])  # 去掉了y开始的<BOS>
    >>>     assert X.shape == (batch_size, num_steps_x)
    >>>     assert y.shape == (batch_size, num_steps_y)
    >>>     assert decoder_len.shape == (batch_size, )
    >>>     break
        tensor([11,  5,  1,  4,  5,  1,  3,  7]) 82-12-04
        tensor([13,  3,  7,  2, 17,  2,  4, 12, 11,  5, 14]) <BOS>04/Dec/1982<EOS>
        tensor(10)
    """
    dataset = DateData(num_examples)
    data_iter = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_iter, dataset


if __name__ == '__main__':
    batch_size = 32
    # 04-07-18, 拆分成token后长度为8
    # <BOS>18/Jul/2004<EOS>, 拆分成token后长度为11
    num_steps_x, num_steps_y = 8, 11
    data_iter, dataset = load_date(batch_size=batch_size)
    for X, y, decoder_len in data_iter:
        print(X[0], dataset.idx2str(X[0].numpy()))
        print(y[0], dataset.idx2str(y[0].numpy()))
        print(decoder_len[0])  # 去掉了y开始的<BOS>
        assert X.shape == (batch_size, num_steps_x)
        assert y.shape == (batch_size, num_steps_y)
        assert decoder_len.shape == (batch_size, )
        break
    # tensor([11,  5,  1,  4,  5,  1,  3,  7]) 82-12-04
    # tensor([13,  3,  7,  2, 17,  2,  4, 12, 11,  5, 14]) <BOS>04/Dec/1982<EOS>
    # tensor(10)
