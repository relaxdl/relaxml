from typing import List, Tuple
import datetime
import sys
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
"""
Simple seq2seq(Luong注意力)

实现说明:
https://tech.foxrelax.com/nlp/simple_seq2seq_luong_attention/
"""
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


class Seq2SeqAttention(nn.Module):
    """
    Encoder-Decoder
    1. Encoder用LSTM实现
    2. Decoder用LSTM+Attention实现
    """

    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int,
                 max_pred_len: int, start_token: int, end_token: int) -> None:
        """
        参数:
        vocab_size: Encoder Embedding输入的维度 & Decoder Embedding输入的维度
        embed_size: Encoder & Decoder Embedding的输出维度
        hidden_size: 中间维度
        max_pred_len: 预测的最大长度
        start_token: <BOS>的token_id
        end_token: <EOS>的token_id
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # encoder
        self.enc_embeddings = nn.Embedding(vocab_size, embed_size)
        self.enc_embeddings.weight.data.normal_(0, 0.1)
        self.encoder = nn.LSTM(embed_size, hidden_size, 1, batch_first=True)

        # decoder
        self.dec_embeddings = nn.Embedding(vocab_size, embed_size)
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.decoder_cell = nn.LSTMCell(embed_size, hidden_size)
        # 要送入`context`和`hx`两个长度为hidden_size的向量
        self.decoder_dense = nn.Linear(hidden_size * 2, vocab_size)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.max_pred_len = max_pred_len
        self.start_token = start_token
        self.end_token = end_token

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        encoder的前向传播
        1. x送入embedding层获得词向量
        2. 词向量送入LSTM输出特征

        输入:
        x: [batch_size, num_steps_x]

        返回:(o, h, c)
        o: [batch_size, num_steps_x, hidden_size]
        h: [1, batch_size, hidden_size]
        c: [1, batch_size, hidden_size]
        """
        # embedded.shape [batch_size, num_steps_x, embed_size]
        embedded = self.enc_embeddings(x)
        # h0.shape [1, batch_size, hidden_size]
        # c0.shape [1, batch_size, hidden_size]
        (h0, c0) = (torch.zeros(1, x.shape[0], self.hidden_size),
                    torch.zeros(1, x.shape[0], self.hidden_size))
        # o.shape [batch_size, num_steps_x, hidden_size]
        # h.shape [1, batch_size, hidden_size]
        # c.shape [1, batch_size, hidden_size]
        o, (h, c) = self.encoder(embedded, (h0, c0))
        return o, h, c

    def inference(self, x: Tensor, return_att_weights: bool = False) -> Tensor:
        """
        `预测模式`的前向传播
        1. Encoder前向传播
        2. Decoder前向传播

        输入:
        x: batch_size, num_steps_x]
        return_att_weights: 是否返回注意力权重. 如果返回注意力权重, 每预测一个token, 就会记录一个注意力权重
            True - 返回att_weights
            False - 返回output

        返回: output或者att_weights
        output: [batch_size, max_pred_len]
        att_weights: [batch_size, max_pred_len, num_steps_x]
        """
        self.eval()
        # o.shape [batch_size, num_steps_x, hidden_size]
        # h.shape [1, batch_size, hidden_size]
        # c.shape [1, batch_size, hidden_size]
        o, hx, cx = self.encode(x)
        # hx.shape: [batch_size, hidden_size]
        # cx.shape: [batch_size, hidden_size]
        hx, cx = hx[0], cx[0]

        # start.shape [batch_size, 1]
        # 内容填充为<BOS>
        start = torch.ones(x.shape[0], 1)
        start[:, 0] = torch.tensor(self.start_token)
        start = start.type(torch.LongTensor)

        # dec_emb_in.shape [batch_size, 1, embed_size]
        dec_emb_in = self.dec_embeddings(start)
        # dec_emb_in.shape [1, batch_size, embed_size]
        dec_emb_in = dec_emb_in.permute(1, 0, 2)
        # dec_in.shape [batch_size, embed_size]
        dec_in = dec_emb_in[0]
        output = []
        att_weights = []  # list of [batch_size, num_steps_x]
        for i in range(self.max_pred_len):
            # 计算注意力(key=hx, querie=o, value=o):
            # hx的形状处理成:    [batch_size, hidden_size]
            #               -> [batch_size, 1, hidden_size]
            # 经过self.atten -> [batch_size, 1, hidden_size]
            #
            # o的形状处理成:     [batch_size, num_steps_x, hidden_size]
            #               -> [batch_size, hidden_size, num_steps_x]
            # attn_prod.shape(注意力分数) [batch_size, 1, num_steps_x]
            attn_prod = torch.matmul(self.attn(hx.unsqueeze(1)),
                                     o.permute(0, 2, 1))

            # att_weight.shape(注意力权重) [batch_size, 1, num_steps_x]
            att_weight = nn.functional.softmax(attn_prod, dim=2)

            # 记录注意力权重: [batch_size, num_steps_x]
            att_weights.append(att_weight.squeeze())

            # att_weight.shape(注意力权重) [batch_size, 1, num_steps_x]
            # o.shape [batch_size, num_steps_x, hidden_size]
            # context.shape [batch_size, 1, hidden_size]
            context = torch.matmul(att_weight, o)

            # dec_in.shape [batch_size, embed_size]
            # hx.shape [batch_size, hidden_size]
            # cx.shape [batch_size, hidden_size]
            hx, cx = self.decoder_cell(dec_in, (hx, cx))

            # context的形状处理成 [batch_size, 1, hidden_size]
            #                -> [batch_size, hidden_size]
            # hx.shape [batch_size, hidden_size]
            # hc.shape [batch_size, hidden_size * 2]
            hc = torch.cat([context.squeeze(1), hx], dim=1)
            # result.shape [batch_size, vocab_size]
            result = self.decoder_dense(hc)
            # result.shape [batch_size, 1]
            result = result.argmax(dim=1).view(-1, 1)
            # 预测出来的词作为下一次预测的输入:
            # dec_in.shape [batch_size, 1, embed_size]
            #           -> [1, batch_size, embed_size]
            #           -> [batch_size, embed_size]
            dec_in = self.dec_embeddings(result).permute(1, 0, 2)[0]
            output.append(result)
        # output.shape [max_pred_len, batch_size, 1]
        output = torch.stack(output, dim=0)

        if return_att_weights:
            # att_weights.shape [max_pred_len, batch_size, num_steps_x]
            #                -> [batch_size, max_pred_len, num_steps_x]
            return torch.stack(att_weights, dim=0).permute(1, 0, 2)
        else:
            # output.shape [max_pred_len, batch_size, 1]
            #           -> [batch_size, max_pred_len, 1)
            #           -> [batch_size, max_pred_len]
            return output.permute(1, 0, 2).view(-1, self.max_pred_len)

    def train_logit(self, x: Tensor, y: Tensor) -> Tensor:
        """
        `训练模式`的前向传播
        1. Encoder前向传播
        2. Decoder前向传播

        输入:
        x: [batch_size, num_steps_x]
        y: [batch_size, num_steps_y]

        返回:
        output: [batch_size, num_steps_y-1, vocab_size]
        """
        # o.shape [batch_size, num_steps_x, hidden_size]
        # h.shape [1, batch_size, hidden_size]
        # c.shape [1, batch_size, hidden_size]
        o, hx, cx = self.encode(x)
        # hx.shape: [batch_size, hidden_size]
        # cx.shape: [batch_size, hidden_size]
        hx, cx = hx[0], cx[0]

        # dec_in.shape [batch_size, num_steps_y-1], 去掉了y中最后一个token: <EOS>
        dec_in = y[:, :-1]
        # dec_emb_in.shape [batch_size, num_steps_y-1, embed_size]
        dec_emb_in = self.dec_embeddings(dec_in)
        # dec_emb_in.shape [num_steps_y-1, batch_size, embed_size]
        dec_emb_in = dec_emb_in.permute(1, 0, 2)
        output = []
        # 遍历dec_emb_in作为输入:
        for i in range(dec_emb_in.shape[0]):
            # 计算注意力(key=hx, querie=o, value=o):
            # hx的形状处理成:    [batch_size, hidden_size]
            #               -> [batch_size, 1, hidden_size]
            # 经过self.atten -> [batch_size, 1, hidden_size]
            #
            # o的形状处理成:     [batch_size, num_steps_x, hidden_size]
            #               -> [batch_size, hidden_size, num_steps_x]
            # attn_prod.shape(注意力分数) [batch_size, 1, num_steps_x]
            attn_prod = torch.matmul(self.attn(hx.unsqueeze(1)),
                                     o.permute(0, 2, 1))

            # att_weight.shape(注意力权重) [batch_size, 1, num_steps_x]
            att_weight = nn.functional.softmax(attn_prod, dim=2)
            # att_weight.shape(注意力权重) [batch_size, 1, num_steps_x]
            # o.shape [batch_size, num_steps_x, hidden_size]
            # context.shape [batch_size, 1, hidden_size]
            context = torch.matmul(att_weight, o)

            # dec_emb_in[i].shape [batch_size, embed_size]
            # hx.shape [batch_size, hidden_size]
            # cx.shape [batch_size, hidden_size]
            hx, cx = self.decoder_cell(dec_emb_in[i], (hx, cx))

            # context的形状处理成 [batch_size, 1, hidden_size]
            #                -> [batch_size, hidden_size]
            # hx.shape [batch_size, hidden_size]
            # hc.shape [batch_size, hidden_size * 2]
            hc = torch.cat([context.squeeze(1), hx], dim=1)
            # result.shape [batch_size, vocab_size]
            result = self.decoder_dense(hc)
            output.append(result)
        # output.shape [num_steps_y-1, batch_size, vocab_size]
        output = torch.stack(output, dim=0)
        # output.shape [batch_size, num_steps_y-1, vocab_size]
        return output.permute(1, 0, 2)

    def step(self, x: Tensor, y: Tensor) -> np.ndarray:
        """
        训练一个批量的数据

        输入:
        x: [batch_size, num_steps_x]
        y: [batch_size, num_steps_y]
        """
        self.optimizer.zero_grad()
        # logit.shape [batch_size, num_steps_y-1, vocab_size]
        logit = self.train_logit(x, y)
        # dec_out.shape [batch_size, num_steps_y-1]
        dec_out = y[:, 1:]
        # 将logit的形状处理成: [batch_size*num_steps_y-1, vocab_size]
        # 将dec_out的形状处理成: [batch_size*num_steps_y-1,]
        # loss: 是一个标量
        loss = nn.functional.cross_entropy(logit.reshape(-1, self.vocab_size),
                                           dec_out.reshape(-1))
        loss.backward()
        self.optimizer.step()
        return loss.detach().numpy()


def train(num_epochs: int = 100) -> Tuple[DateData, nn.Module]:
    data_iter, dataset = load_date()
    print("Chinese time order: yy/mm/dd ", dataset.date_cn[:3],
          "\nEnglish time order: dd/M/yyyy", dataset.date_en[:3])
    print("Vocabularies: ", dataset.vocab)
    print(
        f"x index sample:  \n{dataset.idx2str(dataset.x[0])}\n{dataset.x[0]}",
        f"\ny index sample:  \n{dataset.idx2str(dataset.y[0])}\n{dataset.y[0]}"
    )
    model = Seq2SeqAttention(dataset.num_word,
                             embed_size=16,
                             hidden_size=32,
                             max_pred_len=11,
                             start_token=dataset.start_token,
                             end_token=dataset.end_token)
    for epoch in range(num_epochs):
        data_iter_tqdm = tqdm(data_iter, file=sys.stdout)
        for i, batch in enumerate(data_iter_tqdm):
            bx, by, decoder_len = batch
            bx = bx.type(torch.LongTensor)
            by = by.type(torch.LongTensor)
            loss = model.step(bx, by)
            if i % 70 == 0:
                target = dataset.idx2str(
                    by[0, 1:-1].data.numpy())  # 去掉批量中第一个样本:头(<BOS>)尾(<EOS>)
                pred = model.inference(bx[0:1])  # 预测批量中的一个样本
                res = dataset.idx2str(pred[0].data.numpy())
                src = dataset.idx2str(bx[0].data.numpy())
                data_iter_tqdm.desc = f'epoch {epoch}, step {i}, loss {loss:.3f}, ' \
                    f'input {src}, target {target}, inference {res}'
    return dataset, model


def seq2seq_attention(dataset: DateData, model: nn.Module) -> None:
    batch_size = 6
    # x.shape [batch_size=6, num_steps_x=8]
    # y.shape [batch_size=6, num_steps_y=11]
    # att_weights.shape [batch_size=6, num_steps_y=11, num_steps_x=8]
    id_to_token, x, y, att_weights = dataset.id_to_token, dataset.x[
        0:batch_size], dataset.y[0:batch_size], model.inference(
            torch.from_numpy(dataset.x[0:batch_size]),
            return_att_weights=True).detach().numpy()
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
    for i in range(batch_size):
        plt.subplot(2, 3, i + 1)
        x_vocab = [id_to_token[j] for j in np.ravel(x[i])]
        y_vocab = [id_to_token[j] for j in y[i, 1:]]  # 去掉批量中第i个样本:头(<BOS>)
        plt.imshow(att_weights[i], cmap="Reds", vmin=0., vmax=1.)
        plt.yticks([j for j in range(len(y_vocab))], y_vocab)
        plt.xticks([j for j in range(len(x_vocab))], x_vocab)
        if i == 0 or i == 3:
            plt.ylabel("Output")
        if i >= 3:
            plt.xlabel("Input")
    plt.show()


if __name__ == '__main__':
    dataset, model = train()
    seq2seq_attention(dataset, model)
# Chinese time order: yy/mm/dd  ['31-04-26', '04-07-18', '33-06-06']
# English time order: dd/M/yyyy ['26/Apr/2031', '18/Jul/2004', '06/Jun/2033']
# Vocabularies:  {'-', '/', '8', '9', 'Apr', 'Jan', '4', '5', '6', '7', '0',
#                 '1', '2', '3', 'Feb', '<BOS>', 'Aug', 'Dec', '<PAD>', '<EOS>',
#                 'Oct', 'Nov', 'Mar', 'Jun', 'Sep', 'Jul', 'May'}
# x index sample:
# 31-04-26
# [6 4 1 3 7 1 5 9]
# y index sample:
# <BOS>26/Apr/2031<EOS>
# [13  5  9  2 15  2  5  3  6  4 14]
# epoch 0, step 70, loss 2.366, input 20-02-18, target 18/Feb/2020, inference 1////////20: 100%|███████████| 125/125 [00:00<00:00, 190.74it/s]
# epoch 1, step 70, loss 1.280, input 24-09-16, target 16/Sep/2024, inference 1///2002<EOS>: 100%|█████████| 125/125 [00:00<00:00, 199.03it/s]
# epoch 2, step 70, loss 1.090, input 23-11-22, target 22/Nov/2023, inference 22/Mar/2002<EOS>: 100%|███████| 125/125 [00:01<00:00, 76.39it/s]
# epoch 3, step 70, loss 0.998, input 08-04-29, target 29/Apr/2008, inference 22/Mar/2020<EOS>: 100%|██████| 125/125 [00:00<00:00, 178.23it/s]
# epoch 4, step 70, loss 0.914, input 12-12-16, target 16/Dec/2012, inference 17/Mar/2019<EOS>: 100%|██████| 125/125 [00:00<00:00, 200.89it/s]
# epoch 5, step 70, loss 0.845, input 29-10-28, target 28/Oct/2029, inference 28/Jul/2022<EOS>: 100%|██████| 125/125 [00:00<00:00, 200.42it/s]
# epoch 6, step 70, loss 0.761, input 00-05-15, target 15/May/2000, inference 14/May/2000<EOS>: 100%|██████| 125/125 [00:00<00:00, 198.64it/s]
# epoch 7, step 70, loss 0.672, input 94-08-22, target 22/Aug/1994, inference 22/Jul/1995<EOS>: 100%|██████| 125/125 [00:00<00:00, 191.85it/s]
# epoch 8, step 70, loss 0.619, input 33-06-08, target 08/Jun/2033, inference 08/Mar/2023<EOS>: 100%|██████| 125/125 [00:01<00:00, 102.34it/s]
# epoch 9, step 70, loss 0.557, input 26-04-04, target 04/Apr/2026, inference 04/Aug/2024<EOS>: 100%|██████| 125/125 [00:00<00:00, 171.14it/s]
# ...
# epoch 90, step 70, loss 0.000, input 02-08-24, target 24/Aug/2002, inference 24/Aug/2002<EOS>: 100%|█████| 125/125 [00:00<00:00, 202.78it/s]
# epoch 91, step 70, loss 0.000, input 96-08-20, target 20/Aug/1996, inference 20/Aug/1996<EOS>: 100%|█████| 125/125 [00:00<00:00, 201.55it/s]
# epoch 92, step 70, loss 0.000, input 23-09-19, target 19/Sep/2023, inference 19/Sep/2023<EOS>: 100%|█████| 125/125 [00:00<00:00, 202.10it/s]
# epoch 93, step 70, loss 0.000, input 01-02-19, target 19/Feb/2001, inference 19/Feb/2001<EOS>: 100%|█████| 125/125 [00:00<00:00, 201.68it/s]
# epoch 94, step 70, loss 0.000, input 21-03-31, target 31/Mar/2021, inference 31/Mar/2021<EOS>: 100%|█████| 125/125 [00:00<00:00, 200.67it/s]
# epoch 95, step 70, loss 0.000, input 24-03-31, target 31/Mar/2024, inference 31/Mar/2024<EOS>: 100%|█████| 125/125 [00:00<00:00, 200.66it/s]
# epoch 96, step 70, loss 0.000, input 16-12-20, target 20/Dec/2016, inference 20/Dec/2016<EOS>: 100%|█████| 125/125 [00:00<00:00, 201.75it/s]
# epoch 97, step 70, loss 0.000, input 96-01-03, target 03/Jan/1996, inference 03/Jan/1996<EOS>: 100%|█████| 125/125 [00:00<00:00, 189.59it/s]
# epoch 98, step 70, loss 0.000, input 09-01-17, target 17/Jan/2009, inference 17/Jan/2009<EOS>: 100%|█████| 125/125 [00:00<00:00, 195.88it/s]
# epoch 99, step 70, loss 0.000, input 20-11-19, target 19/Nov/2020, inference 19/Nov/2020<EOS>: 100%|█████| 125/125 [00:00<00:00, 193.42it/s]
