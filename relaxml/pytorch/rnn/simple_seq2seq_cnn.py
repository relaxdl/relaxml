from typing import List, Tuple
import datetime
import sys
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
"""
Simple seq2seq(CNN)

实现说明:
https://tech.foxrelax.com/rnn/simple_seq2seq_cnn/
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


class CNNTranslation(nn.Module):
    """
    Encoder-Decoder
    1. Encoder用CNN + MaxPool实现
    2. Decoder用LSTMCell实现
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
        # 卷积核1: [2, embed_size=16]
        # 卷积核2: [3, embed_size=16]
        # 卷积核3: [4, embed_size=16]
        self.conv2ds = [
            nn.Conv2d(1, 16, (n, embed_size), padding=0) for n in range(2, 5)
        ]
        # max pool kernel_size: [7, 1]
        # max pool kernel_size: [6, 1]
        # max pool kernel_size: [5, 1]
        self.max_pools = [nn.MaxPool2d((n, 1)) for n in [7, 6, 5]]
        self.encoder = nn.Linear(16 * 3, hidden_size)

        # decoder
        self.dec_embeddings = nn.Embedding(vocab_size, embed_size)
        self.dec_embeddings.weight.data.normal_(0, 0.1)
        self.decoder_cell = nn.LSTMCell(embed_size, hidden_size)
        self.decoder_dense = nn.Linear(hidden_size, vocab_size)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.max_pred_len = max_pred_len
        self.start_token = start_token
        self.end_token = end_token

    def encode(self, x: Tensor) -> List[Tensor]:
        """
        encoder的前向传播

        输入:
        x: [batch_size, num_steps_x=8]
            04-07-18 -> ['0', '4', '-', '0', '7', '-', '1', '8'] 
                     -> [3, 7, 1, 3, 10, 1, 4, 11]
        返回:(h, h)
        h: [batch_size, hidden_size]
        """
        # embedded.shape [batch_size, num_steps_x=8, embed_size=16]
        embedded = self.enc_embeddings(x)

        # o.shape [batch_size, channel=1, num_steps_x=8, embed_size=16]
        # 类比成图片: [batch_size, channel, height, width], 就可以应用conv2d来处理了
        o = torch.unsqueeze(embedded, 1)

        # co的形状:
        # [[batch_size, 16, 7, 1], - 卷积核1: [2, embed_size=16]
        #  [batch_size, 16, 6, 1], - 卷积核2: [3, embed_size=16]
        #  [batch_size, 16, 5, 1]] - 卷积核3: [4, embed_size=16]
        co = [nn.functional.relu(conv2d(o)) for conv2d in self.conv2ds]

        # co的形状:
        # [[batch_size, 16, 1, 1], - max pool kernel_size: [7, 1]
        #  [batch_size, 16, 1, 1], - max pool kernel_size: [6, 1]
        #  [batch_size, 16, 1, 1]] - max pool kernel_size: [5, 1]
        co = [self.max_pools[i](co[i]) for i in range(len(co))]

        # co的形状:
        # [[batch_size, 16],
        #  [batch_size, 16],
        #  [batch_size, 16]]
        co = [torch.squeeze(torch.squeeze(c, dim=3), dim=2) for c in co]

        # o.shape [batch_size, 16x3]
        o = torch.cat(co, dim=1)

        # h.shape [batch_size, hidden_size]
        h = self.encoder(o)
        return [h, h]

    def inference(self, x: Tensor) -> Tensor:
        """
        `预测模式`的前向传播

        1. Encoder前向传播
        2. Decoder前向传播

        输入:
        x: [batch_size, num_steps_x]

        返回:
        output: [batch_size, max_pred_len]
        """
        self.eval()
        # hx.shape [batch_size, hidden_size]
        # cx.shape [batch_size, hidden_size]
        hx, cx = self.encode(x)

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
        for i in range(self.max_pred_len):
            # dec_in.shape [batch_size, embed_size]
            # hx.shape [batch_size, hidden_size]
            # cx.shape [batch_size, hidden_size]
            hx, cx = self.decoder_cell(dec_in, (hx, cx))
            # o.shape [batch_size, vocab_size]
            o = self.decoder_dense(hx)
            # o.shape [batch_size, 1]
            # 直接转换为token_id
            o = o.argmax(dim=1).view(-1, 1)

            # 预测出来的词作为下一次预测的输入:
            # dec_in.shape  [batch_size, 1, embed_size]
            #            -> [1, batch_size, embed_size]
            #            -> [batch_size, embed_size]
            dec_in = self.dec_embeddings(o).permute(1, 0, 2)[0]
            output.append(o)
        # output.shape [max_pred_len, batch_size, 1]
        output = torch.stack(output, dim=0)

        # output.shape [max_pred_len, batch_size, 1]
        #           -> [batch_size, max_pred_len, 1]
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

        # hx.shape [batch_size, hidden_size]
        # cx.shape [batch_size, hidden_size]
        hx, cx = self.encode(x)

        # dec_in.shape [batch_size, num_steps_y-1], 去掉了y中最后一个token: <EOS>
        dec_in = y[:, :-1]
        # dec_emb_in.shape [batch_size, num_steps_y-1, embed_size]
        dec_emb_in = self.dec_embeddings(dec_in)
        # dec_emb_in.shape [num_steps_y-1, batch_size, embed_size]
        dec_emb_in = dec_emb_in.permute(1, 0, 2)
        output = []
        # 遍历dec_emb_in作为输入:
        for i in range(dec_emb_in.shape[0]):
            # dec_emb_in[i].shape [batch_size, embed_size]
            # hx.shape [batch_size, hidden_size]
            # cx.shape [batch_size, hidden_size]
            hx, cx = self.decoder_cell(dec_emb_in[i], (hx, cx))
            # o.shape [batch_size, vocab_size]
            o = self.decoder_dense(hx)
            output.append(o)
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


def train(num_epochs: int = 100) -> None:
    data_iter, dataset = load_date()
    print("Chinese time order: yy/mm/dd ", dataset.date_cn[:3],
          "\nEnglish time order: dd/M/yyyy", dataset.date_en[:3])
    print("Vocabularies: ", dataset.vocab)
    print(
        f"x index sample:  \n{dataset.idx2str(dataset.x[0])}\n{dataset.x[0]}",
        f"\ny index sample:  \n{dataset.idx2str(dataset.y[0])}\n{dataset.y[0]}"
    )
    model = CNNTranslation(dataset.num_word,
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


if __name__ == '__main__':
    train()
# Chinese time order: yy/mm/dd  ['31-04-26', '04-07-18', '33-06-06']
# English time order: dd/M/yyyy ['26/Apr/2031', '18/Jul/2004', '06/Jun/2033']
# Vocabularies:  {'Nov', '<BOS>', 'Apr', 'Jan', 'Oct', 'May', '<PAD>', 'Mar',
#                 '8', '9', '2', '3', '0', '1', '6', '7', '4', '5', 'Sep', '/',
#                 '-', 'Aug', '<EOS>', 'Feb', 'Dec', 'Jul', 'Jun'}
# x index sample:
# 31-04-26
# [6 4 1 3 7 1 5 9]
# y index sample:
# <BOS>26/Apr/2031<EOS>
# [13  5  9  2 15  2  5  3  6  4 14]
# epoch 0, step 70, loss 2.561, input 97-05-27, target 27/May/1997, inference ///////////: 100%|███████████████| 125/125 [00:00<00:00, 357.93it/s]
# epoch 1, step 70, loss 1.989, input 09-02-05, target 05/Feb/2009, inference 1////200<EOS>: 100%|█████████████| 125/125 [00:00<00:00, 390.56it/s]
# epoch 2, step 70, loss 1.405, input 15-05-17, target 17/May/2015, inference 1//20000<EOS>: 100%|█████████████| 125/125 [00:00<00:00, 392.70it/s]
# epoch 3, step 70, loss 1.177, input 85-02-22, target 22/Feb/1985, inference 20//20001<EOS>: 100%|████████████| 125/125 [00:00<00:00, 391.58it/s]
# epoch 4, step 70, loss 1.055, input 06-04-27, target 27/Apr/2006, inference 21/Jan/2008<EOS>: 100%|██████████| 125/125 [00:00<00:00, 391.14it/s]
# epoch 5, step 70, loss 0.982, input 77-01-17, target 17/Jan/1977, inference 11/Mar/2001<EOS>: 100%|██████████| 125/125 [00:00<00:00, 393.09it/s]
# epoch 6, step 70, loss 0.962, input 86-12-29, target 29/Dec/1986, inference 29/Mar/2003<EOS>: 100%|██████████| 125/125 [00:00<00:00, 382.47it/s]
# epoch 7, step 70, loss 0.914, input 10-01-29, target 29/Jan/2010, inference 29/Dec/2020<EOS>: 100%|██████████| 125/125 [00:00<00:00, 314.46it/s]
# epoch 8, step 70, loss 0.868, input 31-09-19, target 19/Sep/2031, inference 19/Mar/2001<EOS>: 100%|██████████| 125/125 [00:00<00:00, 145.49it/s]
# epoch 9, step 70, loss 0.850, input 99-07-13, target 13/Jul/1999, inference 11/Jan/2001<EOS>: 100%|██████████| 125/125 [00:00<00:00, 301.98it/s]
# ...
# epoch 91, step 70, loss 0.073, input 89-03-19, target 19/Mar/1989, inference 19/Mar/1989<EOS>: 100%|█████████| 125/125 [00:00<00:00, 378.66it/s]
# epoch 92, step 70, loss 0.076, input 33-01-28, target 28/Jan/2033, inference 28/Jan/2033<EOS>: 100%|█████████| 125/125 [00:00<00:00, 144.09it/s]
# epoch 93, step 70, loss 0.042, input 99-09-29, target 29/Sep/1999, inference 29/Sep/1999<EOS>: 100%|█████████| 125/125 [00:00<00:00, 263.58it/s]
# epoch 94, step 70, loss 0.075, input 10-10-18, target 18/Oct/2010, inference 18/Oct/2010<EOS>: 100%|█████████| 125/125 [00:00<00:00, 382.13it/s]
# epoch 95, step 70, loss 0.046, input 21-11-19, target 19/Nov/2021, inference 19/Nov/2021<EOS>: 100%|█████████| 125/125 [00:00<00:00, 378.17it/s]
# epoch 96, step 70, loss 0.050, input 29-08-18, target 18/Aug/2029, inference 18/Aug/2029<EOS>: 100%|█████████| 125/125 [00:00<00:00, 385.65it/s]
# epoch 97, step 70, loss 0.069, input 30-07-28, target 28/Jul/2030, inference 28/Jul/2030<EOS>: 100%|█████████| 125/125 [00:00<00:00, 384.31it/s]
# epoch 98, step 70, loss 0.073, input 31-03-11, target 11/Mar/2031, inference 11/Mar/2031<EOS>: 100%|█████████| 125/125 [00:00<00:00, 370.75it/s]
# epoch 99, step 70, loss 0.047, input 19-01-31, target 31/Jan/2019, inference 31/Jan/2019<EOS>: 100%|█████████| 125/125 [00:00<00:00, 372.18it/s]