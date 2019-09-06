import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMPR(nn.Module):
    """LSTM for Punctuation Restoration
    """

    def __init__(self,
                 vocab_size,
                 embedding_size,
                 hidden_size,
                 num_layers,
                 num_class):
        super(LSTMPR, self).__init__()
        # hyper parameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_class = num_class

        # 网络中的层
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size*2, num_class)
        self.init_weights()

    def init_weights(self, init_range=0.1):
        """Init weights

        Parameters
        ----------
        init_range : float
            set range of weight, by default 0.1
        """
        for p in self.parameters():
            if p.dim() > 1:
                p.data.uniform_(-init_range, init_range)
            else:
                p.data.fill_(0)

    def init_hidden(self, batch_size):
        h = Variable(torch.zero(self.num_layers*1, batch_size, self.hidden_size).cuda())
        c = Variable(torch.zero(self.num_layers*1, batch_size, self.hidden_size).cuda())
        # h表示隐藏层的保存空间，c是细胞状态的保存空间
        return h, c

    def forward(self, inputs, inputs_lengths):
        """前向传递过程
        
        Parameters
        ----------
        inputs : tensor
            训练数据，padded补齐了的输入，批优先
        inputs_lengths : int
            分数：？？？
        """
        hidden = self.init_hidden(inputs.size(0))
        embedding = self.embedding(inputs)

        # embedding本身是同样长度的，用这个函数主要是为了用pack
        packed = pack_padded_sequence(embedding, inputs_lengths, batch_first=True)
        packed_outputs, hidden = self.lstm(packed, hidden)      # 输入pack，lstm默认输出pack
        out, out_lengths = pad_packed_sequence(packed_outputs, batch_first=True)
        out = out.contiguous()
        score = self.fc(out.view(out.size(0)*out.size(1), out.size(2)))
        return score.view(out.size(0), out.size(1), score.size(1))

        @staticmethod
        def serialize(model, optimizer, epoch):
            """存储模型的信息
            
            Parameters
            ----------
            model : nn.module
                模型
            optimizer : nn.optimizer
                优化器
            epoch : 轮
                迭代轮数
            """
            package = {
                        'state_dict': model.state_dict(),
                        'optim_dict': optimizer.state_dict(),
                        'epoch': epoch
                        }
            return package

