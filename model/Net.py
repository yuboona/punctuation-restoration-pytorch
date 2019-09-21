import torch
import torch.nn as nn
from torch.autograd import Variable


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
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True)
        # 因为不是双向网络，所以LSTM的输出的大小是hidden_size而不是hidden_size*2
        # self.fc = nn.Linear(hidden_size*2, num_class)
        self.fc = nn.Linear(hidden_size, num_class)
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
        h = Variable(torch.zeros(self.num_layers*1, batch_size, self.hidden_size))
        c = Variable(torch.zeros(self.num_layers*1, batch_size, self.hidden_size))
        # h表示隐藏层的保存空间，c是细胞状态的保存空间
        return h, c

    def forward(self, inputs):
        """前向传递过程

        Parameters
        ----------
        inputs : tensor
            训练数据，padded补齐了的输入，批优先
        """
        hidden = self.init_hidden(inputs.size(0))
        embedding = self.embedding(inputs)

        # embedding本身是同样长度的，用这个函数主要是为了用pack
        # packed = pack_sequence(embedding, inputs_lengths, batch_first=True)
        outputs, hidden = self.lstm(embedding, hidden)      # 输入pack，lstm默认输出pack
        outputs = outputs.contiguous()
        print(outputs.size())
        score = self.fc(outputs.view(outputs.size(0)*outputs.size(1), outputs.size(2)))
        return score.view(outputs.size(0), outputs.size(1), score.size(1))

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
